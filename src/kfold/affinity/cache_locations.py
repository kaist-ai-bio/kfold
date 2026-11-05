"""One logical cache catalog with quota-aware, sticky physical placement.

Reservations are conservative: live usage plus the full outstanding reservation
may double-count bytes already being written. No reservation is silently freed
after a failed worker, and no feature or cache file is deleted here.
"""
import json
from pathlib import Path
import sqlite3

from .conversion import file_sha

RESERVE_BYTES = 3_500_000_000_000  # 3 TB group reserve + 0.5 TB concurrent writes
RESERVE_FILES = 1000


class CacheLocations:
    def __init__(self, catalog, roots, *, purpose):
        if purpose not in ('integration_pilot', 'training_cache'):
            raise ValueError('Explicit catalog purpose required')
        self.purpose = purpose
        if list(roots) != ['/appsdata', '/scratch']:
            raise ValueError('Explicit appsdata-first and scratch-overflow roots required')
        self.roots = {mount: str(Path(root).resolve()) for mount, root in roots.items()}
        for mount, root in self.roots.items():
            if not Path(root).is_relative_to(mount) or Path(root) == Path(mount):
                raise ValueError('Cache root must be below its declared filesystem')
        self.path = Path(catalog)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.executescript('''PRAGMA journal_mode=DELETE;
                CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT);
                CREATE TABLE IF NOT EXISTS locations(
                    source_sha TEXT PRIMARY KEY,plan_sha TEXT NOT NULL,mount TEXT NOT NULL,
                    destination TEXT UNIQUE NOT NULL,byte_bound INTEGER NOT NULL,
                    file_bound INTEGER NOT NULL,state TEXT NOT NULL,
                    manifest_path TEXT,manifest_sha TEXT);''')
            settings = json.dumps(dict(roots=self.roots,purpose=purpose), sort_keys=True)
            db.execute('INSERT OR IGNORE INTO metadata VALUES (?,?)', ('roots', settings))
            if db.execute("SELECT value FROM metadata WHERE key='roots'").fetchone()[0] != settings:
                raise ValueError('Catalog roots cannot change on resume')

    def _db(self):
        db = sqlite3.connect(self.path, timeout=60)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA synchronous=FULL')
        return db

    @staticmethod
    def _sha(value):
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Expected lowercase SHA256 identity')

    def reserve(self, source_sha, plan_sha, *, byte_bound, file_bound, quota_provider, future_writes):
        """Reserve once, including projected remaining raw and working bytes.

        quota_provider must query live group quotas and fail on unknown results.
        future_writes covers BOTH filesystems and must not omit remaining raw
        production or working-space estimates. Sticky resume prevents the same
        source shard from acquiring a second physical copy after quota changes.
        """
        self._sha(source_sha); self._sha(plan_sha)
        if type(byte_bound) is not int or byte_bound <= 0 or type(file_bound) is not int or file_bound <= 0:
            raise ValueError('Positive byte/file bounds required')
        if set(future_writes) != set(self.roots) or any(type(n) is not int or n < 0 for n in future_writes.values()):
            raise ValueError('Explicit nonnegative future-write forecast required per filesystem')
        with self._db() as db:
            db.execute('BEGIN IMMEDIATE')
            existing = db.execute('SELECT * FROM locations WHERE source_sha=?', (source_sha,)).fetchone()
            if existing and (existing['plan_sha'] != plan_sha or existing['byte_bound'] != byte_bound or existing['file_bound'] != file_bound):
                raise ValueError('Source plan or reservation bound changed')
            if existing and existing['state'] == 'complete':
                path = Path(existing['manifest_path'])
                if not path.exists() or file_sha(path) != existing['manifest_sha']:
                    raise ValueError('Completed location manifest changed')
                return dict(existing)
            live = quota_provider()
            for mount in self.roots:
                if (mount not in live or type(live[mount].get('free_bytes')) is not int or
                        type(live[mount].get('free_files')) is not int):
                    raise ValueError('Live group quota unavailable')
                if live[mount]['free_bytes'] < RESERVE_BYTES or live[mount]['free_files'] < RESERVE_FILES:
                    raise RuntimeError('Group quota reserve reached')
                outstanding = db.execute('SELECT coalesce(sum(byte_bound),0) FROM locations WHERE state=? AND mount=? AND source_sha!=?',
                    ('reserved',mount,source_sha)).fetchone()[0]
                if live[mount]['free_bytes']-outstanding-future_writes[mount] < RESERVE_BYTES:
                    raise RuntimeError('Projected quota insufficient on one filesystem')
            choices = [existing['mount']] if existing else list(self.roots)
            for mount in choices:
                reserved = db.execute('SELECT coalesce(sum(byte_bound),0),coalesce(sum(file_bound),0) FROM locations WHERE state=? AND mount=? AND source_sha!=?', ('reserved',mount,source_sha)).fetchone()
                available = live[mount]['free_bytes'] - reserved[0] - future_writes[mount]
                free_files = live[mount]['free_files'] - reserved[1]
                if available-byte_bound < RESERVE_BYTES or free_files-file_bound < RESERVE_FILES:
                    continue
                if not existing:
                    destination = str(Path(self.roots[mount])/source_sha)
                    db.execute('INSERT INTO locations VALUES (?,?,?,?,?,?,?,NULL,NULL)',
                        (source_sha,plan_sha,mount,destination,byte_bound,file_bound,'reserved'))
                return dict(db.execute('SELECT * FROM locations WHERE source_sha=?', (source_sha,)).fetchone())
            raise RuntimeError('Projected quota insufficient; preserve current files and request capacity decision')

    def finish(self, source_sha, plan_sha):
        """Link a fully verified converter manifest; never trust an exit code."""
        self._sha(source_sha); self._sha(plan_sha)
        with self._db() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM locations WHERE source_sha=?', (source_sha,)).fetchone()
            if row is None or row['plan_sha'] != plan_sha:
                raise ValueError('Missing matching reservation')
            root = Path(row['destination'])
            path = root/'complete.json'
            result = json.loads(path.read_text())
            if result.get('purpose') != self.purpose or result.get('training_ready') != (self.purpose == 'training_cache'):
                raise ValueError('Pilot and training cache catalogs must remain separate')
            if result['source_sha256'] != source_sha or result['plan_sha256'] != plan_sha:
                raise ValueError('Converter source/plan mismatch')
            if not all(result.get(k) is True for k in ('retained_tensors_verified','readback_verified','head_loader_verified')):
                raise ValueError('Incomplete conversion validation')
            if result['source_deleted'] is not False:
                raise ValueError('Source preservation required')
            if result['bytes'] > row['byte_bound'] or len(result['system_ids']) != result['records'] or len(set(result['system_ids'])) != result['records']:
                raise ValueError('Conversion size or coverage mismatch')
            paths = set(); count = 0; total_bytes = 0
            for shard in result['shards']:
                physical = Path(shard['path']).resolve()
                if not physical.is_relative_to(root) or physical in paths:
                    raise ValueError('Misplaced or duplicate cache shard')
                paths.add(physical)
                if physical.stat().st_size != shard['bytes'] or file_sha(physical) != shard['sha256'] or shard['contract_id'] != plan_sha:
                    raise ValueError('Compact shard checksum or contract mismatch')
                count += shard['records']; total_bytes += shard['bytes']
            if count != result['records'] or total_bytes != result['bytes']:
                raise ValueError('Shard receipt coverage mismatch')
            if sum(1 for p in root.rglob('*') if p.is_file()) > row['file_bound']:
                raise ValueError('Conversion file bound exceeded')
            db.execute('UPDATE locations SET state=?,manifest_path=?,manifest_sha=? WHERE source_sha=?',
                ('complete',str(path),file_sha(path),source_sha))
            return dict(db.execute('SELECT * FROM locations WHERE source_sha=?', (source_sha,)).fetchone())
