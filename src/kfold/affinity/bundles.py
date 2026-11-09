"""Large immutable SQLite bundles, embedded plans, and a record-level catalog.

Storage is independent of tensor serialization. One writer owns each bundle;
only sealed, verified bundles are visible to readers. Quota is an admission
check, not a record operation. The admission budget is deliberately never
replenished during a batch. Shared-group writes can still cause EDQUOT/ENOSPC.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def readonly(path):
    return sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro&immutable=1', uri=True)


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class BatchBudget:
    """One live snapshot, static reservations for DB + rollback journal + metadata.

    The caller holds the campaign lock for the entire batch. No completion
    releases budget. Each DB uses max_page_count; three times its maximum size plus
    16 MiB covers its rollback journal, headers and conservative filesystem overhead.
    Catalog/control space has a separate fixed allowance. No quota subprocess
    is ever invoked here. New admission or a resumed batch needs a NEW snapshot.
    """
    def __init__(self, snapshot, future_writes, *, reserve_bytes=3_500_000_000_000,
                 reserve_files=1000, control_bytes=32*1024**3, control_files=512):
        if set(snapshot) != set(future_writes) or not snapshot:
            raise ValueError('Both mount quotas and future writes required')
        self.remaining = {}
        self.reservations = []
        for mount, live in snapshot.items():
            vals = (live.get('free_bytes'), live.get('free_files'), future_writes[mount])
            if any(type(x) is not int or x < 0 for x in vals):
                raise ValueError('Unknown quota or negative future writes')
            self.remaining[mount] = [vals[0]-reserve_bytes-vals[2]-control_bytes,
                                     vals[1]-reserve_files-control_files]
            if min(self.remaining[mount]) < 0:
                raise RuntimeError('Insufficient batch admission reserve')

    def reserve(self, name, max_bytes, *, mount=None):
        if type(max_bytes) is not int or max_bytes < 1024**2:
            raise ValueError('Invalid bundle bound')
        if any(x['name'] == name for x in self.reservations):
            raise ValueError('Duplicate budget reservation')
        needed = 3*max_bytes+16*1024**2
        for candidate, remaining in self.remaining.items():
            if mount is not None and candidate != mount:
                continue
            if remaining[0] >= needed and remaining[1] >= 4:
                remaining[0] -= needed
                remaining[1] -= 4  # DB, per-bundle lock, journal, spare
                item = dict(name=name, mount=candidate, max_bytes=max_bytes,
                            reserved_bytes=needed, reserved_inodes=4)
                self.reservations.append(item)
                return item
        raise RuntimeError('Finite batch byte/file budget exhausted')


class BundleStore:
    """One bounded database for MANY source shards. No per-source sidecar files."""
    def __init__(self, path, contract, max_bytes):
        self.path = Path(path).resolve()
        self.partial = self.path.with_suffix('.sqlite.partial')
        self.contract = contract
        self.max_bytes = max_bytes
        self.db = None
        self.lock = None
        self.sealed = self.path.exists()
        if self.sealed and self.partial.exists():
            raise ValueError('Both sealed and partial bundle exist')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.sealed:
            self.db = readonly(self.path)
        else:
            self.lock = self.path.with_suffix('.lock').open('a+b')
            try:
                fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BaseException:
                self.lock.close()
                self.lock = None
                raise
            self.db = sqlite3.connect(self.partial)
            self.db.execute('PRAGMA journal_mode=DELETE')
            self.db.execute('PRAGMA synchronous=FULL')
            page = self.db.execute('PRAGMA page_size').fetchone()[0]
            cap = max_bytes//page
            if self.db.execute('PRAGMA page_count').fetchone()[0] > cap:
                self.close()
                raise ValueError('Existing bundle exceeds fixed bound')
            self.db.execute(f'PRAGMA max_page_count={cap}')
            self.db.executescript('''
                CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS sources(source_sha TEXT PRIMARY KEY,plan_sha TEXT NOT NULL,
                    plan TEXT NOT NULL,provenance TEXT NOT NULL,state TEXT NOT NULL,validation TEXT);
                CREATE TABLE IF NOT EXISTS records(system_id TEXT PRIMARY KEY,identity TEXT NOT NULL,
                    blob_sha TEXT NOT NULL,raw_sha TEXT NOT NULL,raw_bytes INTEGER NOT NULL,
                    payload BLOB NOT NULL,source_sha TEXT NOT NULL REFERENCES sources(source_sha));
                CREATE INDEX IF NOT EXISTS by_source ON records(source_sha);
            ''')
            self.db.execute('PRAGMA foreign_keys=ON')
            with self.db:
                self.db.execute('INSERT OR IGNORE INTO metadata VALUES (?,?)',
                                ('contract', canonical(contract)))
                self.db.execute('INSERT OR IGNORE INTO metadata VALUES (?,?)',
                                ('max_bytes', str(max_bytes)))
        try:
            if self.db.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
                raise ValueError('Bundle quick_check failed')
            if self.db.execute("SELECT value FROM metadata WHERE key='contract'").fetchone()[0] != canonical(contract):
                raise ValueError('Bundle contract changed on resume')
            if self.db.execute("SELECT value FROM metadata WHERE key='max_bytes'").fetchone()[0] != str(max_bytes):
                raise ValueError('Bundle bound changed on resume')
        except BaseException:
            self.close()
            raise

    def pin_source(self, plan, provenance):
        sha, psha = plan['source_sha256'], digest(plan)
        row = self.db.execute('SELECT plan_sha,provenance,state FROM sources WHERE source_sha=?', (sha,)).fetchone()
        if row:
            if row[:2] != (psha, canonical(provenance)):
                raise ValueError('Source plan or provenance changed on resume')
            return row[2] == 'complete'
        if self.sealed or self.db.execute("SELECT 1 FROM metadata WHERE key='sealed'").fetchone():
            raise ValueError('Cannot append to sealed bundle')
        with self.db:
            self.db.execute('INSERT INTO sources VALUES (?,?,?,?,?,NULL)',
                            (sha, psha, canonical(plan), canonical(provenance), 'planned'))
        return False

    @contextmanager
    def source_transaction(self, source_sha):
        if self.sealed:
            raise ValueError('Cannot write sealed bundle')
        self.db.execute('BEGIN IMMEDIATE')
        try:
            yield
            row = self.db.execute('SELECT state FROM sources WHERE source_sha=?', (source_sha,)).fetchone()
            if not row or row[0] != 'complete':
                raise ValueError('Source validation missing; transaction not publishable')
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise

    def add_encoded(self, key, identity, blob, blob_sha, raw_sha, raw_bytes, source_sha):
        if not self.db.in_transaction:
            raise RuntimeError('Record write requires source transaction')
        if hashlib.sha256(blob).hexdigest() != blob_sha:
            raise ValueError('Encoded checksum mismatch')
        self.db.execute('INSERT INTO records VALUES (?,?,?,?,?,?,?)',
                        (key, identity, blob_sha, raw_sha, raw_bytes, blob, source_sha))

    def encoded(self, key):
        row = self.db.execute('SELECT payload,blob_sha,raw_sha FROM records WHERE system_id=?', (key,)).fetchone()
        if row is None:
            raise KeyError(key)
        return row

    def finish_source(self, plan, validation):
        sha = plan['source_sha256']
        ids = {x[0] for x in self.db.execute('SELECT system_id FROM records WHERE source_sha=?', (sha,))}
        if ids != set(plan['records']):
            raise ValueError('Exact source coverage mismatch')
        if not all(validation.get(k) is True for k in
                   ('source_sha_verified','retained_tensors_verified','readback_verified','head_loader_verified')):
            raise ValueError('Incomplete source validation')
        self.db.execute("UPDATE sources SET state='complete',validation=? WHERE source_sha=?",
                        (canonical(validation), sha))

    def seal(self):
        if self.db.execute("SELECT count(*) FROM sources WHERE state!='complete'").fetchone()[0]:
            raise ValueError('Pending source prevents bundle publication')
        if not self.db.execute('SELECT count(*) FROM records').fetchone()[0]:
            raise ValueError('Empty bundle cannot be published')
        if not self.sealed:
            if self.db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                raise ValueError('Bundle integrity failed')
            # Check every stored compressed payload before the immutable rename.
            for blob, sha in self.db.execute('SELECT payload,blob_sha FROM records'):
                if hashlib.sha256(blob).hexdigest() != sha:
                    raise ValueError('Stored payload checksum mismatch')
            with self.db:
                self.db.execute("INSERT OR REPLACE INTO metadata VALUES ('sealed','1')")
            self.db.close()
            self.db = None
            with self.partial.open('rb') as stream:
                os.fsync(stream.fileno())
            if self.path.exists():
                raise ValueError('Refuse to overwrite sealed bundle')
            self.partial.rename(self.path)
            sync_directory(self.path.parent)
            self.sealed = True
            self.db = readonly(self.path)
        return dict(path=str(self.path), bytes=self.path.stat().st_size, sha256=file_sha(self.path),
                    records=self.db.execute('SELECT count(*) FROM records').fetchone()[0],
                    sources=self.db.execute('SELECT count(*) FROM sources').fetchone()[0])

    def close(self):
        if self.db is not None:
            self.db.close()
            self.db = None
        if self.lock is not None:
            self.lock.close()
            self.lock = None


class BundleCatalog:
    """One logical index, including old-format entries without copying payloads.

    Only the coordinator writes this DB. Importing legacy metadata is idempotent
    and checks immutable manifests, exact IDs, and duplicate IDs. Bundle claims
    persist before dispatch so an interrupted batch cannot be dispatched twice.
    """
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.executescript('''PRAGMA journal_mode=DELETE; PRAGMA synchronous=FULL;
            CREATE TABLE IF NOT EXISTS bundles(bundle_id TEXT PRIMARY KEY,path TEXT UNIQUE NOT NULL,
                contract TEXT NOT NULL,max_bytes INTEGER NOT NULL,jobs TEXT NOT NULL,state TEXT NOT NULL,
                sha TEXT,bytes INTEGER);
            CREATE TABLE IF NOT EXISTS sources(source_sha TEXT PRIMARY KEY,owner TEXT NOT NULL,plan_sha TEXT);
            CREATE TABLE IF NOT EXISTS records(system_id TEXT PRIMARY KEY,path TEXT NOT NULL,source_sha TEXT NOT NULL,
                identity TEXT NOT NULL,blob_sha TEXT NOT NULL,raw_sha TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS legacy(manifest TEXT PRIMARY KEY,sha TEXT NOT NULL);
        ''')
        self.db.execute('PRAGMA max_page_count=2097152')  # 8 GiB at default 4 KiB pages

    def claim(self, bundle_id, path, contract, max_bytes, jobs):
        frozen = (str(path), canonical(contract), max_bytes, canonical(jobs))
        old = self.db.execute('SELECT path,contract,max_bytes,jobs FROM bundles WHERE bundle_id=?',(bundle_id,)).fetchone()
        if old:
            if old != frozen:
                raise ValueError('Bundle assignment changed on resume')
            return
        with self.db:
            self.db.execute('INSERT INTO bundles VALUES (?,?,?,?,?,?,NULL,NULL)',
                            (bundle_id,*frozen,'pending'))
            for job in jobs:
                self.db.execute('INSERT INTO sources VALUES (?,?,NULL)', (job['receipt']['sha256'],bundle_id))

    def import_legacy(self, catalog_path):
        old = sqlite3.connect(Path(catalog_path).resolve().as_uri()+'?mode=ro', uri=True)
        try:
            for sha, psha, manifest, msha in old.execute("SELECT source_sha,plan_sha,manifest_path,manifest_sha FROM locations WHERE state='complete'"):
                seen = self.db.execute('SELECT sha FROM legacy WHERE manifest=?',(manifest,)).fetchone()
                if seen:
                    if seen[0] != msha:
                        raise ValueError('Legacy manifest identity changed')
                    continue
                if file_sha(manifest) != msha:
                    raise ValueError('Legacy manifest changed')
                m = json.loads(Path(manifest).read_text())
                if m['source_sha256'] != sha or m['plan_sha256'] != psha:
                    raise ValueError('Legacy source/plan identity mismatch')
                found = []
                with self.db:
                    self.db.execute('INSERT INTO sources VALUES (?,?,?)',(sha,'legacy:'+manifest,psha))
                    for shard in m['shards']:
                        if Path(shard['path']).stat().st_size != shard['bytes']:
                            raise ValueError('Legacy payload size mismatch')
                        db = readonly(shard['path'])
                        try:
                            for key,identity,bsha,rsha in db.execute('SELECT system_id,identity,blob_sha,raw_sha FROM records'):
                                found.append(key)
                                self.db.execute('INSERT INTO records VALUES (?,?,?,?,?,?)',(key,shard['path'],sha,identity,bsha,rsha))
                        finally:
                            db.close()
                    if len(found) != m['records'] or set(found) != set(m['system_ids']):
                        raise ValueError('Legacy exact coverage mismatch')
                    self.db.execute('INSERT INTO legacy VALUES (?,?)',(manifest,msha))
        finally:
            old.close()

    def publish(self, bundle_id, receipt):
        row = self.db.execute('SELECT path,contract,max_bytes,jobs,state,sha,bytes FROM bundles WHERE bundle_id=?',(bundle_id,)).fetchone()
        if row is None or row[0] != receipt['path']:
            raise ValueError('Unclaimed bundle')
        path, contract, cap, jobs_json, state, sha, size = row
        stat = Path(path).stat()
        if stat.st_size != receipt['bytes'] or stat.st_size > cap or file_sha(path) != receipt['sha256']:
            raise ValueError('Sealed bundle bytes changed or exceeded bound')
        if state == 'complete':
            if (sha,size) != (receipt['sha256'],receipt['bytes']):
                raise ValueError('Completed bundle changed')
            return
        bundle = readonly(path)
        try:
            if bundle.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
                raise ValueError('Sealed bundle quick_check failed')
            meta = dict(bundle.execute('SELECT key,value FROM metadata'))
            if meta.get('contract') != contract or meta.get('sealed') != '1':
                raise ValueError('Unsealed or changed bundle contract')
            expected = {j['receipt']['sha256'] for j in json.loads(jobs_json)}
            source_rows = list(bundle.execute('SELECT source_sha,plan_sha,plan,state,validation FROM sources'))
            if {r[0] for r in source_rows} != expected or len(source_rows) != receipt['sources']:
                raise ValueError('Assigned source coverage mismatch')
            with self.db:
                count = 0
                for ssha,psha,pjson,sstate,vjson in source_rows:
                    plan = json.loads(pjson)
                    validation = json.loads(vjson) if vjson else {}
                    if (sstate != 'complete' or digest(plan) != psha or not all(validation.get(k) is True for k in
                        ('source_sha_verified','retained_tensors_verified','readback_verified','head_loader_verified'))):
                        raise ValueError('Missing scientific validation')
                    ids = set()
                    for key,ident,bsha,rsha in bundle.execute('SELECT system_id,identity,blob_sha,raw_sha FROM records WHERE source_sha=?',(ssha,)):
                        ids.add(key)
                        self.db.execute('INSERT INTO records VALUES (?,?,?,?,?,?)',(key,path,ssha,ident,bsha,rsha))
                        count += 1
                    if ids != set(plan['records']):
                        raise ValueError('Published exact coverage mismatch')
                    self.db.execute('UPDATE sources SET plan_sha=? WHERE source_sha=? AND owner=?',(psha,ssha,bundle_id))
                if count != receipt['records']:
                    raise ValueError('Bundle record count mismatch')
                after = Path(path).stat()
                if (stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns) != (after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):
                    raise ValueError('Bundle changed during publication')
                self.db.execute("UPDATE bundles SET state='complete',sha=?,bytes=? WHERE bundle_id=?",
                                (receipt['sha256'],receipt['bytes'],bundle_id))
        finally:
            bundle.close()

    def close(self):
        self.db.close()


def iter_encoded(catalog_path):
    """Portable export stream: (ID, identity, blobSHA, rawSHA, rawBytes, payload).

    Also usable for an OFFLINE single-writer LMDB exporter without re-running
    feature inference or changing crops. Plans/provenance are exported separately.
    """
    db = sqlite3.connect(Path(catalog_path).resolve().as_uri()+'?mode=ro', uri=True)
    current = None
    payload_db = None
    try:
        for key,path,identity,bsha,rsha in db.execute('SELECT system_id,path,identity,blob_sha,raw_sha FROM records ORDER BY path,system_id'):
            if path != current:
                if payload_db is not None:
                    payload_db.close()
                payload_db = readonly(path)
                current = path
            row = payload_db.execute('SELECT identity,blob_sha,raw_sha,raw_bytes,payload FROM records WHERE system_id=?',(key,)).fetchone()
            if row is None or row[:3] != (identity,bsha,rsha) or hashlib.sha256(row[4]).hexdigest() != bsha:
                raise ValueError('Indexed record changed')
            yield (key,*row)
    finally:
        if payload_db is not None:
            payload_db.close()
        db.close()


def read_payload(catalog_path, key):
    """Same payload and compact_head_inputs API for legacy and bundle records."""
    from .shards import decode
    index = sqlite3.connect(Path(catalog_path).resolve().as_uri()+'?mode=ro', uri=True)
    try:
        row = index.execute('SELECT path,identity,blob_sha,raw_sha FROM records WHERE system_id=?',(key,)).fetchone()
    finally:
        index.close()
    if row is None:
        raise KeyError(key)
    db = readonly(row[0])
    try:
        payload = db.execute('SELECT payload,identity,blob_sha,raw_sha FROM records WHERE system_id=?',(key,)).fetchone()
        if payload is None or payload[1:] != row[1:]:
            raise ValueError('Indexed record identity mismatch')
        return decode(payload[0],payload[2],payload[3])
    finally:
        db.close()
