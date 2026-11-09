"""Versioned record subsets of immutable sources, with disjoint durable claims.

Existing whole-source entries and bundles are untouched. The sources sentinel
keeps older whole-source dispatchers from selecting a partially covered source.
Completion belongs to each segment; it never asserts whole-source completion.
"""
import json

from .bundles import BundleCatalog, canonical, digest, readonly

SCOPE = 'record_subset_v1'
OWNER = 'record-subsets:v1'


def selection(source_ids, selected_ids):
    all_ids, chosen = sorted(source_ids), sorted(selected_ids)
    if (not chosen or len(all_ids) != len(set(all_ids)) or
            len(chosen) != len(set(chosen)) or not set(chosen) <= set(all_ids)):
        raise ValueError('Nonempty unique source subset required')
    return dict(schema=SCOPE, source_ids=all_ids, selected_ids=chosen,
                unselected_ids=sorted(set(all_ids)-set(chosen)))


def validate_selection(value, source_ids, selected_ids):
    if value != selection(source_ids, selected_ids):
        raise ValueError('Source selection partition changed')


class PartialBundleCatalog(BundleCatalog):
    def __init__(self, path):
        super().__init__(path)
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS source_segments(
                source_sha TEXT NOT NULL, bundle_id TEXT NOT NULL,
                selection_sha TEXT NOT NULL, selection TEXT NOT NULL,
                PRIMARY KEY(source_sha,bundle_id));
            CREATE TABLE IF NOT EXISTS record_claims(
                system_id TEXT PRIMARY KEY, source_sha TEXT NOT NULL,
                bundle_id TEXT NOT NULL);
        ''')

    def claim(self, bundle_id, path, contract, max_bytes, jobs):
        if contract.get('coverage_scope') != SCOPE:
            raise ValueError('Explicit subset bundle contract required')
        frozen = (str(path), canonical(contract), max_bytes, canonical(jobs))
        old = self.db.execute('SELECT path,contract,max_bytes,jobs FROM bundles WHERE bundle_id=?',
                              (bundle_id,)).fetchone()
        if old:
            if old != frozen:
                raise ValueError('Bundle assignment changed on resume')
            return
        with self.db:
            self.db.execute('INSERT INTO bundles VALUES (?,?,?,?,?,?,NULL,NULL)',
                            (bundle_id,*frozen,'pending'))
            for job in jobs:
                sha, subset = job['receipt']['sha256'], job['selection']
                validate_selection(subset, subset['source_ids'], job['identity_keys'])
                if len(subset['source_ids']) != job['receipt']['records']:
                    raise ValueError('Source receipt count mismatch')
                row = self.db.execute('SELECT owner FROM sources WHERE source_sha=?',(sha,)).fetchone()
                if row and row[0] != OWNER:
                    raise ValueError('Source already claimed by whole-source converter')
                self.db.execute('INSERT OR IGNORE INTO sources VALUES (?,?,NULL)',(sha,OWNER))
                # Each later segment must refer to the identical full source inventory.
                previous = self.db.execute('SELECT selection FROM source_segments WHERE source_sha=?',(sha,)).fetchall()
                if any(json.loads(x[0])['source_ids'] != subset['source_ids'] for x in previous):
                    raise ValueError('Source inventory changed between segments')
                self.db.execute('INSERT INTO source_segments VALUES (?,?,?,?)',
                                (sha,bundle_id,digest(subset),canonical(subset)))
                for key in subset['selected_ids']:
                    if self.db.execute('SELECT 1 FROM records WHERE system_id=?',(key,)).fetchone():
                        raise ValueError('Selected record already published')
                    # Unique constraint also rejects overlapping pending segments.
                    self.db.execute('INSERT INTO record_claims VALUES (?,?,?)',(key,sha,bundle_id))

    def publish(self, bundle_id, receipt):
        row = self.db.execute('SELECT path,jobs,contract FROM bundles WHERE bundle_id=?',(bundle_id,)).fetchone()
        if not row or row[0] != receipt['path'] or json.loads(row[2]).get('coverage_scope') != SCOPE:
            raise ValueError('Unclaimed subset bundle')
        jobs = {j['receipt']['sha256']:j for j in json.loads(row[1])}
        db = readonly(row[0])
        try:
            for sha, pjson, vjson in db.execute('SELECT source_sha,plan,validation FROM sources'):
                plan, validation = json.loads(pjson), json.loads(vjson)
                job = jobs[sha]
                if (plan.get('selection') != job['selection'] or
                        set(plan['records']) != set(job['identity_keys']) or
                        validation.get('selected_coverage_verified') is not True):
                    raise ValueError('Published subset differs from frozen assignment')
                claims = {r[0] for r in self.db.execute(
                    'SELECT system_id FROM record_claims WHERE source_sha=? AND bundle_id=?',(sha,bundle_id))}
                if claims != set(plan['records']):
                    raise ValueError('Subset claims changed')
        finally:
            db.close()
        # Parent publishes payload index + complete bundle state in ONE transaction.
        # Its owner-qualified source UPDATE cannot change the subset sentinel.
        super().publish(bundle_id, receipt)
