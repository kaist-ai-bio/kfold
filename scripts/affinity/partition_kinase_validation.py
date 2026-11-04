#!/usr/bin/env python3
"""Preserve the complete screen while preparing the approved single-protein cohort.

Reference sequences are not assay constructs. Known isoform/construct collisions
are deferred rather than silently sharing an input in the primary evaluation.
"""
import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import pyarrow as pa
from audit_validation_overlap import rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('labels', type=Path)
    parser.add_argument('audit', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False, parents=True)
    audit = json.loads(args.audit.read_text())
    # These names explicitly describe complexes in the inspected source catalog.
    complex_names = {r['target_name'] for r in audit['missing_sequence_targets']
                     if r['target_name'].startswith(('AMPK(', 'CDK', 'CDC7_', 'STK38'))
                     or r['target_name'] in {'LKB1', 'TAK1'}}
    collision_names = {name for r in audit['duplicate_input_systems']
                       if len(r['targets']) > 1 for name in r['targets']}
    db = duckdb.connect()
    db.execute('SET threads=2')
    db.execute("SET memory_limit='1GB'")
    db.read_parquet(str(args.labels)).create_view('labels')
    targets = rows(db, 'SELECT DISTINCT target_record_id,target_name,reference_sequence,uniprot,construct,construct_sequence_status FROM labels')
    assert len(targets) == len({r['target_record_id'] for r in targets})
    for target in targets:
        name = target['target_name']
        if name in complex_names:
            category = 'multicomponent_preserved'
        elif not target['reference_sequence']:
            category = 'unresolved_target_alias'
        elif name in collision_names:
            category = 'isoform_or_construct_collision'
        else:
            category = 'single_protein_reference_available'
        target['input_review_status'] = category
        target['input_sequence_scope'] = 'database_reference_not_verified_assay_construct'
    db.register('target_review', pa.Table.from_pylist(targets))
    db.execute("""CREATE VIEW reviewed AS SELECT l.*,t.input_review_status,t.input_sequence_scope,
        input_review_status='single_protein_reference_available' AND uniprot_seen primary_cohort_candidate
        FROM labels l JOIN target_review t USING(target_record_id)""")
    assert db.execute('SELECT count(*) FROM labels').fetchone() == db.execute('SELECT count(*) FROM reviewed').fetchone()
    outputs = []
    def export(name, query):
        p = args.out/name
        db.sql(query).write_parquet(str(p), compression='zstd')
        outputs.append(p)
    export('kinase-reviewed-labels.parquet', 'SELECT * FROM reviewed')
    export('kinase-primary-candidates.parquet', 'SELECT * FROM reviewed WHERE primary_cohort_candidate')
    export('kinase-complex-labels.parquet', "SELECT * FROM reviewed WHERE input_review_status='multicomponent_preserved'")
    export('kinase-input-review.parquet', 'SELECT * FROM target_review')
    summary = {
        'user_decision': 'Single-protein primary evaluation; preserve complexes separately. Training unchanged.',
        'primary_policy': 'Single reference protein, no observed isoform/construct input collision, known training UniProt. Candidate only: drug identity and final input contract remain to be checked.',
        'counts': rows(db, 'SELECT input_review_status,uniprot_seen,count(*) labels,count(DISTINCT target_record_id) targets,count(DISTINCT drug_record_id) drugs,sum(hit_at_1um_lt50::INT) hits FROM reviewed GROUP BY ALL ORDER BY 1,2'),
        'primary': rows(db, 'SELECT count(*) labels,count(DISTINCT target_record_id) targets,count(DISTINCT drug_record_id) drugs,count(DISTINCT candidate_system_id) systems,sum(hit_at_1um_lt50::INT) hits FROM reviewed WHERE primary_cohort_candidate')[0],
        'complex_target_names': sorted(complex_names),
        'collision_target_names': sorted(collision_names),
        'limits': ['Full reference sequence is not proof of exact assay construct', '50 drug identities originated as PubChem name candidates and still need source identity review', 'Missing observations remain in the separate original missing-data file'],
        'files': [{'name': p.name, 'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in outputs],
    }
    (args.out/'kinase-cohort-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('KINASE_PRIMARY',json.dumps(summary['primary']),flush=True)


if __name__ == '__main__':
    main()
