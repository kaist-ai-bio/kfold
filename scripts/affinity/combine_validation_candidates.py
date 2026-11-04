#!/usr/bin/env python3
"""Prepare a shared feature candidate manifest; preserve distinct label cohorts.

This does not finalize model inputs, assign RNG seeds, or launch extraction.
Experimental inequalities remain bounds, not invented point/negative labels.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb


def receipt(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return {'file': str(path), 'bytes': path.stat().st_size,
            'sha256': digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('quantitative', type=Path)
    parser.add_argument('censored', type=Path)
    parser.add_argument('kinase', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    db = duckdb.connect()
    db.execute("SET threads=2")
    db.execute("SET memory_limit='4GB'")
    inputs = [('quantitative', args.quantitative), ('censored', args.censored),
              ('kinase', args.kinase)]
    for name, path in inputs:
        db.read_parquet(str(path)).create_view(name + '_source')
        predicate = 'sair_library_flags_pass' if name != 'kinase' else 'true'
        db.execute(f'CREATE TEMP VIEW {name} AS SELECT * FROM {name}_source WHERE {predicate}')
    assert db.execute("SELECT count(*) FROM quantitative WHERE standard_relation IS DISTINCT FROM '='").fetchone()[0] == 0
    assert db.execute("SELECT count(*) FROM censored WHERE standard_relation IS NULL OR standard_relation NOT IN ('<','<=','>','>=')").fetchone()[0] == 0
    assert db.execute("SELECT count(*) FROM kinase WHERE measurement_status IS DISTINCT FROM 'measured' OR value IS NULL OR dose_uM IS DISTINCT FROM 1 OR unit IS DISTINCT FROM 'percent'").fetchone()[0] == 0

    unions = []
    for name, _ in inputs:
        seq = 'reference_sequence' if name == 'kinase' else 'sequence'
        rid = 'observation_id' if name == 'kinase' else 'record_id'
        unions.append(f"SELECT '{name}' cohort, {rid}::VARCHAR source_record_id, candidate_system_id, {seq} AS \"sequence\", exact_graph FROM {name}")
    db.execute('CREATE TEMP TABLE all_rows AS ' + ' UNION ALL '.join(unions))
    assert db.execute('SELECT count(*) FROM all_rows WHERE candidate_system_id IS NULL OR nullif(sequence,\'\') IS NULL OR nullif(exact_graph,\'\') IS NULL').fetchone()[0] == 0
    assert db.execute('SELECT count(*) FROM (SELECT cohort,source_record_id,count(*) n FROM all_rows GROUP BY 1,2 HAVING n!=1)').fetchone()[0] == 0
    assert db.execute("SELECT count(*) FROM all_rows WHERE candidate_system_id != 'valsys-'||sha256(length(sequence)::VARCHAR||':'||sequence||':'||length(exact_graph)::VARCHAR||':'||exact_graph)").fetchone()[0] == 0
    assert db.execute('SELECT count(*) FROM (SELECT candidate_system_id,count(DISTINCT (sequence,exact_graph)) n FROM all_rows GROUP BY 1 HAVING n!=1)').fetchone()[0] == 0
    db.execute('''CREATE TEMP TABLE systems AS
        SELECT candidate_system_id,sequence,exact_graph canonical_smiles,
               count(*) source_label_rows,
               list_sort(list(DISTINCT cohort)) source_cohorts,
               count(*) FILTER (WHERE cohort='quantitative') quantitative_labels,
               count(*) FILTER (WHERE cohort='censored') censored_labels,
               count(*) FILTER (WHERE cohort='kinase') kinase_labels
        FROM all_rows GROUP BY 1,2,3''')
    outputs = []

    def export(name, query):
        path = args.out / name
        db.sql(query).write_parquet(str(path), compression='zstd')
        outputs.append(path)

    for name, _ in inputs:
        export(name + '-labels.parquet', f'SELECT * FROM {name}')
    export('feature-candidate-systems.parquet', 'SELECT * FROM systems ORDER BY candidate_system_id')
    export('label-system-map.parquet', 'SELECT cohort,source_record_id,candidate_system_id FROM all_rows ORDER BY cohort,source_record_id')
    total_labels = db.execute('SELECT count(*) FROM all_rows').fetchone()[0]
    assert db.execute('SELECT sum(source_label_rows) FROM systems').fetchone()[0] == total_labels
    cohorts = {name: dict(zip(['labels', 'systems'], db.execute(f'SELECT count(*), count(DISTINCT candidate_system_id) FROM {name}').fetchone())) for name, _ in inputs}
    memberships = [dict(zip(['cohorts', 'systems', 'labels'], row)) for row in db.execute('SELECT source_cohorts,count(*),sum(source_label_rows) FROM systems GROUP BY 1 ORDER BY 1').fetchall()]
    bounds = [dict(zip(['relation', 'endpoint', 'labels'], row)) for row in db.execute('SELECT standard_relation,standard_type,count(*) FROM censored GROUP BY 1,2 ORDER BY 1,2').fetchall()]
    summary = {
        'version': 'pli_validation_feature_candidates_v2',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'metadata_prepared_not_production_feature_inputs',
        'cohorts': cohorts, 'labels': total_labels,
        'systems': db.execute('SELECT count(*) FROM systems').fetchone()[0],
        'censored_incremental_systems': db.execute('SELECT count(*) FROM systems WHERE censored_labels>0 AND quantitative_labels=0 AND kinase_labels=0').fetchone()[0],
        'memberships': memberships, 'censored_relations': bounds,
        'policy': [
            'User approved preparation of quantitative plus censored ChEMBL validation and reviewed single-protein kinase primary candidates.',
            'ChEMBL uses existing sair_library_flags_pass; kinase uses reviewed primary cohort unchanged.',
            'Shared exact sequence/graph candidates have one feature candidate row; all original source labels and qualifiers remain in separate cohort files.',
            'Censored values are bounds, not numeric affinity points or automatic negatives.',
            'No training records changed; no extraction or training launched.',
            'Candidate IDs do not encode finalized constructs, model/checkpoint, apo, tokenization or persistent seed. Production input review and validation/test allocation remain pending.'
        ],
        'inputs': {name: receipt(path) for name, path in inputs},
        'files': [receipt(path) for path in outputs],
    }
    (args.out / 'feature-candidate-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
