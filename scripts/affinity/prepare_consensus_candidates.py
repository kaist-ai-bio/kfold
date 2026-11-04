"""Freeze training-only candidates and resolve exact original registry inputs.

Run with node-local label/registry copies. No feature/label/registry mutation.
All ranks are exported, including small targets for an explicit coverage ledger.
"""
import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from kfold.affinity.evidence import freeze_candidates


def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('registry', type=Path)
    parser.add_argument('labels', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False)
    expected = {args.registry: '9d4747129642716cf9aeca2fab30cfe658b9354496a46f0b26b4490cbf8c7118',
                args.labels: '45817c2ead46a3bc49c66b75e0044f371143c788df16419ca1884e56cb81c515'}
    for path, digest in expected.items():
        if sha(path) != digest:
            raise ValueError(f'Input provenance mismatch: {path}')
    columns = ['system_id','source_system_id','protein_key','canonical_smiles',
               'p_activity','eligible_for_training','split']
    def rows():
        for batch in pq.ParquetFile(args.labels).iter_batches(columns=columns, batch_size=20000):
            yield from batch.to_pylist()
    candidates = freeze_candidates(rows())
    print(json.dumps(dict(stage='frozen', candidates=len(candidates))), flush=True)
    d = duckdb.connect()
    d.execute('SET threads=4')
    d.execute("SET memory_limit='6GB'")
    d.execute('SET temp_directory=?', [str(args.out / 'duckdb-tmp')])
    d.register('candidate_input', pa.Table.from_pylist(candidates))
    d.execute('CREATE TABLE candidates AS SELECT * FROM candidate_input')
    d.unregister('candidate_input')
    del candidates
    db = sqlite3.connect(args.registry.resolve().as_uri()+'?mode=ro', uri=True)
    for name, fields in [('aliases','cohort,source_id,input_id'),
                         ('systems','system_index,input_id,protein_sha,ligand_sha,protein_tokens,ligand_tokens'),
                         ('proteins','source_sha,input_sha,apo_sha'), ('ligands','ligand_sha,smiles')]:
        cur = db.execute('SELECT '+fields+' FROM '+name)
        keys = [x[0] for x in cur.description]
        first = True
        while batch := cur.fetchmany(20000):
            d.register('incoming', pa.Table.from_pydict({k:[r[i] for r in batch] for i,k in enumerate(keys)}))
            d.execute(('CREATE TABLE '+name+' AS SELECT * FROM incoming') if first
                      else ('INSERT INTO '+name+' SELECT * FROM incoming'))
            first = False
    db.close()
    d.execute('''CREATE TABLE mapped AS SELECT DISTINCT c.*,a.input_id
      FROM candidates c LEFT JOIN aliases a ON c.manifest_system_id=a.source_id''')
    if d.execute('SELECT count(*) FROM mapped WHERE input_id IS NULL').fetchone()[0]:
        raise ValueError('Candidate manifest ID absent from registry aliases')
    if d.execute('SELECT count(*) FROM (SELECT protein_key,canonical_smiles FROM mapped GROUP BY ALL HAVING count(*)!=1)').fetchone()[0]:
        raise ValueError('Ambiguous manifest ID across registry cohorts')
    d.execute('''CREATE TABLE resolved AS SELECT m.*,s.system_index,s.protein_sha,
       s.ligand_sha,s.protein_tokens,s.ligand_tokens,p.input_sha,p.apo_sha,
       sha256(p.input_sha||':'||p.apo_sha) target_input_key
       FROM mapped m JOIN systems s USING(input_id)
       JOIN proteins p ON s.protein_sha=p.source_sha
       JOIN ligands l ON s.ligand_sha=l.ligand_sha
       WHERE l.smiles=m.canonical_smiles''')
    count = d.execute('SELECT count(*) FROM candidates').fetchone()[0]
    if d.execute('SELECT count(*) FROM resolved').fetchone()[0] != count:
        raise ValueError('Input identity/ligand mismatch')
    if d.execute('SELECT count(*) FROM (SELECT protein_key FROM resolved GROUP BY ALL HAVING count(DISTINCT target_input_key)!=1)').fetchone()[0]:
        raise ValueError('Mixed sequence/apo under a historical protein_key')
    d.sql('SELECT * FROM resolved ORDER BY protein_key,candidate_rank').write_parquet(
        str(args.out/'candidates.parquet'), compression='zstd')
    d.sql('''SELECT protein_key,target_input_key,count(*) training_ligands,
       CASE WHEN count(*)>=10 THEN 'awaiting_profiles' ELSE 'fewer_than_ten_training_ligands' END status
       FROM resolved GROUP BY ALL ORDER BY protein_key''').write_parquet(str(args.out/'target-coverage.parquet'),compression='zstd')
    summary = dict(stage='frozen_candidates_only_no_consensus_yet', candidates=count,
        training_labels=d.execute('SELECT sum(train_label_count) FROM resolved').fetchone()[0],
        targets=d.execute('SELECT count(DISTINCT protein_key) FROM resolved').fetchone()[0],
        qualified_targets=d.execute('SELECT count(DISTINCT protein_key) FROM resolved WHERE target_training_ligands>=10').fetchone()[0],
        selection_seed=20260822, reference_revision='7cd98dd728a95efe6560f0a59a741c7c66bd4c7e',
        validation_labels_used=False, synthetic_labels_used=False, registry_modified=False,
        labels_modified=False, source_sha256={str(p):h for p,h in expected.items()},
        files=[dict(name=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(args.out.glob('*.parquet'))])
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
