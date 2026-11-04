"""Audit measured training-source associations before synthetic-decoy sampling.

This exports no negative labels, changes no registry indices, and assigns no
cutoff eligibility. All measured associations are retained as potential exclusion
evidence. Donor eligibility and final 90%-cluster/similarity screening are later
steps; validation-only ligands are never added to the training-source inventory.
"""
import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

import duckdb
import pyarrow as pa


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('registry', type=Path)
    p.add_argument('labels', type=Path)
    p.add_argument('out', type=Path)
    a = p.parse_args()
    a.out.mkdir(exist_ok=False)
    d = duckdb.connect()
    d.execute('SET threads=4')
    d.execute("SET memory_limit='6GB'")
    d.execute('SET temp_directory=?', [str(a.out / 'duckdb-tmp')])
    db = sqlite3.connect('file:' + str(a.registry) + '?mode=ro', uri=True)
    for name, fields in [('systems', 'input_id,protein_sha,ligand_sha,protein_tokens,ligand_tokens'),
                         ('aliases', 'cohort,source_id,input_id'),
                         ('proteins', 'source_sha,sequence,metadata'),
                         ('ligands', 'ligand_sha,smiles')]:
        cur = db.execute('SELECT ' + fields + ' FROM ' + name)
        columns = [x[0] for x in cur.description]
        first = True
        while batch := cur.fetchmany(20000):
            table = pa.Table.from_pydict({key: [row[i] for row in batch] for i, key in enumerate(columns)})
            d.register('batch_input', table)
            d.execute(('CREATE TABLE ' + name + ' AS SELECT * FROM batch_input') if first
                      else ('INSERT INTO ' + name + ' SELECT * FROM batch_input'))
            first = False
        if first:
            raise ValueError('Empty registry table: ' + name)
    db.close()
    d.read_parquet(str(a.labels)).create_view('labels')
    d.execute('''CREATE TABLE measured AS SELECT system_id source_id,
        count(*) source_rows, min(p_activity) min_p_activity, max(p_activity) max_p_activity,
        bool_or(eligible_for_training AND split='train') historical_train_eligible,
        list(DISTINCT endpoint) endpoints, list(DISTINCT source) sources
        FROM labels GROUP BY system_id''')
    d.execute('''CREATE TABLE measured_inputs AS SELECT DISTINCT a.input_id,m.*
        FROM measured m JOIN aliases a USING(source_id)''')
    unmatched = d.execute('SELECT count(*) FROM measured m WHERE NOT EXISTS(SELECT 1 FROM aliases a WHERE a.source_id=m.source_id)').fetchone()[0]
    d.execute('''CREATE TABLE associations AS SELECT s.input_id,s.protein_sha,s.ligand_sha,
        s.protein_tokens,s.ligand_tokens,sum(m.source_rows) source_rows,
        min(m.min_p_activity) min_p_activity,max(m.max_p_activity) max_p_activity,
        bool_or(m.historical_train_eligible) historical_train_eligible
        FROM systems s JOIN measured_inputs m USING(input_id) GROUP BY ALL''')
    def export(name, query):
        d.sql(query).write_parquet(str(a.out / name), compression='zstd')
    export('measured-associations.parquet', 'SELECT * FROM associations')
    export('training-source-ligands.parquet', '''SELECT l.* FROM ligands l WHERE EXISTS
        (SELECT 1 FROM associations a WHERE a.ligand_sha=l.ligand_sha)''')
    export('training-source-proteins.parquet', '''SELECT p.* FROM proteins p WHERE EXISTS
        (SELECT 1 FROM associations a WHERE a.protein_sha=p.source_sha)''')
    with (a.out / 'training-source-proteins.fasta').open('w') as f:
        for key, sequence in d.execute('''SELECT source_sha,sequence FROM proteins p WHERE EXISTS
            (SELECT 1 FROM associations a WHERE a.protein_sha=p.source_sha) ORDER BY source_sha''').fetchall():
            f.write('>' + key + '\n' + sequence + '\n')
    summary = dict(stage='measured_source_inventory_only_not_decoy_manifest',
        source_registry_sha256=sha(a.registry), source_labels_sha256=sha(a.labels),
        association_systems=d.execute('SELECT count(*) FROM associations').fetchone()[0],
        historical_train_eligible_systems=d.execute('SELECT count(*) FROM associations WHERE historical_train_eligible').fetchone()[0],
        proteins=d.execute('SELECT count(DISTINCT protein_sha) FROM associations').fetchone()[0],
        ligands=d.execute('SELECT count(DISTINCT ligand_sha) FROM associations').fetchone()[0],
        unmatched_source_systems=unmatched,
        registry_cohorts=d.execute('SELECT cohort,count(*) FROM aliases GROUP BY cohort').fetchall(),
        pending=['source cutoff evidence inheritance', 'held-out ligand and chemical identity exclusion',
                 '90 percent sequence clustering', 'known-binder exclusion recipe',
                 'fingerprint version and Tanimoto screening', 'new pair sampling and append-only indices'],
        labels_assigned=False, registry_changed=False, synthetic_pairs_created=0)
    summary['files'] = [dict(name=x.name,bytes=x.stat().st_size,sha256=sha(x))
                        for x in sorted(a.out.iterdir()) if x.is_file()]
    (a.out / 'pool-inventory.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    main()
