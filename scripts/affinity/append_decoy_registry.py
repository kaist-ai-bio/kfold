"""Publish a child registry with new decoy indices; never modify its parent.

This is CPU preparation only. The existing production campaign must continue
using its original registry. Future decoy workers must restrict work to the new
index interval; this command neither submits them nor certifies GPU inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

import pyarrow.parquet as pq


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(8*1024**2), b''):
            h.update(b)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['parent', 'candidates', 'candidate_receipt', 'audit', 'out']:
        parser.add_argument(name, type=Path)
    a = parser.parse_args()
    a.out.mkdir(exist_ok=False)
    parent_sha, candidates_sha = sha(a.parent), sha(a.candidates)
    receipt, audit = [json.loads(p.read_text()) for p in (a.candidate_receipt, a.audit)]
    assert receipt['input_sha256']['registry'] == parent_sha
    assert receipt['output_sha256'] == audit['input_sha256'] == candidates_sha
    assert audit['one_per_training_input_verified'] and audit['all_identity_checks_passed']
    assert receipt['failed_anchors'] == 0 and receipt['all_training']
    parent = sqlite3.connect(a.parent.resolve().as_uri() + '?mode=ro', uri=True)
    out_db = a.out/'input-registry.sqlite'
    db = sqlite3.connect(out_db, uri=True)
    parent.backup(db)
    parent_count, previous_max = db.execute('SELECT count(*),max(system_index) FROM systems').fetchone()
    recipe_sha = db.execute("SELECT value FROM metadata WHERE key='recipe_sha256'").fetchone()[0]
    recipe = json.loads(db.execute("SELECT value FROM metadata WHERE key='recipe'").fetchone()[0])
    assert recipe['schema'] == 'pli_native_input_v1'
    assert hashlib.sha256(json.dumps(recipe,sort_keys=True,separators=(',',':')).encode()).hexdigest() == recipe_sha
    db.create_function('sha256', 1, lambda v: hashlib.sha256(v.encode()).hexdigest(), deterministic=True)
    db.execute('PRAGMA temp_store=FILE')
    db.execute('''CREATE TEMP TABLE candidates(candidate_id TEXT PRIMARY KEY,
        anchor_input_id TEXT UNIQUE,protein_sha TEXT,ligand_sha TEXT,cluster_id TEXT,
        max_cluster_similarity REAL,protein_tokens INTEGER,ligand_tokens INTEGER,total_tokens INTEGER)''')
    columns = ['candidate_id','anchor_input_id','protein_sha','ligand_sha','cluster_id',
               'max_cluster_similarity','protein_tokens','ligand_tokens','total_tokens']
    for batch in pq.ParquetFile(a.candidates).iter_batches(batch_size=10000, columns=columns):
        db.executemany('INSERT INTO candidates VALUES (?,?,?,?,?,?,?,?,?)',
                       ([r[k] for k in columns] for r in batch.to_pylist()))
    db.commit()
    assert db.execute('SELECT count(*) FROM candidates').fetchone()[0] == receipt['candidates']
    assert db.execute('''SELECT count(*) FROM candidates c LEFT JOIN systems s
        ON c.anchor_input_id=s.input_id WHERE s.input_id IS NULL OR s.protein_sha!=c.protein_sha
        OR s.protein_tokens!=c.protein_tokens''').fetchone()[0] == 0
    assert db.execute('''SELECT count(*) FROM candidates c LEFT JOIN proteins p
        ON c.protein_sha=p.source_sha LEFT JOIN ligands l USING(ligand_sha)
        WHERE p.source_sha IS NULL OR l.ligand_sha IS NULL''').fetchone()[0] == 0
    db.execute('''CREATE TEMP TABLE incoming AS SELECT c.*,
        sha256('pli_native_input_v1:'||p.input_sha||':'||c.ligand_sha||':'||p.apo_sha||':'||?) input_id
        FROM candidates c JOIN proteins p ON p.source_sha=c.protein_sha''', (recipe_sha,))
    db.execute('CREATE UNIQUE INDEX incoming_identity ON incoming(input_id)')
    assert db.execute('SELECT count(*) FROM incoming i JOIN systems s USING(input_id)').fetchone()[0] == 0
    assert previous_max + receipt['candidates'] < 2**32
    db.execute('''INSERT INTO systems SELECT ?+row_number() OVER(ORDER BY input_id),
        input_id,protein_sha,ligand_sha,protein_tokens,ligand_tokens,total_tokens FROM incoming''', (previous_max,))
    db.execute('''INSERT INTO aliases SELECT 'synthetic_decoy_assumed_negative',candidate_id,input_id FROM incoming''')
    db.execute('''CREATE TABLE synthetic_evidence AS SELECT candidate_id,anchor_input_id,input_id,
        cluster_id,max_cluster_similarity,'synthetic_assumed_negative' evidence_type,
        CAST(0 AS INTEGER) assumed_binary_label,CAST(NULL AS REAL) affinity_value FROM incoming''')
    db.execute('CREATE UNIQUE INDEX synthetic_identity ON synthetic_evidence(candidate_id)')
    lineage = dict(parent_sha256=parent_sha,candidates_sha256=candidates_sha,
        previous_max_index=previous_max,first_new_index=previous_max+1,
        new_systems=receipt['candidates'],last_new_index=previous_max+receipt['candidates'],
        seed_rule='persistent system_index; original indices unchanged',
        native_full_input_qc='pending; chain token counts inherited from audited parents')
    db.execute('INSERT INTO metadata VALUES (?,?)', ('decoy_lineage',json.dumps(lineage,sort_keys=True)))
    db.commit()
    db.execute('ATTACH DATABASE ? AS parent', (a.parent.resolve().as_uri()+'?mode=ro',))
    for table in ['systems','aliases','proteins','ligands','metadata']:
        assert db.execute('SELECT count(*) FROM (SELECT * FROM parent.'+table+
                          ' EXCEPT SELECT * FROM main.'+table+')').fetchone()[0] == 0, table
    assert db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    total = db.execute('SELECT count(*) FROM systems').fetchone()[0]
    assert total == parent_count + receipt['candidates']
    pilot = []
    boundaries = [0,256,512,768,1024,1536,2048,2304,2560]
    distribution = []
    for lo, hi in zip(boundaries,boundaries[1:]):
        indices = db.execute('''SELECT system_index FROM systems WHERE system_index>?
            AND total_tokens>? AND total_tokens<=? ORDER BY ligand_tokens,input_id''', (previous_max,lo,hi)).fetchall()
        distribution.append(dict(min_exclusive=lo,max_inclusive=hi,systems=len(indices)))
        for n in sorted({round(i*(len(indices)-1)/7) for i in range(8)}) if indices else []:
            row = db.execute('''SELECT system_index,input_id,p.sequence,l.smiles,p.path,p.apo_sha,
                total_tokens,protein_tokens,ligand_tokens,p.metadata FROM systems s JOIN proteins p
                ON p.source_sha=s.protein_sha JOIN ligands l USING(ligand_sha) WHERE system_index=?''',indices[n]).fetchone()
            task = dict(zip(['system_index','input_id','sequence','smiles','apo_path','apo_sha256',
                            'total_tokens','protein_tokens','ligand_tokens','source_metadata'],row))
            task['source_metadata']=json.loads(task['source_metadata']);task['bucket']=f'{lo+1}-{hi}'
            pilot.append(task)
    assert sum(x['systems'] for x in distribution) == receipt['candidates']
    db.close();parent.close()
    assert sha(a.parent) == parent_sha
    (a.out/'pilot-inputs.json').write_text(json.dumps(pilot,indent=2)+'\n')
    summary = dict(**lineage,total_systems=total,parent_systems=parent_count,
        parent_bytes_unchanged=True,parent_tables_preserved=True,sqlite_integrity='ok',
        child_registry_sha256=sha(out_db),child_registry_bytes=out_db.stat().st_size,
        distribution=distribution,pilot_systems=len(pilot),production_campaign_changed=False,
        gpu_started=False,head_training_started=False,
        native_recipe_sha256=recipe_sha,source_script_sha256=sha(__file__))
    (a.out/'registry-receipt.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    main()
