"""Freeze generation-candidate identities and stable positive index seeds.

Does not redefine training membership or validation labels. Aliases preserve
historical IDs even when new exact input recipes deduplicate old features.
"""
import argparse, hashlib, json, sqlite3
from pathlib import Path
import duckdb

def main():
    p=argparse.ArgumentParser();p.add_argument('layouts');p.add_argument('apos');p.add_argument('out');a=p.parse_args()
    out=Path(a.out);out.mkdir(exist_ok=False)
    apos=json.loads(Path(a.apos).read_text())
    recipe={'schema':'pli_native_input_v1','pipeline_base':'20d7b0d7f48ae2e45713038ce3c8d88cbb94ef56'}
    recipe['ccd_sha256']='445b126be5c2bc003930608ed9e639fd49cfba7bc454b3579fc3bc04be10e919'
    recipe.update(token_padding=64,atom_padding=128,sequence_padding=64,preparation_seed='persistent_system_index',prior_samples=1,conformer='native Tokenizer and seeded RNG; no custom chemistry repair',apo_count=1,protein_chain='A',ligand_chain='B',U648X='Q16881 user-authorized derivative; source labels unchanged')
    recipe_sha=hashlib.sha256(json.dumps(recipe,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    d=duckdb.connect();d.execute('SET threads=2');d.execute("SET memory_limit='6GB'")
    d.execute('CREATE TABLE proteins(source_sha VARCHAR,input_sha VARCHAR,apo_sha VARCHAR,sequence VARCHAR,path VARCHAR,metadata VARCHAR)')
    d.executemany('INSERT INTO proteins VALUES (?,?,?,?,?,?)',[(x['sequence_sha256'],hashlib.sha256(x['input_sequence'].encode()).hexdigest(),x['sha256'],x['input_sequence'],x['path'],json.dumps({k:x[k] for k in ('sequence','input_sequence','exact_source_sequence','apo_source','substitutions') if k in x})) for x in apos])
    d.read_parquet(a.layouts).create_view('layouts')
    assert d.execute('SELECT count(*) FROM layouts l LEFT JOIN proteins p ON sha256(l.sequence)=p.source_sha WHERE p.source_sha IS NULL').fetchone()[0]==0
    d.execute("""CREATE TABLE candidates AS SELECT l.cohort,l.id source_id,p.source_sha,sha256(l.canonical_smiles) ligand_sha,l.canonical_smiles,
      sha256('pli_native_input_v1:'||p.input_sha||':'||sha256(l.canonical_smiles)||':'||p.apo_sha||':'||?) input_id,
      length(p.sequence)::INT protein_tokens,l.ligand_tokens::INT ligand_tokens,(length(p.sequence)+l.ligand_tokens)::INT total_tokens
      FROM layouts l JOIN proteins p ON sha256(l.sequence)=p.source_sha WHERE l.ligand_status='ok'""",[recipe_sha])
    assert d.execute('SELECT count(*) FROM candidates WHERE total_tokens IS NULL').fetchone()[0]==0
    d.execute('CREATE TABLE systems AS SELECT row_number() OVER(ORDER BY input_id)::INT system_index,input_id,min(source_sha) protein_sha,min(ligand_sha) ligand_sha,min(protein_tokens) protein_tokens,min(ligand_tokens) ligand_tokens,min(total_tokens) total_tokens FROM candidates GROUP BY input_id')
    db=sqlite3.connect(out/'input-registry.sqlite');db.executescript('''PRAGMA journal_mode=DELETE;
    CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
    CREATE TABLE proteins(source_sha TEXT PRIMARY KEY,input_sha TEXT,apo_sha TEXT,sequence TEXT,path TEXT,metadata TEXT);
    CREATE TABLE ligands(ligand_sha TEXT PRIMARY KEY,smiles TEXT NOT NULL);
    CREATE TABLE systems(system_index INTEGER PRIMARY KEY,input_id TEXT UNIQUE NOT NULL,protein_sha TEXT,ligand_sha TEXT,protein_tokens INTEGER,ligand_tokens INTEGER,total_tokens INTEGER);
    CREATE TABLE aliases(cohort TEXT,source_id TEXT,input_id TEXT,PRIMARY KEY(cohort,source_id));''')
    db.execute('INSERT INTO metadata VALUES (?,?)',('recipe',json.dumps(recipe,sort_keys=True)));db.execute('INSERT INTO metadata VALUES (?,?)',('recipe_sha256',recipe_sha))
    for table,query,n in [('proteins','SELECT * FROM proteins',6),('ligands','SELECT DISTINCT ligand_sha,canonical_smiles FROM candidates',2),('systems','SELECT * FROM systems ORDER BY system_index',7),('aliases','SELECT DISTINCT cohort,source_id,input_id FROM candidates',3)]:
        cursor=d.execute(query)
        while batch:=cursor.fetchmany(10000):db.executemany('INSERT INTO '+table+' VALUES ('+','.join(['?']*n)+')',batch)
        db.commit();print('REGISTRY',table,flush=True)
    assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    boundaries=[0,256,512,768,1024,1536,2048];pilot=[];distribution=[]
    for lo,hi in zip(boundaries,boundaries[1:]):
        counts=d.execute('SELECT count(*),sum(ceil(total_tokens/64.0)*64*ceil(total_tokens/64.0)*64) FROM systems WHERE total_tokens>? AND total_tokens<=?',[lo,hi]).fetchone()
        distribution.append(dict(min_exclusive=lo,max_inclusive=hi,systems=counts[0],padded_token_squared=counts[1]))
        # Equal ranks across ligand size; no activity-label based selection.
        choices=d.execute('''SELECT s.system_index FROM systems s WHERE s.total_tokens>? AND s.total_tokens<=? ORDER BY ligand_tokens,input_id''',[lo,hi]).fetchall()
        indices=sorted({round(i*(len(choices)-1)/7) for i in range(8)}) if choices else []
        for i in indices:
            row=db.execute('''SELECT s.system_index,s.input_id,p.sequence,l.smiles,p.path,p.apo_sha,s.total_tokens,s.protein_tokens,s.ligand_tokens,p.metadata FROM systems s JOIN proteins p ON s.protein_sha=p.source_sha JOIN ligands l USING(ligand_sha) WHERE system_index=?''',choices[i]).fetchone()
            keys=['system_index','input_id','sequence','smiles','apo_path','apo_sha256','total_tokens','protein_tokens','ligand_tokens','source_metadata']
            task=dict(zip(keys,row));task['source_metadata']=json.loads(task['source_metadata']);task['bucket']=str(lo+1)+'-'+str(hi);pilot.append(task)
    (out/'pilot-inputs.json').write_text(json.dumps(pilot,indent=2)+'\n')
    summary=dict(systems=db.execute('SELECT count(*) FROM systems').fetchone()[0],aliases=db.execute('SELECT count(*) FROM aliases').fetchone()[0],proteins=len(apos),distribution=distribution,pilot_systems=len(pilot),recipe=recipe,recipe_sha256=recipe_sha,training_membership_finalized=False,labels_modified=False,append_only=True,source_files={str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [a.layouts,a.apos]})
    db.close();(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))

if __name__=='__main__':main()
