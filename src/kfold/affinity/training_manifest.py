"""Freeze task-specific train/validation membership without modifying sources."""
import gzip
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import random
import shutil
import sqlite3
import time

from .supervision import chemical_split, task_targets


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def ro(path):
    db=sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro&immutable=1',uri=True)
    db.row_factory=sqlite3.Row
    return db


MEASURED_CHEMICAL_KEYS_SQL="""SELECT DISTINCT g.chemical_key FROM meta.train m
    CROSS JOIN registry.aliases a ON a.cohort='train' AND a.source_id=m.system_id
    CROSS JOIN registry.systems s ON s.input_id=a.input_id
    CROSS JOIN chemical_groups g USING(ligand_sha) WHERE m.eligible_for_training=1"""


def chemistry(item):
    """Grouping only; never replaces the actual ligand in the feature input."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdMolHash
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDLogger.DisableLog('rdApp.*')
    key,smiles=item
    mol=Chem.MolFromSmiles(smiles)
    if mol is None:raise ValueError('Unparseable cached ligand '+key)
    mol=rdMolStandardize.LargestFragmentChooser(preferOrganic=True).choose(mol)
    mol=rdMolStandardize.Uncharger().uncharge(mol)
    Chem.RemoveStereochemistry(mol)
    for atom in mol.GetAtoms():atom.SetIsotope(0)
    text=rdMolHash.MolHash(mol,rdMolHash.HashFunction.HetAtomTautomerv2)
    group=hashlib.sha256(('parent_uncharged_no_stereo_isotope_hetatom_v2\0'+text).encode()).hexdigest()
    return key,group,chemical_split(group)


def prepare(run_root):
    r=Path(run_root).resolve();out=r/'data';out.mkdir(exist_ok=True)
    if (out/'completion.json').exists():
        return json.loads((out/'completion.json').read_text())
    if (out/'labels.sqlite').exists():raise RuntimeError('Partial manifest requires reviewed resume version')
    assert os.environ.get('SLURM_JOB_ID') and len(os.sched_getaffinity(0))>=288
    start=time.time();config=json.loads((r/'data-config.json').read_text())
    def save(name,value):
        p=out/name;t=p.with_suffix('.partial');t.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');t.replace(p)
    def progress(stage,**kw):save('progress.json',dict(epoch=time.time(),stage=stage,seconds=time.time()-start,**kw))
    ram=Path('/dev/shm')/('pli-head-'+os.environ['SLURM_JOB_ID'])
    ram.mkdir(mode=0o700,exist_ok=False)
    assert shutil.disk_usage(ram).free>40*1024**3
    incoming=Path(config.get('metadata_gzip',str(r/'metadata.sqlite.gz.partial')));source=config['metadata']
    assert file_sha(incoming)==source['gzip_sha256']
    with gzip.open(incoming,'rb') as a,(ram/'metadata.sqlite').open('xb') as b:
        shutil.copyfileobj(a,b,8<<20)
    assert file_sha(ram/'metadata.sqlite')==source['decoded_sha256']
    paths={}
    for name in ('registry','new_registry','catalog'):
        src=Path(config[name]);dest=ram/(name+'.sqlite');shutil.copyfile(src,dest);paths[name]=dest
    assert file_sha(paths['registry'])==config['registry_sha256']
    assert file_sha(paths['new_registry'])==config['new_registry_sha256']
    save('source-pins.json',dict(catalog_sha256=file_sha(paths['catalog']),catalog=config['catalog'],
        registry_sha256=config['registry_sha256'],metadata_sha256=source['decoded_sha256'],source_files_unchanged=True))
    progress('chemistry_group_inventory')
    old,new=ro(paths['registry']),ro(paths['new_registry'])
    graphs=dict(old.execute('SELECT ligand_sha,smiles FROM ligands'))
    graphs.update(new.execute('SELECT ligand_sha,smiles FROM ligands'))
    old.close();new.close()
    reuse=config.get('reuse_chemical_groups')
    pool=None if reuse else mp.get_context('fork').Pool(64)
    db=sqlite3.connect(out/'labels.sqlite',uri=True);db.execute('PRAGMA temp_store=MEMORY')
    db.execute('PRAGMA synchronous=FULL');db.execute('PRAGMA max_page_count=4194304');db.execute('PRAGMA cache_size=-524288')
    db.execute('CREATE TABLE chemical_groups(ligand_sha TEXT PRIMARY KEY,chemical_key TEXT NOT NULL,split TEXT NOT NULL)')
    if reuse:
        assert file_sha(reuse['path'])==reuse['sha256']
        prior=ro(reuse['path']);assert prior.execute('PRAGMA quick_check').fetchone()[0]=='ok'
        assert {x[0] for x in prior.execute('SELECT ligand_sha FROM chemical_groups')}==set(graphs)
        db.executemany('INSERT INTO chemical_groups VALUES(?,?,?)',prior.execute('SELECT * FROM chemical_groups'))
        db.commit();prior.close();progress('reused_complete_chemical_groups',records=len(graphs))
    else:
        try:
            batch=[];done=0
            for row in pool.imap_unordered(chemistry,graphs.items(),chunksize=128):
                batch.append(row)
                if len(batch)>=4096:
                    db.executemany('INSERT INTO chemical_groups VALUES(?,?,?)',batch);db.commit();done+=len(batch);batch=[]
                    progress('chemistry_grouping',processed=done,total=len(graphs))
            if batch:db.executemany('INSERT INTO chemical_groups VALUES(?,?,?)',batch);db.commit()
            pool.close();pool.join()
        finally:pool.terminate();pool.join()
    for name,path in [*paths.items(),('meta',ram/'metadata.sqlite')]:
        db.execute('ATTACH DATABASE ? AS '+name,(path.resolve().as_uri()+'?mode=ro&immutable=1',))
    db.execute('ATTACH DATABASE ? AS main_new_labels',(Path(config['new_labels']).resolve().as_uri()+'?mode=ro&immutable=1',))
    db.execute('ATTACH DATABASE ? AS supplemental',(Path(config['supplementary_labels']).resolve().as_uri()+'?mode=ro&immutable=1',))
    db.executescript('''CREATE TABLE examples(row_id INTEGER PRIMARY KEY,label_id TEXT UNIQUE NOT NULL,
        input_id TEXT NOT NULL,chemical_key TEXT NOT NULL,split TEXT NOT NULL,cohort TEXT NOT NULL,
        endpoint TEXT NOT NULL,relation TEXT,p_activity REAL,binary REAL,regression_valid INTEGER NOT NULL,
        classification_valid INTEGER NOT NULL,synthetic INTEGER NOT NULL,assay_key TEXT NOT NULL,
        historical_key TEXT,source_table TEXT NOT NULL,source_row INTEGER,provenance TEXT NOT NULL);
        CREATE TABLE dispositions(source TEXT,source_row INTEGER,label_id TEXT,reason TEXT);
        CREATE TABLE task_index(task TEXT,split TEXT,position INTEGER,row_id INTEGER,
            PRIMARY KEY(task,split,position));''')
    # Membership is derived, not inherited from the leaky legacy val flag.
    # Immutable metadata has no system_id index. Pin m as the outer scan so the
    # optimizer cannot rescan its millions of rows once per registry alias.
    all_measured_keys={x[0] for x in db.execute(MEASURED_CHEMICAL_KEYS_SQL)}
    for key, in db.execute('''SELECT DISTINCT g.chemical_key FROM new_registry.systems s
        JOIN chemical_groups g USING(ligand_sha)'''):all_measured_keys.add(key)
    present={x[0] for x in db.execute('SELECT system_id FROM catalog.records')}
    measured_input_ids={x[0] for x in db.execute("SELECT DISTINCT input_id FROM registry.aliases WHERE cohort='train'")}
    measured_input_ids.update(x[0] for x in db.execute('SELECT input_id FROM new_registry.systems'))
    excluded=[];pending=[];counts={}
    def reject(source,row,label,reason):
        excluded.append((source,row,label,reason))
        if len(excluded)>=1024:
            db.executemany('INSERT INTO dispositions VALUES(?,?,?,?)',excluded);excluded.clear()
    def add(label,identity,chemical,split,cohort,endpoint,relation,value,assay,historical,table,source_row,provenance,synthetic=False,binary_override=None):
        if identity not in present:
            reject(table,source_row,label,'not_in_approved_cache');return
        targets=task_targets(value,relation,synthetic=synthetic)
        if binary_override is not None:
            targets['classification_valid']=True;targets['binary']=float(binary_override)
        pending.append((label,identity,chemical,split,cohort,endpoint,relation,value,targets['binary'],
            int(targets['regression_valid']),int(targets['classification_valid']),int(synthetic),assay,
            historical,table,source_row,json.dumps(provenance,separators=(',',':'))))
        if len(pending)>=4096:
            db.executemany('INSERT INTO examples(label_id,input_id,chemical_key,split,cohort,endpoint,relation,p_activity,binary,regression_valid,classification_valid,synthetic,assay_key,historical_key,source_table,source_row,provenance) VALUES('+','.join('?'*17)+')',pending)
            db.commit();pending.clear()
    progress('joining_measured_training_labels')
    rows=db.execute('''SELECT m.rowid source_row,m.*,a.input_id,g.chemical_key,g.split derived_split
        FROM meta.train m LEFT JOIN registry.aliases a ON a.cohort='train' AND a.source_id=m.system_id
        LEFT JOIN registry.systems s ON s.input_id=a.input_id LEFT JOIN chemical_groups g USING(ligand_sha)''')
    n=0
    for x in rows:
        row=dict(zip([v[0] for v in rows.description],x));n+=1
        if not row['eligible_for_training']:
            reject('train',row['source_row'],row['record_id'],'not_source_training_eligible');continue
        if not row['input_id']:raise ValueError('Unresolved original training alias '+row['system_id'])
        assay=row['assay_key']+'|endpoint='+row['endpoint']+'|target='+row['protein_key']
        if row['origin']=='BindingDB-residual' or 'assay_id=singleton:' in row['assay_key']:
            assay='regression_only:'+str(row['source_row'])+'|'+assay
        add('legacy:'+str(row['source_row'])+':'+row['record_id'],row['input_id'],row['chemical_key'],row['derived_split'],'measured',
            row['endpoint'],'=',float(row['p_activity']),assay,row['protein_key'],'train',row['source_row'],
            dict(record_id=row['record_id'],origin=row['origin'],source=row['source'],legacy_split=row['split'],source_record_key=row['source_record_key'],
                source_assay_key=row['assay_key'],ranking_eligible=not assay.startswith('regression_only:'),
                relation_source='legacy_exact_normalized; SAIR endpoint may be proxy'))
    assert n==2183290
    # New BioLiP rows retain original source JSON, including observed relations.
    for table,db_name in [('biolip3','main_new_labels'),('supplementary','supplemental')]:
        q=db.execute('''SELECT l.rowid source_row,l.*,g.chemical_key,g.split derived_split
            FROM '''+db_name+'''.labels l JOIN new_registry.systems s USING(input_id)
            JOIN chemical_groups g USING(ligand_sha)''') if table=='biolip3' else db.execute('''SELECT l.rowid source_row,l.*,g.chemical_key,g.split derived_split
            FROM supplemental.labels l JOIN registry.systems s USING(input_id) JOIN chemical_groups g USING(ligand_sha)''')
        added=0
        for x in q:
            row=dict(zip([v[0] for v in q.description],x));added+=1
            source_json=json.loads(row['source_json']);source_id=source_json.get('assay_chembl_id') or source_json.get('assay_id') or row['label_id']
            assay='BioLiP3|'+str(source_id)+'|'+row['endpoint']+'|'
            # Input-independent assay key is further scoped to exact target below.
            t=db.execute('SELECT protein_sha FROM '+('new_registry' if table=='biolip3' else 'registry')+'.systems WHERE input_id=?',(row['input_id'],)).fetchone()[0]
            assay+='target='+t
            add('biolip3:'+row['label_id'],row['input_id'],row['chemical_key'],row['derived_split'],'measured',row['endpoint'],row['relation'],row['p_activity'],assay,None,table,row['source_row'],source_json)
        assert added==(23573 if table=='biolip3' else 3)
    progress('joining_external_evaluation')
    # External new-ligand rows never enter training. Collision rows stay in an
    # explicit auxiliary stratum, rather than pretending to be cold-ligand tests.
    validation_rows=0
    for table in ('quantitative','censored','kinase'):
        q=db.execute('''SELECT m.rowid source_row,m.*,a.input_id,g.chemical_key FROM meta.'''+table+''' m
            LEFT JOIN registry.aliases a ON a.cohort='validation' AND a.source_id=m.candidate_system_id
            LEFT JOIN registry.systems s ON s.input_id=a.input_id LEFT JOIN chemical_groups g USING(ligand_sha)''')
        for x in q:
            row=dict(zip([v[0] for v in q.description],x))
            label=str(row.get('record_id') or row.get('observation_id'))
            if row['input_id'] not in present:
                reject(table,row['source_row'],label,'approved_validation383_exclusion_or_no_cache');continue
            validation_rows+=1
            if table=='kinase':
                # Residual activity at fixed dose is not a concentration affinity.
                assert row['endpoint']=='residual_activity' and row['unit']=='percent' and float(row['dose_uM'])==1.
                add('kinase:'+label,row['input_id'],row['chemical_key'],'kinase_aux','known_drug_selectivity',
                    'residual_activity','=',None,'kinase|'+row['source_id']+'|'+row['target_record_id'],None,table,row['source_row'],
                    dict(source_id=row['source_id'],drug=row['drug_name'],value=row['value'],unit='percent',dose_uM=1.,
                         training_input_overlap=row['input_id'] in measured_input_ids),binary_override=bool(row['hit_at_1um_lt50']))
            else:
                assert row['standard_units']=='nM'
                v=9-math.log10(float(row['standard_value']))
                collision=row['chemical_key'] in all_measured_keys
                split='external_overlap_aux' if collision else 'external_novel'
                add('external:'+label,row['input_id'],row['chemical_key'],split,'external_'+table,
                    row['standard_type'],row['standard_relation'],v,
                    'ChEMBL|'+row['assay_chembl_id']+'|'+row['target_chembl_id']+'|'+row['standard_type'],
                    None,table,row['source_row'],dict(activity_id=row['activity_id'],document=row['document_chembl_id'],
                         first_release=row['document_first_chembl_release'],first_publication_lower=row['first_known_publication_lower'],
                         first_publication_upper=row['first_known_publication_upper'],chemical_overlap_with_training_pool=collision))
    assert validation_rows==151128,validation_rows
    progress('joining_synthetic_negatives')
    q=db.execute('''SELECT e.*,g.chemical_key,g.split derived_split FROM registry.synthetic_evidence e
        JOIN registry.systems s USING(input_id) JOIN chemical_groups g USING(ligand_sha)''')
    for x in q:
        row=dict(zip([v[0] for v in q.description],x))
        assert row['assumed_binary_label']==0 and row['affinity_value'] is None
        add('synthetic:'+row['candidate_id'],row['input_id'],row['chemical_key'],row['derived_split'],'synthetic_assumed_negative',
            'synthetic_binding','=',None,'synthetic|'+row['candidate_id'],None,'synthetic_evidence',None,
            dict(anchor_input_id=row['anchor_input_id'],cluster_id=row['cluster_id'],max_cluster_similarity=row['max_cluster_similarity'],
                 evidence_type=row['evidence_type'],experimental_inactive=False),synthetic=True)
    if pending:
        db.executemany('INSERT INTO examples(label_id,input_id,chemical_key,split,cohort,endpoint,relation,p_activity,binary,regression_valid,classification_valid,synthetic,assay_key,historical_key,source_table,source_row,provenance) VALUES('+','.join('?'*17)+')',pending);db.commit()
    if excluded:db.executemany('INSERT INTO dispositions VALUES(?,?,?,?)',excluded);db.commit()
    progress('freezing_task_indices')
    db.executescript('CREATE INDEX example_split ON examples(split,chemical_key); CREATE INDEX example_assay ON examples(split,assay_key);')
    for task,flag in [('regression','regression_valid'),('classification','classification_valid')]:
        for split in ('train','validation','external_novel','external_overlap_aux','kinase_aux'):
            db.execute('INSERT INTO task_index SELECT ?,?,row_number() OVER(ORDER BY row_id)-1,row_id FROM examples WHERE split=? AND '+flag+'=1',(task,split,split));db.commit()
    overlap=db.execute("SELECT count(*) FROM (SELECT DISTINCT chemical_key FROM examples WHERE split='train' INTERSECT SELECT DISTINCT chemical_key FROM examples WHERE split='validation')").fetchone()[0]
    ext_overlap=db.execute("SELECT count(*) FROM (SELECT DISTINCT chemical_key FROM examples WHERE split IN ('train','validation') INTERSECT SELECT DISTINCT chemical_key FROM examples WHERE split='external_novel')").fetchone()[0]
    assert overlap==ext_overlap==0
    shared_inputs=db.execute("SELECT count(*) FROM (SELECT DISTINCT input_id FROM examples WHERE split='train' INTERSECT SELECT DISTINCT input_id FROM examples WHERE split='validation')").fetchone()[0]
    assert shared_inputs==0
    assert db.execute('SELECT count(*) FROM examples WHERE synthetic=1 AND (p_activity IS NOT NULL OR regression_valid=1)').fetchone()[0]==0
    stats=[dict(zip(('split','cohort','endpoint','rows','regression','classification','positive','negative'),x)) for x in db.execute('SELECT split,cohort,endpoint,count(*),sum(regression_valid),sum(classification_valid),sum(binary=1),sum(binary=0) FROM examples GROUP BY split,cohort,endpoint')]
    save('composition.json',stats)
    # The pilot batch order is fixed once for all architectures, with measured
    # and synthetic strata explicit. Production samplers read this full index.
    rng=random.Random(20261109);pilot={}
    for task in ('regression','classification'):
        flag=task+'_valid';train=[]
        if task=='regression':
            groups=list(db.execute("SELECT assay_key FROM examples WHERE split='train' AND regression_valid=1 GROUP BY assay_key HAVING count(DISTINCT chemical_key)>=4 ORDER BY assay_key"))
            assert len(groups)>=8
            rng.shuffle(groups)
            for assay, in groups[:8]:
                items=db.execute("SELECT min(row_id) FROM examples WHERE split='train' AND regression_valid=1 AND assay_key=? GROUP BY chemical_key ORDER BY chemical_key LIMIT 4",(assay,)).fetchall()
                train.append([v[0] for v in items])
        else:
            buckets=[]
            for positive,synthetic,amount in [(1,0,16),(0,0,8),(0,1,8)]:
                ids=[x[0] for x in db.execute("SELECT row_id FROM examples WHERE split='train' AND classification_valid=1 AND binary=? AND synthetic=? ORDER BY row_id LIMIT 512",(positive,synthetic))]
                assert len(ids)>=amount;rng.shuffle(ids);buckets.append(ids[:amount])
            for i in range(8):train.append(buckets[0][2*i:2*i+2]+[buckets[1][i],buckets[2][i]])
        evaluation={}
        for split in ('validation','external_novel','kinase_aux'):
            evaluation[split]=[x[0] for x in db.execute('SELECT row_id FROM task_index WHERE task=? AND split=? ORDER BY position LIMIT 8',(task,split))]
        compatibility={}
        for source in ('biolip3','supplementary','censored','synthetic_evidence'):
            compatibility[source]=[x[0] for x in db.execute('SELECT row_id FROM examples WHERE source_table=? AND '+flag+'=1 ORDER BY row_id LIMIT 2',(source,))]
        pilot[task]=dict(train=train,evaluation=evaluation,compatibility=compatibility)
    save('pilot-batches.json',pilot)
    assert db.execute('PRAGMA quick_check').fetchone()[0]=='ok'
    count=db.execute('SELECT count(*) FROM examples').fetchone()[0]
    dispositions=[list(x) for x in db.execute('SELECT reason,count(*) FROM dispositions GROUP BY reason')]
    db.close()
    result=dict(passed=True,epoch=time.time(),rows=count,manifest=str(out/'labels.sqlite'),
        manifest_sha256=file_sha(out/'labels.sqlite'),chemical_train_val_overlap=overlap,input_train_val_overlap=shared_inputs,
        chemical_external_novel_overlap=ext_overlap,validation_source_rows_preserved=validation_rows,
        dispositions=dispositions,source_labels_modified=False,raw_feature_access=False,
        primary_training_view='query_200',consensus_model_selection_blocked_until_train_only_replan=True,
        split='global normalized chemistry hash90/10, seed20261109',seconds=time.time()-start)
    save('completion.json',result)
    shutil.rmtree(ram)
    save('cleanup.json',dict(node_ram_removed=str(ram),permanent_inputs_preserved=True))
    return result
