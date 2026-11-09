import copy
import hashlib
import sqlite3
import pytest

from kfold.affinity.bundles import BundleStore,BundleCatalog,read_payload,iter_encoded
from kfold.affinity.bundle_conversion import append_source
from kfold.affinity.conversion import file_sha
from kfold.affinity.head_inputs import compact_head_inputs
from kfold.affinity.shards import ShardStore,SHUFFLE_CODEC,decode
from kfold.affinity.preparation import digest
from test_compact import fixture

CONTRACT=dict(purpose='integration_pilot',storage_codec=SHUFFLE_CODEC,schema='pli_bundle_v1')

def source(tmp_path,name,keys):
 s=ShardStore(tmp_path/name,'raw-contract');records={}
 for key in keys:
  p,views,recipe=fixture();p['bin_boundaries'].fill_(12.)
  p['receipt']['system_id']=key;p['cache_identity_sha256']=digest(p['receipt'])
  s.add(key,p)
  records[key]=dict(status='final',source_cache_identity_sha256=p['cache_identity_sha256'],views=views,crop_receipt=recipe,unavailable={})
 s.seal();path=next((tmp_path/name).glob('*.sqlite'))
 return path,dict(purpose='integration_pilot',storage_codec=SHUFFLE_CODEC,source_sha256=file_sha(path),source_contract_id='raw-contract',selector_contract='fixture',records=records)

def validate(s,c):
 for view in c['view_indices']:compact_head_inputs(c,view)

def test_real_tensors_multiple_sources_mixed_reader(tmp_path):
 from kfold.affinity.conversion import convert_sealed_shard
 a,pa=source(tmp_path,'a',['a','b']);b,pb=source(tmp_path,'b',['c','d'])
 c,pc=source(tmp_path,'old',['e'])
 manifest=convert_sealed_shard(c,tmp_path/'legacy',pc,check_quota=lambda:None,head_validator=validate)
 old=sqlite3.connect(tmp_path/'old-catalog.sqlite');old.execute('CREATE TABLE locations(source_sha,plan_sha,manifest_path,manifest_sha,state)')
 mp=tmp_path/'legacy/complete.json'
 old.execute('INSERT INTO locations VALUES (?,?,?,?,?)',(pc['source_sha256'],digest(pc),str(mp),file_sha(mp),'complete'));old.commit();old.close()
 cat=BundleCatalog(tmp_path/'catalog.sqlite');cat.import_legacy(tmp_path/'old-catalog.sqlite')
 store=BundleStore(tmp_path/'bundle.sqlite',CONTRACT,16*1024**2)
 for path,plan in [(a,pa),(b,pb)]:append_source(store,path,plan,head_validator=validate)
 cat.claim('one',store.path,CONTRACT,store.max_bytes,[{'receipt':{'sha256':p['source_sha256']}} for p in (pa,pb)])
 receipt=store.seal();cat.publish('one',receipt)
 for key in ['a','b','c','d','e']:
  payload=read_payload(cat.path,key);assert payload['receipt']['system_id']==key
  validate(None,payload)
 assert len(list(iter_encoded(cat.path)))==5
 assert file_sha(a)==pa['source_sha256'] and file_sha(b)==pb['source_sha256']
 assert append_source(store,a,pa,head_validator=validate)['reused']
 store.close();cat.close()

def test_loader_failure_rolls_back_and_reuses_same_plan(tmp_path):
 path,plan=source(tmp_path,'a',['a','b']);store=BundleStore(tmp_path/'bundle.sqlite',CONTRACT,16*1024**2)
 def fail(s,c):raise RuntimeError('head loader')
 with pytest.raises(RuntimeError):append_source(store,path,plan,head_validator=fail)
 assert store.db.execute('SELECT count(*) FROM records').fetchone()[0]==0
 changed=copy.deepcopy(plan);changed['records']['a']['crop_receipt']['config']['window']=22
 with pytest.raises(ValueError,match='changed on resume'):append_source(store,path,changed,head_validator=validate)
 append_source(store,path,plan,head_validator=validate);assert store.seal()['records']==2;store.close()

@pytest.mark.parametrize('defect',['missing','pending','sha','purpose','codec'])
def test_bad_source_plan_never_publishes(tmp_path,defect):
 path,plan=source(tmp_path,'a',['a']);store=BundleStore(tmp_path/'bundle.sqlite',CONTRACT,16*1024**2)
 if defect=='missing':plan['records']={}
 if defect=='pending':plan['records']['a']['status']='pending'
 if defect=='sha':plan['source_sha256']='0'*64
 if defect=='purpose':plan['purpose']='training_cache'
 if defect=='codec':plan['storage_codec']='zlib1'
 with pytest.raises(ValueError):append_source(store,path,plan,head_validator=validate)
 assert not store.path.exists();store.close()
