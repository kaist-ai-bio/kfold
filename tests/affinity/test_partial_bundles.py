import copy
import json
import sqlite3
import pytest
from test_bundles import source, validate, CONTRACT
from kfold.affinity.bundles import BundleStore, read_payload, file_sha
from kfold.affinity.bundle_conversion import append_source
from kfold.affinity.partial_bundles import PartialBundleCatalog, selection, OWNER

SUB = dict(CONTRACT, coverage_scope='record_subset_v1')

def split_plan(path, full, keys):
 p=copy.deepcopy(full);p['selection']=selection(full['records'],keys)
 p['records']={k:v for k,v in p['records'].items() if k in keys}
 receipt=json.loads(path.with_suffix('.receipt.json').read_text())
 return p,dict(receipt=receipt,identity_keys={k:'target' for k in keys},selection=p['selection'])

def test_deferred_records_can_append_later_without_touching_first_bundle(tmp_path):
 path,full=source(tmp_path,'raw',['a','b','c']);original=file_sha(path)
 cat=PartialBundleCatalog(tmp_path/'catalog.sqlite')
 first_sha=None
 for bid,keys in [('first',['a','c']),('second',['b'])]:
  plan,job=split_plan(path,full,keys);store=BundleStore(tmp_path/(bid+'.sqlite'),SUB,16*1024**2)
  cat.claim(bid,store.path,SUB,store.max_bytes,[job])
  # Exact repeated assignment is idempotent before and after publication.
  cat.claim(bid,store.path,SUB,store.max_bytes,[job])
  assert append_source(store,path,plan,head_validator=validate)['records']==len(keys)
  rec=store.seal();cat.publish(bid,rec);cat.publish(bid,rec);store.close()
  if bid=='first':
   first_sha=file_sha(tmp_path/'first.sqlite')
   with pytest.raises(KeyError):read_payload(cat.path,'b')
  else:assert file_sha(tmp_path/'first.sqlite')==first_sha
 assert cat.db.execute('SELECT owner FROM sources').fetchone()[0]==OWNER
 assert cat.db.execute('SELECT count(*) FROM source_segments').fetchone()[0]==2
 assert cat.db.execute('SELECT count(*) FROM records').fetchone()[0]==3
 for key in ['a','b','c']:validate(None,read_payload(cat.path,key))
 assert file_sha(path)==original
 cat.close()

def test_pending_overlap_and_resume_mutation_rejected_atomically(tmp_path):
 path,full=source(tmp_path,'raw',['a','b']);cat=PartialBundleCatalog(tmp_path/'cat.sqlite')
 plan,job=split_plan(path,full,['a']);cat.claim('first',tmp_path/'first.sqlite',SUB,16*1024**2,[job])
 with pytest.raises(sqlite3.IntegrityError):cat.claim('overlap',tmp_path/'other.sqlite',SUB,16*1024**2,[job])
 assert cat.db.execute('SELECT count(*) FROM bundles').fetchone()[0]==1
 _,changed=split_plan(path,full,['b'])
 with pytest.raises(ValueError,match='changed on resume'):cat.claim('first',tmp_path/'first.sqlite',SUB,16*1024**2,[changed])
 cat.close()

@pytest.mark.parametrize('defect',['overlap','missing','foreign','unversioned'])
def test_bad_partition_rejected_before_payload_write(tmp_path,defect):
 path,full=source(tmp_path,'raw',['a','b']);plan,job=split_plan(path,full,['a'])
 if defect=='overlap':plan['selection']['unselected_ids']=['a','b']
 if defect=='missing':plan['selection']['source_ids']=['a']
 if defect=='foreign':plan['selection']['source_ids'].append('z')
 store=BundleStore(tmp_path/'b.sqlite',CONTRACT if defect=='unversioned' else SUB,16*1024**2)
 with pytest.raises(ValueError):append_source(store,path,plan,head_validator=validate)
 assert store.db.execute('SELECT count(*) FROM records').fetchone()[0]==0
 store.close()

def test_failure_rolls_back_subset_and_frozen_resume_succeeds(tmp_path):
 path,full=source(tmp_path,'raw',['a','b','c']);plan,job=split_plan(path,full,['a','c'])
 store=BundleStore(tmp_path/'b.sqlite',SUB,16*1024**2)
 def fail(s,c):
  if c['receipt']['system_id']=='c':raise RuntimeError('injected')
 with pytest.raises(RuntimeError):append_source(store,path,plan,head_validator=fail)
 assert store.db.execute('SELECT count(*) FROM records').fetchone()[0]==0
 store.close();store=BundleStore(tmp_path/'b.sqlite',SUB,16*1024**2)
 changed,_=split_plan(path,full,['b'])
 with pytest.raises(ValueError,match='changed on resume'):append_source(store,path,changed,head_validator=validate)
 append_source(store,path,plan,head_validator=validate);assert store.seal()['records']==2;store.close()

def test_publish_rejects_plan_outside_claimed_subset(tmp_path):
 path,full=source(tmp_path,'raw',['a','b']);plan,job=split_plan(path,full,['a']);other,_=split_plan(path,full,['b'])
 cat=PartialBundleCatalog(tmp_path/'cat.sqlite');store=BundleStore(tmp_path/'b.sqlite',SUB,16*1024**2)
 cat.claim('x',store.path,SUB,store.max_bytes,[job]);append_source(store,path,other,head_validator=validate)
 with pytest.raises(ValueError,match='differs'):cat.publish('x',store.seal())
 assert cat.db.execute('SELECT count(*) FROM records').fetchone()[0]==0
 assert cat.db.execute('SELECT state FROM bundles').fetchone()[0]=='pending'
 store.close();cat.close()
