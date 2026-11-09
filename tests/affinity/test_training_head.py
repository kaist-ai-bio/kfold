import copy
import pytest
import torch

from kfold.affinity.heads import AffinityHead, HeadConfig, SingleOuterProduct
from kfold.affinity.supervision import chemical_split, task_targets, censored_huber, objective
from kfold.affinity.training_data import collate_cached, TaskBatchSampler, CachedLabelDataset
from kfold.affinity.pilot import metrics
from kfold.model.modules.affinity_pairformer import AffinityPairformer


def features(n=6):
    p=torch.arange(n)[None]<n-2
    return dict(s_inputs=torch.randn(1,n,16),s_lm=torch.randn(1,n,16),z=torch.randn(1,n,n,16),
        distogram_features=torch.randn(1,n,n,3),token_mask=torch.ones(1,n,dtype=torch.bool),
        protein_mask=p,ligand_mask=~p)


def config(variant):
    return HeadConfig(variant=variant,channel_s=16,channel_z=16,blocks=1,heads=4,dropout=0.,outer_rank=3)


def test_baseline_preserves_colleague_forward():
    torch.manual_seed(91);head=AffinityHead(config('colleague_cross')).eval()
    reference=AffinityPairformer(AffinityPairformer.Config(channel_s=16,channel_z=16,
        num_blocks=1,num_heads_attn=4,dropout=0.,blocks_per_ckpt=1)).eval()
    reference.load_state_dict(head.base.state_dict())
    x=features()
    with torch.no_grad():
        torch.nn.init.normal_(head.base.readout[-1].weight,std=.1)
        reference.load_state_dict(head.base.state_dict())
        assert torch.equal(head(**x),reference(**x))


def test_outer_uses_genuine_directed_channel_product_and_protein_mask():
    torch.manual_seed(1);m=SingleOuterProduct(16,8,rank=3,chunk=2)
    torch.nn.init.normal_(m.output.weight)
    s=torch.randn(1,6,16);p=torch.tensor([[1,1,1,1,0,0]],dtype=torch.bool)
    a,b=m.left(m.norm(s)),m.right(m.norm(s))
    expected=m.output(torch.einsum('bir,bjs->bijrs',a,b).flatten(-2))
    expected*= (p[:,:,None]&p[:,None,:])[...,None]
    assert torch.allclose(m(s,p),expected,atol=1e-6)
    assert not torch.allclose(expected,expected.transpose(1,2))


def test_full_pair_control_and_outer_start_identically_then_outer_learns():
    torch.manual_seed(5);a=AffinityHead(config('full_pair_control'))
    torch.manual_seed(5);b=AffinityHead(config('slm_outer'))
    x=features();opt=torch.optim.AdamW(b.parameters(),lr=.01)
    torch.nn.init.normal_(a.base.readout[-1].weight,std=.1)
    b.load_state_dict(a.state_dict())
    assert torch.equal(a(**x),b(**x))
    for _ in range(3):
        opt.zero_grad();(b(**x)-1).square().mean().backward();opt.step()
    assert b.outer.output.weight.grad.abs().sum()>0
    assert b.outer.left.weight.grad.abs().sum()>0
    assert not torch.equal(a(**x),b(**x))


@pytest.mark.parametrize('relation,value,binary',[('=',6,1),('=',5,0),('<',6,1),('<=',6,1),
    ('>',6,0),('>=',6,None),('>=',5,0),('<',5,None),('>',7,None)])
def test_censor_bounds_never_become_point_labels(relation,value,binary):
    row=task_targets(value,relation)
    assert row['regression_valid']==(relation=='=')
    assert row['classification_valid']==(binary is not None)
    assert row['binary']==binary


def test_synthetic_has_no_regression_target():
    row=task_targets(None,synthetic=True)
    assert row['binary']==0 and not row['regression_valid']
    with pytest.raises(ValueError):task_targets(4,synthetic=True)


def test_one_sided_bound_loss_and_zero_valid_handling():
    p=torch.tensor([5.,7.],requires_grad=True)
    assert censored_huber(p,torch.tensor([6.,6.]),['<','>'])==0
    assert censored_huber(p,torch.tensor([6.,6.]),['>','<'])>0
    rows=[dict(task_targets(None,synthetic=True),assay_key='a',chemical_key='x')]*2
    with pytest.raises(ValueError):objective(p,rows,'regression')


def test_chemistry_group_split_reproducible_across_identity_aliases():
    assert chemical_split('same chemical')==chemical_split('same chemical')
    assert {chemical_split(str(i)) for i in range(100)}=={'train','validation'}


@pytest.mark.parametrize('variant',['colleague_cross','full_pair_control','slm_outer'])
def test_collation_padding_does_not_change_prediction(variant):
    torch.manual_seed(17);h=AffinityHead(config(variant)).eval()
    torch.nn.init.normal_(h.base.readout[-1].weight,std=.1)
    x,y=features(6),features(9);batch,_=collate_cached([(x,{}),(y,{})])
    with torch.no_grad():
        assert torch.allclose(h(**batch),torch.cat([h(**x),h(**y)]),atol=1e-5,rtol=1e-5)


def test_classifier_metrics_include_ties_and_single_class():
    rows=[dict(binary=v) for v in [0,1,0,1]]
    result=metrics(torch.tensor([.1,.2,.2,.3]),rows,'classification')
    assert result['AUROC']==pytest.approx(.875)
    assert result['AUPRC']==pytest.approx(5/6)
    result=metrics(torch.zeros(2),[dict(binary=1)]*2,'classification')
    assert result['AUROC'] is None and result['AUPRC']==1.


def test_attached_database_uri_remains_read_only(tmp_path):
    import sqlite3
    source=tmp_path/'source.sqlite'
    d=sqlite3.connect(source);d.execute('CREATE TABLE evidence(x)');d.commit();d.close()
    d=sqlite3.connect(tmp_path/'derived.sqlite',uri=True)
    d.execute('ATTACH DATABASE ? AS source',(source.resolve().as_uri()+'?mode=ro&immutable=1',))
    assert d.execute('SELECT count(*) FROM source.evidence').fetchone()[0]==0
    with pytest.raises(sqlite3.OperationalError,match='readonly'):
        d.execute('INSERT INTO source.evidence VALUES(1)')
    d.close()


def test_task_sampler_preserves_assay_layout_and_negative_provenance(tmp_path):
    import sqlite3
    path=tmp_path/'labels.sqlite';d=sqlite3.connect(path)
    d.executescript('CREATE TABLE examples(row_id INTEGER,assay_key TEXT,input_id TEXT,p_activity REAL,synthetic INTEGER,binary REAL); CREATE TABLE task_index(task TEXT,split TEXT,position INTEGER,row_id INTEGER);')
    rows=[]
    for assay in range(4):
        for ligand in range(5):rows.append((len(rows),str(assay),str(ligand),float(ligand) if assay<3 else 5.,0,float(ligand>=3)))
    for i in range(3):rows.append((len(rows),'regression_only:'+str(i),'single'+str(i),5.,0,0.))
    for i in range(3):rows.append((len(rows),'synthetic'+str(i),'decoy'+str(i),None,1,0.))
    d.executemany('INSERT INTO examples VALUES(?,?,?,?,?,?)',rows)
    for task in ('regression','classification'):
        admitted=[r for r in rows if task=='classification' or not r[4]]
        d.executemany('INSERT INTO task_index VALUES(?,?,?,?)',[(task,'train',i,r[0]) for i,r in enumerate(admitted)])
    d.commit();d.close()
    class Data:
        split='train'
        def connect(self):return sqlite3.connect(path)
    dataset=Data();dataset.task='regression'
    sampler=TaskBatchSampler(dataset,batches=3,groups=2,group_size=3,singleton_slots=2,batch_size=8)
    batches=list(sampler);assert batches==list(sampler)
    assert '3' not in sampler.keys  # Constant assays have zero IQR.
    for batch in batches:
        selected=[rows[i[0] if isinstance(i,tuple) else i] for i in batch]
        assert len({r[1] for r in selected[:3]})==len({r[1] for r in selected[3:6]})==1
        assert selected[0][1]!=selected[3][1]
        assert all(r[1].startswith('regression_only:') or r[1]=='3' for r in selected[6:])
        assert all(isinstance(i,tuple) for i in batch[6:])
        assert len({i[1] for i in batch[6:]})==2
    dataset.task='classification'
    for batch in TaskBatchSampler(dataset,batches=3,batch_size=4):
        selected=[rows[i] for i in batch]
        assert sum(r[5]==1 for r in selected)==2
        assert sum(r[5]==0 and r[4]==0 for r in selected)==1
        assert sum(r[4]==1 for r in selected)==1


def test_metadata_join_scans_unindexed_source_once():
    import sqlite3
    from kfold.affinity.training_manifest import MEASURED_CHEMICAL_KEYS_SQL
    d=sqlite3.connect(':memory:')
    d.execute("ATTACH ':memory:' AS meta");d.execute("ATTACH ':memory:' AS registry")
    d.executescript('CREATE TABLE meta.train(system_id TEXT,eligible_for_training INTEGER); CREATE TABLE registry.aliases(cohort TEXT,source_id TEXT,input_id TEXT,PRIMARY KEY(cohort,source_id)); CREATE TABLE registry.systems(input_id TEXT PRIMARY KEY,ligand_sha TEXT); CREATE TABLE chemical_groups(ligand_sha TEXT PRIMARY KEY,chemical_key TEXT);')
    d.execute("INSERT INTO meta.train VALUES('source',1)")
    d.execute("INSERT INTO registry.aliases VALUES('train','source','input')")
    d.execute("INSERT INTO registry.systems VALUES('input','ligand')")
    d.execute("INSERT INTO chemical_groups VALUES('ligand','group')")
    assert d.execute(MEASURED_CHEMICAL_KEYS_SQL).fetchall()==[('group',)]
    plan=[r[-1] for r in d.execute('EXPLAIN QUERY PLAN '+MEASURED_CHEMICAL_KEYS_SQL)]
    assert plan[0]=='SCAN m'
    assert not any(x=='SCAN a' or x=='SCAN s' for x in plan)
    d.close()


def test_partition_group_does_not_collapse_distinct_ligand_measurements():
    rows=[dict(task_targets(v),assay_key='same_target_and_assay',chemical_key='same_parent_group',
        input_id=identity,synthetic=False) for v,identity in [(5.,'stereo-A'),(7.,'stereo-B')]]
    p=torch.tensor([5.,5.],requires_grad=True)
    loss=objective(p,rows,'regression')
    # Absolute mean Huber: .4375; difference Huber: .875.
    assert float(loss.detach())==pytest.approx(.1*.4375+.9*.875)
    loss.backward();assert p.grad[0]!=p.grad[1]
    absolute_rows=[dict(r,loss_group='absolute:'+str(i)) for i,r in enumerate(rows)]
    assert float(objective(p.detach(),absolute_rows,'regression'))==pytest.approx(.1*.4375)


def test_view_exclusions_translate_positions_without_losing_labels(tmp_path):
    import json,sqlite3
    p=tmp_path/'labels.sqlite';d=sqlite3.connect(p)
    d.executescript('CREATE TABLE examples(row_id INTEGER PRIMARY KEY,input_id TEXT); CREATE TABLE task_index(task TEXT,split TEXT,position INTEGER,row_id INTEGER);')
    d.executemany('INSERT INTO examples VALUES(?,?)',[(i,'input'+str(i)) for i in range(7)])
    d.executemany('INSERT INTO task_index VALUES(?,?,?,?)',[('classification','train',i,i) for i in range(7)])
    d.commit();d.close()
    (tmp_path/'completion.json').write_text(json.dumps(dict(manifest_sha256='pin')))
    with pytest.raises(ValueError,match='availability'):
        CachedLabelDataset(p,'unused',task='classification',split='train')
    (tmp_path/'view-availability.json').write_text(json.dumps(dict(manifest_sha256='pin',excluded_positions={
        'query_200':{'classification':{'train':[1,2,5]}},'query_350':{'classification':{'train':[]}}})))
    a=CachedLabelDataset(p,'unused',task='classification',split='train')
    b=CachedLabelDataset(p,'unused',task='classification',split='train',view='query_350')
    assert [a.label(i)['row_id'] for i in range(len(a))]==[0,3,4,6]
    assert [a.logical_position(i) for i in range(7)]==[0,None,None,1,2,None,3]
    assert len(b)==7 and b.label(5)['row_id']==5
    with pytest.raises(IndexError):a.label(-1)


def test_shared_query_flags_require_alias_agreement():
    from kfold.affinity.view_eligibility import query_flags
    q=dict(status='final',views={'query_350':[1,2]})
    assert query_flags(q)=={'query_200':False,'query_350':True}
    assert query_flags(dict(historical_plans={'a':q,'b':copy.deepcopy(q)}))['query_350']
    with pytest.raises(ValueError,match='disagree'):
        query_flags(dict(historical_plans={'a':q,'b':dict(status='final',views={'query_200':[1],'query_350':[1,2]})}))
