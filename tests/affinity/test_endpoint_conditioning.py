import sqlite3
import pytest
import torch
from torch import nn
from kfold.affinity.endpoint_conditioning import (EndpointJointAffinityHead, EndpointReader,
    conditioned_rows, ENDPOINTS)
from kfold.affinity.joint_training import joint_loss_terms

def row(i=0, **kwargs):
    return dict(row_id=i, input_id=str(i), assay_key='assay', endpoint='IC50', loss_group='g',
                synthetic=False,p_activity=6.+i,relation='=',binary=1,classification_valid=True,**kwargs)

def test_dropout_keeps_actual_labels_and_group_context():
    original=[row(0),row(1)];out,idx=conditioned_rows(original,[0,0],training=True,probability=1)
    assert idx==[4,4] and [r['endpoint'] for r in out]==['Kd','Kd']
    assert all(r['endpoint']=='IC50' for r in original)
    assert [r['p_activity'] for r in out]==[6.,7.]
    assert out[0]['loss_group']==out[1]['loss_group']

def test_unknown_and_evaluation_are_never_randomized():
    out,idx=conditioned_rows([row(),row(1)],[4,1],training=False,probability=1)
    assert idx==[4,1]
    assert conditioned_rows([row()],[4],training=True,probability=1)[1]==[4]

def test_dropout_reproducible_and_rate():
    a=[conditioned_rows([row()],[2],training=True,seed=42,update=i)[1][0] for i in range(10000)]
    b=[conditioned_rows([row()],[2],training=True,seed=42,update=i)[1][0] for i in range(10000)]
    assert a==b and .13 < a.count(4)/len(a) < .17

def test_recovered_endpoints_split_difference_groups():
    rows,_=conditioned_rows([row(0),row(1)],[0,1])
    out=dict(value=torch.tensor([6.,7.],requires_grad=True), logits=torch.zeros(2,requires_grad=True))
    terms=joint_loss_terms(out,rows)
    assert terms.counts['absolute']==2 and terms.counts['difference']==0
    same,_=conditioned_rows([row(0),row(1)],[0,0],training=True,probability=1)
    assert joint_loss_terms(out,same).counts['difference']==1

def test_endpoint_changes_only_numeric_readout_and_unknown_has_gradient():
    # Exercise the exact production forward with controlled layers rather than a costly trunk.
    model=EndpointJointAffinityHead.__new__(EndpointJointAffinityHead);nn.Module.__init__(model)
    class Shared(nn.Module):
        def forward(self, *, x):return x
    model.shared=Shared();model.value_head=nn.Linear(8,1);model.binary_head=nn.Linear(8,1)
    model.endpoint_embedding=nn.Embedding(5,8)
    x=torch.randn(3,8)
    a=model(x=x,endpoint_index=torch.zeros(3,dtype=torch.long))
    b=model(x=x,endpoint_index=torch.full((3,),4,dtype=torch.long))
    assert torch.equal(a['logits'],b['logits']) and not torch.equal(a['value'],b['value'])
    b['value'].sum().backward()
    assert model.endpoint_embedding.weight.grad[4].abs().sum()>0
    assert model.endpoint_embedding.weight.grad[:4].abs().sum()==0
    state=model.state_dict();model.load_state_dict(state,strict=True)

@pytest.mark.parametrize('bad',[-1,5,None])
def test_invalid_condition_fails(bad):
    with pytest.raises(ValueError):conditioned_rows([row()],[bad])

def test_reader_exact_id_order_missing_and_synthetic(tmp_path):
    path=tmp_path/'overlay.sqlite';db=sqlite3.connect(path)
    db.execute('CREATE TABLE endpoint_overlay(row_id INTEGER PRIMARY KEY,endpoint_index INTEGER)')
    db.executemany('INSERT INTO endpoint_overlay VALUES (?,?)',[(0,2),(1,4)]);db.commit();db.close()
    r=EndpointReader(path)
    assert r.indices([row(1),row(0),row(0)])==[4,2,2]
    with pytest.raises(ValueError):r.indices([row(3)])
    bad=row(0);bad['synthetic']=True;bad['p_activity']=None
    with pytest.raises(ValueError):r.indices([bad])
    r.close()

def test_synthetic_mask_and_original_relation_preserved():
    x=row();x.update(synthetic=True,p_activity=None,binary=0)
    rows,idx=conditioned_rows([x],[4],training=True,probability=1)
    out=dict(value=torch.ones(1,requires_grad=True),logits=torch.zeros(1,requires_grad=True))
    terms=joint_loss_terms(out,rows)
    assert terms.counts==dict(absolute=0,difference=0,binary=1) and idx==[4]

def test_real_head_state_roundtrip_and_endpoint_gradient():
    from kfold.affinity.heads import HeadConfig
    config=HeadConfig(variant='colleague_cross',channel_s=16,channel_z=16,blocks=1,heads=4,dropout=0.)
    torch.manual_seed(6);head=EndpointJointAffinityHead(config).train()
    nn.init.normal_(head.value_head[-1].weight,std=.1)
    n=6;p=torch.arange(n)[None]<4
    x=dict(s_inputs=torch.randn(1,n,16),s_lm=torch.randn(1,n,16),z=torch.randn(1,n,n,16),
           distogram_features=torch.randn(1,n,n,3),token_mask=torch.ones(1,n,dtype=torch.bool),protein_mask=p,ligand_mask=~p)
    out=head(**x,endpoint_index=torch.tensor([4]));out['value'].sum().backward()
    assert head.endpoint_embedding.weight.grad[4].abs().sum()>0
    head.eval()
    with torch.no_grad():out=head(**x,endpoint_index=torch.tensor([4]))
    restored=EndpointJointAffinityHead(config).eval();restored.load_state_dict(head.state_dict(),strict=True)
    torch.testing.assert_close(out['value'],restored(**x,endpoint_index=torch.tensor([4]))['value'])
    torch.testing.assert_close(out['logits'],head(**x,endpoint_index=torch.tensor([1]))['logits'],atol=0,rtol=0)

def test_external_metrics_keep_frozen_weighting_and_wandb_keys():
    from kfold.affinity.endpoint_external import reports,scalars
    rows=[dict(benchmark='FEP+4',target_id=t,p_activity=float(i),prediction=float(i)+.5,
               overlap={'chemical_train':False}) for t in ('a','b') for i in range(3)]
    result=reports(rows);scalar=scalars(result)
    assert scalar['external/FEPplus4/n']==6 and scalar['external/FEPplus4/mae']==.5
    assert abs(scalar['external/FEPplus4/target_weighted_pearson']-1)<1e-6
    assert scalar['external/FEPplus4/chemical_nonoverlap/n']==6
