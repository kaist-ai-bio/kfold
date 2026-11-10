import copy
import math
import pytest
import torch
import torch.nn.functional as F

from kfold.affinity.joint_training import (JointAffinityHead,activity_interval,
    joint_loss_terms,normalized_joint_loss,JointBatchSampler)
from kfold.affinity.heads import HeadConfig
from kfold.affinity.reference_losses import boltz2_continuous_affinity_loss


def row(i=0, **kwargs):
    return dict(dict(input_id=str(i),assay_key='assay',endpoint='Ki',p_activity=float(i),
        relation='=',synthetic=False,classification_valid=True,binary=float(i%2)),**kwargs)


@pytest.mark.parametrize('rel,expected',[('=',(6.,6.)),('>',(-math.inf,6.)),
    ('>=',(-math.inf,6.)),('<',(6.,math.inf)),('<=',(6.,math.inf))])
def test_concentration_relation_reverses_on_pactivity(rel,expected):
    assert activity_interval(row(relation=rel,p_activity=6)) == expected


def test_joint_exact_continuous_matches_colleague_reference_and_focal_equation():
    torch.manual_seed(3);values=torch.randn(5,requires_grad=True);logits=torch.randn(5,requires_grad=True)
    rows=[row(i) for i in range(5)]
    terms=joint_loss_terms(dict(value=values,logits=logits),rows)
    total,_=normalized_joint_loss(terms,distributed=False)
    ref=boltz2_continuous_affinity_loss(values,torch.arange(5.),torch.zeros(5,dtype=torch.long),
        torch.arange(5),torch.ones(5,dtype=torch.bool))
    y=torch.tensor([r['binary'] for r in rows]);p=torch.sigmoid(logits)
    focal=-(.8*y*(1-p)*torch.log(p)+.2*(1-y)*p*torch.log(1-p)).mean()
    torch.testing.assert_close(total,ref.loss+focal,check_dtype=False)
    total.backward();assert values.grad.abs().sum()>0 and logits.grad.abs().sum()>0


@pytest.mark.parametrize('relation,pred,expected',[('>',5.,0.),('>',7.,.375),('<',7.,0.),('<',5.,.375)])
def test_censored_absolute_satisfied_bounds_have_no_gradient(relation,pred,expected):
    p=torch.tensor([pred],requires_grad=True)
    terms=joint_loss_terms(dict(value=p,logits=p*0),[row(relation=relation,p_activity=6.,classification_valid=False)])
    assert float(terms.sums['absolute'])==pytest.approx(expected)
    terms.sums['absolute'].backward();assert (p.grad.item()==0)==(expected==0)


def test_censored_differences_and_synthetic_masks():
    # pActivity 5 minus a lower bound6 is <=-1. A predicted difference0 violates it.
    rows=[row(0,p_activity=5.),row(1,p_activity=6.,relation='<')]
    terms=joint_loss_terms(dict(value=torch.zeros(2,requires_grad=True),logits=torch.zeros(2)),rows)
    assert terms.counts['difference']==1 and float(terms.sums['difference'])==pytest.approx(.375)
    rows=[row(0,relation='<'),row(1,relation='<')]
    assert joint_loss_terms(dict(value=torch.zeros(2),logits=torch.zeros(2)),rows).counts['difference']==0
    syn=row(synthetic=True,p_activity=None,binary=0.)
    terms=joint_loss_terms(dict(value=torch.ones(1,requires_grad=True),logits=torch.ones(1,requires_grad=True)),[syn])
    assert terms.counts==dict(absolute=0,difference=0,binary=1)
    loss,_=normalized_joint_loss(terms,distributed=False);loss.backward()
    with pytest.raises(ValueError,match='numeric affinity'):activity_interval(row(synthetic=True))


def test_identical_input_is_never_paired_and_distinct_assays_are_rejected():
    rows=[row(0),row(0,p_activity=1.)];out=dict(value=torch.ones(2),logits=torch.zeros(2))
    assert joint_loss_terms(out,rows).counts['difference']==0
    rows=[row(0,loss_group='same'),row(1,loss_group='same',assay_key='other')]
    with pytest.raises(ValueError,match='distinct assays'):joint_loss_terms(out,rows)


def test_focal_extreme_logits_and_empty_masks_remain_finite():
    out=dict(value=torch.tensor([2.,3.],requires_grad=True),logits=torch.tensor([1000.,-1000.],requires_grad=True))
    rows=[row(0,p_activity=None),row(1,p_activity=None)]
    terms=joint_loss_terms(out,rows);loss,_=normalized_joint_loss(terms,distributed=False)
    loss.backward();assert torch.isfinite(loss) and torch.isfinite(out['logits'].grad).all()
    assert out['value'].grad.eq(0).all()


def test_separate_heads_both_backpropagate_through_the_same_encoder():
    torch.manual_seed(11)
    h=JointAffinityHead(HeadConfig(channel_s=16,channel_z=16,blocks=1,heads=4,dropout=0.))
    torch.nn.init.normal_(h.value_head[-1].weight,std=.1)
    torch.nn.init.normal_(h.binary_head[-1].weight,std=.1)
    n=6;p=torch.tensor([[1,1,1,1,0,0]],dtype=torch.bool)
    x=dict(s_inputs=torch.randn(1,n,16),s_lm=torch.randn(1,n,16),z=torch.randn(1,n,n,16),
        distogram_features=torch.randn(1,n,n,3),token_mask=torch.ones(1,n,dtype=torch.bool),protein_mask=p,ligand_mask=~p)
    out=h(**x);shared=h.shared.base.s_to_z_left.weight
    gv=torch.autograd.grad(out['value'].sum(),shared,retain_graph=True)[0]
    gb=torch.autograd.grad(out['logits'].sum(),shared,retain_graph=True)[0]
    gj=torch.autograd.grad(out['value'].sum()+out['logits'].sum(),shared)[0]
    assert gv.abs().sum()>0 and gb.abs().sum()>0
    torch.testing.assert_close(gj,gv+gb,atol=1e-6,rtol=1e-5)
    assert h.value_head[-1].weight.data_ptr()!=h.binary_head[-1].weight.data_ptr()


def test_joint_batch_sampler_never_splits_groups_and_resume_is_exact():
    class Sampler:
        singleton_pool=[99,100]
        def __init__(self,rows):self.rows=rows
        def __len__(self):return len(self.rows)
        def __iter__(self):return iter(self.rows)
    reg=Sampler([list(range(10))+[(99,'a'),(100,'b')]]*3)
    cl=Sampler([[201,202,203,204]]*3)
    all_batches=list(JointBatchSampler(reg,cl,seed=17,bound_count=1))
    assert list(JointBatchSampler(reg,cl,start=2,seed=17,bound_count=1))==all_batches[2:]
    assert all(len(batch)==16 for batch in all_batches)
    assert all(len({x[2] for x in batch[:5]})==1 and len({x[2] for x in batch[5:10]})==1 for batch in all_batches)


def test_masked_sum_normalization_is_global_not_an_average_of_rank_means():
    # Different label counts on two ranks must recover the full-batch gradient.
    rows=[row(i,assay_key=f'a{i}',classification_valid=i!=0) for i in range(5)]
    p=torch.randn(5,requires_grad=True);b=torch.randn(5,requires_grad=True)
    full=joint_loss_terms(dict(value=p,logits=b),rows);loss,_=normalized_joint_loss(full,distributed=False)
    g=torch.autograd.grad(loss,(p,b),retain_graph=True)
    combined=0
    for indices in ([0],[1,2,3,4]):
        part=joint_loss_terms(dict(value=p[indices],logits=b[indices]),[rows[i] for i in indices])
        combined+=sum(w*part.sums[k]/max(full.counts[k],1) for k,w in [('absolute',.1),('difference',.9),('binary',1.)])
    check=torch.autograd.grad(combined,(p,b))
    for a,c in zip(g,check):torch.testing.assert_close(a,c)
