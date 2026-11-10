import copy
import pytest
import torch
from kfold.affinity.heads import SingleProductDifference,AffinityHead,HeadConfig
from kfold.affinity.joint_training import JointAffinityHead


def features(n=6):
    p=torch.arange(n)[None]<n-2
    return dict(s_inputs=torch.randn(1,n,16),s_lm=torch.randn(1,n,16),z=torch.randn(1,n,n,16),
        distogram_features=torch.randn(1,n,n,3),token_mask=torch.ones(1,n,dtype=torch.bool),
        protein_mask=p,ligand_mask=~p)


def test_proddiff_channels_and_signed_order_are_exact():
    torch.manual_seed(13);m=SingleProductDifference(16,8,chunk=2)
    torch.nn.init.normal_(m.output.weight)
    s=torch.randn(2,7,16);mask=torch.tensor([[1,1,1,1,0,0,0],[1,1,1,0,0,0,0]],dtype=torch.bool)
    x=m.norm(s);p=x[:,:,None,:]*x[:,None,:,:];d=x[:,:,None,:]-x[:,None,:,:]
    expected=m.output(torch.cat((p,d),dim=-1))*(mask[:,:,None]&mask[:,None,:])[...,None]
    assert m.output.in_features==32 and m.output.out_features==8
    torch.testing.assert_close(m(s,mask),expected,atol=1e-6,rtol=1e-5)
    assert not torch.allclose(expected,expected.transpose(1,2))
    assert m.output.weight.shape==(8,32)


def test_checkpointed_chunks_match_dense_gradients():
    torch.manual_seed(9);m=SingleProductDifference(16,8,chunk=2);torch.nn.init.normal_(m.output.weight,std=.1)
    reference=copy.deepcopy(m);s=torch.randn(2,5,16,requires_grad=True);t=s.detach().clone().requires_grad_()
    mask=torch.tensor([[1,1,1,0,0],[1,1,1,1,0]],dtype=torch.bool)
    actual=m(s,mask);x=reference.norm(t)
    expected=reference.output(torch.cat((x[:,:,None,:]*x[:,None,:,:],x[:,:,None,:]-x[:,None,:,:]),-1))
    expected*= (mask[:,:,None]&mask[:,None,:])[...,None]
    actual.square().sum().backward();expected.square().sum().backward()
    torch.testing.assert_close(s.grad,t.grad,atol=1e-5,rtol=1e-5)
    for a,b in zip(m.parameters(),reference.parameters()):torch.testing.assert_close(a.grad,b.grad,atol=1e-5,rtol=1e-5)


def test_zero_initialized_adapter_preserves_same_topology_control_then_learns():
    args=dict(channel_s=16,channel_z=16,blocks=1,heads=4,dropout=0.)
    torch.manual_seed(4);control=AffinityHead(HeadConfig(variant='full_pair_proddiff_control',**args))
    torch.manual_seed(4);model=AffinityHead(HeadConfig(variant='slm_proddiff',**args))
    assert model.outer is None and model.proddiff is not None
    assert not model.base.config.cross_pair_only
    torch.nn.init.normal_(control.base.readout[-1].weight,std=.1);model.load_state_dict(control.state_dict())
    x=features();torch.testing.assert_close(control(**x),model(**x),atol=0,rtol=0)
    opt=torch.optim.AdamW(model.parameters(),lr=.01)
    for _ in range(3):
        opt.zero_grad();(model(**x)-1).square().mean().backward();opt.step()
    assert model.proddiff.output.weight.grad.abs().sum()>0
    assert model.proddiff.norm.weight.grad.abs().sum()>0
    assert not torch.allclose(control(**x),model(**x))


def test_joint_both_outputs_reach_proddiff_adapter():
    torch.manual_seed(12)
    h=JointAffinityHead(HeadConfig(variant='slm_proddiff',channel_s=16,channel_z=16,blocks=1,heads=4,dropout=0.))
    torch.nn.init.normal_(h.value_head[-1].weight,std=.1);torch.nn.init.normal_(h.binary_head[-1].weight,std=.1)
    # Shared attention/triangle output gates start at zero. Warm them as the
    # actual optimizer does before probing indirect PP->PL/readout derivatives.
    x=features();opt=torch.optim.AdamW(h.parameters(),lr=.01)
    for _ in range(3):
        opt.zero_grad();out=h(**x)
        ((out['value']-1).square().mean()+torch.nn.functional.binary_cross_entropy_with_logits(
            out['logits'],torch.ones_like(out['logits']))).backward();opt.step()
    out=h(**x);w=h.shared.proddiff.output.weight
    a=torch.autograd.grad(out['value'].sum(),w,retain_graph=True)[0]
    b=torch.autograd.grad(out['logits'].sum(),w)[0]
    assert a.abs().sum()>0 and b.abs().sum()>0


@pytest.mark.parametrize('variant',['slm_proddiff','full_pair_proddiff_control'])
def test_padding_and_entity_masks_preserve_proddiff_predictions(variant):
    from kfold.affinity.training_data import collate_cached
    h=AffinityHead(HeadConfig(variant=variant,channel_s=16,channel_z=16,blocks=1,heads=4,dropout=0.)).eval()
    torch.nn.init.normal_(h.base.readout[-1].weight,std=.1);torch.nn.init.normal_(h.proddiff.output.weight,std=.1)
    x,y=features(),features(9);z,_=collate_cached([(x,{}),(y,{})])
    with torch.no_grad():
        torch.testing.assert_close(h(**z),torch.cat((h(**x),h(**y))),atol=1e-5,rtol=1e-5)
