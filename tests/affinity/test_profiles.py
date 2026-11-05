import pytest
import torch

from kfold.affinity.profiles import distogram_profile


def source(seed=1):
    g=torch.Generator().manual_seed(seed)
    p=torch.tensor([True,False,True,False,True,True])
    n=len(p)
    logits=torch.randn(n,n,64,generator=g)*3
    logits=(logits+logits.transpose(0,1)).to(torch.bfloat16)
    idx=torch.triu_indices(n,n).T
    idx=idx[~(p[idx[:,0]] & p[idx[:,1]])]
    return dict(protein_mask=p,ligand_mask=~p,distogram_tri_indices=idx,
                distogram_tri_logits=logits[idx[:,0],idx[:,1]],
                bin_boundaries=torch.linspace(2+20/64,22-20/64,63)),logits


@pytest.mark.parametrize('seed', range(5))
def test_sparse_profile_matches_dense_reference_reduction(seed):
    data,dense=source(seed)
    result=distogram_profile(data)
    p=data['protein_mask']; prob=dense[p][:,~p].float().softmax(-1)
    centers=torch.linspace(2+10/64,22-10/64,64)
    distance=(prob*centers).sum(-1).amin(-1)
    entropy=-(prob*prob.clamp_min(1e-8).log()).sum(-1)/torch.log(torch.tensor(64.))
    pocket=distance<15
    torch.testing.assert_close(result['protein_min_expected_distance'],distance)
    torch.testing.assert_close(result['protein_mean_normalized_entropy'],entropy.mean(-1))
    assert result['pocket_hlp_15a']==pytest.approx(float(entropy[pocket].mean()),abs=1e-6)


def test_empty_pocket_and_missing_pairs_are_explicit():
    data,_=source()
    data['distogram_tri_logits'].fill_(-100)
    data['distogram_tri_logits'][:,-1]=100
    result=distogram_profile(data)
    assert result['pocket_residue_count']==0 and result['pocket_hlp_15a'] is None
    data['distogram_tri_indices']=data['distogram_tri_indices'][1:]
    data['distogram_tri_logits']=data['distogram_tri_logits'][1:]
    with pytest.raises(ValueError,match='coverage'):
        distogram_profile(data)
