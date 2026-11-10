import copy
import pytest
import torch
from kfold.affinity.distributed_training import microbatch_positions,objective_details,colleague_validation_metrics,rank_seed
from kfold.affinity.supervision import objective


def rows64(task):
    if task=='classification':
        return [dict(binary=float(i<32),classification_valid=True,synthetic=i>=48) for i in range(64)]
    result=[dict(assay_key=f'a{i//5}',input_id=f'lig{i}',p_activity=(i%5)*.7+i//5,
        regression_valid=True,synthetic=False) for i in range(60)]
    result.extend(dict(assay_key='original_singleton',loss_group=f'__absolute_only:{i}',
        input_id=f's{i}',p_activity=float(i),regression_valid=True,synthetic=False) for i in range(4))
    return result


@pytest.mark.parametrize('task',['regression','classification'])
def test_logical64_micro16_preserves_full_objective_and_gradients(task):
    torch.manual_seed(37);rows=rows64(task);full=torch.randn(64,requires_grad=True);micro=full.detach().clone().requires_grad_()
    loss=objective(full,rows,task);loss.backward();parts=[]
    positions=microbatch_positions(task)
    assert sorted(sum(positions,[]))==list(range(64))
    for indices in positions:
        value,_=objective_details(micro[indices],[rows[i] for i in indices],task)
        parts.append(value.detach()/4);(value/4).backward()
    torch.testing.assert_close(sum(parts),loss.detach(),atol=1e-7,rtol=1e-6)
    torch.testing.assert_close(full.grad,micro.grad,atol=1e-7,rtol=1e-6)
    if task=='regression':
        for indices in positions:
            assert len(set(rows[i]['assay_key'] for i in indices[:15]))==3
            assert 'loss_group' in rows[indices[-1]]


def test_colleague_validation_distinguishes_pooled_and_assay_metrics():
    rows=[dict(p_activity=float(x),assay_key=a,input_id=str(i)) for a,values in [('a',[0,1,2]),('b',[3,4,5])] for i,x in enumerate(values)]
    p=torch.tensor([0.,1.,2.,5.,4.,3.]);r=colleague_validation_metrics(p,rows)
    assert r['mean_assay_pearson']==pytest.approx(0.)
    assert r['pairwise_accuracy']==pytest.approx(.5) and r['pairwise_comparisons']==6
    duplicate=rows+[dict(rows[0])];q=torch.cat([p,p[:1]])
    assert colleague_validation_metrics(q,duplicate)['ranking_replicate_records']==1


def test_rank_seed_reproducible_and_distinct():
    assert len({rank_seed(20260822,i) for i in range(8)})==8
    assert rank_seed(20260822,0)==rank_seed(20260822,0)
