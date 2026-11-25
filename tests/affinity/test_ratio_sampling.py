from collections import Counter
import pytest
from kfold.affinity.ratio_sampling import ratio_layout,RatioJointBatchSampler


def test_global_10000_update_exposure_and_provenance_are_exact():
    totals=Counter()
    for update in range(10000):
        p=ratio_layout(update)
        assert len(p['regression_cells'])==p['regression_groups']
        assert all(cell%3<2 for cell in p['regression_cells'])
        assert len(p['regression_singles'])==p['continuous_singletons']
        assert p['regression_singles'].isdisjoint(p['binary_single_roles'])
        assert p['regression_singles']|set(p['binary_single_roles'])==set(range(64))
        assert p['continuous']+p['binary']==1024
        totals['continuous']+=p['continuous'];totals['groups']+=p['regression_groups']
        totals['singles']+=p['continuous_singletons']
        binary_groups=192-p['regression_groups']
        totals['positive']+=binary_groups
        totals['measured_negative']+=2*binary_groups
        totals['synthetic_negative']+=2*binary_groups
        roles=Counter(p['binary_single_roles'].values())
        totals['positive']+=roles[1,0];totals['measured_negative']+=roles[0,0]
        totals['synthetic_negative']+=roles[0,1]
    assert totals['continuous']==2611200
    assert totals['positive']==1525760
    assert totals['measured_negative']==totals['synthetic_negative']==3051520
    assert abs(5*totals['groups']-10*totals['singles'])<=50


class Fake:
    singleton_pool=[91,92]
    def __init__(self,batches,rows):self.batches,self.rows=batches,rows
    def __len__(self):return self.batches
    def __iter__(self):yield from [self.rows]*self.batches


def sampler(rank,start=0):
    return RatioJointBatchSampler(Fake(8,list(range(10))+[(91,'absolute')]),
        Fake(24,[201,202,203,204,205]),{(1,0):[301],(0,0):[302],(0,1):[303]},
        rank=rank,start=start,bound_count=1,seed=rank+33)


def test_complete_groups_physical_batches_global_counts_and_deterministic_resume():
    batches=[list(sampler(rank)) for rank in range(64)]
    for rank in range(64):
        assert list(sampler(rank,3))==batches[rank][3:]
    for update in range(8):
        continuous=0
        for rank in range(64):
            batch=batches[rank][update];assert len(batch)==16
            continuous+=sum(task!='classification' for task,_,_ in batch)
            for i in range(3):
                group=batch[i*5:i*5+5]
                assert len({x[0] for x in group})==len({x[2] for x in group})==1
        assert continuous==ratio_layout(update)['continuous']


def test_layout_seed_and_rank_validation():
    assert ratio_layout(10)==ratio_layout(10)
    assert ratio_layout(10,seed=1)!=ratio_layout(10,seed=2)
    with pytest.raises(ValueError):ratio_layout(-1)
    with pytest.raises(ValueError):sampler(64)
