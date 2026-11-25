"""Fixed-B16 source-ratio ablation with complete five-ligand groups.

Match Boltz-2's aggregate continuous/binary probability, not its unavailable
PubChem/CeMM/MIDAS corpus or same-assay binary-negative construction. Loss
normalization remains unchanged, so this tests exposure, not task loss weights.
"""
import random


def ratio_layout(update, *, world_size=64, seed=20260822):
    """Global deterministic packing, exact 25.5% continuous over 10k updates.

    Each rank has three intact groups of five plus one singleton. Continuous
    groups/singletons retain the original 10:1 exposure ratio up to integer
    rounding. Binary singleton roles preserve the original 1:2:2 provenance
    mixture cumulatively. All ranks derive the same plan without communication.
    """
    if update < 0 or world_size <= 0:
        raise ValueError('Nonnegative update and positive world required')
    def cumulative(t):
        continuous = t * world_size * 16 * 255 // 1000
        groups = continuous * 2 // 11
        return continuous, groups, continuous - 5*groups
    c0,g0,s0 = cumulative(update)
    c1,g1,s1 = cumulative(update+1)
    groups, singles = g1-g0, s1-s0
    if not (0 <= groups <= world_size*3 and 0 <= singles <= world_size):
        raise ValueError('Packing cannot fit complete groups')
    rng = random.Random(f'ratio255-v1:{seed}:{update}')
    # Reuse the exact two original regression draws per rank; the third group
    # is always binary. Do not change the within-assay sampler to three draws.
    cells = [rank*3+group for rank in range(world_size) for group in range(2)]
    rng.shuffle(cells)
    ranks = list(range(world_size));rng.shuffle(ranks)
    regression_cells = frozenset(cells[:groups])
    regression_singles = frozenset(ranks[:singles])
    roles = {};cycle = ((1,0),(0,0),(0,1),(0,0),(0,1))
    prior_binary_singles = update*world_size-s0
    for i,rank in enumerate(ranks[singles:]):
        roles[rank] = cycle[(prior_binary_singles+i)%5]
    return dict(regression_cells=regression_cells, regression_singles=regression_singles,
                binary_single_roles=roles, continuous=c1-c0,
                binary=world_size*16-(c1-c0), regression_groups=groups,
                continuous_singletons=singles)


class RatioJointBatchSampler:
    def __init__(self, regression, binary_groups, pools, *, rank, world_size=64,
                 layout_seed=20260822, seed=0, start=0, bound_count=0):
        if not 0 <= rank < world_size or len(binary_groups) != 3*len(regression):
            raise ValueError('Rank or group budget mismatch')
        self.regression,self.binary_groups,self.pools = regression,binary_groups,pools
        self.rank,self.world_size,self.layout_seed = rank,world_size,layout_seed
        self.seed,self.start,self.bound_count = seed,start,bound_count

    def __len__(self):
        return len(self.regression)-self.start

    def __iter__(self):
        binary = iter(self.binary_groups)
        for update,reg in enumerate(self.regression):
            cl = [next(binary) for _ in range(3)]
            if update < self.start:
                continue
            if len(reg) != 11 or any(len(x) != 5 for x in cl):
                raise ValueError('Complete groups and B16 required')
            plan = ratio_layout(update,world_size=self.world_size,seed=self.layout_seed)
            batch = []
            for group in range(3):
                continuous = self.rank*3+group in plan['regression_cells']
                rows = reg[group*5:group*5+5] if continuous else cl[group]
                for position in rows:
                    if isinstance(position,tuple):position=position[0]
                    batch.append(('regression' if continuous else 'classification',position,
                                  f'assay:{group}' if continuous else f'binary:{group}'))
            if self.rank in plan['regression_singles']:
                p = reg[10];p = p[0] if isinstance(p,tuple) else p
                rng = random.Random(f'{self.seed}:bound:{update}')
                probability = self.bound_count/(self.bound_count+len(self.regression.singleton_pool))
                if self.bound_count and rng.random() < probability:
                    batch.append(('bound',rng.randrange(self.bound_count),'absolute:bound'))
                else:batch.append(('regression',p,'absolute:singleton'))
            else:
                rng = random.Random(f'{self.seed}:binary-single:{update}')
                batch.append(('classification',rng.choice(self.pools[plan['binary_single_roles'][self.rank]]),
                              'binary:singleton'))
            if len(batch) != 16:raise ValueError('Physical B16 required')
            yield batch
