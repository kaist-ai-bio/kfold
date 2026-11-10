"""Shared affinity encoder with Boltz-2 Appendix C.2.5 supervision.

Raw relations are concentration relations. pActivity reverses inequalities;
synthetic negatives never acquire a numeric affinity. Frozen labels stay intact.
"""
import copy
import math
import random
from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from .heads import AffinityHead


class JointAffinityHead(nn.Module):
    """One colleague encoder and two independent scalar MLP readouts."""
    def __init__(self, config):
        super().__init__()
        self.shared = AffinityHead(config)
        self.value_head = self.shared.base.readout
        self.binary_head = copy.deepcopy(self.value_head)
        self.shared.base.readout = nn.Identity()

    def forward(self, **features):
        pooled = self.shared(**features)
        return dict(value=self.value_head(pooled).squeeze(-1),
                    logits=self.binary_head(pooled).squeeze(-1))


def activity_interval(row):
    """Return pActivity bounds, never infer a value for synthetic examples."""
    value = row.get('p_activity')
    if row['synthetic']:
        if value is not None:
            raise ValueError('Synthetic negative has a numeric affinity')
        return None
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('Nonfinite measured affinity')
    relation = row.get('relation')
    if relation == '=':
        return value, value
    if relation in ('<', '<=', '≤'):
        return value, math.inf
    if relation in ('>', '>=', '≥'):
        return -math.inf, value
    raise ValueError('Unknown original concentration qualifier: '+repr(relation))


def interval_huber(prediction, lower, upper, delta=.5):
    error = torch.where(prediction < lower, prediction-lower,
                        torch.where(prediction > upper, prediction-upper, torch.zeros_like(prediction)))
    return F.huber_loss(error, torch.zeros_like(error), delta=delta, reduction='none')


@dataclass
class JointLossTerms:
    sums: dict
    counts: dict


def joint_loss_terms(outputs, rows, *, delta=.5, gamma=1., alpha=.8):
    """Unnormalized sums for exact cross-rank masked normalization.

    Difference loss averages pairs inside each sampled assay group, then groups.
    Same exact input is never ranked against its own replicate. Satisfied bounds
    contribute zero but retain their denominator; unidentifiable differences do
    not contribute. Focal alpha weights positives, 1-alpha negatives.
    """
    value, logits = outputs['value'].float(), outputs['logits'].float()
    if value.shape != logits.shape or value.shape != (len(rows),):
        raise ValueError('Expected two [B] outputs and B provenance rows')
    if not 0 < alpha < 1 or gamma < 0 or delta <= 0:
        raise ValueError('Invalid Boltz-2 loss hyperparameters')
    zero_v, zero_b = value.sum()*0, logits.sum()*0
    intervals = [activity_interval(r) for r in rows]
    intervals = [bounds if r.get('joint_continuous',True) else None for bounds,r in zip(intervals,rows)]
    valid = [i for i, bounds in enumerate(intervals) if bounds is not None]
    sums = dict(absolute=zero_v, difference=zero_v, binary=zero_b)
    counts = dict(absolute=len(valid), difference=0, binary=0)
    if valid:
        lo = value.new_tensor([intervals[i][0] for i in valid])
        hi = value.new_tensor([intervals[i][1] for i in valid])
        sums['absolute'] = interval_huber(value[valid], lo, hi, delta).sum()
    groups = defaultdict(list)
    for i in valid:
        row = rows[i]
        groups[row.get('loss_group', row['assay_key'])].append(i)
    for indices in groups.values():
        # Sampler group IDs must never join endpoints or targets accidentally.
        if len({(rows[i]['assay_key'], rows[i]['endpoint']) for i in indices}) != 1:
            raise ValueError('Difference group spans distinct assays or endpoints')
        pairs = [(i,j) for n,i in enumerate(indices) for j in indices[n+1:]
                 if rows[i]['input_id'] != rows[j]['input_id']]
        comparable = [(i,j) for i,j in pairs if
            math.isfinite(intervals[i][0]-intervals[j][1]) or
            math.isfinite(intervals[i][1]-intervals[j][0])]
        if not comparable:
            continue
        left, right = zip(*comparable)
        lo = value.new_tensor([intervals[i][0]-intervals[j][1] for i,j in comparable])
        hi = value.new_tensor([intervals[i][1]-intervals[j][0] for i,j in comparable])
        sums['difference'] = sums['difference'] + interval_huber(
            value[list(left)]-value[list(right)], lo, hi, delta).mean()
        counts['difference'] += 1
    binary = [i for i,r in enumerate(rows) if r['classification_valid'] and r.get('joint_binary',True)]
    if binary:
        target = logits.new_tensor([rows[i]['binary'] for i in binary])
        if not ((target == 0) | (target == 1)).all():
            raise ValueError('Binary supervision requires explicit 0/1 labels')
        if any(rows[i]['synthetic'] and rows[i]['binary'] != 0 for i in binary):
            raise ValueError('Synthetic example must remain an assumed negative')
        logit = logits[binary]
        bce = F.binary_cross_entropy_with_logits(logit, target, reduction='none')
        pt = torch.sigmoid(torch.where(target == 1, logit, -logit))
        alpha_t = torch.where(target == 1, alpha, 1-alpha)
        sums['binary'] = (alpha_t * (1-pt).pow(gamma) * bce).sum()
        counts['binary'] = len(binary)
    return JointLossTerms(sums, counts)


def normalized_joint_loss(terms, *, distributed=True):
    """DDP averages gradients: scale each local sum by world/global_count."""
    names = ('absolute','difference','binary')
    first = terms.sums['absolute']
    totals = first.new_tensor([terms.counts[k] for k in names], dtype=torch.float64)
    world = dist.get_world_size() if distributed else 1
    if distributed:
        dist.all_reduce(totals)
    means = {k:terms.sums[k] * world / totals[i].clamp_min(1) for i,k in enumerate(names)}
    loss = .1*means['absolute'] + .9*means['difference'] + means['binary']
    return loss, totals


class JointDataset:
    """Tagged positions preserve the two original immutable task indices."""
    def __init__(self, regression, classification, bound_rows=()):
        self.datasets = dict(regression=regression, classification=classification)
        self.bound_rows = bound_rows

    def __getitem__(self, tagged):
        task, position, group = tagged
        if task == 'bound':
            row = dict(self.bound_rows[position]);row['loss_group'] = group
            x,row = self.datasets['regression'].load_row(row)
        else:
            x,row = self.datasets[task][(position,group)]
        row=dict(row,joint_continuous=task!='classification',joint_binary=task=='classification')
        return x,row


class BinaryFiveSampler:
    """One measured binder +4 negatives; retain measured/synthetic provenance.

    The approved corpus has no experimental assay ID for synthetic negatives.
    Do not invent one: unlike the original Boltz sampler this draws from the
    existing global pools, with two measured and two synthetic negatives.
    """
    def __init__(self, pools, *, batches, seed):
        self.pools,self.batches,self.seed = pools,batches,seed
        if any(not pools[k] for k in ((1,0),(0,0),(0,1))):
            raise ValueError('All three approved binary provenance pools required')

    def __len__(self):return self.batches

    def __iter__(self):
        rng=random.Random(self.seed)
        for _ in range(self.batches):
            result=[]
            for key,count in [((1,0),1),((0,0),2),((0,1),2)]:
                pool=self.pools[key]
                result.extend(rng.sample(pool,count) if len(pool)>=count else rng.choices(pool,k=count))
            rng.shuffle(result)
            yield result


class JointBatchSampler:
    """B16: 2 complete 5-ligand assays +1 absolute slot +5 binary slots.

    Existing measured/synthetic pools and internal split remain authoritative.
    This is an adaptation to our corpus, not Boltz-2's PubChem source mixture.
    Distinct supervision flags avoid changing the 1:4 focal sampling ratio by
    adding threshold-derived labels from the continuous sampling stream.
    """
    def __init__(self, regression, classification, *, start=0, bound_count=0, seed=0):
        if len(regression) != len(classification):
            raise ValueError('Equal finite joint sampler budgets required')
        self.regression,self.classification = regression,classification
        self.start,self.bound_count = start,bound_count
        self.seed = seed

    def __len__(self):
        return len(self.regression)-self.start

    def __iter__(self):
        for update,(reg,cl) in enumerate(zip(self.regression,self.classification)):
            if update < self.start:
                continue
            batch=[]
            for slot,p in enumerate(reg):
                if isinstance(p,tuple):
                    p,_ = p
                group = 'assay:'+str(slot//5) if slot < 10 else f'absolute:{slot}'
                batch.append(('regression',p,group))
            # Bound labels join the absolute-only pool without upweighting a tiny
            # pool to one label per rank/update. Draws remain restart deterministic.
            rng = random.Random(f'{self.seed}:bound:{update}')
            probability = self.bound_count / (self.bound_count+len(self.regression.singleton_pool))
            if self.bound_count and rng.random() < probability:
                batch[10] = ('bound',rng.randrange(self.bound_count),'absolute:bound')
            batch += [('classification',p,f'binary:{slot}') for slot,p in enumerate(cl)]
            if len(batch) != 16:
                raise ValueError('Physical batch16 required; no gradient accumulation')
            yield batch


def collate_joint(examples, *, pad_to=None):
    from .training_data import collate_cached
    x,rows = collate_cached(examples)
    n = x['token_mask'].shape[1]
    if len(rows) != 16 or n > 256:
        raise ValueError('Physical B16 and the frozen 256-token budget required')
    if pad_to is not None:
        if n > pad_to:
            raise ValueError('Padding cannot crop the frozen input')
        for key,tensor in list(x.items()):
            if key in ('z','distogram_features'):
                dest=tensor.new_zeros((tensor.shape[0],pad_to,pad_to,tensor.shape[-1]));dest[:,:n,:n]=tensor
            else:
                dest=tensor.new_zeros((tensor.shape[0],pad_to,*tensor.shape[2:]));dest[:,:n]=tensor
            x[key]=dest
    return x,rows
