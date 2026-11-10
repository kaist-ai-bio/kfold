"""DDP64 helpers; preserve complete assay groups during microbatching."""
from collections import defaultdict
import hashlib
import math
import torch
import torch.nn.functional as F

from .training_data import collate_cached
from .reference_losses import boltz2_continuous_affinity_loss


def rank_seed(seed, rank):
    return int.from_bytes(hashlib.sha256(f'{seed}:affinity-rank:{rank}'.encode()).digest()[:8], 'big')


def microbatch_positions(task, batch_size=64, microbatch_size=16):
    if batch_size != 64 or microbatch_size != 16:
        raise ValueError('Validated logical64/micro16 contract required')
    if task == 'classification':
        return [list(range(i,i+16)) for i in range(0,64,16)]
    if task != 'regression':
        raise ValueError('Explicit task required')
    # Twelve contiguous assays of five, followed by four absolute-only slots.
    # Each microbatch receives three WHOLE assays and one absolute-only slot.
    return [list(range(i*15,(i+1)*15))+[60+i] for i in range(4)]


def collate_logical64(examples, *, task):
    if len(examples) != 64:
        raise ValueError('Exactly64 examples per rank/update required')
    if task == 'regression':
        assays=[]
        for i in range(12):
            keys={r.get('loss_group',r['assay_key']) for _,r in examples[i*5:(i+1)*5]}
            identities={r['input_id'] for _,r in examples[i*5:(i+1)*5]}
            if len(keys)!=1 or len(identities)!=5:
                raise ValueError('Each ranking group requires five distinct exact inputs')
            assays.extend(keys)
        if len(set(assays)) != 12 or any('loss_group' not in r for _,r in examples[60:]):
            raise ValueError('Distinct ranking assays and four isolated absolute slots required')
    return [collate_cached([examples[i] for i in indices]) for indices in microbatch_positions(task)]


def objective_details(prediction, rows, task):
    if task=='classification':
        y=prediction.new_tensor([r['binary'] for r in rows]);loss=F.binary_cross_entropy_with_logits(prediction,y)
        return loss,dict(loss=float(loss.detach()),BCE=float(loss.detach()))
    keys=[r.get('loss_group',r['assay_key']) for r in rows]
    groups={k:i for i,k in enumerate(sorted(set(keys)))}
    ligands={k:i for i,k in enumerate(sorted({r['input_id'] for r in rows}))}
    result=boltz2_continuous_affinity_loss(prediction,prediction.new_tensor([r['p_activity'] for r in rows]),
        torch.tensor([groups[k] for k in keys],device=prediction.device),
        torch.tensor([ligands[r['input_id']] for r in rows],device=prediction.device),
        torch.tensor([bool(r['regression_valid']) and not r['synthetic'] for r in rows],device=prediction.device))
    return result.loss,dict(loss=float(result.loss.detach()),huber=float(result.regression.detach()),
        ranking=float(result.ranking.detach()),active_ranking_groups=result.active_ranking_groups,
        difference_pairs=result.difference_pairs)


def colleague_validation_metrics(prediction, rows, *, near_tie_delta=.1):
    """Original assay-mean Pearson and exhaustive pairwise accuracy, bounded RAM.

    Unlike source canonical chemistry, ranking identity remains the approved
    exact input identity. Identical replicates collapse only inside an assay.
    """
    p=prediction.detach().float().cpu();y=torch.tensor([r['p_activity'] for r in rows])
    grouped=defaultdict(lambda:defaultdict(list))
    for pred,value,row in zip(p.tolist(),y.tolist(),rows):
        grouped[row['assay_key']][row['input_id']].append((pred,value))
    pearsons=[];correct=0;total=0;replicates=0
    for values in grouped.values():
        replicates+=sum(len(v)-1 for v in values.values())
        means=[(sum(x for x,_ in v)/len(v),sum(x for _,x in v)/len(v)) for _,v in sorted(values.items())]
        a=torch.tensor([v[0] for v in means]);b=torch.tensor([v[1] for v in means])
        ac,bc=a-a.mean(),b-b.mean();den=ac.norm()*bc.norm()
        if len(a)>=3 and den>torch.finfo(torch.float32).eps:pearsons.append(float((ac*bc).sum()/den))
        # Vectorize rows, with at most128*N comparisons resident in memory.
        for left in range(0,len(a),128):
            stop=min(left+128,len(a));delta=b[left:stop,None]-b[None,:]
            mask=(torch.arange(len(a))[None,:]>torch.arange(left,stop)[:,None]) & (delta.abs()>=near_tie_delta)
            total+=int(mask.sum());correct+=int(((torch.sign(a[left:stop,None]-a[None,:])==torch.sign(delta))&mask).sum())
    return dict(huber=float(F.huber_loss(p,y,delta=.5)),mae=float((p-y).abs().mean()),
        mean_assay_pearson=sum(pearsons)/len(pearsons) if pearsons else None,
        pairwise_accuracy=correct/total if total else None,pearson_assays=len(pearsons),
        pairwise_comparisons=total,ranking_replicate_records=replicates)


def initialize_worker(_):
    torch.set_num_threads(1)
