"""Bounded real-cache optimizer, held-out evaluation and checkpoint-resume pilot."""
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import time

import torch

from .heads import AffinityHead, HeadConfig
from .supervision import objective
from .training_data import CachedLabelDataset, collate_cached


def save(path, value):
    tmp=path.with_suffix('.partial');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def parameter_sha(module):
    h=hashlib.sha256()
    for key,tensor in sorted(module.state_dict().items()):
        x=tensor.detach().cpu().contiguous()
        h.update((key+str(x.dtype)+str(list(x.shape))).encode());h.update(x.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def metrics(prediction, rows, task):
    p=prediction.detach().float().cpu()
    if task=='regression':
        y=torch.tensor([r['p_activity'] for r in rows]);d=p-y
        a,b=p-p.mean(),y-y.mean();den=a.norm()*b.norm()
        return dict(n=len(rows),MAE=float(d.abs().mean()),RMSE=float(d.square().mean().sqrt()),
            Pearson=float((a*b).sum()/den) if den>1e-12 else None)
    y=torch.tensor([r['binary'] for r in rows],dtype=torch.float32);positive=p[y==1];negative=p[y==0]
    auc=None
    if len(positive) and len(negative):
        order=torch.argsort(p);scores=p[order];labels=y[order].double()
        _,counts=torch.unique_consecutive(scores,return_counts=True)
        ends=counts.cumsum(0).double();starts=ends-counts.double()
        average_ranks=(starts+1+ends)/2
        ranks=torch.repeat_interleave(average_ranks,counts)
        auc=float(((ranks*labels).sum()-len(positive)*(len(positive)+1)/2)/(len(positive)*len(negative)))
    ap=None
    if len(positive):
        order=torch.argsort(p,descending=True);scores=p[order];labels=y[order].double()
        _,counts=torch.unique_consecutive(scores,return_counts=True)
        groups=torch.repeat_interleave(torch.arange(len(counts)),counts)
        group_positive=torch.zeros(len(counts),dtype=torch.float64).index_add_(0,groups,labels)
        precision=group_positive.cumsum(0)/counts.cumsum(0).double()
        ap=float((precision*group_positive/len(positive)).sum())
    return dict(n=len(rows),positive=int((y==1).sum()),negative=int((y==0).sum()),AUROC=auc,AUPRC=ap,
        BCE=float(torch.nn.functional.binary_cross_entropy_with_logits(p,y)),
        accuracy=float(((p>=0)==(y==1)).float().mean()))


def run_condition(root, task, variant, *, kernel_backend='torch'):
    r=Path(root);out=r/'pilot'/f'{task}-{variant}';out.mkdir(parents=True,exist_ok=True)
    if (out/'completion.json').exists():raise RuntimeError('Completed condition must not repeat')
    began=time.time();data=json.loads((r/'data-config.json').read_text())
    gate=json.loads((r/'data/completion.json').read_text());assert gate['passed']
    assert gate['input_train_val_overlap']==gate['chemical_train_val_overlap']==0
    batches=json.loads((r/'data/pilot-batches.json').read_text())[task]
    torch.manual_seed(20261109);torch.cuda.manual_seed_all(20261109);torch.set_num_threads(4)
    config=HeadConfig(variant=variant,kernel_backend=kernel_backend)
    model=AffinityHead(config).cuda().train()
    initial=parameter_sha(model.base)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-3)
    train=CachedLabelDataset(r/'data/labels.sqlite',data['catalog'],task=task,split='train',cache_records=4)
    db=train.connect()
    def row(identity):
        found=db.execute('SELECT * FROM examples WHERE row_id=?',(identity,)).fetchone()
        if found is None:raise ValueError('Missing frozen label row')
        return dict(found)
    def load(ids, split):
        rows=[row(i) for i in ids]
        assert all(x['split']==split for x in rows)
        x,rows=collate_cached([train.load_row(value) for value in rows])
        return {k:v.cuda(non_blocking=False) for k,v in x.items()},rows
    history=[];gradient_groups=set()
    def step(network,opt,x,rows,record=False):
        network.train();opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            y=network(**x);loss=objective(y.float(),rows,task)
        if not torch.isfinite(loss):raise ValueError('Nonfinite pilot loss')
        loss.backward()
        for name,parameter in network.named_parameters():
            if parameter.grad is not None:
                if not torch.isfinite(parameter.grad).all():raise ValueError('Nonfinite gradient '+name)
                if record and bool(parameter.grad.abs().sum()>0):gradient_groups.add(name)
        norm=torch.nn.utils.clip_grad_norm_(network.parameters(),10.,error_if_nonfinite=True)
        opt.step()
        return float(loss.detach()),float(norm),y.detach().float().cpu()
    torch.cuda.reset_peak_memory_stats()
    for index,ids in enumerate(batches['train']):
        x,rows=load(ids,'train')
        loss,norm,_=step(model,optimizer,x,rows,record=True)
        history.append(dict(update=index+1,loss=loss,gradient_norm=norm,label_rows=ids,
            measured=sum(not x['synthetic'] for x in rows),synthetic=sum(x['synthetic'] for x in rows)))
        save(out/'progress.json',dict(epoch=time.time(),updates=len(history),history=history))
        del x
    assert len(history)==8 and parameter_sha(model.base)!=initial
    assert any('s_lm_to_s' in k for k in gradient_groups)
    if variant=='slm_outer':
        assert any('outer.left' in k for k in gradient_groups) and any('outer.output' in k for k in gradient_groups)
    # Fixed update8 checkpoint; resume probe branches below are not a production fit.
    checkpoint=out/'step-000008.pt'
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),updates=8,
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),config=asdict(config),
        manifest_sha256=gate['manifest_sha256'],task=task,view='query_200'),checkpoint)
    probe_x,probe_rows=load(batches['train'][0],'train')
    forward_a=step(model,optimizer,probe_x,probe_rows)
    reference={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    state=torch.load(checkpoint,map_location='cuda',weights_only=False)
    model_b=AffinityHead(config).cuda();opt_b=torch.optim.AdamW(model_b.parameters(),lr=1e-4,weight_decay=1e-3)
    model_b.load_state_dict(state['model']);opt_b.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['torch_rng'].cpu());torch.cuda.set_rng_state_all([t.cpu() for t in state['cuda_rng']])
    forward_b=step(model_b,opt_b,probe_x,probe_rows)
    differences=[float((reference[k]-v.detach().cpu()).abs().max()) for k,v in model_b.state_dict().items()]
    assert all(torch.allclose(reference[k],v.detach().cpu(),atol=1e-6,rtol=1e-5) for k,v in model_b.state_dict().items()),max(differences)
    assert torch.allclose(forward_a[2],forward_b[2],atol=1e-5,rtol=1e-5)
    del model_b,opt_b,state,reference,probe_x
    # Evaluate update8, not the extra continuation probe update.
    state=torch.load(checkpoint,map_location='cuda',weights_only=False);model.load_state_dict(state['model']);model.eval()
    evaluation={}
    for split,ids in batches['evaluation'].items():
        if not ids:
            evaluation[split]=dict(n=0,metrics=None);continue
        predictions=[];labels=[]
        for start in range(0,len(ids),4):
            x,rows=load(ids[start:start+4],split)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):prediction=model(**x)
            assert torch.isfinite(prediction).all();predictions.append(prediction.float().cpu());labels+=rows
        evaluation[split]=metrics(torch.cat(predictions),labels,task)
    compatibility={}
    for source,ids in batches.get('compatibility',{}).items():
        observed=[]
        for identity in ids:
            label=row(identity);x,rows=load([identity],label['split'])
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):prediction=model(**x)
            assert torch.isfinite(prediction).all()
            observed.append(dict(row_id=identity,input_id=label['input_id'],split=label['split'],
                relation=label['relation'],synthetic=bool(label['synthetic']),finite=True))
        compatibility[source]=observed
    db.close()
    result=dict(passed=True,epoch=time.time(),task=task,variant=variant,updates=8,batch_size=4,blocks=6,
        view='query_200',initial_common_parameter_sha256=initial,train_loss_history=history,
        gradient_groups=sorted(gradient_groups),all_gradients_finite=True,
        optimizer_changed_parameters=True,resume_matches_uninterrupted=True,resume_max_parameter_difference=max(differences),
        checkpoint=str(checkpoint),evaluation=evaluation,source_compatibility=compatibility,
        evaluation_scope='small fixed held-out compatibility pilot, not a performance benchmark',
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),seconds=time.time()-began,
        parameters=sum(p.numel() for p in model.parameters()),config=asdict(config),
        full_trunk_present=False,raw_features_read=False,source_deleted=False)
    save(out/'completion.json',result)
    return result
