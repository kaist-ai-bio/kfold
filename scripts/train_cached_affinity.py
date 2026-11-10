"""Single-GPU cached affinity training; full runs require an explicit launch flag.

The bounded multi-condition pilot is separate. External evaluation is never used
to select checkpoints here. Existing raw features and payloads are read-only.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import torch
from torch.utils.data import DataLoader, Subset

from kfold.affinity.heads import AffinityHead, HeadConfig
from kfold.affinity.pilot import metrics, save
from kfold.affinity.supervision import objective
from kfold.affinity.training_data import CachedLabelDataset, TaskBatchSampler, collate_cached
from kfold.affinity.training_monitor import validation_positions, scalar_report, start_tracking


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--task',choices=['regression','classification'],required=True)
    p.add_argument('--variant',choices=['colleague_cross','full_pair_control','slm_outer'],required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--resume')
    p.add_argument('--allow-full-training',action='store_true')
    a=p.parse_args()
    if not a.allow_full_training:raise RuntimeError('Only pilot was launched; explicit full-training flag required')
    if int(os.environ.get('WORLD_SIZE','1'))!=1:raise RuntimeError('This entrypoint is single GPU; no duplicated DDP sampling')
    c=json.loads(Path(a.config).read_text());out=Path(a.output)
    data=c['data']
    gate=json.loads(Path(data['manifest']).with_name('completion.json').read_text())
    if not gate['passed'] or gate['chemical_train_val_overlap'] or gate['input_train_val_overlap']:
        raise ValueError('An audited leakage-safe manifest is required')
    manifest_hash=hashlib.sha256()
    with Path(data['manifest']).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):manifest_hash.update(b)
    if manifest_hash.hexdigest()!=gate['manifest_sha256']:raise ValueError('Manifest changed after audit')
    availability=Path(data['manifest']).with_name('view-availability.json')
    if hashlib.sha256(availability.read_bytes()).hexdigest()!=data['availability_sha256']:
        raise ValueError('View availability changed after preparation')
    contract=hashlib.sha256(json.dumps(dict(task=a.task,variant=a.variant,settings=c,
        manifest_sha256=gate['manifest_sha256']),sort_keys=True).encode()).hexdigest()
    if out.exists() and not a.resume:raise RuntimeError('Existing run requires explicit resume')
    out.mkdir(parents=True,exist_ok=bool(a.resume))
    torch.set_num_threads(4);torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed'])
    config=HeadConfig(variant=a.variant,**c['model'])
    model=AffinityHead(config).cuda();opt=torch.optim.AdamW(model.parameters(),**c['optimizer'])
    start=0
    if a.resume:
        s=torch.load(a.resume,map_location='cuda',weights_only=False)
        if s['task']!=a.task or s['config']!=asdict(config) or s['training_contract_sha256']!=contract:
            raise ValueError('Resume task/model/data/optimizer/sampling contract changed')
        model.load_state_dict(s['model']);opt.load_state_dict(s['optimizer']);start=s['updates']
        torch.set_rng_state(s['torch_rng'].cpu());torch.cuda.set_rng_state_all([t.cpu() for t in s['cuda_rng']])
    train=CachedLabelDataset(data['manifest'],data['catalog'],task=a.task,split='train',view=data['view'])
    validation=CachedLabelDataset(data['manifest'],data['catalog'],task=a.task,split='validation',view=data['view'])
    sampler=TaskBatchSampler(train,batches=c['max_steps'],seed=c['seed'],**c['sampling'][a.task])
    save(out/'resolved-config.json',dict(task=a.task,variant=a.variant,training_contract_sha256=contract,**c))
    panel=validation_positions(validation,per_stratum=c.get('validation_per_stratum',1024),seed=c['seed'])
    save(out/'validation-panel.json',dict(positions=panel,split='validation',
        stratification='cohort and original endpoint',external_used=False))
    tracker=start_tracking(c.get('wandb',{}),out,task=a.task,variant=a.variant,
        contract=contract,config=c,resume=bool(a.resume))
    if tracker:save(out/'wandb-run.json',dict(id=tracker.id,url=tracker.url,project=tracker.project,entity=tracker.entity))
    stopped=[]
    signal.signal(signal.SIGUSR1,lambda *_:stopped.append('allocation_time_limit_warning'))
    signal.signal(signal.SIGTERM,lambda *_:stopped.append('termination_requested'))
    def checkpoint(updates):
        file=out/f'step-{updates:08d}.pt';partial=file.with_suffix('.partial')
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),updates=updates,
            config=asdict(config),task=a.task,training_contract_sha256=contract,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),partial)
        partial.replace(file)
    def evaluate(updates,full=False):
        model.eval();by_cohort={}
        val=DataLoader(validation if full else Subset(validation,panel),
            batch_size=c['validation_batch_size'],shuffle=False,collate_fn=collate_cached,num_workers=0)
        for vx,vr in val:
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                y=model(**{k:v.cuda() for k,v in vx.items()}).float().cpu()
            for prediction,row in zip(y,vr):
                for key in (row['cohort'],row['cohort']+'__'+row['endpoint']):
                    values,labels=by_cohort.setdefault(key,([],[]));values.append(prediction);labels.append(row)
            if stopped:return False
        report={k:metrics(torch.stack(v),labels,a.task) for k,(v,labels) in by_cohort.items()}
        prefix='validation_full' if full else 'validation_panel'
        save(out/f'{prefix}-{updates:08d}.json',dict(updates=updates,cohorts=report,
            full_internal_validation=full,source='internal chemical-group holdout',external_used_for_selection=False))
        if tracker:tracker.log(dict(update=updates,**scalar_report(report,prefix)))
        return True
    began=time.monotonic();last=time.monotonic();updates=start
    for step,batch_indices in enumerate(sampler):
        if step<start:continue
        # Skip resumed positions before payload I/O; keep the frozen sampler.
        x,rows=collate_cached([train[position] for position in batch_indices])
        loaded=time.monotonic()
        model.train();opt.zero_grad(set_to_none=True);x={k:v.cuda() for k,v in x.items()}
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=objective(model(**x).float(),rows,a.task)
        if not torch.isfinite(loss):raise ValueError('Nonfinite training objective')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),c['gradient_clip'],error_if_nonfinite=True);opt.step()
        loss_value=float(loss);norm_value=float(norm);now=time.monotonic();updates=step+1
        progress=dict(epoch=time.time(),updates=updates,loss=loss_value,gradient_norm=norm_value,
            data_seconds=loaded-last,compute_seconds=now-loaded,elapsed_seconds=now-began,
            max_steps=c['max_steps'],steps_per_second=(updates-start)/(now-began))
        if updates%c.get('log_every',10)==0 or updates==start+1 or updates==c['max_steps']:
            save(out/'progress.json',progress)
            if tracker:tracker.log(dict(update=updates,**{'train/'+k:progress[k] for k in
                ('loss','gradient_norm','data_seconds','compute_seconds','steps_per_second')},
                **{'train/lr':opt.param_groups[0]['lr']}))
        if (step+1)%c['checkpoint_every']==0 or step+1==c['max_steps']:
            checkpoint(updates)
        if (step+1)%c['validate_every']==0 or step+1==c['max_steps']:
            if not stopped:evaluate(updates)
        if stopped:break
        last=time.monotonic()
    full_complete=False
    if not stopped and updates==c['max_steps']:full_complete=evaluate(updates,full=True)
    if stopped:
        checkpoint(updates)
        save(out/'interrupted.json',dict(epoch=time.time(),updates=updates,reason=stopped,passed=False))
        if tracker:tracker.finish(exit_code=75)
        raise SystemExit(75)
    assert updates==c['max_steps'] and full_complete
    save(out/'completion.json',dict(epoch=time.time(),updates=updates,task=a.task,variant=a.variant,
        passed=True,full_internal_validation=True,external_used_for_selection=False))
    if tracker:tracker.finish()


if __name__=='__main__':main()
