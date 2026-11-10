"""Eight-rank cached head training: logical64/rank, intact micro16 assays."""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
from datetime import timedelta
from functools import partial
import hashlib,json,math,os,signal,socket,time
from pathlib import Path
if os.environ.get('KFOLD_AFFINITY_OVERLAY'):
    import bootstrap  # Also runs in spawn DataLoader workers before kfold imports.
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from kfold.affinity.heads import AffinityHead,HeadConfig
from kfold.affinity.training_data import CachedLabelDataset,TaskBatchSampler
from kfold.affinity.distributed_training import (rank_seed,collate_logical64,objective_details,
    colleague_validation_metrics,initialize_worker)
from kfold.affinity.training_monitor import validation_positions,scalar_report,start_tracking
from kfold.affinity.pilot import metrics,save,parameter_sha

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    p.add_argument('--task',choices=['regression','classification'],default='regression')
    p.add_argument('--output',required=True);p.add_argument('--resume');p.add_argument('--preflight',action='store_true')
    p.add_argument('--pilot',action='store_true');p.add_argument('--allow-full-training',action='store_true');a=p.parse_args()
    rank=int(os.environ['SLURM_PROCID']);world=int(os.environ['SLURM_NTASKS']);local=int(os.environ['SLURM_LOCALID'])
    assert world==8 and torch.cuda.device_count()==4
    torch.cuda.set_device(local);device=torch.device('cuda',local);torch.set_num_threads(4)
    dist.init_process_group('nccl',rank=rank,world_size=world,timeout=timedelta(seconds=240),device_id=device)
    c=json.loads(Path(a.config).read_text());out=Path(a.output)
    if rank==0:
        assert not out.exists(), 'Every launch phase gets a fresh output directory'
        out.mkdir(parents=True)
    dist.barrier()
    def gather(value):
        result=[None]*world;dist.all_gather_object(result,value);return result
    proof=gather(dict(rank=rank,host=socket.gethostname(),local_rank=local,
        uuid=str(torch.cuda.get_device_properties(local).uuid),cpus=len(os.sched_getaffinity(0)),torch=torch.__version__))
    value=torch.tensor(float(rank),device=device);dist.all_reduce(value);assert value.item()==28
    assert len({r['uuid'] for r in proof})==8 and len({r['host'] for r in proof})==2
    if a.preflight:
        if rank==0:save(out/'completion.json',dict(passed=True,ranks=proof,all_reduce=28,phase='communication_preflight'))
        dist.destroy_process_group();return
    assert a.pilot or a.allow_full_training
    if rank==0:
        data=c['data'];gate=json.loads(Path(data['manifest']).with_name('completion.json').read_text())
        assert gate['passed'] and gate['chemical_train_val_overlap']==gate['input_train_val_overlap']==0
        assert sha(data['manifest'])==gate['manifest_sha256']
        assert sha(Path(data['manifest']).with_name('view-availability.json'))==data['availability_sha256']
    else:gate=None
    objects=[gate];dist.broadcast_object_list(objects,src=0);gate=objects[0];data=c['data']
    assert data['view']=='query_200' and c['sampling'][a.task]['batch_size']==64
    contract=hashlib.sha256(json.dumps(dict(task=a.task,config=c,world=world,pilot=a.pilot,
        manifest_sha256=gate['manifest_sha256']),sort_keys=True).encode()).hexdigest()
    torch.manual_seed(c['seed']);torch.cuda.manual_seed(c['seed'])
    hc=HeadConfig(variant='colleague_cross',**c['model']);bare=AffinityHead(hc).to(device)
    model=DDP(bare,device_ids=[local],broadcast_buffers=False,gradient_as_bucket_view=True)
    opt=torch.optim.AdamW(bare.parameters(),**c['optimizer']);start=0
    if a.resume:
        state=torch.load(a.resume,map_location='cpu',weights_only=False)
        assert state['contract']==contract and state['world_size']==world and state['task']==a.task
        bare.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);start=state['updates']
    train=CachedLabelDataset(data['manifest'],data['catalog'],task=a.task,split='train',view=data['view'])
    val=CachedLabelDataset(data['manifest'],data['catalog'],task=a.task,split='validation',view=data['view'])
    target=c['pilot_steps'] if a.pilot else c['max_steps']
    sampler=TaskBatchSampler(train,batches=target,seed=rank_seed(c['seed'],rank),**c['sampling'][a.task])
    class Remaining:
        def __iter__(self):
            for i,batch in enumerate(sampler):
                if i>=start:yield batch
        def __len__(self):return target-start
    loader=DataLoader(train,batch_sampler=Remaining(),collate_fn=partial(collate_logical64,task=a.task),
        num_workers=c['num_workers'],persistent_workers=bool(c['num_workers']),pin_memory=True,
        multiprocessing_context='spawn' if c['num_workers'] else None,
        worker_init_fn=initialize_worker,prefetch_factor=1 if c['num_workers'] else None,
        generator=torch.Generator().manual_seed(rank_seed(c['seed'],rank)))
    panel=validation_positions(val,per_stratum=16 if a.pilot else c['validation_per_stratum'],seed=c['validation_seed'])
    if rank==0:save(out/'resolved-config.json',dict(task=a.task,world_size=world,contract=contract,
        logical_batch_per_rank=64,microbatch_per_rank=16,global_batch=512,**c))
    tracker=None
    if rank==0 and not a.pilot:
        tracker=start_tracking(c['wandb'],out,task=a.task,variant='colleague_cross',contract=contract,config=c,resume=False)
        save(out/'wandb-run.json',dict(id=tracker.id,url=tracker.url,project=tracker.project,entity=tracker.entity))
    stop=[]
    signal.signal(signal.SIGUSR1,lambda *_:stop.append('requested_checkpoint_drain'))
    signal.signal(signal.SIGTERM,lambda *_:stop.append('termination_requested'))
    def stopped():
        v=torch.tensor(int(bool(stop)),device=device);dist.all_reduce(v,op=dist.ReduceOp.MAX);return bool(v.item())
    def checkpoint(updates):
        rng=gather(dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(local).cpu()))
        if rank==0:
            dest=out/f'step-{updates:08d}.pt';tmp=dest.with_suffix('.partial')
            torch.save(dict(model=bare.state_dict(),optimizer=opt.state_dict(),updates=updates,
                contract=contract,task=a.task,config=asdict(hc),world_size=world,rng=rng),tmp);tmp.replace(dest)
        dist.barrier()
    def reports(predictions,rows):
        groups={}
        for i,row in enumerate(rows):
            for key in (row['cohort'],row['cohort']+'__'+row['endpoint']):groups.setdefault(key,[]).append(i)
        answer={}
        for key,indices in groups.items():
            labels=[rows[i] for i in indices];pr=predictions[indices];m=metrics(pr,labels,a.task)
            if a.task=='regression':m.update(colleague_validation_metrics(pr,labels))
            answer[key]=m
        return answer
    def evaluate(updates,full=False):
        bare.eval();positions=list(range(len(val))) if full else panel;local_records=[]
        width=c['validation_batch_size'];steps=math.ceil(len(positions)/(world*width))
        from kfold.affinity.training_data import collate_cached
        for i in range(steps):
            indices=positions[(i*world+rank)*width:(i*world+rank+1)*width]
            if indices:
                x,rows=collate_cached([val[j] for j in indices])
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    prediction=bare(**{k:v.to(device,non_blocking=True) for k,v in x.items()}).float().cpu().tolist()
                for position,pred,row in zip(indices,prediction,rows):
                    local_records.append((position,pred,{k:row[k] for k in
                        ('row_id','input_id','assay_key','cohort','endpoint','p_activity','binary')}))
            if stopped():return False
        records=[v for part in gather(local_records) for v in part]
        if rank==0:
            records.sort(key=lambda x:x[0]);assert [v[0] for v in records]==sorted(positions)
            result=reports(torch.tensor([v[1] for v in records]),[v[2] for v in records])
            prefix='validation_full' if full else 'validation_panel'
            save(out/f'{prefix}-{updates:08d}.json',dict(updates=updates,rows=len(records),cohorts=result,
                full_internal_validation=full,external_used_for_selection=False))
            if tracker:tracker.log(dict(update=updates,**scalar_report(result,prefix)))
        dist.barrier();return True
    # An initial real held-out panel gives validation curves a genuine step0.
    # Restore per-rank RNG AFTER initialization and any DataLoader setup.
    if a.resume:
        torch.set_rng_state(state['rng'][rank]['cpu']);torch.cuda.set_rng_state(state['rng'][rank]['cuda'],local)
    else:torch.cuda.manual_seed(rank_seed(c['seed'],rank))
    if not a.resume:assert evaluate(0)
    began=time.monotonic();updates=start;last=began;sample_proofs=[]
    for updates,microbatches in enumerate(loader,start=start+1):
        loaded=time.monotonic();model.train();opt.zero_grad(set_to_none=True);local_stats={};row_ids=[]
        for i,(x,rows) in enumerate(microbatches):
            context=model.no_sync() if i<3 else nullcontext()
            with context:
                x={k:v.to(device,non_blocking=True) for k,v in x.items()}
                with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(**x).float();loss,details=objective_details(pred,rows,a.task)
                assert torch.isfinite(loss);(loss/4).backward()
            for key,v in details.items():
                local_stats[key]=local_stats.get(key,0)+(v if key in ('active_ranking_groups','difference_pairs') else v/4)
            row_ids.extend(row['row_id'] for row in rows)
            del x,rows,pred,loss
        norm=torch.nn.utils.clip_grad_norm_(bare.parameters(),c['gradient_clip'],error_if_nonfinite=True);opt.step()
        sample_proofs.append(hashlib.sha256(json.dumps(row_ids).encode()).hexdigest())
        now=time.monotonic()
        if updates==start+1 or updates%c['log_every']==0 or updates==target:
            names=sorted(local_stats);values=torch.tensor([local_stats[k] for k in names],device=device);dist.all_reduce(values);values/=world
            timing=torch.tensor([loaded-last,now-loaded],device=device);dist.all_reduce(timing,op=dist.ReduceOp.MAX)
            progress=dict(epoch=time.time(),updates=updates,max_steps=target,rank=rank,logical_batch_per_rank=64,
                global_batch=512,microbatch_per_rank=16,gradient_norm=float(norm),metrics=dict(zip(names,values.tolist())),
                data_seconds=float(timing[0]),compute_seconds=float(timing[1]),elapsed_seconds=now-began,
                steps_per_second=(updates-start)/(now-began))
            save(out/f'progress-rank{rank}.json',progress)
            if rank==0:
                save(out/'progress.json',progress)
                if tracker:tracker.log(dict(update=updates,**{'train/'+k:v for k,v in progress['metrics'].items()},
                    **{'train/'+k:progress[k] for k in ('gradient_norm','data_seconds','compute_seconds','steps_per_second')},
                    **{'train/lr':opt.param_groups[0]['lr']}))
        stop_now=stopped()
        if updates%c['checkpoint_every']==0 or updates==target or stop_now or (a.pilot and updates==2):checkpoint(updates)
        if not stop_now and (updates%c['validate_every']==0 or updates==target):
            if not evaluate(updates):stop_now=True
        if stop_now:stop.append('coordinated_stop');break
        last=time.monotonic()
    if stop:
        checkpoint(updates)
        if rank==0:
            save(out/'interrupted.json',dict(epoch=time.time(),updates=updates,reason=stop,passed=False))
            if tracker:tracker.finish(exit_code=75)
        dist.destroy_process_group();raise SystemExit(75)
    assert updates==target
    if not a.pilot and not evaluate(updates,full=True):
        if rank==0:
            save(out/'interrupted.json',dict(epoch=time.time(),updates=updates,stage='full_internal_validation',passed=False))
            if tracker:tracker.finish(exit_code=75)
        dist.destroy_process_group();raise SystemExit(75)
    if a.pilot:
        states=gather(dict(rank=rank,parameter_sha256=parameter_sha(bare),updates=updates,
            sample_proofs=sample_proofs,peak_reserved_bytes=torch.cuda.max_memory_reserved(local)))
        assert len({s['parameter_sha256'] for s in states})==1
    else:states=None
    if rank==0:
        save(out/'completion.json',dict(passed=True,epoch=time.time(),task=a.task,updates=updates,
            phase='pilot' if a.pilot else 'training',world_size=world,logical_batch_per_rank=64,global_batch=512,
            rank_states=states,full_internal_validation=not a.pilot))
        if tracker:tracker.finish()
    dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
