"""64-rank joint regression/classification, one physical B16 forward per update."""
import argparse
from dataclasses import asdict
from datetime import timedelta
from functools import partial
import hashlib,json,math,os,pickle,signal,socket,time
from pathlib import Path
if os.environ.get('KFOLD_AFFINITY_OVERLAY'):
    import bootstrap
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from kfold.affinity.heads import HeadConfig
from kfold.affinity.training_data import CachedLabelDataset,TaskBatchSampler,collate_cached
from kfold.affinity.distributed_training import rank_seed,initialize_worker,colleague_validation_metrics
from kfold.affinity.joint_training import (JointAffinityHead,JointDataset,JointBatchSampler,
    BinaryFiveSampler,collate_joint,joint_loss_terms,normalized_joint_loss)
from kfold.affinity.training_monitor import validation_positions,scalar_report,start_tracking
from kfold.affinity.pilot import metrics,save,parameter_sha


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--resume')
    parser.add_argument('--pilot',action='store_true');parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--allow-full-training',action='store_true');a=parser.parse_args()
    rank=int(os.environ['SLURM_PROCID']);world=int(os.environ['SLURM_NTASKS']);local=int(os.environ['SLURM_LOCALID'])
    assert world==64 and torch.cuda.device_count()==4
    c=json.loads(Path(a.config).read_text());out=Path(a.output)
    assert c['distributed']['gradient_accumulation']==1 and c['distributed']['batch_per_rank']==16
    torch.cuda.set_device(local);device=torch.device('cuda',local);torch.set_num_threads(4)
    dist.init_process_group('nccl',rank=rank,world_size=world,timeout=timedelta(seconds=600),device_id=device)
    if rank==0:
        assert not out.exists(),'Each launch phase requires a fresh output directory'
        out.mkdir(parents=True)
    dist.barrier()
    def gather(value):
        result=[None]*world;dist.all_gather_object(result,value);return result
    proof=gather(dict(rank=rank,host=socket.gethostname(),local_rank=local,
        uuid=str(torch.cuda.get_device_properties(local).uuid),cpus=len(os.sched_getaffinity(0)),torch=torch.__version__))
    check=torch.tensor(float(rank),device=device);dist.all_reduce(check)
    assert check.item()==2016 and len({r['uuid'] for r in proof})==64 and len({r['host'] for r in proof})==16
    if rank==0:save(out/'communication.json',dict(passed=True,ranks=proof,all_reduce=2016))
    if a.preflight:
        dist.destroy_process_group();return
    assert a.pilot or a.allow_full_training
    data=c['data'];gate=None
    if rank==0:
        gate=json.loads(Path(data['manifest']).with_name('completion.json').read_text())
        assert gate['passed'] and gate['chemical_train_val_overlap']==gate['input_train_val_overlap']==0
        assert sha(data['manifest'])==gate['manifest_sha256']
        assert sha(Path(data['manifest']).with_name('view-availability.json'))==data['availability_sha256']
    obj=[gate];dist.broadcast_object_list(obj,src=0);gate=obj[0]
    assert data['view']=='query_200'
    prepared=json.loads(Path(c['sampler_summary']).read_text())
    assert prepared['passed'] and prepared['manifest_sha256']==gate['manifest_sha256']
    assert sha(c['sampler_state'])==prepared['sampler_state_sha256']
    with open(c['sampler_state'],'rb') as f:sample_state=pickle.load(f)
    contract=hashlib.sha256(json.dumps(dict(config=c,world=world,pilot=a.pilot,
        manifest_sha256=gate['manifest_sha256']),sort_keys=True).encode()).hexdigest()
    torch.manual_seed(c['seed']);torch.cuda.manual_seed(c['seed'])
    hc=HeadConfig(variant=c.get('variant','colleague_cross'),**c['model']);bare=JointAffinityHead(hc).to(device)
    model=DDP(bare,device_ids=[local],broadcast_buffers=False,gradient_as_bucket_view=True)
    opt=torch.optim.AdamW(bare.parameters(),**c['optimizer']);start=0;state=None
    if a.resume:
        state=torch.load(a.resume,map_location='cpu',weights_only=False)
        assert state['contract']==contract and state['world_size']==world and state['task']=='joint'
        bare.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);start=state['updates']
    train={task:CachedLabelDataset(data['manifest'],data['catalog'],task=task,split='train')
           for task in ('regression','classification')}
    val={task:CachedLabelDataset(data['manifest'],data['catalog'],task=task,split='validation')
         for task in ('regression','classification')}
    target=c['pilot_steps'] if a.pilot else c['max_steps'];samplers={}
    for task in ('regression','classification'):
        sampler=TaskBatchSampler.__new__(TaskBatchSampler)
        sampler.__dict__.update(sample_state[task]);sampler.dataset=train[task]
        sampler.seed=rank_seed(c['seed'],rank)+(0 if task=='regression' else 1);sampler.batches=target
        if task=='regression':
            sampler.groups=2;sampler.group_size=5;sampler.singletons=1;sampler.batch_size=11
        else:
            sampler=BinaryFiveSampler(sampler.class_pools,batches=target,seed=sampler.seed)
        samplers[task]=sampler
    joint=JointDataset(train['regression'],train['classification'],sample_state['bound_rows'])
    sampler=JointBatchSampler(samplers['regression'],samplers['classification'],start=start,
        bound_count=len(sample_state['bound_rows']),seed=rank_seed(c['seed'],rank))
    loader=DataLoader(joint,batch_sampler=sampler,collate_fn=partial(collate_joint,pad_to=256 if a.pilot else None),
        num_workers=c['num_workers'],persistent_workers=bool(c['num_workers']),pin_memory=True,
        multiprocessing_context='spawn' if c['num_workers'] else None,worker_init_fn=initialize_worker,
        prefetch_factor=1 if c['num_workers'] else None,generator=torch.Generator().manual_seed(rank_seed(c['seed'],rank)))
    panels={task:validation_positions(d,per_stratum=16 if a.pilot else c['validation_per_stratum'],
        seed=c['validation_seed']) for task,d in val.items()}
    if rank==0:
        save(out/'resolved-config.json',dict(task='joint',world_size=world,contract=contract,
            gradient_accumulation=1,global_batch=1024,**c))
    tracker=None
    if rank==0 and not a.pilot:
        tracker=start_tracking(c['wandb'],out,task='joint',variant=hc.variant,
            contract=contract,config=c,resume=bool(a.resume))
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
                contract=contract,task='joint',config=asdict(hc),world_size=world,rng=rng),tmp);tmp.replace(dest)
        dist.barrier()
    def reports(predictions,rows,task):
        groups={}
        for i,row in enumerate(rows):
            for key in (row['cohort'],row['cohort']+'__'+row['endpoint']):groups.setdefault(key,[]).append(i)
        result={}
        for key,indices in groups.items():
            labels=[rows[i] for i in indices];p=predictions[indices];m=metrics(p,labels,task)
            if task=='regression':m.update(colleague_validation_metrics(p,labels))
            else:
                terms=joint_loss_terms(dict(value=p*0,logits=p),[
                    dict(r,p_activity=None,synthetic=bool(r['synthetic']),classification_valid=True) for r in labels],
                    **c['loss'])
                m['focal']=float(terms.sums['binary'])/max(terms.counts['binary'],1)
            result[key]=m
        return result
    def evaluate(updates,full=False):
        bare.eval();results={};prefix='validation_full' if full else 'validation_panel'
        for task,dataset in val.items():
            positions=list(range(len(dataset))) if full else panels[task];local_records=[]
            width=c['validation_batch_size'];steps=math.ceil(len(positions)/(world*width))
            for i in range(steps):
                indices=positions[(i*world+rank)*width:(i*world+rank+1)*width]
                if indices:
                    x,rows=collate_cached([dataset[j] for j in indices])
                    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                        outputs=bare(**{k:v.to(device,non_blocking=True) for k,v in x.items()})
                        prediction=outputs['value' if task=='regression' else 'logits'].float().cpu().tolist()
                    for position,pred,row in zip(indices,prediction,rows):
                        local_records.append((position,pred,{k:row[k] for k in
                            ('row_id','input_id','assay_key','cohort','endpoint','p_activity','binary','synthetic')}))
                if stopped():return False
            records=[v for part in gather(local_records) for v in part]
            if rank==0:
                records.sort(key=lambda x:x[0]);assert [v[0] for v in records]==sorted(positions)
                results[task]=dict(rows=len(records),cohorts=reports(torch.tensor([v[1] for v in records]),
                    [v[2] for v in records],task))
                if tracker:tracker.log(dict(update=updates,**scalar_report(results[task]['cohorts'],prefix+'/'+task)))
        if rank==0:save(out/f'{prefix}-{updates:08d}.json',dict(updates=updates,tasks=results,
            full_internal_validation=full,external_used_for_selection=False))
        dist.barrier();return True
    if a.resume:
        torch.set_rng_state(state['rng'][rank]['cpu']);torch.cuda.set_rng_state(state['rng'][rank]['cuda'],local)
    else:torch.cuda.manual_seed(rank_seed(c['seed'],rank))
    if not a.resume:assert evaluate(0)
    began=time.monotonic();updates=start;last=began;sample_proofs=[];nonzero=set()
    torch.cuda.reset_peak_memory_stats()
    for updates,(x,rows) in enumerate(loader,start=start+1):
        loaded=time.monotonic();model.train();opt.zero_grad(set_to_none=True)
        tokens=x['token_mask'].shape[1];x={k:v.to(device,non_blocking=True) for k,v in x.items()}
        # Exactly one physical forward/backward and optimizer step; no no_sync,
        # microbatch loop, or loss scaling by an accumulation factor.
        with torch.autocast('cuda',dtype=torch.bfloat16):outputs=model(**x)
        terms=joint_loss_terms(outputs,rows,**c['loss']);loss,counts=normalized_joint_loss(terms)
        assert torch.isfinite(loss);loss.backward()
        if a.pilot:
            modules=[('shared',bare.shared),('value_head',bare.value_head),('binary_head',bare.binary_head)]
            if bare.shared.proddiff is not None:
                modules.extend([('proddiff',bare.shared.proddiff),('proddiff_norm',bare.shared.proddiff.norm),
                                ('proddiff_projection',bare.shared.proddiff.output)])
            for label,module in modules:
                if any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in module.parameters()):nonzero.add(label)
        norm=torch.nn.utils.clip_grad_norm_(bare.parameters(),c['gradient_clip'],error_if_nonfinite=True);opt.step()
        sample_proofs.append(hashlib.sha256(json.dumps([r['row_id'] for r in rows]).encode()).hexdigest())
        now=time.monotonic()
        if updates==start+1 or updates%c['log_every']==0 or updates==target:
            sums=torch.stack([terms.sums[k].detach().double() for k in ('absolute','difference','binary')]);dist.all_reduce(sums)
            means=sums/counts.clamp_min(1);timing=torch.tensor([loaded-last,now-loaded],device=device)
            dist.all_reduce(timing,op=dist.ReduceOp.MAX)
            progress=dict(epoch=time.time(),updates=updates,max_steps=target,variant=hc.variant,gradient_accumulation=1,
                batch_per_rank=16,world_size=world,global_batch=1024,tokens=tokens,
                gradient_norm=float(norm),metrics=dict(absolute_huber=float(means[0]),difference_huber=float(means[1]),
                    focal=float(means[2]),loss=float(.1*means[0]+.9*means[1]+means[2])),
                supervision_counts=dict(zip(('absolute','difference_groups','binary'),counts.tolist())),
                data_seconds=float(timing[0]),compute_seconds=float(timing[1]),elapsed_seconds=now-began,
                steps_per_second=(updates-start)/(now-began),peak_reserved_bytes=torch.cuda.max_memory_reserved(local))
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
        del x,rows,outputs,terms,loss
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
    states=gather(dict(rank=rank,parameter_sha256=parameter_sha(bare),updates=updates,sample_proofs=sample_proofs,
        nonzero_gradient_modules=sorted(nonzero),peak_reserved_bytes=torch.cuda.max_memory_reserved(local))) if a.pilot else None
    if a.pilot:
        assert len({s['parameter_sha256'] for s in states})==1
        expected=['binary_head','shared','value_head']
        if bare.shared.proddiff is not None:expected.extend(['proddiff','proddiff_norm','proddiff_projection'])
        if not a.resume:assert all(s['nonzero_gradient_modules']==sorted(expected) for s in states)
    if rank==0:
        save(out/'completion.json',dict(passed=True,epoch=time.time(),task='joint',updates=updates,
            phase='pilot' if a.pilot else 'training',world_size=world,batch_per_rank=16,global_batch=1024,
            gradient_accumulation=1,rank_states=states,full_internal_validation=not a.pilot))
        if tracker:tracker.finish()
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
