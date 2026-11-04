"""Real frozen71K extraction and bounded native-input/storage validation."""
import argparse, hashlib, json, time, gc
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch
import torch
from kfold.affinity.features import FeatureContract, extract_indexed_features, indexed_rng, pack_full_cross, hash_prepared_input
from kfold.affinity.preparation import digest
from kfold.affinity.shards import ShardStore
from kfold.data.types.ccd import CCD
from kfold.inference.query import Query, ProteinSequence, LigandSequence
from kfold.inference.runner import KFoldRunner
from kfold.model import KFold
import kfold.constants as C

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def save(p,x):
    t=p.with_suffix('.partial');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)

def input_hash(f):
    return hash_prepared_input(f)

def native_reference(model,f,seed,cycles):
    """Run the normal inference entrypoint and stop immediately before diffusion."""
    class BeforeDiffusion(Exception):pass
    saved={}; original=model.run_trunk
    def capture(inp,n):
        singles,lm,z=original(inp,n);valid=inp.token.pad_mask[0];pos=valid.nonzero().flatten();chain=inp.token.chain_type[0,valid]
        with torch.autocast(f.device.type,enabled=False):dist=model.distogram_head(z.float())
        saved.update(pack_full_cross(singles[0,valid],lm[0,valid],z[0].index_select(0,pos).index_select(1,pos),dist['logits'][0].index_select(0,pos).index_select(1,pos),protein_mask=chain==C.ChainType.PROTEIN.value,ligand_mask=chain==C.ChainType.LIGAND.value,bin_boundaries=dist['bin_boundaries'],token_mapping={}))
        return singles,lm,z
    with indexed_rng(seed),torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16),patch.object(model,'run_trunk',side_effect=capture),patch.object(model.diffusion_head,'sample_structure',side_effect=BeforeDiffusion):
        try:model.inference(f,num_recycles=cycles,num_steps=1,num_samples=1)
        except BeforeDiffusion:pass
        else:raise AssertionError('Expected pre-diffusion stop')
    assert saved
    return saved

def compare(a,b):
    keys=['s_inputs','s_lm','z_pair_indices','z_pair_values','distogram_tri_indices','distogram_tri_logits','bin_boundaries']
    for k in keys:assert torch.equal(a[k],b[k]),'Mismatch: '+k

def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--registry-sha',required=True);p.add_argument('--model',required=True);p.add_argument('--ccd',required=True);p.add_argument('--cache',required=True);p.add_argument('--revision',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    root=Path(a.out);root.mkdir(parents=True,exist_ok=True)
    tasks=json.loads(Path(a.inputs).read_text());assert len(tasks)<=56
    expected_w='4607f5e7d34923c9f03aab53d3967d160cf385ddaecbe776ebc0d080d0e711f9';expected_c='e8efcc29a4e04b624e10f6ac625751c91ae5853431fb42437bf6a1da0278b3bc'
    assert sha(Path(a.model)/'weights/kfold.pth')==expected_w
    assert sha(Path(a.model)/'config.yaml')==expected_c
    assert sha(a.ccd)=='445b126be5c2bc003930608ed9e639fd49cfba7bc454b3579fc3bc04be10e919'
    contract=FeatureContract(expected_w,expected_c,a.revision,a.registry_sha)
    run_contract=dict(feature=asdict(contract),inputs_sha256=sha(a.inputs),padding='native64/128/64',codec='sqlite+zlib1',kernel='triton',apo_source='pinned manifest',input_recipe='pli_native_input_v1')
    store=ShardStore(root/'shards',digest(run_contract),max_bytes=1024**3)
    save(root/'contract.json',run_contract)
    torch.set_num_threads(4)
    model=KFold.from_pretrained(a.model,device='cuda:0',cache_dir=a.cache,use_rna_encoder=False,kernel_backend='triton')
    runner=KFoldRunner(model,ccd=CCD.load(a.ccd),cache_dir=a.cache,verbose=False)
    save(root/'hardware.json',dict(gpu=torch.cuda.get_device_name(),vram_bytes=torch.cuda.get_device_properties(0).total_memory,torch=str(torch.__version__),lm_dropout=float(model.lm_dropout)))
    # First small, second small, maximum-length stress case; then remaining inputs.
    ordered=sorted(tasks,key=lambda x:(x['total_tokens'],x['system_index']))
    ordered=ordered[:2]+ordered[-1:]+ordered[2:-1]
    rows=[];successes=0
    for task in ordered:
        key=task['input_id'];start=time.perf_counter();record=dict(system_index=task['system_index'],input_id=key,bucket=task['bucket'],native_layout_tokens=task['total_tokens'])
        if key in store.entries:
            cached=store.get(key);assert cached['receipt']['system_index']==task['system_index'];record.update(status='reused');rows.append(record);continue
        print('START',task['system_index'],task['total_tokens'],flush=True)
        f=None;result=None
        try:
            assert sha(task['apo_path'])==task['apo_sha256']
            query=Query(name='pli_'+str(task['system_index']),sequences=[ProteinSequence(id=['A'],sequence=task['sequence'],apo=[Path(task['apo_path'])]),LigandSequence(id=['B'],smiles=task['smiles'])])
            torch.cuda.reset_peak_memory_stats()
            with indexed_rng(task['system_index']):
                apos,priors=runner.load_apo_and_prior(query)
                item=runner.build_input(query,task['system_index'],num_samples=1,apos=apos,priors=priors)
            prepared_sha=input_hash(item.f_input)
            f=item.f_input.add_batch_dim().to('cuda:0');torch.cuda.synchronize()
            prep_time=time.perf_counter()-start
            kwargs=dict(system_id=key,system_index=task['system_index'],input_sha256=prepared_sha,contract=contract)
            t=time.perf_counter();result=extract_indexed_features(model,f,**kwargs);torch.cuda.synchronize();trunk_time=time.perf_counter()-t
            assert result['receipt']['actual_tokens']==task['total_tokens']
            record.update(preparation_seconds=prep_time,trunk_and_pack_seconds=trunk_time,actual_tokens=result['receipt']['actual_tokens'],padded_tokens=result['receipt']['padded_tokens'],peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
            if successes<2:
                reference=native_reference(model,f,task['system_index'],3);compare(reference,result)
                again=extract_indexed_features(model,f,**kwargs);compare(again,result)
                alternate=extract_indexed_features(model,f,**(kwargs|{'system_index':task['system_index']+2**31}))
                assert not torch.equal(result['z_pair_values'],alternate['z_pair_values']),'Dropout view unchanged'
                # Rebuild preparation too, rather than checking only RNG on cached tensors.
                with indexed_rng(task['system_index']):
                    a2,p2=runner.load_apo_and_prior(query);i2=runner.build_input(query,task['system_index'],num_samples=1,apos=a2,priors=p2)
                assert input_hash(i2.f_input)==prepared_sha,'Prepared input not reproducible'
                del reference,again,alternate,i2,a2,p2
                record['validation']='native inference pre-diffusion equality + same-seed replay + different-seed dropout + preparation replay passed'
            atom=item.f_input.atom;valid=atom.pad_mask
            result['atom_mapping']={k:getattr(atom,k)[valid].cpu().clone() for k in ['atom_type','token_index','atom_index','ref_charge','ref_pos','ref_mask','ref_space_uid']}
            result['atom_mapping']['name_chars']=atom.ref_atom_name_chars[valid].argmax(-1).to(torch.uint8)
            result['atom_mapping']['element']=atom.ref_element[valid].argmax(-1).to(torch.uint8)
            result['source_input']=task
            t=time.perf_counter();sizes=store.add(key,result);record.update(sizes,write_seconds=time.perf_counter()-t)
            t=time.perf_counter();restored=store.get(key);compare(restored,result);record['read_decode_seconds']=time.perf_counter()-t
            for k,v in result['atom_mapping'].items():assert torch.equal(v,restored['atom_mapping'][k])
            del restored;successes+=1;record['status']='passed'
        except torch.cuda.OutOfMemoryError as exc:
            record.update(status='cuda_oom',error=str(exc))
        except Exception as exc:
            record.update(status='failed',error=repr(exc));save(root/'blocking-error.json',record);raise
        finally:
            record['elapsed_seconds']=time.perf_counter()-start;rows.append(record);save(root/'progress.json',dict(attempted=len(rows),total=len(tasks),rows=rows))
            del f,result;gc.collect();torch.cuda.empty_cache()
        print('END',json.dumps(record),flush=True)
    store.seal();save(root/'final.json',dict(contract=run_contract,rows=rows,total=len(tasks),passed=sum(x['status']=='passed' for x in rows),reused=sum(x['status']=='reused' for x in rows)))

if __name__=='__main__':main()
