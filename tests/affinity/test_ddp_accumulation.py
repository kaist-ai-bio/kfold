"""CPU Gloo proof of no_sync accumulation, averaging and checkpoint continuation."""
from pathlib import Path
import json
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from contextlib import nullcontext
from kfold.affinity.distributed_training import microbatch_positions,objective_details


def worker(rank,root):
    torch.set_num_threads(1);dist.init_process_group('gloo',rank=rank,world_size=2,
        init_method=(Path(root)/'rendezvous').as_uri())
    torch.manual_seed(41);model=DDP(torch.nn.Linear(3,1));opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
    def step(index):
        opt.zero_grad(set_to_none=True)
        x=torch.randn(64,3,generator=torch.Generator().manual_seed(1000+rank*100+index))
        rows=[dict(binary=float(i<32)) for i in range(64)]
        for i,positions in enumerate(microbatch_positions('classification')):
            with model.no_sync() if i<3 else nullcontext():
                loss,_=objective_details(model(x[positions]).flatten(),[rows[j] for j in positions],'classification');(loss/4).backward()
        opt.step()
    step(1);step(2)
    state={'model':{k:v.detach().clone() for k,v in model.module.state_dict().items()},'optimizer':opt.state_dict()}
    path=Path(root)/f'checkpoint-{rank}.pt';torch.save(state,path)
    step(3);expected={k:v.detach().clone() for k,v in model.module.state_dict().items()}
    state=torch.load(path,weights_only=False);model.module.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);step(3)
    error=max(float((p-expected[k]).abs().max()) for k,p in model.module.state_dict().items())
    flat=torch.cat([v.flatten() for v in model.module.state_dict().values()]);all_weights=[torch.empty_like(flat) for _ in range(2)];dist.all_gather(all_weights,flat)
    (Path(root)/f'proof-{rank}.json').write_text(json.dumps({'resume_error':error,'rank_parameter_error':float((all_weights[0]-all_weights[1]).abs().max())}))
    dist.destroy_process_group()


def test_two_rank_ddp_accumulation_and_resume(tmp_path):
    mp.spawn(worker,args=(str(tmp_path),),nprocs=2,join=True)
    for rank in range(2):
        proof=json.loads((tmp_path/f'proof-{rank}.json').read_text())
        assert proof=={'resume_error':0.,'rank_parameter_error':0.}
