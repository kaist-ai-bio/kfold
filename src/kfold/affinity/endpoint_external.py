"""Full frozen external benchmarks, endpoint-conditioned, distributed within training."""
import hashlib,json,math,sqlite3
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr,kendalltau
import torch
from .head_inputs import compact_head_inputs
from .bundles import read_payload
from .shards import decode

def metric(rows):
    if not rows:return {'n':0}
    a=np.array([r['p_activity'] for r in rows]);b=np.array([r['prediction'] for r in rows])
    out={'n':len(rows),'mae':float(np.mean(abs(a-b)))}
    if len(rows)>=2 and np.std(a)>0 and np.std(b)>0:
        out.update(pearson=float(pearsonr(a,b).statistic),kendall=float(kendalltau(a,b).statistic))
    return out

def reports(rows):
    result={}
    for benchmark in sorted({r['benchmark'] for r in rows}):
        group=[r for r in rows if r['benchmark']==benchmark];bytarget=defaultdict(list)
        for row in group:bytarget[row['target_id']].append(row)
        values=[metric(v) for v in bytarget.values()];pool=metric(group)
        for key in ('pearson','kendall'):
            eligible=[v for v in values if key in v];weight=sum(v['n'] for v in eligible)
            pool['target_weighted_'+key]=sum(v['n']*v[key] for v in eligible)/weight if weight else None
        pool['chemical_nonoverlap']=metric([r for r in group if not r['overlap']['chemical_train']])
        result[benchmark]=pool
    return result

def scalars(result):
    out={}
    for benchmark,values in result.items():
        prefix='external/'+benchmark.replace('+','plus')
        for key,value in values.items():
            if isinstance(value,dict):
                for child,v in value.items():
                    if isinstance(v,(int,float)) and math.isfinite(v):out[prefix+'/'+key+'/'+child]=v
            elif isinstance(value,(int,float)) and math.isfinite(value):out[prefix+'/'+key]=value
    return out

class ExternalEvaluation:
    def __init__(self, config, catalog):
        self.config=config;self.catalog=catalog
        read=lambda key:json.loads(Path(config[key]).read_text())
        lines=lambda key:[json.loads(x) for x in Path(config[key]).read_text().splitlines()]
        assert read('completion')['passed']
        self.cache=read('index');self.tasks={(x['sequence_sha256'],x['ligand_sha256']):x for x in lines('input_map')}
        self.new={(hashlib.sha256(x['sequence'].encode()).hexdigest(),hashlib.sha256(x['smiles'].encode()).hexdigest()):x['input_id'] for x in lines('gpu_inputs')}
        self.labels=lines('labels');self.endpoints={x['record_id']:x for x in lines('endpoints')}
        self.overlap={x['record_id']:x for x in read('overlap')['rows']}
        assert len(self.labels)==len({x['record_id'] for x in self.labels})==1100
        assert set(self.endpoints)==set(self.overlap)=={x['record_id'] for x in self.labels}
        assert len(self.tasks)==997
        # Never merge predictions across distinct requested endpoints of the same input.
        self.requests=sorted({(x['sequence_sha256'],x['ligand_sha256'],self.endpoints[x['record_id']]['endpoint_index']) for x in self.labels})
        assert all(0<=x[2]<=4 for x in self.requests)

    def local_predictions(self, head, device, rank, world):
        head.eval();outputs=[]
        for sequence,ligand,endpoint in self.requests[rank::world]:
            key=(sequence,ligand);task=self.tasks[key]
            ident=task['existing_input']['input_id'] if task['cache_reusable'] else self.new[key]
            if task['cache_reusable']:payload=read_payload(self.catalog,ident)
            else:
                db=sqlite3.connect(Path(self.cache[ident]).as_uri()+'?mode=ro&immutable=1',uri=True)
                try:row=db.execute('SELECT payload,blob_sha,raw_sha FROM records WHERE system_id=?',(ident,)).fetchone()
                finally:db.close()
                assert row is not None
                payload=decode(*row)
            features=compact_head_inputs(payload,'query_200')
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                pred=head(**{k:v.to(device,non_blocking=True) for k,v in features.items()},
                          endpoint_index=torch.tensor([endpoint],device=device,dtype=torch.long))
            value=float(pred['value'].float().cpu().item());logit=float(pred['logits'].float().cpu().item())
            assert math.isfinite(value) and math.isfinite(logit)
            outputs.append(((sequence,ligand,endpoint),value,logit))
        return outputs

    def finalize(self, parts):
        flat=[r for part in parts for r in part];pred={tuple(r[0]):r[1:] for r in flat}
        assert len(flat)==len(pred)==len(self.requests) and set(pred)==set(self.requests)
        rows=[]
        for x in self.labels:
            ep=self.endpoints[x['record_id']];value,logit=pred[(x['sequence_sha256'],x['ligand_sha256'],ep['endpoint_index'])]
            rows.append(dict(record_id=x['record_id'],benchmark=x['benchmark'],target_id=x['target_id'],
                             p_activity=x['p_activity'],prediction=value,binary_logit=logit,
                             endpoint=ep['endpoint'],endpoint_index=ep['endpoint_index'],overlap=self.overlap[x['record_id']]))
        return dict(passed=True,rows=rows,reports=reports(rows),evaluated_labels=len(rows),
                    unique_inputs=len(self.tasks),input_endpoint_requests=len(self.requests),
                    endpoint_dropout=False,used_for_optimizer=False,used_for_checkpoint_selection=False,
                    development_benchmark=True)
