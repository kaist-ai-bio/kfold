"""Read-only cached features plus auditable label rows; no backbone dependency."""
from collections import OrderedDict
import json
import sqlite3
from pathlib import Path
import random
from collections import defaultdict
from bisect import bisect_left, bisect_right

import torch
from torch.utils.data import Dataset

from .bundles import read_payload
from .head_inputs import compact_head_inputs


class CachedLabelDataset(Dataset):
    def __init__(self, manifest, catalog, *, task, split, view='query_200', cache_records=4):
        if task not in ('regression', 'classification') or split not in ('train','validation','external_novel','external_overlap_aux','kinase_aux'):
            raise ValueError('Explicit task and evaluation stratum required')
        # Frozen consensus used pre-split source evidence. It is not an internal
        # validation-safe training feature without a new training-only plan.
        if view not in ('query_200','query_350'):
            raise ValueError('Consensus views require split-specific evidence plans; no silent fallback')
        self.manifest, self.catalog = str(manifest), str(catalog)
        self.task, self.split, self.view = task, split, view
        self.cache_records, self._cache = cache_records, OrderedDict()
        availability_path=Path(manifest).with_name('view-availability.json')
        if not availability_path.exists():
            raise ValueError('Audited view availability is required before sampling')
        availability=json.loads(availability_path.read_text())
        gate=json.loads(Path(manifest).with_name('completion.json').read_text())
        if availability['manifest_sha256']!=gate['manifest_sha256']:
            raise ValueError('View availability belongs to a different manifest')
        self.excluded_positions=availability['excluded_positions'][view][task][split]
        if self.excluded_positions!=sorted(set(self.excluded_positions)):
            raise ValueError('Invalid view exclusion index')
        self.adjusted_exclusions=[p-i for i,p in enumerate(self.excluded_positions)]
        db = self.connect()
        self.length = db.execute('SELECT count(*) FROM task_index WHERE task=? AND split=?', (task,split)).fetchone()[0]-len(self.excluded_positions)
        db.close()

    def connect(self):
        db = sqlite3.connect(Path(self.manifest).resolve().as_uri()+'?mode=ro&immutable=1',uri=True)
        db.row_factory = sqlite3.Row
        return db

    def __len__(self):
        return self.length

    def label(self, position):
        if not 0<=position<self.length:raise IndexError(position)
        position+=bisect_right(self.adjusted_exclusions,position)
        db = self.connect()
        try:
            row = db.execute('SELECT e.* FROM task_index i JOIN examples e ON e.row_id=i.row_id '
                'WHERE i.task=? AND i.split=? AND i.position=?', (self.task,self.split,position)).fetchone()
            if row is None:
                raise IndexError(position)
            return dict(row)
        finally:
            db.close()

    def logical_position(self, original_position):
        offset=bisect_left(self.excluded_positions,original_position)
        if offset<len(self.excluded_positions) and self.excluded_positions[offset]==original_position:
            return None
        return original_position-offset

    def load_row(self, row):
        identity = row['input_id']
        if identity not in self._cache:
            self._cache[identity] = read_payload(self.catalog,identity)
            while len(self._cache) > self.cache_records:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(identity)
        compact = self._cache[identity]
        historical = row.get('historical_key') or None
        # Query indices are explicitly shared across alias keys. A missing label
        # key never selects an arbitrary consensus; all routes remain metadata.
        if compact.get('schema') != 'pli_historical_crop_union_bf16_v1':
            historical = None
        result = compact_head_inputs(compact,self.view,historical_key=historical)
        return result,row

    def __getitem__(self, position):
        return self.load_row(self.sample_label(position))

    def sample_label(self, position):
        if isinstance(position,tuple):
            position,loss_group=position
            row=self.label(position);row['loss_group']=loss_group
            return row
        return self.label(position)


def collate_cached(examples):
    """Pad each axis with a matching token mask; labels keep measurement identity."""
    if not examples:
        raise ValueError('Empty batch')
    count = len(examples)
    length = max(x['token_mask'].shape[1] for x,_ in examples)
    single = examples[0][0]['s_inputs'].shape[-1]
    pair = examples[0][0]['z'].shape[-1]
    result = dict(s_inputs=torch.zeros(count,length,single),s_lm=torch.zeros(count,length,single),
        z=torch.zeros(count,length,length,pair),distogram_features=torch.zeros(count,length,length,3),
        token_mask=torch.zeros(count,length,dtype=torch.bool),protein_mask=torch.zeros(count,length,dtype=torch.bool),
        ligand_mask=torch.zeros(count,length,dtype=torch.bool))
    for i,(x,_) in enumerate(examples):
        n=x['token_mask'].shape[1]
        for key in ('s_inputs','s_lm','token_mask','protein_mask','ligand_mask'):
            result[key][i,:n]=x[key][0]
        for key in ('z','distogram_features'):
            result[key][i,:n,:n]=x[key][0]
    return result,[r for _,r in examples]


class TaskBatchSampler:
    """Deterministic task sampling, independent of model variant.

    Regression: distinct same-assay groups weighted by observed IQR, plus
    regression-only singleton slots. Classification: equal active/inactive;
    one half of negative slots are measured and one half synthetic.
    Partition chemical groups never collapse distinct exact ligand inputs.
    """
    def __init__(self, dataset, *, batches, seed=20261109, groups=12, group_size=5,
                 singleton_slots=4, batch_size=64):
        self.dataset,self.batches,self.seed=dataset,batches,seed
        self.groups,self.group_size,self.singletons=groups,group_size,singleton_slots
        self.batch_size=batch_size
        if batches<=0 or batch_size<=0:
            raise ValueError('Finite positive sampling budget required')
        if dataset.task=='regression' and groups*group_size+singleton_slots!=batch_size:
            raise ValueError('Regression layout must exactly fill the batch')
        db=dataset.connect()
        rows=db.execute('SELECT i.position,e.assay_key,e.input_id,e.p_activity,e.synthetic,e.binary '
            'FROM task_index i JOIN examples e USING(row_id) WHERE i.task=? AND i.split=? ORDER BY i.position',
            (dataset.task,dataset.split))
        self.assays=defaultdict(lambda:defaultdict(list));self.class_pools=defaultdict(list)
        for position,assay,ligand,value,synthetic,binary in rows:
            if hasattr(dataset,'logical_position'):
                position=dataset.logical_position(position)
                if position is None:continue
            if dataset.task=='regression':self.assays[assay][ligand].append((position,value))
            else:self.class_pools[(int(binary),int(synthetic))].append(position)
        db.close()
        self.keys=[];self.weights=[]
        for key in sorted(self.assays):
            if key.startswith('regression_only:') or len(self.assays[key])<group_size:continue
            values=sorted(sum(v for _,v in rows)/len(rows) for rows in self.assays[key].values())
            q=torch.quantile(torch.tensor(values,dtype=torch.float64),torch.tensor([.25,.75],dtype=torch.float64))
            weight=float(q[1]-q[0])
            if weight>0:self.keys.append(key);self.weights.append(weight)
        eligible=set(self.keys)
        self.singleton_pool=[p for key,v in self.assays.items() if key not in eligible
            for entries in v.values() for p,_ in entries]
        if dataset.task=='regression' and (len(self.keys)<groups or (self.singletons and not self.singleton_pool)):
            raise ValueError('Insufficient rankable assay groups')
        if dataset.task=='classification' and any(not self.class_pools[k] for k in [(1,0),(0,0),(0,1)]):
            raise ValueError('Measured positive/negative and synthetic negative pools required')

    def __len__(self):return self.batches

    def __iter__(self):
        rng=random.Random(self.seed)
        for batch_number in range(self.batches):
            if self.dataset.task=='regression':
                available=list(range(len(self.keys)));chosen=[]
                for _ in range(self.groups):
                    index=rng.choices(available,weights=[self.weights[i] for i in available],k=1)[0]
                    available.remove(index);chosen.append(index)
                batch=[]
                for index in chosen:
                    by_ligand=self.assays[self.keys[index]]
                    for ligand in rng.sample(sorted(by_ligand),self.group_size):
                        batch.append(rng.choice(by_ligand[ligand])[0])
                batch.extend((position,f'__absolute_only:{batch_number}:{slot}')
                    for slot,position in enumerate(rng.choices(self.singleton_pool,k=self.singletons)))
            else:
                positive=self.batch_size//2;measured_negative=(self.batch_size-positive)//2
                synthetic_negative=self.batch_size-positive-measured_negative
                batch=[]
                for key,count in [((1,0),positive),((0,0),measured_negative),((0,1),synthetic_negative)]:
                    pool=self.class_pools[key]
                    batch.extend(rng.sample(pool,count) if len(pool)>=count else rng.choices(pool,k=count))
                rng.shuffle(batch)
            yield batch
