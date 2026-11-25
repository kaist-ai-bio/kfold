"""Versioned endpoint conditioning; immutable labels and synthetic masks preserved."""
import hashlib
import json
import sqlite3
from pathlib import Path
import torch
from torch import nn
from .joint_training import JointAffinityHead

ENDPOINTS = ('Kd', 'Ki', 'IC50', 'EC50', 'unknown')
UNKNOWN = 4

class EndpointJointAffinityHead(JointAffinityHead):
    """Endpoint embedding conditions numeric readout only, never binary logits."""
    def __init__(self, config):
        super().__init__(config)
        self.endpoint_embedding = nn.Embedding(len(ENDPOINTS), config.channel_z)
        nn.init.normal_(self.endpoint_embedding.weight, std=.02)

    def forward(self, *, endpoint_index, **features):
        pooled = self.shared(**features)
        if endpoint_index.dtype != torch.long or endpoint_index.shape != pooled.shape[:-1]:
            raise ValueError('Expected one int64 endpoint index per example')
        return dict(value=self.value_head(pooled + self.endpoint_embedding(endpoint_index)).squeeze(-1),
                    logits=self.binary_head(pooled).squeeze(-1))

class EndpointReader:
    """Small indexed reads after rank-zero full SHA verification, no writes/copies."""
    def __init__(self, path):
        self.db = sqlite3.connect(Path(path).as_uri()+'?mode=ro&immutable=1', uri=True)
    def indices(self, rows):
        ids = [r['row_id'] for r in rows]
        if not ids: return []
        found = dict(self.db.execute('SELECT row_id,endpoint_index FROM endpoint_overlay WHERE row_id IN ('+
                                    ','.join('?' for _ in ids)+')', ids))
        if set(found) != set(ids): raise ValueError('Endpoint overlay has missing row IDs')
        values = [found[i] for i in ids]
        if any(type(v) is not int or not 0 <= v < len(ENDPOINTS) for v in values):
            raise ValueError('Invalid endpoint index')
        if any(r['synthetic'] and v != UNKNOWN for r,v in zip(rows, values)):
            raise ValueError('Synthetic row must retain unknown endpoint')
        return values
    def close(self): self.db.close()

def conditioned_rows(rows, indices, *, training=False, probability=.15, seed=0, rank=0, update=0):
    """15% group-level masking: pairs in one recovered assay endpoint share context.

    Actual endpoint stays available to loss/metrics. Only model input is masked.
    Deterministic hash of seed/rank/update/group makes restart reproduce masking.
    Split old groups by recovered type so incomparable endpoints are never paired.
    """
    if len(rows) != len(indices) or not 0 <= probability <= 1:
        raise ValueError('Invalid conditioning request')
    copied, effective = [], []
    for row, idx in zip(rows, indices):
        if type(idx) is not int or not 0 <= idx < len(ENDPOINTS): raise ValueError('Invalid endpoint index')
        group = (row.get('loss_group', row['assay_key']), row['assay_key'], idx)
        condition = idx
        if training and idx != UNKNOWN:
            key = json.dumps([seed, rank, update, group], sort_keys=True).encode()
            uniform = int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') / 2**64
            if uniform < probability: condition = UNKNOWN
        copied.append(dict(row, original_endpoint=row['endpoint'], endpoint=ENDPOINTS[idx],
                           endpoint_index=idx, condition_index=condition, loss_group=group))
        effective.append(condition)
    return copied, effective
