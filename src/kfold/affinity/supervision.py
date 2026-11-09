"""Explicit task masks; original endpoint, censoring and synthetic status survive."""
import hashlib
import math

import torch
import torch.nn.functional as F

from .reference_losses import boltz2_continuous_affinity_loss


def chemical_split(key, *, seed=20261109, validation_fraction=.1):
    """One derived split per normalized chemical group, across every target/source."""
    if not key or not 0 < validation_fraction < 1:
        raise ValueError('Explicit chemistry key and split fraction required')
    value = int.from_bytes(hashlib.sha256(f'{seed}\0{key}'.encode()).digest()[:8], 'big')
    return 'validation' if value / 2**64 < validation_fraction else 'train'


def task_targets(p_activity, relation='=', *, synthetic=False, threshold=6.):
    """Relations are ORIGINAL concentration relations; pActivity reverses them.

    Censored values remain bounds, never point regression labels. Binary labels
    are available only if the bound logically fixes activity at pActivity>=6.
    Synthetic decoys have assumed binary0 and no numeric affinity.
    """
    if synthetic:
        if p_activity is not None:
            raise ValueError('Synthetic labels cannot carry a numeric affinity')
        return dict(regression_valid=False, classification_valid=True,
                    binary=0., p_activity=None, p_relation=None, synthetic=True)
    if p_activity is None or not math.isfinite(float(p_activity)):
        return dict(regression_valid=False, classification_valid=False,
                    binary=None, p_activity=None, p_relation=None, synthetic=False)
    value = float(p_activity)
    relation = {'≤':'<=', '≥':'>='}.get(relation, relation)
    p_relation = {'=':'=', '<':'>', '<=':'>=', '>':'<', '>=':'<='}.get(relation)
    if p_relation is None:
        raise ValueError('Unknown censoring relation')
    binary = None
    if relation == '=':
        binary = float(value >= threshold)
    elif relation in ('<', '<=') and value >= threshold:
        binary = 1.
    elif relation == '>' and value <= threshold:
        binary = 0.
    elif relation == '>=' and value < threshold:
        binary = 0.
    return dict(regression_valid=relation == '=', classification_valid=binary is not None,
                binary=binary, p_activity=value, p_relation=p_relation, synthetic=False)


def censored_huber(prediction, bound, p_relations, *, delta=.5):
    """Optional one-sided bound objective, separate from exact-label baseline."""
    errors = []
    for pred, limit, relation in zip(prediction, bound, p_relations):
        if relation == '=':
            errors.append(pred-limit)
        elif relation in ('>', '>='):
            errors.append(torch.minimum(pred-limit, pred.new_zeros(())))
        elif relation in ('<', '<='):
            errors.append(torch.maximum(pred-limit, pred.new_zeros(())))
        else:
            raise ValueError('Unknown pActivity relation')
    e = torch.stack(errors)
    return F.huber_loss(e, torch.zeros_like(e), delta=delta)


def objective(prediction, rows, task):
    if task == 'classification':
        mask = torch.tensor([bool(r['classification_valid']) for r in rows], device=prediction.device)
        if not mask.any():
            raise ValueError('No logically defined classification labels')
        y = prediction.new_tensor([r['binary'] if r['binary'] is not None else 0 for r in rows])
        return F.binary_cross_entropy_with_logits(prediction[mask], y[mask])
    if task != 'regression':
        raise ValueError('Unknown task')
    mask = torch.tensor([bool(r['regression_valid']) and not r['synthetic'] for r in rows], device=prediction.device)
    if not mask.any():
        raise ValueError('No exact measured regression labels')
    groups=[r.get('loss_group',r['assay_key']) for r in rows]
    group = {x:i for i,x in enumerate(sorted(set(groups)))}
    # Partition groups deliberately merge stereo/protonation/tautomer variants.
    # They must not collapse distinct measured ligands in the ranking loss.
    # Assays are already scoped to target; use exact cached-input identity here.
    ligand = {x:i for i,x in enumerate(sorted({r['input_id'] for r in rows}))}
    target = prediction.new_tensor([r['p_activity'] if r['p_activity'] is not None else 0 for r in rows])
    return boltz2_continuous_affinity_loss(prediction, target,
        torch.tensor([group[key] for key in groups], device=prediction.device),
        torch.tensor([ligand[r['input_id']] for r in rows], device=prediction.device), mask,
        absolute_weight=.1, difference_weight=.9, huber_delta=.5).loss
