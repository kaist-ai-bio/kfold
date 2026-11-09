"""Preserve historical target representations in one lossless physical union.

Label routing must supply its original historical key. No first-key default,
consensus averaging, or relabeling is permitted. Queries are shared only after
their selectors have been shown identical across all historical plans.
"""
import copy
import hashlib

import torch

from .compact import VIEW_NAMES, _subset, pack_crop_union, verify_against_source
from .preparation import digest

SCHEMA = 'pli_historical_crop_union_bf16_v1'
POLICY = 'preserve_historical_target_views_v1'


def representation_id(source_system_id, historical_key):
    # Exactly the colleague's logical (source system, historical key) identity.
    return 'pocket80k-' + hashlib.sha256(
        (source_system_id + '\0' + historical_key).encode()).hexdigest()


def pack_alias_union(source, plans):
    if len(plans) < 2:
        raise ValueError('Multiple complete historical plans required')
    views, unavailable, routing, representations = {}, {}, {}, {}
    target = None
    for key, plan in sorted(plans.items()):
        if not isinstance(key, str) or len(key) != 64 or any(c not in '0123456789abcdef' for c in key):
            raise ValueError('Historical key must be a SHA256')
        receipt = plan['crop_receipt']
        if (plan['status'] != 'final' or
                plan['source_cache_identity_sha256'] != source['cache_identity_sha256'] or
                receipt.get('historical_target_key') != key or
                receipt.get('training_ready') is not True):
            raise ValueError('Historical source identity or finalized plan mismatch')
        current_target = receipt.get('target_input_key')
        if not current_target or (target is not None and target != current_target):
            raise ValueError('Historical aliases must share exact sequence/apo')
        target = current_target
        # Reuse every existing three-view validation, including full ligand.
        single = pack_crop_union(source, plan['views'], crop_receipt=receipt,
                                 unavailable=plan.get('unavailable', {}))
        routing[key] = {}
        for name in VIEW_NAMES:
            physical = f'{name}/{key}' if name == 'target_consensus_200' else name
            routing[key][name] = physical
            if name in single['crop_receipt']['source_view_indices']:
                indices = single['crop_receipt']['source_view_indices'][name]
                if physical in unavailable or (physical in views and views[physical] != indices):
                    raise ValueError('Shared query selectors disagree')
                views[physical] = indices
            else:
                reason = single['unavailable_views'][name]
                if physical in views or (physical in unavailable and unavailable[physical] != reason):
                    raise ValueError('Shared query availability disagrees')
                unavailable[physical] = reason
        representations[representation_id(source['receipt']['system_id'], key)] = key
    union = torch.tensor(sorted({i for ids in views.values() for i in ids}), dtype=torch.long)
    out = _subset(source, union)
    reverse = torch.full((len(source['protein_mask']),), -1, dtype=torch.long)
    reverse[union] = torch.arange(len(union))
    receipt = dict(schema=SCHEMA, policy=POLICY,
        source_cache_identity_sha256=source['cache_identity_sha256'],
        source_view_indices=views, unavailable_views=unavailable,
        historical_plans=copy.deepcopy(plans), historical_views=routing,
        representation_keys=representations, target_input_key=target)
    out.update(schema=SCHEMA, source_token_indices=union,
        view_indices={name: reverse[torch.tensor(ids)] for name, ids in views.items()},
        unavailable_views=unavailable, crop_receipt=receipt,
        receipt=copy.deepcopy(source['receipt']), cache_identity_sha256=digest(receipt))
    return out


def resolve_historical_view(compact, historical_key, view, *, source_system_id=None):
    """Resolve an explicitly retained label key; unknown keys never fall back."""
    if compact.get('schema') != SCHEMA or view not in VIEW_NAMES:
        raise ValueError('Historical union and approved logical view required')
    if digest(compact['crop_receipt']) != compact['cache_identity_sha256']:
        raise ValueError('Historical routing receipt changed')
    if source_system_id is not None and source_system_id != compact['receipt']['system_id']:
        raise ValueError('Label belongs to a different physical input')
    mapping = compact['crop_receipt']['historical_views']
    if historical_key not in mapping:
        raise ValueError('Unknown historical label key; no default consensus')
    rid = representation_id(compact['receipt']['system_id'], historical_key)
    if compact['crop_receipt']['representation_keys'].get(rid) != historical_key:
        raise ValueError('Historical representation identity changed')
    return mapping[historical_key][view]


def verify_alias_against_source(source, compact):
    if compact.get('schema') != SCHEMA:
        raise ValueError('Unexpected historical union schema')
    expected = pack_alias_union(source, compact['crop_receipt']['historical_plans'])
    if expected['crop_receipt'] != compact['crop_receipt']:
        raise ValueError('Historical plan or label routing changed')
    return verify_against_source(source, compact)


def load_historical_head_inputs(catalog_path, source_system_id, historical_key, view):
    """Label adapter: retain both source ID and historical key through loading."""
    from .bundles import read_payload
    from .head_inputs import compact_head_inputs
    compact = read_payload(catalog_path, source_system_id)
    resolve_historical_view(compact, historical_key, view, source_system_id=source_system_id)
    return compact_head_inputs(compact, view, historical_key=historical_key)
