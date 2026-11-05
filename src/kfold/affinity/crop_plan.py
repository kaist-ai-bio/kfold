"""Bind frozen training evidence to an exact full-input feature record.

Unresolved aliases and missing earlier ranks remain pending. Unstable
consensuses can be retained with explicit quality flags when requested.
"""
import hashlib

import torch

from .crops import query_views, select_windows
from .preparation import digest
from .profile_corpus import consensus_from_prefix, validate_profile_input


def stable_record_plan(source, candidates, profiles, *, historical_keys,
                       feature_contract, candidate_provenance, retain_unstable=False):
    """Build an unambiguous three-view storage plan, preserving stability.

    The caller supplies the complete historical-key inventory for this exact
    sequence/apo, and verifies the candidate/profile artifact checksums before
    calling. Candidate rows must originate from the frozen training-only export;
    validation and decoy records may be queries, never candidate evidence.
    """
    for field in ('candidate_manifest_sha256', 'frontier_sha256'):
        value = candidate_provenance.get(field, '')
        if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Verified candidate artifact provenance required')
    raw = source['source_input']
    sha = lambda value: hashlib.sha256(value.encode()).hexdigest()
    descriptor = dict(input_id=raw['input_id'], system_index=raw['system_index'],
        input_sha=sha(raw['sequence']), ligand_sha=sha(raw['smiles']),
        apo_sha=raw['apo_sha256'], protein_tokens=int(source['protein_mask'].sum()))
    descriptor['target_input_key'] = sha(descriptor['input_sha']+':'+descriptor['apo_sha'])
    validate_profile_input(source, descriptor, feature_contract)
    keys = set(historical_keys)
    if len(keys) != 1:
        return dict(status='pending',reason='historical_target_alias_review' if keys else 'training_inventory_unresolved')
    if not candidates:
        return dict(status='pending',reason='training_evidence_not_available')
    if any(c['target_input_key'] != descriptor['target_input_key'] or c['protein_key'] not in keys
           or c['input_sha'] != descriptor['input_sha'] or c['apo_sha'] != descriptor['apo_sha']
           for c in candidates):
        raise ValueError('Training evidence belongs to a different sequence/apo/source target')
    result = consensus_from_prefix(candidates, profiles)
    if result['status'] != 'ready':
        return dict(status='pending',reason=result.get('reason','consensus_unresolved'),evidence_status=result['status'])
    consensus = result['consensus']
    if not consensus['strict_stable'] and not retain_unstable:
        return dict(status='pending',reason='unstable_consensus_policy_review',consensus_sha256=digest(result))
    medoid = profiles[consensus['medoid_system_id']]
    if (medoid['target_input_key'] != descriptor['target_input_key'] or
            medoid['input_sha'] != descriptor['input_sha'] or medoid['apo_sha'] != descriptor['apo_sha']):
        raise ValueError('Medoid profile identity mismatch')
    mapping = medoid['protein_token_mapping']
    if not mapping:
        raise ValueError('Missing medoid protein token mapping')
    for name, values in mapping.items():
        if name not in source['token_mapping'] or source['token_mapping'][name][source['protein_mask']].tolist() != values:
            raise ValueError('Medoid and query protein token layouts differ')
    if not torch.equal(torch.tensor(consensus['distance_profile']),torch.tensor(medoid['protein_min_expected_distance'])):
        raise ValueError('Medoid distance profile changed')
    views, unavailable, _ = query_views(source)
    try:
        views['target_consensus_200'] = select_windows(source,consensus['distance_profile'],
            max_protein_tokens=200,max_tokens=256)
    except ValueError as exc:
        unavailable['target_consensus_200'] = str(exc)
    if not views:
        return dict(status='unsupported',reason='no_approved_crop_fits',unavailable=unavailable)
    receipt = dict(selector_version='quality_flagged_training_consensus_v2',training_ready=True,
        training_inclusion_decided=False,
        unstable_consensus_policy='retain_flagged' if retain_unstable else 'pending_review',
        config=dict(query_200=[200,256],target_consensus_200=[200,256],query_350=[350,512]),
        target_input_key=descriptor['target_input_key'],historical_target_key=next(iter(keys)),
        consensus_sha256=digest(result),candidate_provenance=candidate_provenance,
        medoid_system_id=consensus['medoid_system_id'],
        evidence_system_ids=consensus['evidence_system_ids'],
        evidence_candidate_ranks=consensus['evidence_candidate_ranks'],
        medoid_mean_jaccard=consensus['medoid_mean_jaccard'],
        split_half_jaccard=consensus['split_half_jaccard'],
        strict_stable=consensus['strict_stable'],
        consensus_quality='stable' if consensus['strict_stable'] else 'unstable')
    return dict(status='final',source_cache_identity_sha256=source['cache_identity_sha256'],
        views={name:indices.tolist() for name,indices in views.items()},
        unavailable=unavailable,crop_receipt=receipt)
