"""Validate an additive frozen prefix without modifying the primary profile DB.

Callers must verify the input artifact hashes before passing decoded rows.
The returned digest identifies the exact extended candidate inventory.
"""
import hashlib
import json


def merge_profile_overlay(parent_rows, extended_rows, primary_profiles, staged_profiles):
    if not parent_rows or len(extended_rows) <= len(parent_rows):
        raise ValueError('An additive candidate prefix is required')
    if extended_rows[:len(parent_rows)] != parent_rows:
        raise ValueError('Earlier frozen candidates changed')
    key = parent_rows[0]['protein_key']
    target = parent_rows[0]['target_input_key']
    total = parent_rows[0]['target_training_ligands']
    if ([c['candidate_rank'] for c in extended_rows] != list(range(len(extended_rows)))
            or len(extended_rows) > total
            or any(c['protein_key'] != key or c['target_input_key'] != target
                   or c['target_training_ligands'] != total for c in extended_rows)):
        raise ValueError('Mixed target or noncontiguous candidate prefix')
    candidates = {c['input_id']: c for c in extended_rows}
    if len(candidates) != len(extended_rows) or set(staged_profiles) - candidates.keys():
        raise ValueError('Duplicate or unrelated overlay input')
    merged = {k: v for k, v in primary_profiles.items() if k in candidates}
    for identity, profile in staged_profiles.items():
        if identity in merged and merged[identity] != profile:
            raise ValueError('Existing verified profile changed')
        c = candidates[identity]
        if profile.get('status') == 'excluded':
            if profile.get('reason') != 'source_exclusion_ledger' or not profile.get('source_ledger'):
                raise ValueError('Unverified exclusion')
        elif (profile.get('record_checksums_verified') is not True
              or any(profile.get(k) != c[k] for k in
                     ('input_id', 'system_index', 'target_input_key', 'input_sha', 'apo_sha', 'ligand_sha'))):
            raise ValueError('Overlay profile identity or verification mismatch')
        merged[identity] = profile
    digest = hashlib.sha256(json.dumps(extended_rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return merged, digest
