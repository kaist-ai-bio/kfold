"""Frozen training-only consensus candidate order and incremental selection.

Ordering and gates follow recovered revision 7cd98dd. An absent profile blocks
selection: asynchronous feature completion must not change the frozen cohort.
These gates apply to consensus evidence only, never to decoy labels/features.
"""
import hashlib
import math
from collections import defaultdict

SELECTION_SEED = 20260822


def stable_ligand_key(protein_key, canonical_smiles):
    return hashlib.sha256(
        f'{SELECTION_SEED}\0{protein_key}\0{canonical_smiles}'.encode()
    ).hexdigest()


def freeze_candidates(rows):
    """Aggregate only eligible measured training labels, then freeze ALL ranks.

Caller resolves source_system_id to the exact registry input/apo/construct.
Original protein_key and canonical_smiles are retained for reference ordering.
Returned rows include targets with fewer than ten ligands for coverage auditing.
"""
    groups = {}
    for row in rows:
        if row['split'] != 'train' or row['eligible_for_training'] is not True:
            continue
        if row.get('synthetic', False) is not False:
            raise ValueError('Synthetic labels cannot enter consensus evidence')
        key = (row['protein_key'], row['canonical_smiles'])
        source_id = str(row.get('source_system_id') or row['system_id'])
        manifest_id = str(row['system_id'])
        value = float(row['p_activity'])
        if not all(key) or not source_id or not math.isfinite(value):
            raise ValueError('Missing identity or nonfinite training label')
        if key not in groups:
            groups[key] = dict(protein_key=key[0], canonical_smiles=key[1],
                source_system_id=source_id, manifest_system_id=manifest_id, min_p_activity=value,
                max_p_activity=value, train_label_count=0,
                selection_key=stable_ligand_key(*key))
        entry = groups[key]
        if (source_id, manifest_id) < (entry['source_system_id'], entry['manifest_system_id']):
            entry['source_system_id'], entry['manifest_system_id'] = source_id, manifest_id
        entry['min_p_activity'] = min(entry['min_p_activity'], value)
        entry['max_p_activity'] = max(entry['max_p_activity'], value)
        entry['train_label_count'] += 1
    targets = defaultdict(list)
    for entry in groups.values():
        targets[entry['protein_key']].append(entry)
    result = []
    for key in sorted(targets):
        ordered = sorted(targets[key], key=lambda x: (x['selection_key'], x['canonical_smiles']))
        result.extend(dict(entry, candidate_rank=rank, selection_seed=SELECTION_SEED,
                           target_training_ligands=len(ordered))
                      for rank, entry in enumerate(ordered))
    return result


def evidence_rejection(candidate, profile):
    """Require nonempty <15A pocket; strong-only entropy gate matches reference."""
    if profile.get('status') == 'excluded':
        if not profile.get('reason'):
            raise ValueError('An excluded feature requires a ledger reason')
        return 'feature_excluded:' + profile['reason']
    if profile.get('status') != 'passed':
        raise ValueError('Only verified passed/excluded profiles may be consumed')
    entropy = profile['pocket_hlp_15a']
    if profile['pocket_residue_count'] <= 0 or entropy is None:
        return 'empty_15a_pocket'
    if not math.isfinite(entropy) or not 0 <= entropy <= 1:
        raise ValueError('Invalid normalized pocket entropy')
    if candidate['min_p_activity'] > 6.0 and entropy > .7:
        return 'strong_high_entropy'
    return None


def select_evidence(candidates, profiles):
    """Select first ten valid ranks for ONE target; profiles keyed by frozen rank.

Profile identity/provenance must be verified by the corpus adapter before this
call. Missing earlier ranks remain pending, never silently become rejections.
"""
    ordered = sorted(candidates, key=lambda c: c['candidate_rank'])
    if not ordered:
        return dict(status='unsupported', reason='no_training_ligands', selected=[], rejected=[])
    if len({c['protein_key'] for c in ordered}) != 1:
        raise ValueError('Mixed target candidates')
    if [c['candidate_rank'] for c in ordered] != list(range(len(ordered))):
        raise ValueError('Complete contiguous frozen candidate ranks required')
    if len({c['canonical_smiles'] for c in ordered}) != len(ordered):
        raise ValueError('Duplicate candidate ligand')
    if len(ordered) < 10:
        return dict(status='unsupported', reason='fewer_than_ten_training_ligands', selected=[], rejected=[])
    selected, rejected = [], []
    for candidate in ordered:
        rank = candidate['candidate_rank']
        if rank not in profiles:
            return dict(status='pending', next_candidate_rank=rank,
                        selected=selected, rejected=rejected)
        reason = evidence_rejection(candidate, profiles[rank])
        if reason:
            rejected.append(dict(candidate_rank=rank, reason=reason))
        else:
            selected.append(candidate)
        if len(selected) == 10:
            return dict(status='ready', selected=selected, rejected=rejected)
    return dict(status='unsupported', reason='fewer_than_ten_valid_profiles',
                selected=selected, rejected=rejected)
