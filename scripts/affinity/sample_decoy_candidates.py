"""Deterministic synthetic-pair candidates, with conservative exclusion evidence.

Uses ALL measured training-source associations in a target's 90% cluster as a
superset of known-binder exclusions. This does not assert that every measured
association is a binder. No validation activity values are used. Existing input
pairs (including evaluation pairs) are excluded by identity only. Input chemistry,
feature registry indices, and numeric affinity labels are never modified.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import multiprocessing
from pathlib import Path
import random
import sqlite3
import time

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import DataStructs, rdBase

FPS = []
KEYS = []
DONORS = []
KNOWN = {}
EXISTING = {}
CLUSTERS = {}
LENGTHS = {}
FIXED = {}
SEED = 20261105


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''):
            h.update(b)
    return h.hexdigest()


def sample_target(task):
    protein, anchors, max_attempts = task
    cluster = CLUSTERS[protein]
    exclusion = [FPS[i] for i in KNOWN[cluster]]
    if not exclusion:
        raise ValueError('Missing measured association evidence')
    out = list(FIXED.get(protein, []))
    selected = {row['ligand_sha'] for row in out}
    failed, reasons = [], Counter()
    for anchor, p_tokens in anchors:
        rng = random.Random(str(SEED) + ':' + anchor)
        for attempt in range(1, max_attempts + 1):
            idx = DONORS[rng.randrange(len(DONORS))]
            ligand = KEYS[idx]
            if ligand in EXISTING[protein]:
                reasons['existing_pair'] += 1
                continue
            if ligand in selected:
                reasons['duplicate_proposed_pair'] += 1
                continue
            maximum = max(DataStructs.BulkTanimotoSimilarity(FPS[idx], exclusion))
            if maximum >= .3:
                reasons['cluster_similarity_ge_0.3'] += 1
                continue
            selected.add(ligand)
            identity = hashlib.sha256(('pli_decoy_candidate_v1:' + protein + ':' + ligand).encode()).hexdigest()
            out.append(dict(candidate_id=identity, anchor_input_id=anchor, protein_sha=protein,
                ligand_sha=ligand, cluster_id=cluster, max_cluster_similarity=maximum,
                cluster_measured_ligands=len(exclusion), sampling_attempt=attempt,
                sampling_seed=SEED, protein_tokens=p_tokens, ligand_tokens=LENGTHS[ligand],
                total_tokens=p_tokens + LENGTHS[ligand], evidence_type='synthetic_assumed_negative',
                assumed_binary_label=0, affinity_value=None))
            break
        else:
            failed.append(dict(anchor_input_id=anchor, protein_sha=protein, reason='sampling_attempt_limit'))
    return out, failed, dict(reasons)


def main():
    global FPS, KEYS, DONORS, KNOWN, EXISTING, CLUSTERS, LENGTHS, SEED, FIXED
    parser = argparse.ArgumentParser()
    for name in ['associations', 'fingerprints', 'clusters', 'registry', 'out']:
        parser.add_argument(name, type=Path)
    parser.add_argument('--target-count', type=int, default=250000)
    parser.add_argument('--all-training', action='store_true', help='One candidate per unique eligible source input')
    parser.add_argument('--existing-candidates', type=Path, help='Preserve already audited candidates exactly')
    parser.add_argument('--pilot-limit', type=int)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=20261105)
    parser.add_argument('--max-attempts', type=int, default=256)
    a = parser.parse_args()
    if a.target_count <= 0 or (a.pilot_limit is not None and a.pilot_limit <= 0):
        raise ValueError('Invalid candidate count')
    if a.existing_candidates and (not a.all_training or a.pilot_limit):
        raise ValueError('Preserving a candidate set requires full training expansion')
    a.out.mkdir(exist_ok=False)
    SEED = a.seed
    started = time.time()
    for line in a.clusters.read_text().splitlines():
        representative, member = line.split('\t')
        if member in CLUSTERS:
            raise ValueError('Duplicate cluster membership')
        CLUSTERS[member] = representative
    rows = pq.read_table(a.fingerprints, columns=['ligand_sha','fingerprint','donor_eligible','fingerprint_error']).to_pylist()
    lookup = {}
    for row in rows:
        if row['fingerprint_error'] is not None or row['fingerprint'] is None:
            raise ValueError('Incomplete exclusion fingerprints; cannot silently omit evidence')
        idx = len(KEYS);lookup[row['ligand_sha']] = idx;KEYS.append(row['ligand_sha'])
        FPS.append(DataStructs.CreateFromBinaryText(row['fingerprint']))
        if row['donor_eligible']:
            DONORS.append(idx)
    del rows
    assert len(lookup) == len(KEYS) and DONORS
    associations = pq.read_table(a.associations, columns=['input_id','protein_sha','ligand_sha','protein_tokens','historical_train_eligible']).to_pylist()
    known = defaultdict(set)
    anchors = []
    for row in associations:
        cluster = CLUSTERS[row['protein_sha']]
        known[cluster].add(lookup[row['ligand_sha']])
        if row['historical_train_eligible']:
            anchors.append((row['input_id'], row['protein_sha'], row['protein_tokens']))
    KNOWN = {k: sorted(v) for k, v in known.items()}
    del associations, known
    anchors.sort()
    if a.all_training:
        a.target_count = len(anchors)
    if a.target_count > len(anchors):
        raise ValueError('Requested count exceeds unique eligible training anchors')
    sampled = anchors if a.all_training else random.Random(SEED).sample(anchors, a.target_count)
    if a.pilot_limit is not None:
        sampled = sampled[:a.pilot_limit]
    del anchors
    db = sqlite3.connect('file:' + str(a.registry) + '?mode=ro', uri=True)
    existing = defaultdict(set)
    for protein, ligand, tokens in db.execute('SELECT protein_sha,ligand_sha,ligand_tokens FROM systems'):
        existing[protein].add(ligand)
        if ligand in LENGTHS and LENGTHS[ligand] != tokens:
            raise ValueError('Ambiguous ligand token count')
        LENGTHS[ligand] = tokens
    db.close();EXISTING = dict(existing)
    fixed_anchors = set()
    if a.existing_candidates:
        anchor_lookup = {anchor: (protein, tokens) for anchor, protein, tokens in sampled}
        fixed = defaultdict(list)
        pairs = set()
        donors = set(DONORS)
        for row in pq.read_table(a.existing_candidates).to_pylist():
            anchor, protein, ligand = row['anchor_input_id'], row['protein_sha'], row['ligand_sha']
            if anchor in fixed_anchors or anchor_lookup.get(anchor) != (protein, row['protein_tokens']):
                raise ValueError('Existing candidate anchor is duplicated or outside training population')
            if (protein, ligand) in pairs or ligand in EXISTING[protein] or lookup[ligand] not in donors:
                raise ValueError('Existing candidate violates pair or donor identity exclusion')
            if row['sampling_seed'] != SEED or row['cluster_id'] != CLUSTERS[protein]:
                raise ValueError('Existing candidate recipe differs')
            if not 0 <= row['max_cluster_similarity'] < .3 or row['affinity_value'] is not None:
                raise ValueError('Existing candidate violates similarity or label contract')
            fixed_anchors.add(anchor);pairs.add((protein, ligand));fixed[protein].append(row)
        FIXED = dict(fixed)
        del anchor_lookup, fixed, donors, pairs
    grouped = defaultdict(list)
    for anchor, protein, tokens in sampled:
        grouped[protein]  # Include targets whose anchors were all already preserved.
        if anchor not in fixed_anchors:
            grouped[protein].append((anchor, tokens))
    schema = pa.schema([(k, pa.string()) for k in ['candidate_id','anchor_input_id','protein_sha','ligand_sha','cluster_id']] +
        [('max_cluster_similarity',pa.float64())] + [(k,pa.int64()) for k in ['cluster_measured_ligands','sampling_attempt','sampling_seed','protein_tokens','ligand_tokens','total_tokens']] +
        [('evidence_type',pa.string()),('assumed_binary_label',pa.int64()),('affinity_value',pa.float64())])
    count = 0; seen = set(); failures = []; reasons = Counter(); long_inputs = 0; max_similarity = 0.
    with pq.ParquetWriter(a.out/'candidates.parquet', schema, compression='zstd') as writer:
        with multiprocessing.get_context('fork').Pool(a.workers) as pool:
            tasks = [(protein, grouped[protein], a.max_attempts) for protein in sorted(grouped)]
            for results, failed, rejected in pool.imap(sample_target, tasks, chunksize=1):
                failures.extend(failed);reasons.update(rejected)
                if results:
                    for row in results:
                        assert row['candidate_id'] not in seen and row['max_cluster_similarity'] < .3
                        assert row['ligand_sha'] not in EXISTING[row['protein_sha']]
                        seen.add(row['candidate_id']);long_inputs += row['total_tokens'] > 2048
                        max_similarity = max(max_similarity, row['max_cluster_similarity'])
                    writer.write_table(pa.Table.from_pylist(results,schema=schema));count += len(results)
                print('CANDIDATES',count,'failed',len(failures),'seconds',round(time.time()-started,1),flush=True)
    assert count + len(failures) == len(sampled)
    (a.out/'unfilled-anchors.json').write_text(json.dumps(failures,indent=2)+'\n')
    receipt = dict(stage='candidate_pairs_not_feature_ready',candidates=count,requested_anchors=len(sampled),
        full_target_count=a.target_count,pilot_limit=a.pilot_limit,failed_anchors=len(failures),
        all_training=a.all_training,preserved_candidates=len(fixed_anchors),
        existing_candidates_sha256=sha(a.existing_candidates) if a.existing_candidates else None,
        target_proteins=len(grouped),max_cluster_similarity=max_similarity,above_previous_2048_tokens=long_inputs,
        exclusion_policy='All measured source associations in cluster; conservative superset of known binders, not assertion that every measured pair is a binder',
        existing_pair_scope='All current registry pairs including validation, identity only; no validation activity labels used',
        source_scope='Inherits approved historical SAIR+BindingDB candidate source; missing original per-record dates remain provenance limitation',
        seed=SEED,max_attempts=a.max_attempts,rdkit_version=rdBase.rdkitVersion,rejections=dict(reasons),seconds=time.time()-started,
        input_sha256={name:sha(getattr(a,name)) for name in ['associations','fingerprints','clusters','registry']},
        output_sha256=sha(a.out/'candidates.parquet'),output_bytes=(a.out/'candidates.parquet').stat().st_size,
        feature_registry_modified=False,feature_gpu_started=False,numeric_affinity_assigned=False)
    (a.out/'candidate-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('CANDIDATES_VERIFIED',json.dumps(receipt),flush=True)


if __name__ == '__main__':
    main()
