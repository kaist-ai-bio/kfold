"""Read-only independent identity audit and sampled similarity recomputation."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import sqlite3

import numpy as np
import pyarrow.parquet as pq
from rdkit import DataStructs


def main():
    p = argparse.ArgumentParser()
    for name in ['candidates', 'associations', 'fingerprints', 'clusters', 'registry', 'out']:
        p.add_argument(name, type=Path)
    a = p.parse_args()
    rows = pq.read_table(a.candidates).to_pylist()
    assert rows and len(rows) <= 250000
    clusters = dict(line.split('\t')[::-1] for line in a.clusters.read_text().splitlines())
    fingerprints = {r['ligand_sha']: r for r in pq.read_table(a.fingerprints,
        columns=['ligand_sha', 'fingerprint', 'donor_eligible', 'heldout_identity']).to_pylist()}
    measured = defaultdict(set)
    anchors = {}
    for r in pq.read_table(a.associations).to_pylist():
        measured[clusters[r['protein_sha']]].add(r['ligand_sha'])
        anchors[r['input_id']] = r
    db = sqlite3.connect('file:' + str(a.registry) + '?mode=ro', uri=True)
    existing = set(db.execute('SELECT protein_sha,ligand_sha FROM systems'))
    db.close()
    ids, pairs = set(), set()
    for r in rows:
        protein, ligand = r['protein_sha'], r['ligand_sha']
        pair = (protein, ligand)
        assert pair not in existing and pair not in pairs
        pairs.add(pair)
        expected = hashlib.sha256(('pli_decoy_candidate_v1:' + protein + ':' + ligand).encode()).hexdigest()
        assert r['candidate_id'] == expected and expected not in ids
        ids.add(expected)
        anchor = anchors[r['anchor_input_id']]
        assert anchor['protein_sha'] == protein and anchor['historical_train_eligible']
        assert r['cluster_id'] == clusters[protein]
        assert r['cluster_measured_ligands'] == len(measured[clusters[protein]])
        assert ligand not in measured[clusters[protein]]
        assert fingerprints[ligand]['donor_eligible'] and not fingerprints[ligand]['heldout_identity']
        assert r['evidence_type'] == 'synthetic_assumed_negative'
        assert r['assumed_binary_label'] == 0 and r['affinity_value'] is None
        assert 0 <= r['max_cluster_similarity'] < .3
        assert r['protein_tokens'] == anchor['protein_tokens']
        assert r['total_tokens'] == r['protein_tokens'] + r['ligand_tokens']
    sample = random.Random(71359).sample(rows, min(256, len(rows)))
    cache = {}
    def fp(key):
        if key not in cache:
            cache[key] = DataStructs.CreateFromBinaryText(fingerprints[key]['fingerprint'])
        return cache[key]
    for r in sample:
        value = max(DataStructs.TanimotoSimilarity(fp(r['ligand_sha']), fp(k))
                    for k in measured[r['cluster_id']])
        assert value < .3 and abs(value - r['max_cluster_similarity']) < 1e-12
    quantiles = [0, .5, .9, .95, .99, 1]
    summary = dict(candidates=len(rows), unique_pairs=len(pairs), all_identity_checks_passed=True,
        independently_recomputed_similarity_rows=len(sample), similarity_checks_passed=True,
        target_proteins=len({r['protein_sha'] for r in rows}),
        unique_donor_ligands=len({r['ligand_sha'] for r in rows}),
        max_ligand_reuse=max(Counter(r['ligand_sha'] for r in rows).values()),
        quantile_probabilities=quantiles,
        token_quantiles={k: np.quantile([r[k] for r in rows], quantiles).tolist()
                         for k in ['protein_tokens', 'ligand_tokens', 'total_tokens']},
        above_2048_tokens=sum(r['total_tokens'] > 2048 for r in rows),
        input_sha256=hashlib.sha256(a.candidates.read_bytes()).hexdigest(),
        scope='Identity checks cover all candidates; similarity independently recomputed for256. Token counts inherited from parent inputs, native rebuilt input QC and feature budget remain pending.')
    a.out.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
