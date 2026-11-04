#!/usr/bin/env python3
"""Count native reference-layout tokens without conformer generation or a GPU.

Use the exact inference query parser and prepare_ref_chain implementation. This
does not certify conformers, apo alignment, or GPU featurization. Rejections are
retained, and no input is silently cropped or repaired.
"""
import argparse
import concurrent.futures
import hashlib
import json
import time
from pathlib import Path

CCD_DB = None


def initialize(ccd_path):
    global CCD_DB
    from kfold.data.types.ccd import CCD
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.warning')
    CCD_DB = CCD.load(ccd_path)


def parse_chain(kind, value):
    import kfold.constants as C
    from kfold.inference.query import ProteinSequence, LigandSequence
    from kfold.data.pipelines.structure_preparation import prepare_ref_chain
    if kind == 'protein':
        query = ProteinSequence(id=['A'], sequence=value)
        return prepare_ref_chain(C.ChainType.PROTEIN, query.ccd_sequence, CCD_DB,
                                 entity_id=1, asym_id=1, sym_id=1)
    query = LigandSequence(id=['B'], smiles=value)
    return prepare_ref_chain(C.ChainType.LIGAND, ['LIG'], CCD_DB,
                             smiles=query.smiles, entity_id=2, asym_id=2, sym_id=1)


def count_one(item):
    kind, value = item
    out = {'kind': kind, 'identity_sha256': hashlib.sha256(value.encode()).hexdigest()}
    try:
        chain = parse_chain(kind, value)
        out.update(status='ok', tokens=int(chain.num_tokens), atoms=int(chain.num_atoms),
                   residues=int(chain.num_residues))
    except Exception as exc:
        out.update(status='rejected', error=type(exc).__name__ + ': ' + str(exc))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--targets', type=Path, required=True)
    p.add_argument('--ligands', type=Path, required=True, help='One JSON string per line')
    p.add_argument('--ccd', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--limit', type=int, default=0)
    a = p.parse_args(); a.out.mkdir(exist_ok=False)
    started = time.monotonic()
    targets = json.loads(a.targets.read_text())
    def inputs():
        for row in targets:
            yield 'protein', row['sequence']
        with a.ligands.open() as f:
            for i, line in enumerate(f):
                if a.limit and i >= a.limit:
                    break
                yield 'ligand', json.loads(line)
    counts = {}; total = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,
             initializer=initialize, initargs=(a.ccd,)) as pool:
        # Bounded batches avoid millions of queued futures and excess memory.
        source = iter(inputs())
        import itertools
        with (a.out/'native-layout-counts.jsonl').open('w') as out:
            while batch := list(itertools.islice(source, 4096)):
                for row in pool.map(count_one, batch, chunksize=64):
                    out.write(json.dumps(row, separators=(',', ':'))+'\n')
                    key = row['kind'] + ':' + row['status']
                    counts[key] = counts.get(key, 0)+1; total += 1
                print('NATIVE_LAYOUT', total, round(time.monotonic()-started, 1), flush=True)
    summary = {'counts': counts, 'elapsed_seconds': time.monotonic()-started,
               'workers': a.workers, 'ligand_limit': a.limit,
               'method': 'native inference query parser + prepare_ref_chain + Chain.num_tokens',
               'full_tokenizer_coordinate_validation': False,
               'conformer_generation': False, 'gpu_started': False}
    (a.out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
