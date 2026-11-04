#!/usr/bin/env python3
"""Validate downloaded exact-sequence AFDB monomers, retaining all failures."""
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import gemmi


def validate(row, structure_root):
    path = structure_root/Path(row['path']).name
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == row['sha256'], 'File hash mismatch'
    block = gemmi.cif.read(str(path)).sole_block()
    sequences = [''.join(gemmi.cif.as_string(x).split()) for x in block.find_values('_entity_poly.pdbx_seq_one_letter_code_can')]
    assert sequences == [row['sequence']], 'mmCIF polymer sequence mismatch'
    columns = ['label_asym_id', 'label_seq_id', 'label_comp_id', 'label_atom_id',
               'Cartn_x', 'Cartn_y', 'Cartn_z']
    atoms = block.find('_atom_site.', columns)
    assert len(atoms)>0, 'No atoms'
    chains, ca = set(), {}
    for atom in atoms:
        assert all(math.isfinite(float(atom[i])) for i in (4,5,6)), 'Nonfinite coordinate'
        chains.add(atom[0])
        if atom[3] == 'CA':
            i = int(atom[1]); assert i not in ca, 'Duplicate CA'
            ca[i] = gemmi.find_tabulated_residue(atom[2]).one_letter_code.upper()
    assert len(chains)==1, 'Not a single-chain structure'
    assert set(ca)==set(range(1,len(row['sequence'])+1)), 'Incomplete CA coverage'
    assert ''.join(ca[i] for i in sorted(ca)) == row['sequence'], 'Coordinate residue mismatch'
    return {'coordinate_qc':'passed','atom_count':len(atoms),'ca_count':len(ca)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest',type=Path);p.add_argument('structures',type=Path)
    p.add_argument('out',type=Path);a=p.parse_args();a.out.mkdir(exist_ok=False)
    rows=json.loads(a.manifest.read_text())
    for row in rows:
        if row['status'] == 'selected_pending_coordinate_qc':
            try: row.update(validate(row,a.structures))
            except Exception as exc:
                row.update(coordinate_qc='failed',qc_error=type(exc).__name__+': '+str(exc))
        else: row['coordinate_qc']='unavailable'
    (a.out/'target-priors-qc.json').write_text(json.dumps(rows,separators=(',',':'))+'\n')
    unresolved=[r for r in rows if r['coordinate_qc']!='passed']
    (a.out/'unresolved-targets.json').write_text(json.dumps(unresolved,separators=(',',':'))+'\n')
    summary={'sequences':len(rows),'coordinate_qc':dict(Counter(r['coordinate_qc'] for r in rows)),
      'unresolved_reasons':dict(Counter(r.get('qc_error',r['status']) for r in unresolved)),
      'input_manifest_sha256':hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
      'input_preparation_and_conformer_qc_complete':False,'atlasfold_gpu_started':False}
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))


if __name__=='__main__':main()
