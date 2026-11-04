"""Pin donor chemistry and fingerprints without assigning synthetic labels.

Morgan radius2/2048-bit/no-chirality is an explicit implementation choice, not
claimed to be Boltz-2's unpublished fingerprint recipe. Keep exact input chemistry
unchanged. Existing parent/tautomer keys are used only for held-out exclusion.
"""
import argparse
import hashlib
import json
import multiprocessing
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, DataStructs, RDLogger, rdBase
from rdkit.Chem import rdFingerprintGenerator

GENERATOR = None


def init():
    global GENERATOR
    RDLogger.DisableLog('rdApp.*')
    GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False)


def fingerprint(row):
    key, smiles, graph, parent, tautomer, heldout = row
    try:
        molecule = Chem.MolFromSmiles(graph or '')
        if molecule is None:
            raise ValueError('Missing/unparseable exact graph')
        fp = GENERATOR.GetFingerprint(molecule)
        blob, popcount, error = DataStructs.BitVectToBinaryText(fp), fp.GetNumOnBits(), None
        if popcount == 0:
            raise ValueError('Empty fingerprint')
    except Exception as exc:
        blob, popcount, error = None, None, str(exc)
    return dict(ligand_sha=key, smiles=smiles, exact_graph=graph, parent_no_stereo=parent,
                tautomer_hash=tautomer, heldout_identity=bool(heldout),
                donor_eligible=not heldout and tautomer is not None and error is None,
                fingerprint=blob, fingerprint_popcount=popcount, fingerprint_error=error)


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    for name in ['pool','keys','tautomers','quantitative','censored','out']:
        p.add_argument(name,type=Path)
    p.add_argument('--workers',type=int,default=8)
    a=p.parse_args();a.out.mkdir(exist_ok=False)
    d=duckdb.connect();d.execute('SET threads=4');d.execute("SET memory_limit='6GB'")
    for name in ['pool','keys','tautomers','quantitative','censored']:
        d.read_parquet(str(getattr(a,name))).create_view(name)
    d.execute('''CREATE TABLE heldout AS SELECT exact_graph,parent_no_stereo,tautomer_hash FROM quantitative
        UNION SELECT exact_graph,parent_no_stereo,tautomer_hash FROM censored''')
    d.execute('''CREATE TABLE inputs AS SELECT p.ligand_sha,p.smiles,k.exact_graph,k.parent_no_stereo,t.tautomer_hash,
        (EXISTS(SELECT 1 FROM heldout h WHERE h.exact_graph=k.exact_graph)
        OR EXISTS(SELECT 1 FROM heldout h WHERE h.parent_no_stereo=k.parent_no_stereo)
        OR EXISTS(SELECT 1 FROM heldout h WHERE h.tautomer_hash=t.tautomer_hash)) heldout_identity
        FROM pool p LEFT JOIN keys k ON p.smiles=k.canonical_smiles
        LEFT JOIN tautomers t USING(parent_no_stereo)''')
    n=d.execute('SELECT count(*) FROM pool').fetchone()[0]
    assert d.execute('SELECT count(*),count(DISTINCT ligand_sha) FROM inputs').fetchone()==(n,n), 'Duplicate chemistry join'
    schema=pa.schema([(x,pa.string()) for x in ['ligand_sha','smiles','exact_graph','parent_no_stereo','tautomer_hash']]+
        [('heldout_identity',pa.bool_()),('donor_eligible',pa.bool_()),('fingerprint',pa.binary()),
         ('fingerprint_popcount',pa.int64()),('fingerprint_error',pa.string())])
    inputs=d.execute('SELECT * FROM inputs ORDER BY ligand_sha').fetchall()
    written=eligible=blocked=errors=0
    with pq.ParquetWriter(a.out/'fingerprints.parquet',schema,compression='zstd') as writer:
        with multiprocessing.Pool(a.workers,initializer=init) as pool:
            batch=[]
            for row in pool.imap(fingerprint,inputs,chunksize=128):
                batch.append(row);written+=1;eligible+=row['donor_eligible'];blocked+=row['heldout_identity'];errors+=row['fingerprint_error'] is not None
                if len(batch)==10000:
                    writer.write_table(pa.Table.from_pylist(batch,schema=schema));batch=[]
                    print('FINGERPRINTS',written,flush=True)
            if batch:writer.write_table(pa.Table.from_pylist(batch,schema=schema))
    assert written==n
    # Check binary encoding is stable under RDKit decoding before publication.
    sample=pq.ParquetFile(a.out/'fingerprints.parquet').read_row_group(0,columns=['fingerprint','fingerprint_popcount']).to_pylist()
    for row in sample:
        if row['fingerprint'] is not None:
            fp=DataStructs.CreateFromBinaryText(row['fingerprint'])
            assert fp.GetNumBits()==2048 and fp.GetNumOnBits()==row['fingerprint_popcount']
    receipt=dict(stage='fingerprints_and_donor_identity_qc_only_no_negative_labels',rows=written,
        donor_eligible=eligible,heldout_identity_excluded=blocked,fingerprint_errors=errors,
        rdkit_version=rdBase.rdkitVersion,recipe=dict(type='Morgan_bit',radius=2,bits=2048,include_chirality=False,input='existing exact RDKit graph; original model chemistry unchanged'),
        scope='Fingerprint parameters are our pinned implementation choice; all measured associations remain exclusion evidence candidates, not asserted binders. Known-drug kinase ligands are not held out.',
        input_sha256={name:sha(getattr(a,name)) for name in ['pool','keys','tautomers','quantitative','censored']},
        output_sha256=sha(a.out/'fingerprints.parquet'),output_bytes=(a.out/'fingerprints.parquet').stat().st_size,
        synthetic_pairs_created=0)
    (a.out/'fingerprint-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
