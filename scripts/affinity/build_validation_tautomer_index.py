#!/usr/bin/env python3
"""Conservative tautomer-overlap audit keys; never change model-input chemistry."""
import argparse
import json
import multiprocessing
from pathlib import Path
import time

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import rdMolHash
from rdkit.Chem.MolStandardize import rdMolStandardize


def tautomer_key(smiles, parent_no_stereo=None):
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None or mol.GetNumAtoms() == 0:
        return {"parent_no_stereo": parent_no_stereo or smiles, "tautomer_hash": None}
    fragments=Chem.GetMolFrags(mol,asMols=True,sanitizeFrags=True)
    organic=[p for p in fragments if any(a.GetAtomicNum()==6 for a in p.GetAtoms())]
    parent=max(organic or fragments,key=lambda p:(p.GetNumHeavyAtoms(),Chem.MolToSmiles(p)))
    parent=rdMolStandardize.Uncharger().uncharge(parent)
    Chem.RemoveStereochemistry(parent)
    for atom in parent.GetAtoms():
        atom.SetIsotope(0)
    return {"parent_no_stereo": parent_no_stereo or Chem.MolToSmiles(parent,isomericSmiles=False),
            "tautomer_hash": rdMolHash.MolHash(parent, rdMolHash.HashFunction.HetAtomTautomer)}


def from_parent_record(item):
    return tautomer_key(item[1],item[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("keys", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    RDLogger.DisableLog("rdApp.*")
    start=time.time()
    db=duckdb.connect()
    db.execute("SET memory_limit='4GB'")
    db.execute(f"SET threads={args.workers}")
    db.read_parquet(str(args.keys)).create_view("keys")
    parents=db.execute("SELECT parent_no_stereo,any_value(canonical_smiles) FROM keys WHERE parse_ok GROUP BY 1 ORDER BY 1").fetchall()
    schema=pa.schema([("parent_no_stereo",pa.string()),("tautomer_hash",pa.string())])
    missing=0
    with pq.ParquetWriter(args.out,schema,compression="zstd") as writer:
        with multiprocessing.Pool(args.workers) as pool:
            batch=[]
            for i,row in enumerate(pool.imap(from_parent_record,parents,chunksize=256),1):
                missing+=row["tautomer_hash"] is None
                batch.append(row)
                if len(batch)==10000:
                    writer.write_table(pa.Table.from_pylist(batch,schema=schema));batch=[]
                    print("TAUTOMER_KEYS",i,round(time.time()-start,1),flush=True)
            if batch:writer.write_table(pa.Table.from_pylist(batch,schema=schema))
    receipt={"rdkit_version":rdBase.rdkitVersion,"parents":len(parents),"missing":missing,"seconds":time.time()-start,"method":"v2: parse original canonical SMILES, select largest organic fragment, uncharge, remove stereo/isotope on the molecular graph, then RDKit HetAtomTautomer hash; retain original parent-string lookup key. Avoid reparsing a non-isomeric SMILES that can lose isotope-associated hydrogen information. Audit only; source/model chemistry unchanged."}
    args.out.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__=="__main__":
    main()
