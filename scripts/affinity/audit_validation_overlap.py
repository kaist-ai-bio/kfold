#!/usr/bin/env python3
"""Audit the existing kinase screen against the full candidate training manifest.

Run on an allocated CPU node. The manifest is NOT a verified final training set.
Chemical keys flag overlap; they do not replace source molecular structures.
"""
import argparse
from collections import Counter
import hashlib
import json
import multiprocessing
from pathlib import Path
import sqlite3
import time

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize


def chemical_keys(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles or "")
        if mol is None or mol.GetNumAtoms() == 0:
            raise ValueError("empty or unparseable")
        exact = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        parts = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        # Conservative overlap screen only: largest carbon-containing fragment,
        # neutralized when possible, stereochemistry removed. No tautomer collapse.
        carbon = [p for p in parts if any(a.GetAtomicNum() == 6 for a in p.GetAtoms())]
        parent = max(carbon or parts, key=lambda p: (p.GetNumHeavyAtoms(), Chem.MolToSmiles(p)))
        parent = rdMolStandardize.Uncharger().uncharge(parent)
        loose = Chem.MolToSmiles(parent, canonical=True, isomericSmiles=False)
        return {"canonical_smiles": smiles, "exact_graph": exact, "parent_no_stereo": loose, "parse_ok": True}
    except Exception:
        return {"canonical_smiles": smiles, "exact_graph": None, "parent_no_stereo": None, "parse_ok": False}


def rows(db, query):
    cur = db.execute(query)
    names = [x[0] for x in cur.description]
    return [dict(zip(names, row)) for row in cur.fetchall()]


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("catalog", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    RDLogger.DisableLog("rdApp.*")
    args.out.mkdir(exist_ok=True, parents=True)
    start = time.time()
    db = duckdb.connect()
    db.execute(f"SET threads={args.workers}")
    db.execute("SET memory_limit='8GB'")
    db.read_parquet(str(args.manifest)).create_view("labels")
    cat = sqlite3.connect(f"file:{args.catalog}?mode=ro", uri=True)
    summary = {
        "manifest_sha256": hashlib.file_digest(args.manifest.open("rb"), "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "catalog_bytes": args.catalog.stat().st_size,
        "rdkit_version": rdBase.rdkitVersion,
        "comparison_scope": "Full feature-generation candidate manifest, not final filtered training membership",
        "chemical_scope": "exact RDKit graph including stereo; conservative largest organic fragment uncharged no-stereo screen. Neither is tautomer invariant or an identity adjudication.",
        "sources": rows(cat, "SELECT source_id, endpoint, unit, measurement_status, count(*) n FROM observations GROUP BY 1,2,3,4"),
        "wt_drug_records": rows(cat, "SELECT d.drug_record_id, d.source_name, d.structure_status, d.identity_resolution_level, count(o.observation_id) n FROM drug_records d LEFT JOIN observations o ON d.drug_record_id=o.drug_record_id AND o.source_id='clinical2026_wt' WHERE d.source_id='clinical2026_wt' GROUP BY d.drug_record_id ORDER BY d.source_name"),
        "wt_overview": rows(cat, "SELECT count(*) cells, count(DISTINCT drug_record_id) drugs, count(DISTINCT target_record_id) targets, sum(measurement_status='measured') measured, sum(measurement_status='measured' AND value<50) hits_lt50 FROM observations WHERE source_id='clinical2026_wt'")[0],
    }
    write_json(args.out / "kinase-inventory.json", summary)
    print("INVENTORY", json.dumps(summary["wt_overview"]), flush=True)
    drugs = rows(cat, "SELECT d.* FROM drug_records d WHERE d.drug_record_id IN (SELECT DISTINCT drug_record_id FROM observations WHERE source_id='clinical2026_wt')")
    for drug in drugs:
        drug.update(chemical_keys(drug["canonical_isomeric_smiles"] or drug["smiles"]))
    pq.write_table(pa.Table.from_pylist(drugs), args.out / "kinase-drugs.parquet")
    # Repeated observations keep their source location, conditions and missingness.
    obs = rows(cat, """SELECT o.*, d.source_name drug_name, t.source_name target_name,
        t.uniprot, t.gene, t.reference_sequence, t.reference_sequence_sha256,
        t.construct, t.construct_sequence_status, t.mutation
        FROM observations o JOIN drug_records d USING(drug_record_id)
        JOIN target_records t USING(target_record_id) WHERE o.source_id='clinical2026_wt'""")
    pq.write_table(pa.Table.from_pylist(obs), args.out / "kinase-wt-candidates.parquet", compression="zstd")
    smiles = [r[0] for r in db.execute("SELECT DISTINCT canonical_smiles FROM labels ORDER BY canonical_smiles").fetchall()]
    assert chemical_keys("C[NH3+].[Cl-]")["parent_no_stereo"] == chemical_keys("CN")["parent_no_stereo"]
    assert chemical_keys("N[C@@H](C)O")["exact_graph"] != chemical_keys("N[C@H](C)O")["exact_graph"]
    assert chemical_keys("N[C@@H](C)O")["parent_no_stereo"] == chemical_keys("N[C@H](C)O")["parent_no_stereo"]
    print("CHEMICAL_KEYS", len(smiles), flush=True)
    key_path = args.out / "training-chemical-keys.parquet"
    schema = pa.schema([(k, pa.string()) for k in ["canonical_smiles", "exact_graph", "parent_no_stereo"]] + [("parse_ok", pa.bool_())])
    with pq.ParquetWriter(key_path, schema, compression="zstd") as writer:
        with multiprocessing.Pool(args.workers) as pool:
            batch = []
            for i, key in enumerate(pool.imap(chemical_keys, smiles, chunksize=256), 1):
                batch.append(key)
                if len(batch) == 10000:
                    writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                    batch = []
                    print("KEY_PROGRESS", i, round(time.time()-start, 1), flush=True)
            if batch:
                writer.write_table(pa.Table.from_pylist(batch, schema=schema))
    db.read_parquet(str(key_path)).create_view("keys")
    db.read_parquet(str(args.out / "kinase-drugs.parquet")).create_view("drugs")
    db.read_parquet(str(args.out / "kinase-wt-candidates.parquet")).create_view("obs")
    db.execute("CREATE TABLE train_keys AS SELECT DISTINCT exact_graph, parent_no_stereo FROM keys WHERE parse_ok")
    db.execute("CREATE TABLE train_pairs AS SELECT DISTINCT l.protein_uniprot, k.exact_graph, k.parent_no_stereo FROM labels l JOIN keys k USING(canonical_smiles) WHERE k.parse_ok")
    drug_overlap = db.execute("""SELECT d.drug_record_id, d.source_name, d.structure_status, d.identity_resolution_level,
        d.parse_ok,
        EXISTS(SELECT 1 FROM train_keys t WHERE t.exact_graph=d.exact_graph) exact_graph_seen,
        EXISTS(SELECT 1 FROM train_keys t WHERE t.parent_no_stereo=d.parent_no_stereo) parent_no_stereo_seen
        FROM drugs d ORDER BY d.source_name""").fetch_arrow_table()
    pq.write_table(drug_overlap, args.out / "kinase-drug-overlap.parquet")
    overlap = db.execute("""SELECT o.*,
        d.parse_ok chemical_parse_ok,
        EXISTS(SELECT 1 FROM train_keys t WHERE t.exact_graph=d.exact_graph) drug_exact_seen,
        EXISTS(SELECT 1 FROM train_keys t WHERE t.parent_no_stereo=d.parent_no_stereo) drug_parent_seen,
        EXISTS(SELECT 1 FROM labels l WHERE l.protein_uniprot=o.uniprot) uniprot_seen,
        EXISTS(SELECT 1 FROM labels l WHERE l.sequence=o.reference_sequence) reference_sequence_seen,
        EXISTS(SELECT 1 FROM train_pairs t WHERE t.protein_uniprot=o.uniprot AND t.exact_graph=d.exact_graph) uniprot_drug_exact_seen,
        EXISTS(SELECT 1 FROM train_pairs t WHERE t.protein_uniprot=o.uniprot AND t.parent_no_stereo=d.parent_no_stereo) uniprot_drug_parent_seen
        FROM obs o JOIN drugs d USING(drug_record_id)""").fetch_arrow_table()
    pq.write_table(overlap, args.out / "kinase-wt-overlap.parquet", compression="zstd")
    db.register("overlap", overlap)
    summary["drug_overlap"] = drug_overlap.to_pylist()
    summary["measurement_overlap"] = rows(db, """SELECT measurement_status, chemical_parse_ok, drug_exact_seen, drug_parent_seen, uniprot_seen, reference_sequence_seen, uniprot_drug_exact_seen, uniprot_drug_parent_seen, count(*) n FROM overlap GROUP BY ALL""")
    summary["training_chemical_keys"] = rows(db, "SELECT parse_ok, count(*) n FROM keys GROUP BY 1")
    summary["elapsed_seconds"] = time.time()-start
    write_json(args.out / "kinase-audit.json", summary)
    print("COMPLETE", summary["elapsed_seconds"], flush=True)


if __name__ == "__main__":
    main()
