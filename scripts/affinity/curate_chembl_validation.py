#!/usr/bin/env python3
"""Make reviewable validation candidates, retaining exclusion reasons and provenance.

No train/validation/test allocation is frozen here. Publication dates and first
document indexing release remain separate. Run on an allocated CPU node with
staged inputs; activity rows and overlap output are streamed to bound memory.
"""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import re

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from audit_validation_overlap import chemical_keys, rows


def load(root, folder, key):
    return list(iter_records(root, folder, key))


def iter_records(root, folder, key):
    for path in sorted((root / folder).glob("*.json.gz")):
        with gzip.open(path, "rt") as stream:
            yield from json.load(stream)[key]


def candidate_schema(row):
    integers = {"activity_id", "document_year", "document_pubmed_id", "src_id",
                "standard_flag", "potential_duplicate", "confidence_score",
                "protein_tokens_full_sequence", "ligand_heavy_atoms"}
    booleans = {"document_indexed_after_35", "known_publication_after_sair_release",
                "known_publication_by_sair_release", "parse_ok", "core_candidate"}
    return pa.schema([(key, pa.int64() if key in integers else
                       pa.bool_() if key in booleans else
                       pa.float64() if key == "pchembl_value" else pa.string())
                      for key in row])


def protein_class(component, classes):
    names = set()
    for item in component.get("protein_classifications", []):
        cid = item["protein_classification_id"]
        seen = set()
        while cid in classes and cid not in seen:
            seen.add(cid)
            row = classes[cid]
            names.add(row["pref_name"])
            cid = row.get("parent_id")
    text = " | ".join(sorted(names)).lower().replace("-", " ")
    groups = []
    if "kinase" in text:
        groups.append("kinase")
    if "g protein coupled receptor" in text or "gpcr" in text:
        groups.append("gpcr")
    if "nuclear receptor" in text:
        groups.append("nuclear_receptor")
    if "ion channel" in text:
        groups.append("ion_channel")
    return ";".join(groups) or "other", ";".join(sorted(names))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("api", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("training_keys", type=Path)
    parser.add_argument("kinase_drugs", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    RDLogger.DisableLog("rdApp.*")
    args.out.mkdir(parents=True, exist_ok=True)
    for name in ["receipt.json", "enrichment-receipt.json", "component-receipt.json", "pubmed-receipt.json"]:
        assert (args.api / name).is_file(), name
    publication_dates = json.loads((args.api / "publication-dates.json").read_text())
    expected_activity_count = json.loads((args.api / "receipt.json").read_text())["activities"]["total"]
    documents = {r["document_chembl_id"]: r for r in load(args.api, "document", "documents")}
    assays = {r["assay_chembl_id"]: r for r in load(args.api, "assay", "assays")}
    targets = {r["target_chembl_id"]: r for r in load(args.api, "target", "targets")}
    components = {r["component_id"]: r for r in load(args.api, "target_component", "target_components")}
    classes = {r["protein_class_id"]: r for r in load(args.api, "protein_classification", "protein_classifications")}
    batch, cache, seen_ids = [], {}, set()
    exclusion_counts = Counter()
    candidate_path = args.out / "chembl37-candidates.parquet"
    writer = None
    scalar_fields = ["activity_id", "assay_chembl_id", "document_chembl_id", "molecule_chembl_id", "parent_molecule_chembl_id", "canonical_smiles", "target_chembl_id", "target_pref_name", "target_organism", "standard_type", "standard_relation", "standard_value", "standard_units", "pchembl_value", "assay_type", "data_validity_comment", "potential_duplicate", "activity_comment", "assay_variant_accession", "assay_variant_mutation", "src_id", "standard_flag"]
    for activity in iter_records(args.api, "activity", "activities"):
        assert activity["activity_id"] not in seen_ids, "Duplicate activity ID"
        seen_ids.add(activity["activity_id"])
        doc = documents[activity["document_chembl_id"]]
        assay = assays[activity["assay_chembl_id"]]
        target = targets[activity["target_chembl_id"]]
        comps = target.get("target_components", [])
        component = components[comps[0]["component_id"]] if len(comps) == 1 else {}
        row = {k: activity.get(k) for k in scalar_fields}
        row["pchembl_value"] = float(row["pchembl_value"]) if row["pchembl_value"] is not None else None
        row["document_year"] = doc.get("year")
        row["document_doi"] = doc.get("doi")
        row["document_pubmed_id"] = doc.get("pubmed_id")
        row["document_first_chembl_release"] = (doc.get("chembl_release") or {}).get("chembl_release")
        row["document_first_release_date"] = (doc.get("chembl_release") or {}).get("creation_date")
        release_number = re.search(r"(\d+)$", row["document_first_chembl_release"] or "")
        row["document_indexed_after_35"] = bool(release_number and int(release_number[1]) > 35)
        dates = publication_dates.get(str(doc.get("pubmed_id")), {})
        row["first_known_publication_lower"] = dates.get("first_known_lower")
        row["first_known_publication_upper"] = dates.get("first_known_upper")
        row["publication_events_json"] = json.dumps(dates.get("publication_events", []))
        row["first_publication_date_status"] = "pubmed_events_retrieved" if dates.get("publication_events") else "not_verified"
        # A scenario for review, not an approved split contract. Preprints and
        # other prior dataset disclosures still need separate adjudication.
        row["known_publication_after_sair_release"] = bool(row["first_known_publication_lower"] and row["first_known_publication_lower"] > "2025-06-18")
        row["known_publication_by_sair_release"] = bool(row["first_known_publication_upper"] and row["first_known_publication_upper"] <= "2025-06-18")
        row["confidence_score"] = assay.get("confidence_score")
        row["assay_description"] = assay.get("description")
        row["target_type"] = target.get("target_type")
        row["protein_uniprot"] = component.get("accession")
        row["sequence"] = component.get("sequence")
        row["protein_tokens_full_sequence"] = len(row["sequence"]) if row["sequence"] else None
        row["protein_class_group"], row["protein_class_ancestors"] = protein_class(component, classes)
        reasons = []
        def require(condition, reason):
            if not condition:
                reasons.append(reason)
        require(row["standard_type"] in {"Kd", "Ki", "IC50"}, "endpoint_not_binding_potency")
        require(row["assay_type"] == "B", "assay_not_B")
        require(row["confidence_score"] == 9, "target_confidence_not_9")
        require(row["target_type"] == "SINGLE PROTEIN" and bool(row["protein_uniprot"]), "not_single_uniprot")
        require(not row["assay_variant_accession"] and not row["assay_variant_mutation"] and not assay.get("variant_sequence"), "annotated_variant")
        require(row["standard_relation"] == "=", "nonexact_relation")
        require(row["standard_flag"] == 1, "nonstandard_measurement")
        require(not row["data_validity_comment"], "validity_comment")
        require(row["pchembl_value"] is not None and 4 < row["pchembl_value"] < 12, "outside_sair_numeric_range")
        smiles = row["canonical_smiles"]
        if smiles not in cache:
            cache[smiles] = chemical_keys(smiles)
            cache[smiles]["ligand_heavy_atoms"] = Chem.MolFromSmiles(smiles).GetNumHeavyAtoms() if cache[smiles]["parse_ok"] else None
        row.update({k: v for k, v in cache[smiles].items() if k != "canonical_smiles"})
        require(row["parse_ok"], "chemical_parse_failure")
        row["core_candidate"] = not reasons
        row["core_exclusion_reasons"] = ";".join(reasons)
        row["source_activity_json"] = json.dumps(activity, separators=(",", ":"))
        exclusion_counts.update(reasons)
        batch.append(row)
        if writer is None:
            schema = candidate_schema(row)
            writer = pq.ParquetWriter(candidate_path, schema, compression="zstd")
        if len(batch) == 10000:
            writer.write_table(pa.Table.from_pylist(batch, schema=schema))
            batch.clear()
            print("CURATED_ROWS", len(seen_ids), flush=True)
    assert writer is not None, "Empty activity population"
    if batch:
        writer.write_table(pa.Table.from_pylist(batch, schema=schema))
        batch.clear()
    writer.close()
    activity_count = len(seen_ids)
    assert activity_count == expected_activity_count
    del cache, seen_ids
    db = duckdb.connect()
    db.execute("SET threads=4")
    db.execute("SET memory_limit='8GB'")
    db.read_parquet(str(candidate_path)).create_view("candidate")
    db.read_parquet(str(args.manifest)).create_view("labels")
    db.read_parquet(str(args.training_keys)).create_view("keys")
    db.read_parquet(str(args.kinase_drugs)).create_view("kinase_drugs")
    db.execute("CREATE TABLE train_keys AS SELECT DISTINCT exact_graph, parent_no_stereo FROM keys WHERE parse_ok")
    db.execute("CREATE TABLE train_pairs AS SELECT DISTINCT l.protein_uniprot, k.exact_graph, k.parent_no_stereo FROM labels l JOIN keys k USING(canonical_smiles) WHERE k.parse_ok")
    overlap_query = """SELECT c.*,
        EXISTS(SELECT 1 FROM train_keys t WHERE t.exact_graph=c.exact_graph) drug_exact_seen,
        EXISTS(SELECT 1 FROM train_keys t WHERE t.parent_no_stereo=c.parent_no_stereo) drug_parent_seen,
        EXISTS(SELECT 1 FROM labels t WHERE t.protein_uniprot=c.protein_uniprot) uniprot_seen,
        EXISTS(SELECT 1 FROM labels t WHERE t.sequence=c.sequence) sequence_seen,
        EXISTS(SELECT 1 FROM train_pairs t WHERE t.protein_uniprot=c.protein_uniprot AND t.exact_graph=c.exact_graph) pair_exact_seen,
        EXISTS(SELECT 1 FROM train_pairs t WHERE t.protein_uniprot=c.protein_uniprot AND t.parent_no_stereo=c.parent_no_stereo) pair_parent_seen
        FROM candidate c"""
    overlap_path = args.out / "chembl37-overlap.parquet"
    db.sql(overlap_query).write_parquet(str(overlap_path), compression="zstd")
    db.read_parquet(str(overlap_path)).create_view("overlap")
    assert db.execute("SELECT count(*) FROM overlap").fetchone()[0] == activity_count
    summary = {
        "counts": {"activity_rows": activity_count, "documents": len(documents), "assays": len(assays), "targets": len(targets), "components": len(components)},
        "year_endpoint": rows(db, "SELECT document_year, standard_type, count(*) n FROM overlap GROUP BY ALL ORDER BY 1,2"),
        "doc_index_release": rows(db, "SELECT document_first_chembl_release, count(*) n, count(DISTINCT document_chembl_id) documents FROM overlap GROUP BY 1"),
        "core_flow": rows(db, """SELECT core_candidate, document_indexed_after_35, drug_exact_seen, drug_parent_seen, uniprot_seen, sequence_seen, count(*) n, count(DISTINCT exact_graph) ligands, count(DISTINCT protein_uniprot) proteins FROM overlap GROUP BY ALL"""),
        "core_by_endpoint": rows(db, "SELECT standard_type, count(*) n FROM overlap WHERE core_candidate GROUP BY 1"),
        "new_ligand_known_target_by_class": rows(db, """SELECT protein_class_group, standard_type, count(*) n, count(DISTINCT exact_graph) ligands, count(DISTINCT protein_uniprot) proteins FROM overlap WHERE core_candidate AND document_indexed_after_35 AND NOT drug_parent_seen AND uniprot_seen GROUP BY 1,2 ORDER BY 1,2"""),
        "temporal_scenario": rows(db, """SELECT known_publication_after_sair_release, known_publication_by_sair_release, count(*) n, count(DISTINCT exact_graph) ligands, count(DISTINCT protein_uniprot) proteins, count(DISTINCT document_chembl_id) documents FROM overlap WHERE core_candidate AND document_indexed_after_35 AND NOT drug_parent_seen AND uniprot_seen GROUP BY ALL"""),
        "post_sair_scenario_by_class": rows(db, """SELECT protein_class_group, standard_type, count(*) n, count(DISTINCT exact_graph) ligands, count(DISTINCT protein_uniprot) proteins FROM overlap WHERE core_candidate AND document_indexed_after_35 AND NOT drug_parent_seen AND uniprot_seen AND known_publication_after_sair_release GROUP BY 1,2 ORDER BY 1,2"""),
        "post_sair_scenario_tokens": rows(db, """SELECT count(*) unique_pairs, min(protein_tokens_full_sequence) protein_min, quantile_cont(protein_tokens_full_sequence,[0.5,0.9,0.99]) protein_quantiles, max(protein_tokens_full_sequence) protein_max, quantile_cont(ligand_heavy_atoms,[0.5,0.9,0.99]) ligand_heavy_atom_quantiles, max(ligand_heavy_atoms) ligand_heavy_atom_max FROM (SELECT DISTINCT sequence, exact_graph, protein_tokens_full_sequence, ligand_heavy_atoms FROM overlap WHERE core_candidate AND document_indexed_after_35 AND NOT drug_parent_seen AND uniprot_seen AND known_publication_after_sair_release)"""),
        "core_exclusion_flags": dict(exclusion_counts),
        "kinase_holdout_training_impact": rows(db, """SELECT count(*) label_rows, count(DISTINCT system_id) systems, count(DISTINCT l.canonical_smiles) source_smiles FROM labels l JOIN keys k USING(canonical_smiles) WHERE EXISTS(SELECT 1 FROM kinase_drugs d WHERE d.parent_no_stereo=k.parent_no_stereo)""")[0],
        "limits": ["Candidate superset only; final split and temporal cutoff not frozen", "PubMed earliest recorded publication events are not proof against earlier preprint or other source disclosures; train-side source dates remain missing", "Overlap against full feature-generation candidate manifest, not verified final train membership", "Largest fragment uncharged no-stereo key is a conservative audit flag, not tautomer-invariant identity", "Reference sequence does not establish assay construct equivalence", "Protein token counts are full database sequences, not finalized model input crops"],
    }
    (args.out / "chembl37-audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
