#!/usr/bin/env python3
"""Download an auditable ChEMBL candidate superset, without choosing a split.

The full pinned database remains the archival source. API status is checked
before and after; publication year is a retrieval bound, not a temporal proof.
"""
import argparse
import concurrent.futures
import datetime
import gzip
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://www.ebi.ac.uk/chembl/api/data/"


def request(url):
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "K-Fold-validation-audit/1.0"})
            with urllib.request.urlopen(req, timeout=120) as response:
                return json.load(response)
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** (attempt + 1))


def collect(root, endpoint, key, filters, year, workers):
    folder = root / endpoint
    folder.mkdir(exist_ok=True)
    def fetch(offset):
        params = dict(filters, limit=1000, offset=offset, order_by=("activity_id" if endpoint == "activity" else "document_chembl_id"))
        url = BASE + endpoint + ".json?" + urllib.parse.urlencode(params)
        path = folder / f"{offset:08d}.json.gz"
        if path.exists():
            with gzip.open(path, "rt") as stream:
                data = json.load(stream)
        else:
            data = request(url)
            with gzip.open(path.with_suffix(".part"), "wt") as stream:
                json.dump(data, stream, separators=(",", ":"))
            path.with_suffix(".part").replace(path)
        rows = data[key]
        assert data["page_meta"]["offset"] == offset
        assert data["page_meta"]["limit"] == 1000
        for row in rows:
            assert int(row["document_year" if endpoint == "activity" else "year"]) >= year, ("Ignored API year filter", row)
            if endpoint == "activity":
                assert row["standard_type"] in {"Kd", "Ki", "IC50", "EC50"}
        receipt = {"url": url, "file": str(path), "rows": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        print(json.dumps({"endpoint": endpoint, "offset": offset, "rows": len(rows)}), flush=True)
        return data["page_meta"]["total_count"], receipt, [row["activity_id" if endpoint == "activity" else "document_chembl_id"] for row in rows]
    total, first, ids = fetch(0)
    receipts = [first]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for count, receipt, page_ids in pool.map(fetch, range(1000, total, 1000)):
            assert count == total, "API result changed during pagination"
            receipts.append(receipt)
            ids.extend(page_ids)
    assert len(ids) == len(set(ids)) == total
    return {"total": total, "pages": receipts}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--release", default="ChEMBL_37")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    before = request(BASE + "status.json")
    assert before["chembl_db_version"] == args.release
    identity = {"release": args.release, "year": args.year, "types": ["Kd", "Ki", "IC50", "EC50"]}
    identity_path = args.out / "query.json"
    if identity_path.exists():
        assert json.loads(identity_path.read_text()) == identity
    else:
        identity_path.write_text(json.dumps(identity, indent=2) + "\n")
    docs = collect(args.out, "document", "documents", {"year__gte": args.year}, args.year, args.workers)
    acts = collect(args.out, "activity", "activities", {"document_year__gte": args.year, "standard_type__in": "Ki,Kd,IC50,EC50"}, args.year, args.workers)
    after = request(BASE + "status.json")
    assert before == after, "API release/counts changed during download"
    receipt = {"status_before": before, "status_after": after, "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "documents": docs, "activities": acts, "scope": "Candidate superset; no temporal novelty, train-overlap, assay-quality or final-split eligibility asserted"}
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("VERIFIED", docs["total"], acts["total"], flush=True)


if __name__ == "__main__":
    main()
