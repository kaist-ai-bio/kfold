#!/usr/bin/env python3
"""Preserve source target sequences and protein classifications for candidates."""
import concurrent.futures
import gzip
import hashlib
import json
from pathlib import Path
import sys
import urllib.parse
from fetch_chembl_validation import BASE, request


def main():
    root = Path(sys.argv[1])
    prior = json.loads((root / "enrichment-receipt.json").read_text())
    assert request(BASE + "status.json") == prior["status"]
    ids = set()
    for page in prior["pages"]["target"]:
        with gzip.open(page["file"], "rt") as stream:
            for target in json.load(stream)["targets"]:
                ids.update(x["component_id"] for x in target["target_components"])
    names = sorted(ids)
    folder = root / "target_component"
    folder.mkdir(exist_ok=True)
    def save(path, url):
        if path.exists():
            with gzip.open(path, "rt") as stream:
                data = json.load(stream)
        else:
            data = request(url)
            with gzip.open(path.with_suffix(".part"), "wt") as stream:
                json.dump(data, stream, separators=(",", ":"))
            path.with_suffix(".part").replace(path)
        return data, {"file": str(path), "url": url, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    def fetch(start):
        batch = names[start:start+50]
        url = BASE + "target_component.json?" + urllib.parse.urlencode({"component_id__in": ",".join(map(str, batch)), "limit": 1000})
        data, receipt = save(folder / f"{start:08d}.json.gz", url)
        got = {r["component_id"] for r in data["target_components"]}
        assert got == set(batch)
        assert len(got) == data["page_meta"]["total_count"]
        receipt["rows"] = len(got)
        print("component", start, len(got), flush=True)
        return receipt
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        pages = list(pool.map(fetch, range(0, len(names), 50)))
    folder = root / "protein_classification"
    folder.mkdir(exist_ok=True)
    first, receipt = save(folder / "00000000.json.gz", BASE + "protein_classification.json?limit=1000&offset=0&order_by=protein_class_id")
    classes = [receipt]
    total = first["page_meta"]["total_count"]
    found = [r["protein_class_id"] for r in first["protein_classifications"]]
    for start in range(1000, total, 1000):
        data, receipt = save(folder / f"{start:08d}.json.gz", BASE + f"protein_classification.json?limit=1000&offset={start}&order_by=protein_class_id")
        assert data["page_meta"]["total_count"] == total
        found.extend(r["protein_class_id"] for r in data["protein_classifications"])
        classes.append(receipt)
    assert len(found) == len(set(found)) == total
    assert request(BASE + "status.json") == prior["status"]
    (root / "component-receipt.json").write_text(json.dumps({"status": prior["status"], "components": pages, "classes": classes, "counts": {"components": len(ids), "classes": total}}, indent=2) + "\n")
    print("COMPONENTS_VERIFIED", len(ids), total, flush=True)


if __name__ == "__main__":
    main()
