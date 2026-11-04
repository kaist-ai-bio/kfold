#!/usr/bin/env python3
"""Fetch assay and target metadata for the downloaded API candidate superset."""
import concurrent.futures
import gzip
import hashlib
import json
from pathlib import Path
import sys
import urllib.parse
import shutil

from fetch_chembl_validation import BASE, request


def main():
    root = Path(sys.argv[1])
    parent = json.loads((root / "receipt.json").read_text())
    status = request(BASE + "status.json")
    assert status == parent["status_after"]
    ids = {"assay": set(), "target": set()}
    for page in parent["activities"]["pages"]:
        with gzip.open(page["file"], "rt") as stream:
            for row in json.load(stream)["activities"]:
                for kind in ids:
                    ids[kind].add(row[kind + "_chembl_id"])
    output = {}
    for kind, values in ids.items():
        folder = root / kind
        folder.mkdir(exist_ok=True)
        reused = []
        known = set()
        if len(sys.argv) > 2:
            old = json.loads((Path(sys.argv[2]) / "enrichment-receipt.json").read_text())
            assert old["status"] == status
            for index, page in enumerate(old["pages"][kind]):
                src = Path(page["file"])
                dst = folder / f"reused-{index:08d}.json.gz"
                if not dst.exists():
                    shutil.copyfile(src, dst)
                assert hashlib.sha256(dst.read_bytes()).hexdigest() == page["sha256"]
                with gzip.open(dst, "rt") as stream:
                    saved = json.load(stream)[kind + "s"]
                got = {r[kind + "_chembl_id"] for r in saved}
                assert got <= values
                known.update(got)
                reused.append(dict(page, file=str(dst), reused_from=str(src)))
        names = sorted(x for x in values - known if x)
        id_key = kind + "_chembl_id"
        def fetch(start):
            batch = names[start:start+50]
            params = {id_key + "__in": ",".join(batch), "limit": 1000}
            url = BASE + kind + ".json?" + urllib.parse.urlencode(params)
            path = folder / f"{start:08d}.json.gz"
            if path.exists():
                with gzip.open(path, "rt") as stream:
                    data = json.load(stream)
            else:
                data = request(url)
                with gzip.open(path.with_suffix(".part"), "wt") as stream:
                    json.dump(data, stream, separators=(",", ":"))
                path.with_suffix(".part").replace(path)
            rows = data[kind + "s"]
            got = {row[id_key] for row in rows}
            assert got <= set(batch), "API filter ignored"
            assert len(rows) == len(got) == data["page_meta"]["total_count"]
            print(kind, start, len(rows), flush=True)
            return {"file": str(path), "url": url, "rows": len(rows), "missing_ids": sorted(set(batch)-got), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            output[kind] = reused + list(pool.map(fetch, range(0, len(names), 50)))
    after = request(BASE + "status.json")
    assert after == status
    (root / "enrichment-receipt.json").write_text(json.dumps({"status": after, "pages": output}, indent=2) + "\n")
    print("ENRICHMENT_VERIFIED", {k: len(v) for k, v in ids.items()}, flush=True)


if __name__ == "__main__":
    main()
