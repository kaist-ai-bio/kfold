"""Read a joined manifest and report composition, dates and repeat exposure.

Example: PYTHONPATH=src python scripts/affinity/audit_manifest.py joined.jsonl
No implicit date filtering, dataset download, or feature extraction is done.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from kfold.affinity.preparation import audit_records, exposure_budget


def rows(path):
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8192):
            yield from batch.to_pylist()
    else:
        with path.open(encoding="utf-8") as handle:
            if path.suffix in (".csv", ".tsv"):
                yield from csv.DictReader(handle, delimiter="\t" if path.suffix == ".tsv" else ",")
            else:
                for line in handle:
                    if line.strip():
                        yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--cutoff", help="Inclusive YYYY-MM-DD; omission means audit without temporal eligibility")
    parser.add_argument("--date-field", default="publication_date")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--global-batch", type=int, default=512)
    args = parser.parse_args()
    report = audit_records(rows(args.manifest), cutoff=args.cutoff, date_field=args.date_field)
    with args.manifest.open("rb") as handle:
        report["input_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
    report["input_path"] = str(args.manifest.resolve())
    if report["unique_system_ids"] and report["unique_record_ids"]:
        report["arithmetic_exposure"] = exposure_budget(args.steps, args.global_batch,
            report["unique_system_ids"], report["unique_record_ids"])
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
