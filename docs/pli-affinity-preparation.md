# PLI affinity preparation — 2026-11-04

## Scope and decisions

Branch `codex/pli-affinity-predictor` starts at freshly fetched `origin/main`
`a443fc81`. Existing main checkout and running remote experiments are untouched.
This implementation prepares manifest audits and the single-system extraction
boundary; it is not a launched feature-generation or head-training campaign.

Use the colleague's SAIR + BindingDB residual corpus, current Hangang backbone
(exact checkpoint/config still pending), and `feat/affinity` head design. The head
module was ported from `86c10b5c389fbcae8c3f22d24802e764223cd120`, adapting
kernel selection to current main's constructor-based backend API. Its initial
unchanged copy failed a real CPU forward test with the removed
`use_cuequiv_kernels` forward argument; both stack paths now use the main API.
The corpus/sampler/packing reference is
`codex/pr-affinity-core-config@0327f8cb95e74ede66dd52bbc4d4dc1c73a32895`.
Its historical nearest-500 pocket and assay+ligand split are not silently adopted.
PPI remains a later project, not part of this extraction plan.

Native LM-to-pair dropout stays on. Use one immutable unique system index directly
as the RNG seed, with one cached view per system. Assay duplicates refer to the
same cache. Persist the append-only registry before sharding; never renumber after
filtering, reordering, adding decoys, or restarting. Resolve duplicate source system
IDs for the same canonical input before creating the registry. Label/split must not
affect seed. Fingerprint exact apo/MSA/ligand-conformer/tokenization input as well
as checkpoint, config, code and registry. Index alone is not a cache identity.

The extraction function uses one unpadded system per call with isolated Python,
NumPy, CPU and CUDA RNG states. It preserves native inference LM dropout while
keeping ordinary training dropout disabled via eval. Global RNG scopes must not
overlap in threads. Different kernels/hardware may still change numeric results.
Pre-tokenization stochastic input preparation also needs its own recorded seed
and input digest. Default `num_recycles=3` follows the colleague's API argument;
current main runs `num_recycles + 1` trunk iterations before refinement.

Full-context trunk precedes crop. Persist BF16 full singles, directional PL/LP/LL
z, symmetric triangular raw distance logits, actual bin boundaries and token
mapping. Never persist PP. The new extraction payload is versioned
`pli_full_cross_bf16_v1`; it is NOT byte/schema compatible with historical LMDB.
A compatible sharded writer, codec, resume receipt, loader and crop/head adapter
are still required before production. Do not publish millions of per-system files.

## Manifest access and source provenance

Messi master / UID1020 verified. These directories exist but hwkim cannot list them
(mode 0700, owner mseok; ACL checked for the affinity directory):

```text
/backup/mseok/affinity_20260911
/backup/mseok/pocket80k_full_cross_full350_20260822_r2
/backup/mseok/pocket80k_target_full350_allprofiles_20260822_r1
```

Request directory traversal plus read access to manifests, date metadata, source
receipts, configuration, exclusion reports and archive inventories. Feature arrays
are not required for this audit. Do not bypass owner permissions or substitute a
newly downloaded dataset as the colleague's exact snapshot.

The readable conversation archive establishes these ORIGINAL paths (on the
colleague's former machine, not verified as current Messi paths):

```text
/mnt/parallel_storage/wykim_lab/icl_mseok/artifacts/kfold/sair_source_date_metadata_20260819_r1/systems.parquet
/mnt/parallel_storage/wykim_lab/icl_mseok/artifacts/kfold/sair_source_date_metadata_20260819_r1/affinity_labels.parquet
/mnt/parallel_storage/wykim_lab/icl_mseok/artifacts/kfold/external/bindingdb/BindingDB_All_2025-04-14.tsv.gz
```

Archived source receipts report SHA256
`e20af4fdf1cd76452e05ff4ee3da3ee4c07d4d7baf3742a59e91871453dcf2b2`
for systems and
`54ac3fee50e3145ef8a1c5116faa38b5e77d22903b7637e637e9ee6e0d5a2161`
for labels. These hashes are historical evidence, not fresh file verification.
Evidence: data-preparation conversation `01a01dd5-d611-7781-b49f-b2a37641c511`,
lines 2703/3798; PLI conversation `01a0200f-92f3-7b72-aa47-7972bc5523e3`, line 6643,
inside the readable `conversations.tar.zst` backup. No private transcript copied.

## Time cutoff audit

Keep separate: (1) experimental-label publication/first-public-availability date,
(2) database curation date, (3) database snapshot/release date, (4) structure
pretraining cutoff, (5) apo/template source dates. A 2025-04-14 filename is a
snapshot identifier, not proof that every measurement is date-filtered or dated.
The current checkpoint's pretraining cutoff requires its own training receipt;
the historical 2021-09-30 structure cutoff is not an affinity-label cutoff.

The inspected colleague `AffinityRecord` normalization preserves document IDs but
does not carry publication/curation date fields. Recover dates from the original
date-enriched sidecar before eligibility/split decisions. `IC50` on SAIR records
is an explicit proxy label in that code; the original endpoint/relation was not
preserved by the released sidecar. Do not infer inactive status from entropy.

The audit CLI accepts an explicit inclusive cutoff and explicit date field. It
reports missing, invalid, straddling, before/on and after dates without deleting
records. Year/month-only dates become intervals. It never replaces publication
dates with file timestamps or download dates. A strict temporal experiment needs
a chosen cutoff, a rule for missing dates and a holdout policy for repeated
measurements of a system/drug across the boundary. These remain decisions.

## Exposure and evaluation

At historical global batch 512, 20,000 optimizer steps consume 10,240,000 label
slots. Against 1,092,142 systems / 1,781,883 labels, arithmetic averages are
9.376 system visits / 5.747 label visits. These historical denominators may include
validation and are not the new training-only population. ActivityCliffBucketSampler
weights eligible assays by IQR and reserves 60 grouped + 4 singleton labels per
rank. Therefore averages do not establish the exposure tail, coverage or effective
epochs. Count actual sampler draws by system, including zero-visit systems, and
report P50/P90/P99/max once the final manifest and sampling schedule are known.
Head epochs do not create new trunk dropout views in a fixed cache.

Proposed evaluation: known targets with held-out drug identities, plus family-wise
reports for kinase, GPCR, nuclear receptor and ion channel. Family reports are
strata of the held-out ligand set, not a replacement for a split. Keep sequence
construct identity, target-family annotation provenance and unknown family values.
Do not tune checkpoints/thresholds on final test. Exact fractions, temporal cutoff,
benchmark roster and synthetic-negative inclusion remain pending.

## Commands and verification

```bash
PYTHONPATH=src python scripts/affinity/audit_manifest.py joined.jsonl
# An explicit user-chosen temporal audit; no cutoff is silently selected:
PYTHONPATH=src python scripts/affinity/audit_manifest.py joined.parquet --cutoff YYYY-MM-DD --date-field publication_date
PYTHONPATH=src python -m pytest tests/affinity -q
```

Joined records require `system_id`, `record_id`, `sequence`, `canonical_smiles`.
Optional columns include origin, endpoint, relation, split, target_family,
publication_date, protein_tokens and ligand_tokens. Preserve all original label
columns upstream. Parquet reading requires pyarrow. Estimated payload excludes
indices/metadata, compression, staging and replicas. Audit readiness stays false.

Tests use small synthetic tensors and a stochastic toy trunk; a real checkpoint,
GPU, full corpus, codec throughput, and training/resume remain unverified.
