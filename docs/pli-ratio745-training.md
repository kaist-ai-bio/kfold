# Ratio745 training and provenance

Ratio745 uses the shared cross-pair head with regression and binary binding readouts. Across the 10,000-update schedule, 25.5% of sampled entries are from the regression stream and 74.5% are from the binary stream. The sampler keeps five-entry assay groups intact and preserves the 1:2:2 exposure ratio for measured positives, measured negatives, and synthetic negatives. The [joint objective](pli-joint-boltz2-ddp64-20261110.md) applies the same loss normalization as the baseline.

| Step | Source |
|---|---|
| Freeze input identities, task labels, and splits | [Data preparation](pli-affinity-preparation.md) and [frozen data release](pli-data-release-20261109.md) |
| Read the frozen cache and select the training view | `src/kfold/affinity/training_data.py` and `head_inputs.py`, with `query_200` |
| Pack deterministic 25.5:74.5 distributed batches | `src/kfold/affinity/ratio_sampling.py` |
| Run joint optimization, validation, and checkpointing | `scripts/train_cached_affinity_joint.py` |
| Validate sampler totals, batch integrity, and replay from an update | `tests/affinity/test_ratio_sampling.py` |

The [config template](../configs/affinity/ratio745.json) retains the scientific settings for a fresh run. Supply the local paths to the catalog, label manifest, and trusted sampler state and summary. Preserve the label manifest's companion `completion.json` and `view-availability.json`. The trainer checks their identities, split disjointness, and sampler checksums. Set a new tracking run ID and project for production.

The historical execution used 16 nodes with four visible GPUs per node, 64 Slurm ranks, batch 16 per rank, and one optimizer update per forward/backward cycle. Its launcher provides the Slurm rank variables and NCCL rendezvous. `--preflight` verifies that allocation and communication. `--pilot` runs a bounded training check, and `--allow-full-training` starts production. Each phase uses a fresh output directory.

For an exact resume, use the archived launch configuration, manifest and sampler digests, and rank layout. The trainer binds the resume contract to the full configuration, so remapping paths with the template yields a different contract.

## Historical checksums

The archived production snapshot contains the original launcher and configuration. Its exact trainer and sampler are retained in this branch.

| Artifact | SHA-256 |
|---|---|
| Original Ratio745 `config.json` | `9ae46fb2270661d742b7d5cb1b838fea838c346058711286292a1318f99980ac` |
| `train_cached_affinity_joint.py` | `8a17af95ee7bd3644401cd568d0b4c1a860a41e6d2d560fd25372b51472ab204` |
| `ratio_sampling.py` | `99f5c6113a5dff3d772f9c38f266bd714fff016e6198761124fd764f37b88d78` |
| Shared `step-00001000.pt` | `e323ab653071c264c4dcd71009a99a5d4141dad3973877de7f8896048260a910` |

The head was evaluated using `query_200` features from the frozen 71K EMA backbone. The v1.1 inference branch consumes a new backbone path. Benchmarking it requires a separate evaluation with the same external rows.
