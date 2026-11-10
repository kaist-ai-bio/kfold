# Affinity head: baseline, controlled outer-product ablation and training preparation

## Initial production launch, 2026-11-10

The user authorized initial head training after data preservation and cache relocation completed. The first launch is the `colleague_cross` baseline with separate regression and classification models, one GPU each, seed20261109, batch16, AdamW1e-4/weight-decay1e-3 and a finite10,000-update budget. It is an initial optimization stage, not a claim of converged or full-epoch training. The prepared `full_pair_control` and `slm_outer` ablations are not launched in this stage.

The private Forge/W&B project is `K-Fold-Affinity/initial training`. Scalar losses, gradient norms, learning rate, throughput and internal-validation metrics are logged online; code snapshots, source labels, cache payloads and model artifacts are not uploaded. Checkpoints stay in scratch every500updates. Run IDs and resolved config are frozen in the launch receipts; resume requires the same contract and existing online run.

Every1,000updates, a deterministic panel of up to1,024eligible labels per original cohort/endpoint monitors the internal chemical-group holdout. The final checkpoint receives full internal-validation evaluation. Neither external novelty evaluation nor the kinase auxiliary set selects checkpoints. The original chemical split, view exclusions, task sampler, network and objectives remain unchanged. Allocation warnings drain at a checkpoint; an interrupted run is not reported complete or automatically resubmitted. Full internal validation must finish for a successful completion receipt.

Launch root: `/scratch/t530a05/kfold/experiments/pli-initial-training-20261110`; initial jobs1249389(regression) and1249390(classification), each1GPU/72allocatedCPU/24h cap. The first real optimizer update and online run registration were verified. The launch adds fixed-panel/tracking helpers and skips resumed sampler positions before payload I/O. CPU affinity tests:193passed. The pinned launch source is independent of the earlier pilot directory. Earlier preparation/pilot descriptions below are historical evidence.

## Frozen data and supervision

Data preparation is frozen at commit f2775385. The logical release contains 2,817,891 cached inputs. The backbone is not optimized or rerun for head training. Raw features, source labels, immutable historical routes, original relations and experimental/synthetic provenance are preserved. No additional apo side-cache is produced: `s_lm` already contains mixed sequence and projected apo information. This experiment tests direct head access through a new pair construction, not the presence versus absence of apo information in the overall model.

Training labels are derived from the frozen 2,183,290-row legacy training manifest, 23,576 admitted BioLiP3 measurements and 1,337,265 synthetic negatives, with explicit dispositions for any source-ineligible rows. Label rows and unique cached inputs are different units. Synthetic negatives receive binary zero and no numeric affinity. Existing external evaluation rows remain external; the known-drug kinase panel is a distinct selectivity stratum. The 383 previously excluded validation inputs remain excluded.

For model selection, canonical ligand graphs are reduced to their largest organic parent fragment, uncharged, stripped of stereochemical and isotope annotations, and grouped by RDKit HetAtomTautomerv2 graph hash. This transformation defines partition membership only; molecular inputs and cached features remain unchanged. A deterministic chemical-group hash (seed 20261109) allocates 90%/10% to training/internal validation globally across targets, aliases and data sources. Input and chemical-group intersection checks must both be zero. This is a ligand-group split, not a protein-family or scaffold holdout. Library versions and chemical-group assignments are frozen in the manifest receipts.

External candidate ligands that collide with the measured training source pool are retained in an explicit overlap auxiliary stratum. The novel-ligand stratum must have no normalized chemical-group overlap with the training/internal-validation pool. Known-drug kinase evaluation can intentionally reuse known training drugs and is not represented as novel-ligand generalization. Publication/release metadata are preserved without inventing a measurement's first-availability date.

The frozen consensus crop was constructed from the whole legacy training evidence pool, before the new internal split. Consequently internal model-selection training is restricted to **query200** (query350 is available as an explicit alternative). A consensus-training experiment requires a new split-specific training-only evidence plan and cache coverage review. Existing consensus tensors and historical views remain preserved; they are not silently replaced or merged.

## Colleague baseline

The implementation follows the colleague's Pairformer head (`86c10b5c`) and the recovered continuous-supervision reference (`7cd98dd7`). It uses 384 single channels, 256 pair channels, six pair/single blocks, 16 attention heads and dropout 0.1. Inputs are `s_inputs`, `s_lm`, directed PL/LP and LL representations, and three distogram summaries: contact probability at 8 Å, expected distance, and normalized entropy. Padding masks propagate through the head. The baseline masks out PP pairs throughout the stack and pools directed PL and off-diagonal LL pairs plus projected token singles into a scalar readout.

The default scalar predicts pActivity in the regression task and an uncalibrated binary logit in the classification task. Separate task-specific model instances are trained; no multitask or calibrated inhibition-probability claim is made. Cached PP backbone features do not exist and are not reconstructed.

## Outer-product ablation

Three architectures are prepared and receive the same immutable feature and label batches:

1. `colleague_cross`: unchanged cross-only baseline.
2. `full_pair_control`: full pair topology, with a null-input adapter of the same shape and parameter count as the outer adapter.
3. `slm_outer`: full pair topology and a directed protein–protein outer residual derived from `s_lm`.

The adapter normalizes `s_lm`, projects it independently to left/right rank-16 channels, forms the actual 16-by-16 channel outer product for each protein-token pair, flattens those channels and projects them to 256 pair channels. The residual is zero outside PP pairs. Row chunking limits intermediate memory. Its output projection is initially zero. The two full-pair variants start with identical common parameters/output; the control supplies zeros rather than the observed single embeddings to the adapter. It can learn an input-independent offset and therefore is a topology/nominal-capacity control, not a perfect functional-capacity match.

A PP residual would be discarded by the original cross-only mask. The third condition is therefore interpreted relative to the full-pair control, with the original colleague baseline also reported. This avoids assigning the whole topology change to the outer-product information.

## Task objectives and sampling

Regression uses exact numeric measurements only. The colleague objective is 0.1 times absolute Huber plus 0.9 times same-assay prediction-difference Huber, with delta 0.5. Ranking comparisons are scoped to assay, target and endpoint. Repeated exact cached ligand-input identities within a target-scoped assay are averaged only within the ranking term; partition groups do not collapse distinct stereochemical measurements. All original measurement rows remain available for the absolute term. BindingDB residual/source-record singletons do not acquire invented cross-record ranking pairs. Censored observations remain bounds and are not turned into numeric point targets; an optional one-sided bound helper is separate from this exact-label baseline. Legacy normalized endpoint proxies retain their original provenance and are not promoted to measured Kd.

The single-GPU launch configuration uses three distinct IQR-weighted assay groups of five exact ligand inputs plus one absolute-only slot (batch size16). This preserves the15:1 label layout ratio of the colleague-style reference layout12×5+4 (batch64), which is retained as an unlaunched alternative. Zero-IQR assays do not supply ranking groups. All non-rankable assays, including small/zero-IQR groups, BindingDB residual singletons and BioLiP3 rows without shared assay identifiers, supply the absolute-only pool. A per-slot loss-group override prevents accidental ranking among these slots while preserving original assay provenance. Every eligible regression label belongs to either a ranking pool or this absolute-only pool. This sampler adapts the colleague layout to the expanded dataset; it is not a byte-for-byte reproduction of the colleague's historical sampler or corpus. The pilot instead uses batch size four to validate the complete six-block forward/backward path.

Classification uses BCE with logits. Exact measured pActivity at least 6 defines the active proxy class; weaker measurements define the negative proxy class. A concentration bound supplies a binary target only when it logically determines which side of that threshold it lies on. Synthetic binary zero remains explicitly assumed nonbinding and separate from experimental weak/inactive observations. Training batches contain 50% measured positives, 25% measured negatives and 25% synthetic negatives. This deliberate sampled class prior is not an estimate of population prevalence or calibrated binding probability. Endpoint/source-specific evaluation remains necessary. The kinase panel uses its original residual-activity-below-50%-at-1µM criterion, not a fabricated numeric pAffinity.

## Optimization, pilot and checkpoints

The prepared single-GPU entrypoint uses AdamW (learning rate 1e-4, weight decay 1e-3), gradient clipping at 10, BF16 autocast and checkpointed six-block execution. The full-training configuration is a template, not an executed production run or tuned protocol. The reference batch64 memory/throughput and multi-node DDP are not validated by a batch4 pilot. The batch16 launch configuration passed a separate real-sampler GPU readiness check (job1246559): one six-block BF16 optimizer update per condition, all six conditions passed, with43.87–44.53GB peak reserved memory. Full-corpus throughput, convergence, query350 batch memory and multi-node scaling remain unmeasured.

A pilot consists of eight optimizer updates per task/architecture, followed by held-out cached-input evaluation, source-stratum compatibility checks and a saved-checkpoint continuation comparison. Original-trunk/raw-shard reads are prohibited by the pilot launcher. Tests check finite gradients, nonzero `s_lm` projection gradients, learned outer-projection gradients, changed model parameters, and checkpoint/optimizer/RNG restoration. Pilot metrics are compatibility diagnostics, not affinity performance estimates, confidence intervals or evidence for the scientific ablation.

CPU tests: 191 affinity tests passed on the MacMini CPU test environment. The initial allocated preparation job1246324 completed 967,284 chemical-group assignments, then failed at a read-only SQLite ATTACH URI setting before GPU training. Its database and logs are preserved. Job1246382 reused the verified group table but was stopped before GPU training after a quadratic SQLite query plan was diagnosed. Job1246447 uses a source-ordered indexed join in a new run version; the original source files and both diagnostic runs remain preserved. The completed 3,695,259-row manifest is reused without recomputation. The first GPU attempt in1246447 failed before any optimizer step because the frozen-source Triton triangle multiplication backend is inference-only; it was replaced with the supported PyTorch autograd path, preserving the failed outputs. Job1246503 then completed0:0 in1m53s. All six task/architecture conditions passed eight optimizer updates, finite gradients, source-stratum readback, held-out loader evaluation and checkpoint continuation with maximum parameter difference0. The common initial head parameter SHA was identical across all six conditions. Batch4 peak reserved GPU memory ranged from11.34GB to16.87GB; these are pilot-specific measurements, not batch64 predictions. Scientific performance is not inferred from these small diagnostic evaluations.


The full-data sampler coverage check (job1246538, COMPLETED0:0) verified that all1,977,805regression training labels belong to either49,615ranking-assay pools or827,608absolute-only labels. Classification pools contain1,339,192measured positives,649,660measured negatives and1,203,007synthetic negatives, covering all3,191,859eligible labels. The CPU check does not claim that batch64 tensors fit a GPU.

## Audited task membership (query200)

| Task | Training label rows | Internal-validation label rows | External novel-ligand rows | Known-drug kinase auxiliary rows |
|---|---:|---:|---:|---:|
| Regression | 1,977,805 | 216,032 | 55,531 | 0 numeric affinity targets |
| Classification | 3,191,859 | 351,414 | 118,618 | 26,832 |

Classification counts include the separately marked synthetic-negative cohort. The internal split is approximately90/10 by normalized chemical group; label-row fractions need not equal exactly10%. Five external overlap rows remain an explicit auxiliary stratum. All151,128 retained external source rows stay in the manifest, including bounds that cannot supply a particular task label. No source label is removed by a task mask.

Frozen crop metadata establish209physical inputs without query200; they are excluded only for that view, with an explicit per-task position map. Query350 is available for all2,817,891cached inputs. This avoids silent crop substitution and preserves every measurement and query350 representation. The old complete view audit is reused; source payloads are not redecoded or rehashed for this filter. Train/internal-validation input and normalized-chemical overlap, and external-novel/training-pool overlap, are all zero.

## Reproducibility paths

- Implementation branch: `codex/pli-affinity-predictor`.
- Data release: `configs/affinity/data_release_v1.json`; data-preparation commit `f2775385`.
- Prepared training config: `configs/affinity/head_training.json`.
- Full-run entrypoint: `scripts/train_cached_affinity.py` (explicit full-training launch flag required; not run).
- Remote batch16 training entrypoint: `/scratch/t530a05/kfold/experiments/pli-head-training-20261110-retry3/training-ready-v1/train.py`; full training requires `--allow-full-training`.
- Remote pilot: `/scratch/t530a05/kfold/experiments/pli-head-training-20261110-retry3`.
- Frozen child catalog: `/scratch/t530a05/kfold/experiments/biolip3-feature-production-20261109/training-cache-retry1/catalog.sqlite`.
- Original source metadata receipts and chemistry-reuse proofs: MacMini `ops/preparations/pli-head-training-20261109` and `pli-head-training-20261110-retry3`.
- No source features, caches, model weights, checkpoint binaries or credentials are committed to Git. Raw archival copying remains a separate preservation workflow, and raw deletion is disabled.
