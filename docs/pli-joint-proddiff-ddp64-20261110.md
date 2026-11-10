# Joint affinity proddiff ablation, 2026-11-10

The user requested replacing the proposed single outer-product route with product–difference and training that route on64 GPUs, in parallel with the existing joint baseline. Baseline job1250353 and its immutable source remain untouched. Both runs retain physical batch16/rank, global1024 and gradient accumulation1. Other campaigns are unchanged; PLI running+pending<=2 and totalGPU<=128, account jobs<=5.

## Pair construction

The cached mixed sequence/apo single representation `s_lm` has384channels. Per-token LayerNorm produces `x`. For each ordered protein-token pair:

`p_ij = x_i * x_j` (Hadamard/element-wise product,384channels)

`d_ij = x_i - x_j` (signed difference,384channels)

`delta_z_ij = W concat(p_ij,d_ij)` (768->256, bias-free projection).

The residual is masked to protein–protein positions and added to the existing head pair input. Invalid/padded/protein-ligand/ligand-ligand positions receive no adapter residual. The original ligand and cross representations remain the cached values plus the original single/distogram projections. The head then runs the same six-block Pairformer with PP interactions enabled and supplies a shared pooled representation to independent regression and binary output MLPs.

The adapter never constructs384x384 channel interactions. Its projection has256x768=196,608weights plus768LayerNorm parameters. Token axes still form an N-by-N pair grid. Projection chunks contain32rows of token pairs, and non-reentrant activation checkpointing recomputes concatenated features during backward instead of retaining every N*N*768activation. This is activation recomputation, not gradient accumulation. Projection weights start at zero; common encoder/readout initialization order is preserved. The original head/attention gates also start at zero, so indirect PP-to-output derivatives are checked after optimizer warm-up rather than incorrectly requiring nonzero adapter gradients before the gates learn.

The earlier unlaunched outer adapter already projected384->16 before its16x16outer product. Consequently the new adapter is not claimed to be a measured memory/speed reduction relative to that low-rank implementation. Historical `slm_outer` remains available for provenance/replay; the new run explicitly selects `slm_proddiff` and contains no trainable outer adapter.

## Controlled comparison and limitations

Baseline uses cross-only protein-ligand/ligand-ligand pair updates. Proddiff enables PP propagation as required for the added PP features. Therefore a difference from baseline can reflect topology and added parameter/input effects, not solely the product–difference operation. `full_pair_proddiff_control` is prepared with identical proddiff parameter count and full-pair topology but zeroed adapter input; no third GPU run is launched. `s_lm` mixes sequence and apo information, so this remains a direct mixed-single input ablation rather than pure apo causal isolation.

The frozen labels, chemical-group split, query200/256-token contract, availability exclusions, binary1:4 provenance pools, IQR assay groups and sampling indices are unchanged. The completed370.7MBsampler state is read-only reused from the baseline root, not recopied or regenerated. Numeric bounds remain one-sided; synthetic negatives have no numeric affinity. Loss remains Boltz-2 `0.9 difference Huber +0.1 absolute Huber + focal`, delta0.5/gamma1/alpha0.8. DDP uses global task-mask denominators. AdamW1e-4/weight-decay1e-3, gradient clip10, BF16, seed20260822 and10,000updates remain unchanged. Internal validation seed20261109 and external-holdout exclusion are preserved.

## Verification and execution

221CPU affinity tests pass (prior loopback-only Gloo integration test excluded from this invocation). Tests compare chunked product/difference values and gradients against direct dense concatenation, verify protein/padding masks and signed direction, check zero-residual initialization and actual learning, and confirm both output derivatives reach the adapter after shared-gate warm-up. Tracking handles an explicit variant in the resolved config without duplicate-key errors.

The64GPU pilot runs4updates at maximum256tokens, resumes checkpoint2 through4 in a second process, verifies identical64-rank model parameters, finite/nonzero gradients in shared encoder/both heads/proddiff norm/projection, and nonzero learned projection weights of shape(256,768). Every rank's first3sample hashes must match the already completed baseline pilot; the final2hashes must match the resumed run. Pilot is a runtime/memory/continuation check, not a benchmark of scientific performance. Only a passed receipt plus Slurm COMPLETED0:0 permits production.

Runtime: `/scratch/t530a05/kfold/experiments/pli-joint-proddiff-ddp64-20261110`.
Record: `ops/runs/pli-joint-proddiff-ddp64-20261110.json`.
Source: `src/kfold/affinity/heads.py`, `scripts/train_cached_affinity_joint.py`.
W&B: same private `K-Fold-Affinity/initial training` project, separate variant/group/run. Both tasks' train components, real initial holdout panels, subsequent1,000update panels and final full internal validation are logged. No raw/cache/source/label/model artifact uploads and no new automation are introduced. New code/receipts and test temporary files stay on MacMini external SSD; low internal disk space is not used for task outputs.
