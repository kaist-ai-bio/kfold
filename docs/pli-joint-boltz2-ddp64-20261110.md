# Joint affinity baseline: Boltz-2 supervision, 64 ranks

The user authorized joint regression/classification with Boltz-2 loss functions and weights, 64 GPUs and gradient accumulation exactly one on 2026-11-10. This supersedes the two independent DDP8 task runs as the primary baseline. Existing checkpoints, W&B histories and immutable runtime sources are retained. New training starts from a common fresh initialization rather than treating a single-output checkpoint as a joint-model continuation.

## Architecture and objective

A single six-block colleague Pairformer encoder (single384, pair256, heads16, dropout0.1, cross-only protein-ligand/ligand-ligand pairs, block checkpointing) supplies a pooled representation to two independent scalar MLP readouts. Both readouts backpropagate into the shared encoder. This preserves the colleague encoder rather than claiming to reproduce the complete Boltz-2 architecture.

Boltz-2 Appendix C.2.5 specifies `L = 0.9 L_difference + 0.1 L_absolute + L_binary`. Both continuous terms use Huber delta0.5. The binary term is focal loss with gamma1, positive alpha0.8 and negative weight0.2, matching the first ensemble member in Appendix B.5.2. Alpha0.6 is the other published member, not an additional simultaneous task model. We train one joint model in this run.

Stored targets remain pActivity and original concentration qualifiers. A concentration `<` becomes a pActivity lower bound, and `>` becomes an upper bound. The one-sided Huber penalty is zero on the permitted side. Pairwise bounds are formed only within the same assay, endpoint and exact target scope; unidentifiable differences between like-sided bounds are masked. Distinct exact cached inputs are required for ranking comparisons; an input is not compared with its replicate. Synthetic assumed negatives retain binary0 and null numeric affinity. Classification uses only explicitly valid 0/1 labels. No threshold, chemistry, raw relation, provenance, or source label is overwritten.

Absolute loss is averaged over eligible numeric rows, pairwise Huber over comparable pairs within each sampled assay group and then over groups, focal loss over valid binary labels. Distributed normalization uses global supervision counts, not the mean of rank-local masked means. The local sum is multiplied by world_size/global_count so DDP gradient averaging produces the global masked objective. Empty task masks retain zero-gradient graph links to both outputs. The invariant loss weights are not altered by the number of labels available on a rank.

## Sampling, batches and retained boundaries

64 ranks on16 GH200 nodes, four ranks per node. Physical batch16/rank gives global1024; each update has one forward, one backward and one optimizer step. No gradient accumulation or no_sync microbatch loop is used. Each local batch contains two intact five-ligand assay groups selected by the existing IQR sampler, two absolute-only slots and four dedicated binary slots drawn from the approved measured-positive/measured-negative/synthetic-negative pools. Available binary labels on continuous rows also supervise the binary output. Numeric bound rows from the same training manifest join the absolute-only pool without amplifying their sampling fraction.

This sampling is an adaptation to our available corpus, not a reproduction of Boltz-2 Table8's PubChem/CeMM/MIDAS source mixture or a claim that synthetic decoys share an experimental assay. The frozen manifest, chemical holdout, explicit view exclusions and query200/256-token contract remain unchanged. External validation is excluded from fitting and checkpoint selection. Frozen consensus is not used in model-selection training until split-specific evidence is planned. The s_lm outer-product ablation remains available for subsequent matched comparisons; it is not enabled in this baseline.

AdamW lr1e-4, weight decay1e-3, gradient clip10, BF16, constant LR, model/sampler seed20260822 and10,000 updates are retained. Data split/panel seed20261109 is unchanged. Global batch increases from512 in the earlier DDP8 accumulated runs to1024 here, so equal update count does not imply equal sample exposure. Memory pilot uses actual cached inputs padded to the maximum256 tokens to test physical B16 without accumulating.

## Validation, reproducibility and operations

Train total/absolute/difference/focal losses, gradient norm, LR and timings are logged separately. Each output has its own real step0 internal holdout evaluation and panel every1,000updates. Regression reports MAE/RMSE/Pearson and assay metrics; classification reports BCE/focal/AUROC/AUPRC, with measured and synthetic cohorts distinct. Full internal validation is mandatory before scientific completion. No cache, source labels, code or model artifact is uploaded to W&B; scalar-only logging uses the private `initial training` project.

64GPU pilot verifies16hosts/64distinctGPU, NCCL all-reduce2016, physical batch16 at256tokens, finite gradients through the shared encoder and both output heads, all-rank parameter agreement and checkpoint2→3 model/optimizer/RNG/sampler continuation. The pilot is not evidence of convergence or performance improvement. Production requires the pilot and its Slurm terminal result to pass. A resumable checkpoint contains all64rank RNG states and the frozen training contract; future online continuation must rejoin the existing W&B run explicitly.

Runtime root: `/scratch/t530a05/kfold/experiments/pli-joint-boltz2-ddp64-20261110`.
Record: `ops/runs/pli-joint-boltz2-ddp64-20261110.json`.
Source: `src/kfold/affinity/joint_training.py`, `scripts/train_cached_affinity_joint.py`.
Local mathematical/gradient tests:213 passing CPU affinity tests, excluding the prior loopback-only Gloo integration test from this invocation. Actual64rank validation is a separate gate.
Source/runtime pins and single-attempt transition journals preserve uncertain-submission evidence. PLI active+pending<=2, totalGPU<=128 and account jobs<=5. Other campaigns, raw feature archives, cached payloads and original labels are preserved. No scheduler automation is added.

Primary reference: [Boltz-2 original manuscript, Appendix B.5 and C.2](https://jeremywohlwend.com/assets/boltz2.pdf).
