# Candidate-conditioned apo relation adapters

These opt-in training recipes compare a shallow relation encoder with two extra
relation updates. They start independently from the original stage-3 **71K full
checkpoint**. The old 5K apo-mean-mixer experiment is a separate baseline and is
not a continuation checkpoint for either recipe. This document describes code
and launch settings; it is not evidence that distributed validation or a full
training run has finished.

## Paths and learned parameters

The original multi-apo trunk and the separately sampled single-apo-combination
trunks remain frozen. Each branch is a **complete complex assignment** of one apo
per independent apo group; it is not necessarily one isolated molecule. At most
three branches are sampled during training. Multimer snapshot groups retain
their coupled apo indices. All molecules, chemical inputs, token layouts and
the original ECSI prior path are preserved.

The four independently parameterized groups are:

| Group under `model.apo_repr_mix` | Inputs and purpose | Learning signal |
| --- | --- | --- |
| `structure_z` | Post-trunk branch pairs; local learned apo weights and residual to the original multi-apo trunk output | Existing structure loss |
| `confidence_relation.confidence_z` | Pre-trunk apo single/pair evidence, partner context, detached trunk query and the particular candidate geometry; apo-axis pooling produces a pair residual | Existing confidence loss |
| `confidence_relation.confidence_s` | Independent relation encoder and token pooling produce a confidence single residual | Existing confidence loss |
| `apo_selector` | Separate shallow pre-candidate apo/partner projection and branch-quality predictor | Generated-quality regression |

The explicit apo inputs are `StructureEncoder` and `ApoModule` taps before their
apo-axis means and before the folding trunk. Capturing them during a branch
`run_trunk()` call does not turn their source into post-trunk features. They
retain the apo axis until candidate-conditioned relation encoding and pooling.
Candidate coordinates and standard trunk queries are stopped-gradient inputs.
The standard trunk-to-confidence path is retained.

The confidence pair/single encoders share no learned weights. The selector has
its **own** `PreApoProjection`; it does not reuse or update a confidence encoder
or generation mixer. Frozen raw features and `s_inputs` provide its evidence.
The selector reads both structural singles and observed apo-pair summaries; the candidate does not enter it. All original encoders, trunk,
diffusion and confidence parameters remain fixed, while derivatives through
the frozen heads still train the new input adapters.

The structure and confidence residual outputs are zero-initialized. Thus those
paths initially reproduce the explicitly initialized 71K model, despite the new
utility predictor having fresh parameters. For the production launch, the reference uses **all 71K checkpoint EMA
parameters**, including trunk, distogram, diffusion and confidence heads. This
matches the published final K-Fold tensors. The earlier eight-GPU validation used
trunk EMA with raw heads; it validates architecture, gradients and checkpointing,
while production now explicitly initializes all four parameter groups from EMA.
The initializer asserts that every checkpoint EMA tensor is covered. External
pretrained encoders retain their pinned assets. New adapters and optimizer state
are initialized afresh; no trained apo mixer or later confidence checkpoint is
loaded.

## Controlled shallow/deep comparison

`configs/train/apo_joint_relation_shallow.yaml` uses `relation_depth: 1`: one
candidate-conditioned apo relation encoding followed by local pooling.
`apo_joint_relation_deep.yaml` uses depth 3: the same encoder with two additional
pair-to-token-to-pair relation updates before pooling. The updates use partner
and candidate-contact context. Pooling is across apos for each token/pair, not a
single choice of one apo for the whole complex. Neither architecture runs a
separate full confidence head for every apo.

All parameters common to the two new architectures have identical initialization; only the deep-only update blocks are additional. A zero-valued null reference lets local attention suppress every apo when none is useful.

Both variants retain the existing structure/confidence objectives and targets,
including the inherited stage-3 **bond weight 1**, weighted coordinate MSE and
smooth lDDT. No representation matching, apo identity target, posterior KL or
physical pairing label is introduced. Existing confidence output meanings,
including the quantities used to derive pTM/ipTM, are retained. New utility
scores are returned separately; they do not modify the default ranking score.

## Utility target without extra teacher rollouts

On 25% of confidence mini-rollout batches, one valid apo branch is sampled at
random and the frozen original 71K generator produces the candidate from that
branch. This **replaces**, rather than adds to, the normal mini-rollout. The
same candidate is used for confidence learning. Remaining mini-rollouts use the
mixed generation representation. The structure training loss always uses the
mixed path. No learned utility score routes training inputs.

Only the observed branch receives a utility target. Its target is the generated
candidate's native-contact representative-atom lDDT: interchain contacts for
multichain systems, intrachain contacts for single-chain systems. The cutoff is
15 Angstrom, or 30 Angstrom for pairs involving nucleic acids. Only resolved,
valid pairs contribute. An example without evaluable contacts has no utility
target; it is not assigned zero quality. Existing confidence supervision
eligibility is respected. The selector's bounded quality prediction is trained
by MSE, with utility loss weight 0.01. Bounded quality is not a calibrated success
probability. Unobserved batches keep a zero-valued gradient path for DDP.

This learns expected generated quality under the fixed base generator and its
sampling procedure. It does **not** establish a physical apo--holo pair, infer
state populations or identify each molecule's causal contribution to a complex
branch. It also does not directly measure an apo's marginal benefit inside the
new mixer. A single generated observation is noisy; the target definition is
shared by both variants.

The inference result exposes `apo_utility.scores` in branch order, corresponding
to `last_apo_mix_assignments`. Default generation uses all capped branches.
Optional inference-only preselection is available through:

```python
model.score_apo_combinations(f_input, max_combinations=32)
model.inference(f_input, apo_selection_top_k=3, apo_selection_pool_size=32)
```

Pool scoring runs the input embedding and pre-trunk apo producers only: no
sequence LM, complete folding trunk, diffusion or confidence head. Top-K retains
whole complex assignments and runs only those additional branch trunks. The
original base trunk and ECSI prior remain unchanged. An entirely unobserved pool
falls back to normal capped mixing. Sampling preserves the caller's RNG stream;
preselection is never used during training. Runtime savings are not yet measured,
and pre-trunk scoring itself has a cost. Neither utility nor attention weights
should be presented as a physical posterior.

## Training settings

Both variants use seed 3, crop 768, at most three apos/three complex branches,
one recycle, diffusion training batch 4, and a 30-step/one-sample confidence
mini-rollout. Compilation is on (`mode: default`, `dynamic: false`), using the
existing backbone/head compilation and compilation of the four new adapters.
The structure pair mixer compiles its reusable `_chunk` calculation, retaining
the outer Python loop and per-chunk activation checkpointing. Compiling the whole
768-by-768 pair loop would unroll 576 chunks into an unnecessarily large graph.
Confidence/utility adapters compile in place. Both approaches preserve their
audited parameter/checkpoint names. Existing activation checkpoint settings are
retained. Adam/LR scheduling
is inherited with peak LR 1e-4 and 100-update
warmup. The production proposal is **5,000 updates** on 32 nodes x 4 GPUs,
batch 1/GPU and no accumulation: global batch 128 per variant. Two variants
require 64 nodes/256 GPUs total. This is approximately 640,000 sampled complex
examples per variant, with replacement, not that many unique complexes.

The training wrapper resets the seeded random stream once after model and data
module construction, before `fit()`. This prevents the extra initialization
draws of the deep model from shifting the starting data/augmentation/rollout
stream. The data sampling seed remains 3. There is no epoch-level reset. This
does not promise identical random streams after architecture-dependent random
operations. On full resume this reset also occurs before Lightning restores
checkpoint training state: optimizer/scheduler/EMA restoration is unchanged,
but exact random-stream continuity with an uninterrupted allocation is not
claimed. Subsequent explicit checkpoint RNG restoration, if provided by the
runtime, takes precedence over this pre-fit seed.

`scripts/train_apo_joint_relation.py` checks that the optimizer contains exactly
the new trainable allowlist, audits every frozen parameter's version and absent
gradient, and records finite gradients for each new group. Utility gradients
may be zero on individual ranks/batches; the audit accumulates evidence and
reports its distributed union. A completed validation requires nonzero gradients
and changed parameters in all four groups across the run, not every rank.
`JOINT_REQUIRE_UTILITY_GRAD=0` can preserve a very short diagnostic run that did
not exercise utility, but its `all_groups_exercised: false` is **not** a complete
adapter validation. Groupwise frozen versions, finite loss and exact optimizer
membership remain mandatory.
`JOINT_UTILITY_FRACTION` optionally overrides the common rollout fraction
(default 0.25). A short diagnostic with fraction 0 must also explicitly set
`JOINT_REQUIRE_UTILITY_GRAD=0`; it cannot validate utility training. Production
comparisons must keep the fraction identical across variants.

Per-rank JSON progress, separate loss components in CSV, resolved configuration,
source commit, initialization checkpoint and cache paths are recorded. Full
periodic checkpoints are saved every 500 updates. At 5,000 updates or the
23-hour timer, an atomic `checkpoints/progress.ckpt` includes full training state.
The script never resubmits itself. A timer pause is not completion.

## Launch preparation and bounded smoke checks

Use the reviewed clean source checkout on Hangang and set `KFOLD_EXPECTED_COMMIT`
to its full commit. The launcher reuses the existing training environment,
`hangang_rank.py`, all-rank IPC/NCCL preflight and CXI/libfabric settings. Cache
directories are unique per job/rank/step under `/appsdata`; short worker socket
paths are under `/scratch`. Exact paths are recorded; no broad cleanup occurs.
The launcher keeps the previously validated AWS Libfabric/CXI transport and
bounded OMP/BLAS/compiler thread counts. Its all-rank preflight additionally times
five 4-MiB all-reduces after warmup; this is a small payload probe, not proof of
optimal network saturation or GPU Direct RDMA. Runtime receipts record selected
transport variables and compile registrations. CUDA launch blocking is unset
unless `JOINT_CUDA_LAUNCH_BLOCKING=1` explicitly requests a diagnostic run.
`TRITON_STORE_BINARY_ONLY=1` retains executable kernels and metadata while
omitting intermediate compiler IR cache files, reducing per-rank inode use.
Lightning's automatic Slurm requeue signal handler is explicitly disabled;
continuation after a timer pause remains a separate full-state resume.

`JOINT_COMPILE_ENABLED=false` provides an otherwise identical eager control.
`JOINT_PROFILE_STEP=16` profiles one rank-zero optimizer update and writes a
compact `kernel-profile.json` with actual attention/cuEquivariance/compiled-region
operator counts; no large trace is exported. This is for bounded validation, not
continuous production profiling. SDPA remains automatic: learned additive pair
bias and its derivatives must remain intact. No Flash/cuDNN kernel, low-precision
diffusion or static DDP graph is forced merely to report a faster backend.

The following are example commands, **not a record of submitted jobs**. Submit
from the remote scratch checkout after reviewing the settings and live quota.

```bash
export KFOLD_EXPECTED_COMMIT=<reviewed-full-commit>
# Four-GPU, eight-update smoke; repeat with deep, then with --nodes=2 for 8 GPUs.
sbatch --nodes=1 --time=02:00:00 \
  --export=ALL,JOINT_VARIANT=shallow,JOINT_VALIDATION_STEPS=8,JOINT_TRAIN_MAX_TIME=00:01:50:00 \
  scripts/slurm/hangang_apo_joint_relation.sbatch

# Independent production variants, 128 GPUs each. Do not submit until validated.
sbatch --export=ALL,JOINT_VARIANT=shallow scripts/slurm/hangang_apo_joint_relation.sbatch
sbatch --export=ALL,JOINT_VARIANT=deep scripts/slurm/hangang_apo_joint_relation.sbatch
```

Fresh runs initialize from
`/scratch/t530a05/kfold/checkpoints/training-candidates/0908-72k/s3-71k.ckpt`,
resetting optimizer, LR/global step and adapter EMA state. Distinct variants,
smokes and Slurm jobs have distinct default experiment roots. To continue after
inspecting a timer pause, set both `JOINT_EXPERIMENT_ROOT` to the original root
and `JOINT_FULL_RESUME_CKPT` to its full progress or periodic checkpoint. This
restores optimizer/scheduler/EMA/global step and uses a fresh allocation timer.
It does not reinitialize from 71K or load the old 5K mixer. The target remains
5,000 cumulative new updates, and a checkpoint at/above the target is rejected.

Local syntax checks, unit tests and a successful preflight do not establish
distributed training success. Verify terminal Slurm state, all-rank progress,
all four gradient/update groups, frozen audit and readable full checkpoint
before advancing from smoke tests to a long run.
