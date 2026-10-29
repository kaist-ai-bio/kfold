# Apo representation mixing

Opt-in implementation based on main `20d7b0d7`. The unchanged default has no new
parameters. `configs/train/apo_repr_mix.yaml` enables three independent residual
set mixers: structure pair, confidence pair, and confidence single. Existing
trunk, encoders, diffusion and confidence weights are frozen. The unchanged
`s_inputs` is passed directly to both heads.

## Identity initialization and objective

Each mixer adds a zero-initialized residual to the **existing multi-apo trunk
output**, not to the mean of single-apo trunk outputs. Hidden layers are nonzero;
the final linear weight and bias are zero. Separate apo branches run inside an
isolated RNG context, with the same dropout stream per branch. They do not consume
the baseline diffusion RNG stream. Thus the initial forward path reproduces the
legacy model for the same input, mode, seed and precision.

The original multi-apo trunk costs one pass, followed by at most K single-complex
branch passes. This is K+1 trunk evaluations, not K. Both heads reuse the same
frozen features, but no mixer parameters are shared. Masks exclude padded branches.
Per-branch nonlinear transforms precede masked mean pooling; network dimensions
are independent of K. Chunked, activation-checkpointed residual MLPs limit memory.

The structure mixer learns through the frozen diffusion operations with existing
coordinate/smooth-lDDT/bond losses inherited from stage 3. Confidence mixers learn
through the frozen confidence head with the existing confidence objective and
supervision eligibility. Generated coordinates are detached; confidence loss
cannot reach the structure mixer. No representation-MSE objective is added.
Frozen head parameters stay fixed, but heads use their autograd-safe training
paths while training the mixers. Inference uses normal eval paths.

## Apo combinations

Training loads up to three apos per group and at most three complex branches.
Each `apo_uid` group is sampled independently; chains from the same multimer
snapshot group retain their shared source index. Independent random permutations
provide balanced first combinations, followed by unique sampled combinations when
more are requested. The full Cartesian product is never materialized. Geometry,
atom apo coordinates and structure tokens are sliced consistently. Chemical
reference features, chain/token layouts, crop and ECSI priors remain unchanged.

`max_apo_combinations` in `KFold.inference`, `KFold.predict`, and runner methods
caps the number of complex branches. The CLI exposes `--max-apo-combinations`;
`--model-path` selects a local exported mixer model. The model configuration
`apo_repr_mix.max_combinations` is the default when no override is supplied.
A cap is not a promise that this many unique combinations exist. Counts above
three are supported mechanically but require separate generalization evaluation.
Assignments are available as `model.last_apo_mix_assignments` and in prediction
settings as `apo_mix_assignments` for provenance.

## Initialize and resume

Use the existing full checkpoint with `scripts/train.py --config
configs/train/apo_repr_mix.yaml --resume_from_checkpoint PATH`. A new optimizer
is created; only the complete absence of new mixer keys is allowed during legacy
initialization. Other model keys are checked strictly. Partial mixer checkpoints
are rejected. Trunk EMA initialization follows the reused stage-3 policy; heads
use raw checkpoint values. EMA is rebuilt from the actual initialized parameters.
The identity reference is this explicitly initialized base, not an unrelated
inference export.

To resume a mixer checkpoint including optimizer, scheduler and EMA, override
`train.load_opt_state=true`. Do not use the legacy initialization path for an
optimizer continuation. Learning step starts at zero for a new adapter experiment.

To export for the existing inference loader, write the resolved `config.model`
to `config.yaml` and the checkpoint EMA `shadow_params` (or raw `state_dict`
with `model.` prefixes removed) to `weights/kfold.pth`. Keep the source checkpoint
and record whether raw or EMA was exported. The usual pretrained encoder assets
are still required. Load with `KFold.from_pretrained(export_directory)`.

## Validation

`tests/test_apo_repr_mix.py` covers zero-output identity, gradient separation,
variable branch count, permutation/padding invariance, state-dict round trip,
balanced unique combination sampling, and correlated multimer slicing with
unchanged priors.

`scripts/slurm/hangang_apo_repr_mix.sbatch` reuses the confidence task's cuEq runtime,
Slingshot/NCCL rendezvous helper and archived-LMDB compatibility fixes. It requests
two nodes/eight GPUs for a bounded ten-update real-data check. It uses crop 768,
30-step confidence rollout, diffusion batch 4, and one recycle; compile is off.
These are validation settings, not a production throughput benchmark. No other
job is modified. `KFOLD_EXPECTED_COMMIT` must match the clean checkout.

The validation callback checks exact zero-residual head-input identity and
legacy/mixed output differences relative to repeated legacy GPU execution, finite
loss/gradients, the optimizer parameter allowlist, all three mixers changing,
frozen parameters receiving no gradients or in-place changes, and checkpoint
contents. Per-rank evidence is written under the job's `validation/` directory.
Successful unit tests or submission alone are not distributed-training success.

### Verified run (2026-10-29)

Hangang job `1176574`, source `c91d1715`, completed with exit code `0:0` on
`node0689` and `node0693`. All eight ranks completed ten optimizer updates on
real training examples. All three mixers received nonzero gradients and changed;
all existing parameters remained frozen and unchanged. The optimizer contains
only the 2,628,992 mixer parameters. Peak allocated memory was 42.11 GiB per GPU.
The five focused unit tests also passed.

Zero-initialized mixer outputs matched baseline head inputs bitwise. Full GPU
forward outputs were checked against separately measured repeated-baseline
numerical jitter, rather than claimed to be bitwise reproducible. The frozen
diffusion bias-only SDPA backward path required a local query gradient leaf to
allocate the efficient-attention backward workspace; no pretrained parameter is
unfrozen. Its GPU gradient regression against math SDPA passed.

The saved checkpoint was read back and mixer tensors matched exactly; optimizer
and EMA states were present. It remains on Hangang at
`/scratch/t530a05/kfold/experiments/apo-repr-mix-8gpu-1176574/validation/validated.ckpt`.
Small per-rank reports are retained in the project operations checkout under
`outputs/remote/Hangang/apo-repr-mix-20261029/`; the run receipt is
`ops/runs/apo-repr-mix-20261029.json`.

This verifies distributed execution with the settings above, not convergence,
accuracy improvement, compiled execution, or long-run training stability.
