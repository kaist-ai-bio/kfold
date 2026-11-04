#!/bin/bash
#SBATCH --job-name=apo-joint-relation
#SBATCH --partition=normal
#SBATCH --account=bs0247
#SBATCH --comment="field=sciai;appl=pytorch-ddp"
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=apo-joint-relation-%j.out
#SBATCH --error=apo-joint-relation-%j.err

# This script never submits another job. Resume explicitly after inspecting state.
set -euo pipefail
ulimit -c 0

CODE=${JOINT_CODE:-/scratch/t530a05/kfold/worktrees/apo-joint-relation}
export JOINT_VARIANT=${JOINT_VARIANT:-shallow}
case "$JOINT_VARIANT" in
    shallow|deep) ;;
    *) echo 'JOINT_VARIANT must be shallow or deep' >&2; exit 2 ;;
esac
CONFIG="configs/train/apo_joint_relation_${JOINT_VARIANT}.yaml"
ACTIVATE=${JOINT_ACTIVATE:-/scratch/t530a05/kfold/ops/activate-train-cueq010.sh}
source "$ACTIVATE"
module load libfabric/2.3.1 aws-ofi-nccl/1.20.0

export NCCL_NET="AWS Libfabric" FI_PROVIDER=cxi NCCL_IB_DISABLE=1
export NCCL_PROTO='^LL128'
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=4 MAX_JOBS=4
# CUDA launch blocking is a debugging aid, not a production optimization.
if [[ ${JOINT_CUDA_LAUNCH_BLOCKING:-0} == 1 ]]; then
    export CUDA_LAUNCH_BLOCKING=1
else
    unset CUDA_LAUNCH_BLOCKING
fi
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=disabled

cd "$CODE"
NNODES=${SLURM_JOB_NUM_NODES:?}
NRANKS=${SLURM_NTASKS:?}
test "$NRANKS" -eq "$((NNODES * 4))"
test "$(git rev-parse HEAD)" = "${KFOLD_EXPECTED_COMMIT:?Set the reviewed source commit}"
git diff --exit-code HEAD --

MAX_STEPS=5000
if [[ -n ${JOINT_VALIDATION_STEPS:-} ]]; then
    [[ "$JOINT_VALIDATION_STEPS" =~ ^[0-9]+$ ]]
    test "$JOINT_VALIDATION_STEPS" -gt 0
    test "$JOINT_VALIDATION_STEPS" -lt 5000
    MAX_STEPS=$JOINT_VALIDATION_STEPS
    RUN_KIND="smoke${MAX_STEPS}"
else
    RUN_KIND=5k
fi
export JOINT_REQUIRE_UTILITY_GRAD=${JOINT_REQUIRE_UTILITY_GRAD:-1}

INIT_CKPT=${JOINT_INIT_CKPT:-/scratch/t530a05/kfold/checkpoints/training-candidates/0908-72k/s3-71k.ckpt}
RESUME_OVERRIDES=(train.load_opt_state=false train.load_global_step=false)
if [[ -n ${JOINT_FULL_RESUME_CKPT:-} ]]; then
    : "${JOINT_EXPERIMENT_ROOT:?Full resume requires the original experiment root}"
    INIT_CKPT=$JOINT_FULL_RESUME_CKPT
    RESUME_OVERRIDES=(train.load_opt_state=true)
fi
test -r "$INIT_CKPT"
export JOINT_EXPERIMENT_ROOT=${JOINT_EXPERIMENT_ROOT:-/scratch/t530a05/kfold/experiments/apo-joint-${JOINT_VARIANT}-${RUN_KIND}-g${NRANKS}-b${NRANKS}-seed3-${SLURM_JOB_ID}}
if [[ -z ${JOINT_FULL_RESUME_CKPT:-} && -e "$JOINT_EXPERIMENT_ROOT/progress/rank-0.json" ]]; then
    echo 'Fresh initialization cannot overwrite an existing experiment; use a new root or explicit full resume' >&2
    exit 2
fi
export RUN_ROOT="$JOINT_EXPERIMENT_ROOT/segments/$SLURM_JOB_ID"
export JOINT_CACHE_ROOT=${JOINT_CACHE_ROOT:-/appsdata/t530a05/kfold/cache/apo-joint-${JOINT_VARIANT}-${SLURM_JOB_ID}}
mkdir -p "$RUN_ROOT" "$JOINT_EXPERIMENT_ROOT/checkpoints" "$JOINT_CACHE_ROOT"
git rev-parse HEAD > "$RUN_ROOT/source-commit.txt"
cp "$CONFIG" "$RUN_ROOT/launch-config.yaml"
printf '%s\n' "$INIT_CKPT" > "$RUN_ROOT/initialization-checkpoint.txt"
printf '%s\n' "$JOINT_CACHE_ROOT" > "$RUN_ROOT/cache-root.txt"
printf '%s\n' "/scratch/t530a05/kfold/t/$SLURM_JOB_ID" > "$RUN_ROOT/temporary-root.txt"
export PYTHONPATH="$CODE/src${PYTHONPATH:+:$PYTHONPATH}"
export TRAIN_PYTHON=${JOINT_TRAIN_PYTHON:-/scratch/t530a05/kfold/envs/train-cueq010/bin/python}
test -x "$TRAIN_PYTHON"

run_step() {
    local log_prefix=$1
    shift
    srun --mpi=none --network=disable_rdzv_get --kill-on-bad-exit=1 \
        --distribution=block:block --cpu-bind=none --gpu-bind=none \
        --output="$RUN_ROOT/${log_prefix}rank-%t.out" \
        --error="$RUN_ROOT/${log_prefix}rank-%t.err" \
        bash -c '
            set -euo pipefail
            # Caches are isolated by job, rank and srun step; none use $HOME.
            step=${SLURM_STEP_ID:-${SLURM_STEPID:?}}
            rank_root="$JOINT_CACHE_ROOT/$SLURM_PROCID/$step"
            # Keep multiprocessing Unix-socket paths below Linux sockaddr_un limits.
            export TMPDIR="/scratch/t530a05/kfold/t/$SLURM_JOB_ID/$SLURM_PROCID/$step"
            export TRITON_CACHE_DIR="$rank_root/triton"
            export TORCHINDUCTOR_CACHE_DIR="$rank_root/inductor"
            export CUDA_CACHE_PATH="$rank_root/cuda"
            mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH"
            ulimit -l unlimited
            exec "$TRAIN_PYTHON" -B scripts/slurm/hangang_rank.py "$@"
        ' kfold-joint "$@"
}

# Bounded IPC + NCCL all-reduce check on all allocated GPUs before model loading.
run_step preflight- --preflight
run_step '' -- scripts/train_apo_joint_relation.py \
    --config "$CONFIG" --out_dir "$RUN_ROOT" \
    --resume_from_checkpoint "$INIT_CKPT" \
    --override train.trainer.num_nodes="$NNODES" train.trainer.devices=4 \
    train.trainer.strategy=ddp train.trainer.max_steps="$MAX_STEPS" \
    train.compile.enabled="${JOINT_COMPILE_ENABLED:-true}" \
    train.trainer.max_time="${JOINT_TRAIN_MAX_TIME:-00:23:00:00}" \
    train.trainer.limit_val_batches=0 train.trainer.enable_checkpointing=true \
    train.trainer.checkpoint_dir="$JOINT_EXPERIMENT_ROOT/checkpoints" \
    train.data.num_workers=2 train.data.train_batch_size=1 \
    model.apo_repr_mix.utility_rollout_fraction="${JOINT_UTILITY_FRACTION:-0.25}" \
    train.data.data_root=/scratch/t530a05/kfold/data/kfold_data_archive/dataset \
    train.data.ccd_path=/scratch/t530a05/kfold/data/kfold_data_archive/ccd-train.pkl \
    "${RESUME_OVERRIDES[@]}"

# The 23-hour training timer saves progress.ckpt. No automatic resubmission.
"$TRAIN_PYTHON" -c '
import json, os
from pathlib import Path
root = Path(os.environ["JOINT_EXPERIMENT_ROOT"])
report = json.loads((root / "progress/rank-0.json").read_text())
assert report["status"] in {"complete", "paused"}, report
assert report["global_step"] <= report["target_steps"], report
assert (root / "checkpoints/progress.ckpt").is_file()
print("JOINT_SEGMENT_FINISHED", json.dumps(report))
'
