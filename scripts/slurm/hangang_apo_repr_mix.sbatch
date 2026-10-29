#!/bin/bash
#SBATCH --job-name=apo-repr-mix
#SBATCH --partition=normal
#SBATCH --account=bs0247
#SBATCH --comment="field=sciai;appl=pytorch-ddp"
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=00:45:00
#SBATCH --output=apo-repr-mix-%j.out
#SBATCH --error=apo-repr-mix-%j.err

set -euo pipefail
ulimit -c 0

CODE=/scratch/t530a05/kfold/worktrees/apo-repr-mix
CONFIG=configs/train/apo_repr_mix.yaml
source /scratch/t530a05/kfold/ops/activate-train-cueq010.sh
module load libfabric/2.3.1 aws-ofi-nccl/1.20.0

# Force the Slingshot/CXI path rather than silently falling back to TCP.
export NCCL_NET="AWS Libfabric" FI_PROVIDER=cxi NCCL_IB_DISABLE=1
export NCCL_PROTO='^LL128'
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1

cd "$CODE"
test "${SLURM_JOB_NUM_NODES:?}" -eq 2
test "${SLURM_NTASKS:?}" -eq 8
test "$(git rev-parse HEAD)" = "${KFOLD_EXPECTED_COMMIT:?Set the reviewed commit before sbatch}"
git diff --exit-code HEAD --

export RUN_ROOT=/scratch/t530a05/kfold/experiments/apo-repr-mix-8gpu-${SLURM_JOB_ID}
mkdir "$RUN_ROOT"
export WANDB_MODE=disabled
git rev-parse HEAD > "$RUN_ROOT/source-commit.txt"
cp "$CONFIG" "$RUN_ROOT/launch-config.yaml"
export PYTHONPATH="$CODE/src${PYTHONPATH:+:$PYTHONPATH}"
export TRAIN_PYTHON=/scratch/t530a05/kfold/envs/train-cueq010/bin/python

RESUME_CKPT=${RESUME_CKPT:-/scratch/t530a05/kfold/checkpoints/training-candidates/0908-72k/s3-71k.ckpt}
test -r "$RESUME_CKPT"

# Rank zero may not live on the first host in SLURM_JOB_NODELIST. Discover the
# actual step's rank-zero host for both the bounded NCCL check and training.
run_step() {
    local log_prefix=$1
    shift
    srun --mpi=none --network=disable_rdzv_get --kill-on-bad-exit=1 \
    --distribution=block:block --cpu-bind=none --gpu-bind=none \
    --output="$RUN_ROOT/${log_prefix}rank-%t.out" --error="$RUN_ROOT/${log_prefix}rank-%t.err" \
    bash -c '
        set -euo pipefail
        rank_root="$RUN_ROOT/runtime/$SLURM_PROCID"
        # multiprocessing appends pymp-*/listener-* Unix socket names.
        # Keep the prefix short enough for Linux sockaddr_un (108 bytes).
        export TMPDIR="/scratch/t530a05/kfold/t/$SLURM_JOB_ID/$SLURM_PROCID"
        export TRITON_CACHE_DIR="$rank_root/triton"
        export TORCHINDUCTOR_CACHE_DIR="$rank_root/inductor"
        export CUDA_CACHE_PATH="$rank_root/cuda"
        mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH"
        ulimit -l unlimited
        exec "$TRAIN_PYTHON" -B scripts/slurm/hangang_rank.py "$@"
    ' kfold "$@"
}

# Verify all eight GPUs before loading the model.
run_step preflight- --preflight
run_step '' -- scripts/validate_apo_repr_mix.py \
    --config "$CONFIG" --out_dir "$RUN_ROOT" \
    --resume_from_checkpoint "$RESUME_CKPT" \
    --override train.trainer.num_nodes=2 train.trainer.devices=4 \
    train.trainer.strategy=ddp train.trainer.limit_val_batches=0 \
    train.trainer.enable_checkpointing=false train.trainer.log_every_n_steps=1 \
    train.data.num_workers=2 train.data.train_batch_size=1 \
    train.data.data_root=/scratch/t530a05/kfold/data/kfold_data_archive/dataset \
    train.data.ccd_path=/scratch/t530a05/kfold/data/kfold_data_archive/ccd-train.pkl \
    train.training.parcae.min_recycles=1 train.training.parcae.max_recycles=1
