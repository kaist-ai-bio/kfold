#!/usr/bin/env bash
# Run from this source snapshot, without replacing an installed main checkout.
set -euo pipefail

if [[ $# -lt 4 ]]; then
  cat >&2 <<'USAGE'
Usage: run_apo_joint_inference.sh MODEL_DIR mix|single INPUT OUT_DIR [CLI options]
Example: ... models/apo-joint-shallow-update4000 mix query.yaml results/mix \
  --seeds 1 2 3 4 5 6 7 8 9 10 --gpu-ids 0
Set KFOLD_PYTHON to a Python >=3.11 environment with K-Fold dependencies.
This script runs inference immediately. On Messi, run inside a Slurm allocation.
USAGE
  exit 2
fi

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
model_dir="$(cd -- "$1" && pwd -P)"
mode="$2"
input="$3"
out_dir="$4"
shift 4
python_bin="${KFOLD_PYTHON:-python}"
[[ -f "$model_dir/config.yaml" && -f "$model_dir/weights/kfold.pth" ]] || {
  echo 'MODEL_DIR must contain config.yaml and weights/kfold.pth.' >&2; exit 2;
}
case "$mode" in
  mix) trunk_flag=--apo-trunk ;;
  single) trunk_flag=--no-apo-trunk ;;
  *) echo 'Mode must be mix or single.' >&2; exit 2 ;;
esac
# Keep these controls in one place; change the source command for other protocols.
for arg in "$@"; do
  case "$arg" in
    --model-path|--model-path=*|--apo-trunk|--no-apo-trunk|--apo-trunk-recycles|--apo-trunk-recycles=*|--max-apo-combinations|--max-apo-combinations=*|--num-recycles|--num-recycles=*|--num-steps|--num-steps=*|--num-samples|--num-samples=*|--input|--input=*|-i|--out-dir|--out-dir=*|-o|--disable-struct-encoder|--overwrite)
      echo "Protocol/output override not accepted by this wrapper: $arg" >&2; exit 2 ;;
  esac
done
export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
"$python_bin" -c 'import pathlib,sys,kfold; assert sys.version_info >= (3,11); assert pathlib.Path(kfold.__file__).resolve().is_relative_to(pathlib.Path(sys.argv[1])/"src"), kfold.__file__; print("K-Fold source:", kfold.__file__)' "$repo_dir"
exec "$python_bin" "$repo_dir/run_kfold.py" \
  --model-path "$model_dir" --input "$input" --out-dir "$out_dir" \
  "$trunk_flag" --apo-trunk-recycles 2 --max-apo-combinations 3 \
  --num-recycles 10 --num-steps 100 --num-samples 5 "$@"
