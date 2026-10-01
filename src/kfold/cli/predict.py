# Copyright 2026 Korea Advanced Institute of Science and Technology (KAIST)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run predictions for each query and seed."""

import argparse
import logging
import os
from pathlib import Path

from kfold.inference.query import Query

logger = logging.getLogger("cli")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    # Stage selection, query paths, and generation seeds.
    parser.add_argument(
        "--stage",
        choices=("all", "apo", "complex"),
        default="all",
        help="Run all stages, prepare apos, or predict complexes from --out-dir.",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Query JSON/YAML file or directory.",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1],
        help="Complex inference seeds (default: [1]).",
    )
    # Apo generation settings.
    parser.add_argument(
        "--apo-config",
        type=Path,
        help="YAML settings for apo batching and AtlasFold prediction.",
    )
    parser.add_argument(
        "--num-apos",
        type=int,
        default=1,
        help="Generated apos per protein entry per inference seed "
        "without shared seeds (1–5; default: 1).",
    )
    parser.add_argument(
        "--share-apo-seeds",
        type=int,
        nargs="+",
        help="Generate apos with these unique positive seeds "
        "and reuse them across inference seeds.",
    )
    # Complex prediction settings.
    parser.add_argument(
        "--num-shared-apos",
        type=int,
        default=5,
        help="Maximum apo candidates per protein entry with shared seeds "
        "(1–5; default: 5).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Predictions per query/seed (default: 5).",
    )
    parser.add_argument(
        "--num-recycles",
        type=int,
        default=10,
        help="Number of recycling iterations (default: 10).",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=100,
        help="Number of diffusion steps (default: 100).",
    )
    # Devices, model components, and model cache.
    parser.add_argument(
        "--kernel",
        dest="kernel_backend",
        type=str,
        default="auto",
        choices=("auto", "torch", "triton", "cuequiv"),
        help="Kernel backend for K-Fold and AtlasFold apo preparation.",
    )
    parser.add_argument(
        "--gpu-ids",
        type=int,
        nargs="+",
        default=[0],
        help="Visible CUDA device IDs (default: [0]).",
    )
    parser.add_argument(
        "--distribution",
        choices=("size", "round-robin"),
        default="size",
        help="Multi-GPU job distribution",
    )
    parser.add_argument(
        "--disable-struct-encoder",
        action="store_true",
        help="Disable the apo structure encoder to save ~6 GB of GPU memory.",
    )
    parser.add_argument(
        "--disable-rna-encoder",
        action="store_true",
        help="Disable the RNA encoder; only for queries without RNA.",
    )
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Offload pretrained encoders to CPU to save GPU memory (slower).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Hugging Face cache directory for models and CCD.",
    )
    # Execution control and optional output files.
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check query files and paths; count pending jobs without loading models.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rerun completed query/seed jobs.",
    )
    parser.add_argument(
        "--save-confidence",
        action="store_true",
        help="Save per-atom and token-pair confidence arrays.",
    )
    parser.add_argument(
        "--save-embeddings",
        action="store_true",
        help="Save internal trunk embeddings.",
    )
    parser.add_argument(
        "--save-distogram",
        action="store_true",
        help="Save distogram.",
    )
    parser.add_argument(
        "--save-trajectory",
        action="store_true",
        help="Save diffusion trajectories.",
    )


def _download_models(cache_dir: Path | None) -> None:
    """Cache all model weights and assets."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import tqdm

    progress_bars = []

    class DownloadProgress(tqdm):
        def __init__(self, *args, **kwargs):
            kwargs.update(leave=False, dynamic_ncols=True)
            super().__init__(*args, **kwargs)
            progress_bars.append(self)

    logger.info("Downloading all model weights and assets.")
    if cache_dir is not None:
        logger.info("Cache directory: %s", cache_dir.resolve())

    for repo_id in (
        "SeonghwanSeo/atlaslm-3b-base",
        "SeonghwanSeo/atlasfold-260703",
        "SeonghwanSeo/atlasfold-m-260725",
        "SeonghwanSeo/kfold-assets",
        "SeonghwanSeo/kfold",
    ):
        try:
            snapshot_download(
                repo_id,
                revision="v1.0.0" if repo_id == "SeonghwanSeo/kfold" else None,
                cache_dir=cache_dir,
                tqdm_class=DownloadProgress,
            )
        finally:
            for bar in reversed(progress_bars):
                bar.close()
            progress_bars.clear()
    logger.info("All model weights and assets are cached.")


def _load_queries(args: argparse.Namespace) -> list[Query]:
    """Validate runtime arguments and load queries in prediction order."""
    # Validate all runtime arguments, regardless of the selected stages.
    # GPU selection and generation seeds.
    if not args.gpu_ids or any(gpu_id < 0 for gpu_id in args.gpu_ids):
        raise ValueError("--gpu-ids must be a non-empty list of non-negative indices.")
    if len(set(args.gpu_ids)) != len(args.gpu_ids):
        raise ValueError("--gpu-ids values must be unique.")
    for name in ("seeds", "share_apo_seeds"):
        seeds = getattr(args, name)
        if seeds is not None and (
            not seeds or any(seed < 1 for seed in seeds) or len(set(seeds)) != len(seeds)
        ):
            raise ValueError(
                f"--{name.replace('_', '-')} requires one or more unique positive seeds."
            )

    # Apo and complex sampling counts.
    for name in ("num_apos", "num_shared_apos"):
        if not 1 <= getattr(args, name) <= 5:
            raise ValueError(f"--{name.replace('_', '-')} must be between 1 and 5.")
    for name in ("num_samples", "num_recycles", "num_steps"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")

    # Discover query files from a single file or directory.
    if args.input.is_dir():
        paths = sorted(
            path
            for path in args.input.iterdir()
            if path.is_file() and path.suffix in {".json", ".yaml", ".yml"}
        )
    else:
        paths = [args.input]
    if not paths:
        raise ValueError(f"No JSON/YAML query files found in {args.input}.")

    # Load queries and reject names that would share an output directory.
    queries = [Query.load(path) for path in paths]
    names = set()
    for query in queries:
        if query.name in names:
            raise ValueError(f"Duplicate query name: {query.name!r}.")
        names.add(query.name)

    queries.sort(key=lambda query: query.priority)
    return queries


def dry_run(args: argparse.Namespace) -> None:
    """Validate selected queries and report pending work without writing outputs."""
    from huggingface_hub import snapshot_download

    from kfold.cli.predict_complex import dry_run as predict_complex
    from kfold.cli.prepare_apo import dry_run as prepare_apo
    from kfold.data.types.ccd import CCD
    from kfold.inference.runner import ASSETS_REPO_ID

    queries = _load_queries(args)

    logger.info("Checking query CCD codes.")
    ccd_path = (
        Path(snapshot_download(ASSETS_REPO_ID, cache_dir=args.cache_dir))
        / "assets/ccd.pkl"
    )
    ccd = CCD.load(ccd_path)
    for query in queries:
        query.validate_ccd_codes(ccd.keys())

    logger.info("Input: %s; output: %s.", args.input, args.out_dir)
    if args.stage in ("all", "apo"):
        prepare_apo(args, queries)
    if args.stage in ("all", "complex"):
        predict_complex(args, queries)
    logger.info("Dry run complete; no models loaded or outputs written.")


def run(args: argparse.Namespace) -> None:
    """Validate arguments, load queries, and run the selected prediction stages."""
    from huggingface_hub.utils import disable_progress_bars

    from kfold.cli.predict_complex import run as predict_complex
    from kfold.cli.prepare_apo import run as prepare_apo
    from kfold.utils.runtime import select_kernel_backend

    if os.environ.get("OMP_NUM_THREADS") != "1":
        logger.warning(
            "OMP_NUM_THREADS is not set to 1; model initialization may be slow. "
            "Set OMP_NUM_THREADS=1 to avoid this slowdown."
        )

    # Populate the shared cache before any GPU worker loads a model.
    _download_models(args.cache_dir)

    input_path = args.input.resolve()
    output_path = args.out_dir.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path {input_path} does not exist.")

    # Print the resolved input and output paths for clarity.
    logger.info("Input path: %s", input_path)
    logger.info("Output path: %s", output_path)

    # Resolve the kernel backend, preferring Triton when unspecified.
    args.kernel_backend = select_kernel_backend(args.kernel_backend)
    logger.info("Kernel backend: %s.", args.kernel_backend)

    # Load queries
    queries = _load_queries(args)

    # Disable Hugging Face progress bars to avoid cluttering the output.
    disable_progress_bars()

    # Prepare provided and generated apo structures.
    if args.stage in ("all", "apo"):
        prepare_apo(args, queries)

    # Predict complexes from prepared apo structures.
    if args.stage in ("all", "complex"):
        predict_complex(args, queries)

    # Report completion after all selected stages finish.
    logger.info("Output path: %s", output_path)
