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

"""Predict complex structures from prepared apo inputs."""

import argparse
import csv
import json
import logging
import math
import random
import shutil
from pathlib import Path
from time import perf_counter

import torch

from kfold import __version__
from kfold.cli.multigpu import distribute, launch
from kfold.cli.prepare_apo import is_apo_prepared
from kfold.inference.query import Query
from kfold.inference.runner import KFoldRunner
from kfold.model import KFold

logger = logging.getLogger("complex")


def _load_prepared_query(query: Query, apo_dir: Path) -> Query:
    """Load a completed preparation and validate its target and protein entries."""
    if not is_apo_prepared(query, apo_dir):
        raise ValueError(
            f"Apo preparation is incomplete: {apo_dir}. "
            "Run kfold --stage apo with the same inputs and --share-apo-seeds "
            "setting first, and the same seeds unless apos are shared."
        )
    prepared = Query.load(apo_dir / "query.json")
    if prepared.name != query.name:
        raise ValueError(f"Prepared query name does not match {query.name}.")
    if any(not entry.apo for entry in prepared.protein_entries):
        raise ValueError(f"Prepared query has missing apo structures: {apo_dir}.")
    return prepared


def dry_run(args: argparse.Namespace, queries: list[Query]) -> None:
    """Validate saved preparations and count pending inference jobs without writing."""
    pending = 0
    for query in queries:
        for seed in args.seeds:
            job_dir = args.out_dir / query.name / f"{query.name}_seed-{seed}"
            apo_dir = job_dir.parent if args.share_apo_seeds is not None else job_dir
            apo_ready = is_apo_prepared(query, apo_dir)
            # Complex-only runs require prepared inputs, including completed jobs.
            if args.stage == "complex":
                _load_prepared_query(query, apo_dir)
            elif apo_ready and not args.overwrite:
                Query.load(apo_dir / "query.json")

            if (job_dir / "done.txt").is_file() and not args.overwrite and apo_ready:
                continue

            pending += 1

    logger.info(
        "K-Fold inference: %d jobs pending (%d complete); GPUs %s.",
        pending,
        len(queries) * len(args.seeds) - pending,
        args.gpu_ids[:pending],
    )


def run(args: argparse.Namespace, queries: list[Query]) -> None:
    """Predict unfinished complex jobs and summarize their ranked outputs."""
    start = perf_counter()

    # Load prepared inputs for unfinished query/seed jobs.
    pending = []
    for query in queries:
        for seed in args.seeds:
            job_dir = args.out_dir / query.name / f"{query.name}_seed-{seed}"
            apo_dir = job_dir.parent if args.share_apo_seeds is not None else job_dir
            if (
                (job_dir / "done.txt").is_file()
                and not args.overwrite
                and is_apo_prepared(query, apo_dir)
            ):
                continue
            prepared = _load_prepared_query(query, apo_dir)
            pending.append((prepared, seed, job_dir))

    n_jobs = len(queries) * len(args.seeds)
    n_pending = len(pending)
    n_prev = n_jobs - n_pending

    logger.info(
        "K-Fold inference: %d jobs pending (%d complete); GPUs %s.",
        n_pending,
        n_prev,
        args.gpu_ids[:n_pending],
    )

    # Invalidate previous completion records and run the pending GPU jobs.
    if pending:
        logger.info(
            "Settings: struct_encoder=%s, rna_encoder=%s, kernel=%s, cpu_offload=%s.",
            "disabled" if args.disable_struct_encoder else "enabled",
            "disabled" if args.disable_rna_encoder else "enabled",
            args.kernel_backend,
            "enabled" if args.cpu_offload else "disabled",
        )
        logger.info(
            "Starting complex inference: seeds=%s, num_samples=%d, "
            "num_recycles=%d, num_steps=%d.",
            args.seeds,
            args.num_samples,
            args.num_recycles,
            args.num_steps,
        )
        for _, _, job_dir in pending:
            (job_dir / "done.txt").unlink(missing_ok=True)
        num_workers = min(len(args.gpu_ids), n_pending)
        costs = [query.priority[0] ** 2 for query, _, _ in pending]
        groups = distribute(costs, num_workers, args.distribution)
        worker_jobs = [[pending[index] for index in group] for group in groups]
        launch(_worker, args, worker_jobs, stage="K-Fold inference")

    # Rank all requested seeds and save one representative result per query.
    logger.info("Summarizing predictions.")
    for query in queries:
        name = query.name
        logger.info("Ranking samples and saving best prediction for %s.", name)
        summary_start = perf_counter()
        if not _summarize_predictions(
            args.out_dir / name, name, args.seeds, args.num_samples
        ):
            continue

        # Record the K-Fold settings requested in this CLI invocation.
        settings = {
            "version": __version__,
            "seeds": args.seeds,
            "num_samples": args.num_samples,
            "num_recycles": args.num_recycles,
            "num_steps": args.num_steps,
            "use_struct_encoder": not args.disable_struct_encoder,
            "use_rna_encoder": not args.disable_rna_encoder,
            "kernel_backend": args.kernel_backend,
        }
        if args.share_apo_seeds is not None:
            settings["share_apo_seeds"] = args.share_apo_seeds
            settings["num_shared_apos"] = args.num_shared_apos

        (args.out_dir / name / "kfold_settings.json").write_text(
            json.dumps(settings, indent=2) + "\n"
        )
        logger.info(
            "Saved ranked results for %s to %s in %.1f s.",
            name,
            args.out_dir / name,
            perf_counter() - summary_start,
        )

    n_succ = sum((job_dir / "done.txt").is_file() for _, _, job_dir in pending)
    n_fail = n_pending - n_succ
    summary = [f"{n_jobs} jobs total"]
    if n_prev:
        summary.append(f"{n_prev} previously completed")
    summary.append(f"{n_succ} succeeded")
    if n_fail:
        summary.append(f"{n_fail} incomplete")
    logger.info(
        "K-Fold inference finished in %.1f s: %s.",
        perf_counter() - start,
        ", ".join(summary),
    )


def _summarize_predictions(
    out_dir: Path, name: str, seeds: list[int], num_samples: int
) -> bool:
    """Rank completed predictions and return whether results were saved."""
    # Collect confidence summaries from completed prediction jobs.
    records = []
    for seed in seeds:
        job_dir = out_dir / f"{name}_seed-{seed}"
        if not (job_dir / "done.txt").is_file():
            logger.warning("Skipping incomplete prediction job: %s.", job_dir)
            continue
        prefix = f"{job_dir.name}_sample-"
        for sample in range(num_samples):
            path = job_dir / f"{prefix}{sample}_confidence.json"
            model_path = job_dir / f"{prefix}{sample}_model.cif"
            if not path.is_file() or not model_path.is_file():
                logger.warning(
                    "Skipping sample %s in %s: missing structure or confidence output.",
                    sample,
                    job_dir,
                )
                continue
            with path.open() as f:
                scores = json.load(f)["complex"]
            if not math.isfinite(scores["ranking_score"]):
                raise ValueError(f"Non-finite ranking score in {path}.")
            records.append({"seed": seed, "sample": sample, **scores})

    if not records:
        logger.warning(
            "Skipping summary for %s: no completed predictions in %s.", name, out_dir
        )
        return False

    # Rank samples and copy the best structure and available confidence outputs.
    records.sort(key=lambda row: (-row["ranking_score"], row["seed"], row["sample"]))
    best = records[0]
    rank_dir = out_dir / f"{name}_seed-{best['seed']}"
    prefix = f"{name}_seed-{best['seed']}_sample-{best['sample']}"
    for suffix in ("model.cif", "confidence.json"):
        shutil.copyfile(rank_dir / f"{prefix}_{suffix}", out_dir / f"{name}_{suffix}")

    confidence = rank_dir / f"{prefix}_confidence.npz"
    best_confidence = out_dir / f"{name}_confidence.npz"
    if confidence.is_file():
        shutil.copyfile(confidence, best_confidence)
    else:
        best_confidence.unlink(missing_ok=True)

    # Write the complete ranking across seeds and samples.
    with (out_dir / f"{name}_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed",
                "sample",
                "ranking_score",
                "plddt",
                "ptm",
                "iptm",
                "pde",
                "has_clash",
            ],
        )
        writer.writeheader()
        writer.writerows(records)
    return True


def _worker(
    gpu_id: int, args: argparse.Namespace, jobs: list[tuple[Query, int, Path]]
) -> None:
    try:
        # Load one model and runner for all jobs assigned to this GPU.
        logger.info("Loading K-Fold model and runner.")
        init_start = perf_counter()
        model = KFold.from_pretrained(
            args.model_path,
            device=torch.device("cuda", gpu_id),
            cache_dir=args.cache_dir,
            use_struct_encoder=not args.disable_struct_encoder,
            use_rna_encoder=not args.disable_rna_encoder,
            cpu_offload=args.cpu_offload,
            kernel_backend=args.kernel_backend,
        )
        runner = KFoldRunner(model, cache_dir=args.cache_dir, verbose=False)
        logger.info(
            "Model and runner initialized in %.1f s.", perf_counter() - init_start
        )
        for job_index, (query, seed, job_dir) in enumerate(jobs, start=1):
            progress = f"[{job_index}/{len(jobs)}]"
            target = f"{query.name} (seed={seed})"
            job_start = perf_counter()
            logger.info("Job started: %s %s.", target, progress)

            # Featurize the query.
            stage_start = perf_counter()
            logger.info("Featurization started.")
            # Load aligned apo and prior candidates from the prepared query.
            apos, priors = runner.load_apo_and_prior(query)
            if args.share_apo_seeds is not None:
                # Select candidates within each protein entry, preserving entry order.
                rng = random.Random(seed)
                apos = [
                    rng.sample(candidates, args.num_shared_apos)
                    if len(candidates) > args.num_shared_apos
                    else candidates
                    for candidates in apos
                ]
            # Prepare the input for complex prediction
            item = runner.build_input(
                query, seed, args.num_samples, apos=apos, priors=priors
            )
            logger.info(
                "Featurization completed: chains=%d, tokens=%d, atoms=%d, %.1f s.",
                item.ref_struct.num_chains,
                item.ref_struct.num_tokens,
                item.ref_struct.num_atoms,
                perf_counter() - stage_start,
            )

            # Run complex prediction and report the highest-ranked sample.
            logger.info("Inference started.")
            stage_start = perf_counter()
            result = runner.predict_from_input(
                item,
                seed=seed,
                num_samples=args.num_samples,
                num_recycles=args.num_recycles,
                max_apo_combinations=args.max_apo_combinations,
                num_steps=args.num_steps,
                return_trajectory=args.save_trajectory,
                return_embeddings=args.save_embeddings,
                return_distogram=args.save_distogram,
            )
            best_sample, best_summary = max(
                enumerate(result.confidence_summary),
                key=lambda item: item[1]["complex"]["ranking_score"],
            )
            scores = best_summary["complex"]
            logger.info(
                "Inference completed: %.1f s, best sample=%d, "
                "ipTM=%.3f, pTM=%.3f, pLDDT=%.2f, ranking_score=%.3f.",
                perf_counter() - stage_start,
                best_sample,
                scores["iptm"],
                scores["ptm"],
                scores["plddt"],
                scores["ranking_score"],
            )
            # Save outputs before marking the query/seed job complete.
            logger.info("Saving started.")
            stage_start = perf_counter()
            if args.share_apo_seeds is not None:
                result.settings["share_apo_seeds"] = args.share_apo_seeds
                result.settings["num_shared_apos"] = args.num_shared_apos
            result.save(
                job_dir,
                save_confidence=args.save_confidence,
                save_embeddings=args.save_embeddings,
                save_distogram=args.save_distogram,
                save_trajectory=args.save_trajectory,
            )
            (job_dir / "done.txt").touch()
            logger.info(
                "Saving completed in %.1f s.",
                perf_counter() - stage_start,
            )
            del result, item, apos, priors
            logger.info(
                "Job completed in %.1f s: %s %s.",
                perf_counter() - job_start,
                target,
                progress,
            )
    except torch.cuda.OutOfMemoryError:
        logger.error(
            f"K-Fold GPU {gpu_id}: out of memory. "
            "Enable --cpu-offload to reduce inference memory use."
        )
