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

"""Score one protein and one ligand per prepared K-Fold query."""

import argparse
import hashlib
import json
import logging
import random
import tempfile
from pathlib import Path

import torch

from kfold.cli.multigpu import distribute, launch
from kfold.cli.predict_complex import _load_prepared_query
from kfold.inference.affinity import load_affinity_head, predict_affinity
from kfold.inference.query import LigandSequence, ProteinSequence, Query
from kfold.inference.runner import KFoldRunner
from kfold.model import KFold

logger = logging.getLogger("affinity")


def _query_identity(query: Query) -> tuple:
    proteins = [entry for entry in query.sequences if isinstance(entry, ProteinSequence)]
    ligands = [entry for entry in query.sequences if isinstance(entry, LigandSequence)]
    if (
        len(query.sequences) != 2
        or len(proteins) != 1
        or len(ligands) != 1
        or any(len(entry.id) != 1 for entry in query.sequences)
        or len(ligands[0]) != 1
        or query.bonds
    ):
        raise ValueError(
            f"{query.name}: expected one protein chain and one noncovalent ligand."
        )
    protein = proteins[0]
    ligand = ligands[0]
    return (
        protein.id,
        protein.sequence,
        protein.modifications,
        ligand.id,
        ligand.smiles,
        ligand.ccd,
    )


def _pending_jobs(
    args: argparse.Namespace, queries: list[Query]
) -> list[tuple[Query, int, Path]]:
    jobs = []
    for query in queries:
        identity = _query_identity(query)
        for seed in args.seeds:
            job_dir = args.out_dir / query.name / f"{query.name}_seed-{seed}"
            apo_dir = job_dir.parent if args.share_apo_seeds is not None else job_dir
            prepared = _load_prepared_query(query, apo_dir)
            if _query_identity(prepared) != identity:
                raise ValueError(
                    f"{query.name}: prepared query differs from the requested input. "
                    "Rerun --stage apo --overwrite for the changed query."
                )
            output = job_dir / "affinity.json"
            if args.overwrite or not output.is_file():
                jobs.append((prepared, seed, output))
    return jobs


def dry_run(args: argparse.Namespace, queries: list[Query]) -> None:
    jobs = _pending_jobs(args, queries)
    logger.info("Affinity inference: %d pending query/seed jobs.", len(jobs))


def run(args: argparse.Namespace, queries: list[Query]) -> None:
    jobs = _pending_jobs(args, queries)
    logger.info("Affinity inference: %d pending query/seed jobs.", len(jobs))
    if not jobs:
        return

    with args.affinity_checkpoint.open("rb") as stream:
        args.affinity_checkpoint_sha256 = hashlib.file_digest(
            stream, "sha256"
        ).hexdigest()
    costs = [query.priority[0] ** 2 for query, _, _ in jobs]
    groups = distribute(costs, min(len(jobs), len(args.gpu_ids)), args.distribution)
    launch(
        _worker,
        args,
        [[jobs[index] for index in group] for group in groups],
        stage="Affinity",
    )


def _worker(
    gpu_id: int, args: argparse.Namespace, jobs: list[tuple[Query, int, Path]]
) -> None:
    device = torch.device("cuda", gpu_id)
    head = load_affinity_head(args.affinity_checkpoint, device)
    model = KFold.from_pretrained(
        args.model_path,
        device=device,
        cache_dir=args.cache_dir,
        use_rna_encoder=not args.disable_rna_encoder,
        cpu_offload=args.cpu_offload,
        kernel_backend=args.kernel_backend,
    )
    runner = KFoldRunner(model, cache_dir=args.cache_dir, verbose=False)
    for query, seed, output in jobs:
        apos, priors = runner.load_apo_and_prior(query)
        if args.share_apo_seeds is not None:
            rng = random.Random(seed)
            apos = [
                rng.sample(candidates, args.num_shared_apos)
                if len(candidates) > args.num_shared_apos
                else candidates
                for candidates in apos
            ]
        item = runner.build_input(query, seed, num_samples=1, apos=apos, priors=priors)
        scores = predict_affinity(
            model, head, item.f_input, seed=seed, num_recycles=args.num_recycles
        )
        result = {
            "query": query.name,
            "seed": seed,
            "num_recycles": args.num_recycles,
            "model_path": args.model_path,
            "affinity_checkpoint_sha256": args.affinity_checkpoint_sha256,
            **scores,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=".affinity-",
            delete=False,
        ) as stream:
            partial = Path(stream.name)
            try:
                json.dump(result, stream, indent=2, allow_nan=False)
                stream.write("\n")
                stream.close()
                partial.replace(output)
            finally:
                partial.unlink(missing_ok=True)
        logger.info(
            "Saved affinity prediction for %s (seed=%d) to %s.", query.name, seed, output
        )
