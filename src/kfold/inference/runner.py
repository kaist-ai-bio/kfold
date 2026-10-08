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

"""Prepare query structures and generate K-Fold predictions."""

import copy
import json
import logging
import re
from dataclasses import asdict, dataclass
from functools import cached_property
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from huggingface_hub import snapshot_download

from kfold import __version__
from kfold.data.types.ccd import CCD
from kfold.data.types.structure import RefStructure
from kfold.data.utils.writer import KFoldWriter
from kfold.inference.apo_runner import (
    ApoChain,
    ApoConfig,
    ApoMultimer,
    ApoRunner,
    apo_output_prefix,
    load_monomer_apo,
    load_multimer_apo,
)
from kfold.inference.data_pipeline import InferenceInput, InputDataPipeline
from kfold.inference.query import (
    ProteinPair,
    ProteinSequence,
    Query,
)
from kfold.model import KFold
from kfold.utils import confidence_metrics

ASSETS_REPO_ID = "kaist-ai-bio/kfold-assets"
logger = logging.getLogger(__name__)


@dataclass
class FoldingResult:
    """Predictions, confidence metrics, and the apo/prior candidates used as inputs.

    Embeddings and distograms are shared across prediction samples.
    """

    query: Query
    seed: int
    structure: RefStructure
    coordinates: list[np.ndarray]
    confidence_summary: list[dict]
    confidence_scores: list[dict]
    # Candidate ensembles in query.protein_entries order.
    apos: list[list[ApoChain] | list[ApoMultimer]]
    priors: list[list[ApoChain] | list[ApoMultimer]]
    settings: dict
    apo_settings: dict | None = None
    embeddings: dict[str, np.ndarray] | None = None
    distogram: dict[str, np.ndarray] | None = None
    trajectory: np.ndarray | None = None

    @property
    def num_samples(self) -> int:
        """Return the number of predicted coordinate samples."""
        return len(self.coordinates)

    def save(
        self,
        out_dir: str | Path,
        *,
        save_confidence: bool = True,
        save_embeddings: bool = False,
        save_distogram: bool = False,
        save_trajectory: bool = False,
    ) -> None:
        """Write predictions to an output directory.

        Predictions, confidence summaries, and K-Fold settings are always written.
        Available apo settings are saved in apo_setting.json. Existing files
        with matching names are replaced. Optional outputs not written for these
        samples, and unwritten shared embeddings or distograms, are removed.
        Previous sample outputs beyond the new sample count for this query and
        seed are also removed. Other files in the directory are retained.

        Parameters
        ----------
        out_dir : str | Path
            Destination directory, created if needed.
        save_confidence : bool
            Write per-sample plddt, pae, and pde arrays as NPZ.
        save_embeddings : bool
            Write shared embeddings if present in this result.
        save_distogram : bool
            Write the shared distogram if present in this result.
        save_trajectory : bool
            Write per-sample trajectories if present in this result.
        """
        out_dir = Path(out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        name, seed = self.query.name, self.seed

        # Save per-sample structures, confidence summaries, and optional outputs.
        for i in range(self.num_samples):
            prefix = f"{name}_seed-{seed}_sample-{i}"
            coords = self.coordinates[i]
            confidence_summary = self.confidence_summary[i]
            confidence_scores = self.confidence_scores[i]

            structure = self.structure.copy_with_new_coords(
                coords, b_factors=confidence_scores["plddt"]
            )

            # Write the structure and confidence summary for every sample.
            KFoldWriter.write_mmcif(structure, out_dir / f"{prefix}_model.cif")

            with open(out_dir / f"{prefix}_confidence.json", "w") as f:
                json.dump(confidence_summary, f, indent=2)

            # Replace requested optional outputs and remove obsolete ones.
            if save_confidence:
                np.savez_compressed(
                    out_dir / f"{prefix}_confidence.npz",
                    **{key: confidence_scores[key] for key in ("plddt", "pae", "pde")},
                )
            else:
                (out_dir / f"{prefix}_confidence.npz").unlink(missing_ok=True)
            if save_trajectory and self.trajectory is not None:
                KFoldWriter.write_trajectory(
                    self.structure,
                    self.trajectory[i],
                    out_dir / f"{prefix}_trajectory.cif",
                )
            else:
                (out_dir / f"{prefix}_trajectory.cif").unlink(missing_ok=True)

        # Save or remove embeddings and distograms shared across all samples.
        if save_embeddings and self.embeddings is not None:
            np.savez_compressed(
                out_dir / f"{name}_seed-{seed}_embeddings.npz", **self.embeddings
            )
        else:
            (out_dir / f"{name}_seed-{seed}_embeddings.npz").unlink(missing_ok=True)

        if save_distogram and self.distogram is not None:
            np.savez_compressed(
                out_dir / f"{name}_seed-{seed}_distogram.npz", **self.distogram
            )
        else:
            (out_dir / f"{name}_seed-{seed}_distogram.npz").unlink(missing_ok=True)

        # A smaller rerun replaces the whole sample set for this query and seed.
        sample_pattern = re.compile(
            rf"{re.escape(name)}_seed-{seed}_sample-(\d+)_"
            r"(?:model\.cif|confidence\.(?:json|npz)|trajectory\.cif)"
        )
        for path in out_dir.glob(f"{name}_seed-{seed}_sample-*"):
            match = sample_pattern.fullmatch(path.name)
            if match and int(match[1]) >= self.num_samples and path.is_file():
                path.unlink()

        # Save model settings separately from prediction outputs.
        (out_dir / "kfold_settings.json").write_text(
            json.dumps(self.settings, indent=2) + "\n"
        )
        if self.apo_settings is not None:
            (out_dir / "apo_setting.json").write_text(
                json.dumps(self.apo_settings, indent=2) + "\n"
            )


class KFoldRunner:
    """Prepare query structures and generate K-Fold predictions."""

    def __init__(
        self,
        model: KFold,
        *,
        ccd: CCD | None = None,
        apo_config: ApoConfig | None = None,
        cache_dir: str | Path | None = None,
        share_atlaslm: bool = True,
        verbose: bool = True,
    ):
        """Create a runner for a loaded K-Fold model.

        Parameters
        ----------
        model : KFold
            KFold model already placed on a CUDA device.
        ccd : CCD | None
            Chemical component dictionary. If omitted, load the release CCD.
        apo_config : ApoConfig | None
            Settings for automatic apo generation. Uses defaults when omitted.
        share_atlaslm : bool
            Reuse KFold's AtlasLM for apo sampling. If false, the
            apo samplers share a separately loaded AtlasLM.
        cache_dir : str | Path | None
            Hugging Face cache directory for model and CCD downloads.
        verbose : bool
            Emit INFO logs for initialization, prediction stages, and elapsed times.

        Raises
        ------
        NotImplementedError
            The model is not on a CUDA device.
        """
        self.verbose = verbose
        self.model: KFold = model
        self.device: torch.device = self.model.device
        if self.device.type != "cuda":
            raise NotImplementedError("KFoldRunner requires a CUDA device.")

        self.cache_dir: Path | None = Path(cache_dir) if cache_dir is not None else None

        # Load the CCD used for query validation and reference structures.
        self.ccd: CCD = ccd if ccd is not None else self.load_default_ccd()

        # Initialize the input pipeline.
        self.input_pipeline: InputDataPipeline = InputDataPipeline(self.ccd)

        self.share_atlaslm = share_atlaslm
        self.apo_config = apo_config if apo_config is not None else ApoConfig()

    def _log(self, message: str, *args: object) -> None:
        """Emit an INFO message when verbose output is enabled."""
        if self.verbose:
            logger.info(message, *args)

    def load_default_ccd(self) -> CCD:
        """Load the release CCD, downloading it if needed.

        Returns
        -------
        CCD
            The loaded chemical component dictionary.
        """
        self._log("Loading CCD from release assets.")
        assets_dir = Path(snapshot_download(ASSETS_REPO_ID, cache_dir=self.cache_dir))
        return CCD.load(assets_dir / "assets/ccd.pkl")

    @cached_property
    def apo_runner(self) -> ApoRunner:
        """Create an apo runner on demand for automatic structure generation."""
        return ApoRunner(
            self.device,
            kernel_backend=self.model.kernel_backend,
            config=self.apo_config,
            cache_dir=self.cache_dir,
            lm=self.model.prot_seq_encoder.lm if self.share_atlaslm else None,
            verbose=self.verbose,
        )

    @torch.inference_mode()
    def tokenize_apo(self, apo: ApoChain | ApoMultimer) -> None:
        """Populate an apo candidate's BB/FA structure tokens in place.

        If the model has no protein structure encoder, the candidate is unchanged.

        Parameters
        ----------
        apo : ApoChain | ApoMultimer
            Aligned candidate whose chain token fields will be replaced.
        """
        encoder = self.model.prot_struct_encoder
        if encoder is not None:
            chains = (apo.chain1, apo.chain2) if isinstance(apo, ApoMultimer) else (apo,)
            for chain in chains:
                encoded = encoder.tokenize(chain.sequence, chain.coordinates)
                chain.bb_tokens = encoded["bb_token_id"].cpu().numpy().copy()
                chain.fa_tokens = encoded["fa_token_id"].cpu().numpy().copy()

    def load_apo_and_prior(
        self,
        query: Query,
    ) -> tuple[
        list[list[ApoChain] | list[ApoMultimer]], list[list[ApoChain] | list[ApoMultimer]]
    ]:
        """Load provided apo and prior structures after checking query CCD codes.

        Parameters
        ----------
        query : Query
            Query describing protein entries and optional structure paths.

        Returns
        -------
        apos : list[list[ApoChain] | list[ApoMultimer]]
            Untokenized apo candidates in query.protein_entries order.
            Entries selected for automatic generation have empty candidate lists.
        priors : list[list[ApoChain] | list[ApoMultimer]]
            Prior candidates in the same entry order. Provided apos are reused
            when no prior paths are given.

        Raises
        ------
        ValueError
            Required CCD codes are missing or provided structures cannot be aligned.
        """
        query.validate_ccd_codes(self.ccd.keys())
        # Load and align provided structures in protein-entry order.
        apo_ensembles, prior_ensembles = [], []
        for entry in query.protein_entries:
            apos, priors = [], []
            if isinstance(entry, ProteinPair):
                for path in entry.apo or []:
                    apos.extend(load_multimer_apo(path, entry.sequence1, entry.sequence2))
                for path in entry.prior or []:
                    priors.extend(
                        load_multimer_apo(path, entry.sequence1, entry.sequence2)
                    )
            else:
                for path in entry.apo or []:
                    apos.extend(load_monomer_apo(path, entry.sequence))
                for path in entry.prior or []:
                    priors.extend(load_monomer_apo(path, entry.sequence))
            if entry.prior is None:
                priors = apos
            apo_ensembles.append(apos)
            prior_ensembles.append(priors)
        return apo_ensembles, prior_ensembles

    @torch.inference_mode()
    def build_input(
        self,
        query: Query,
        seed: int,
        num_samples: int = 5,
        *,
        apos: list[list[ApoChain] | list[ApoMultimer]],
        priors: list[list[ApoChain] | list[ApoMultimer]],
    ) -> InferenceInput:
        """Tokenize prepared apo candidates and build K-Fold input features.

        Parameters
        ----------
        query : Query
            Validated query describing the complex.
        seed : int
            Seed for input construction.
        num_samples : int
            Number of prior coordinate samples to prepare.
        apos : list[list[ApoChain] | list[ApoMultimer]]
            Apo ensembles in query.protein_entries order.
        priors : list[list[ApoChain] | list[ApoMultimer]]
            Prior ensembles in the same order.

        Returns
        -------
        InferenceInput
            Input ready for predict_from_input, including prepared structures.
        """
        # Encode apo candidates before building the complex input features.
        for entry_index, ensemble in enumerate(apos, start=1):
            if len(ensemble) > 5:
                logger.warning(
                    "%s (seed=%d): protein entry %d uses %d apo candidates, exceeding 5.",
                    query.name,
                    seed,
                    entry_index,
                    len(ensemble),
                )
            for apo in ensemble:
                self.tokenize_apo(apo)
        return self.input_pipeline.build_input(
            query, seed, num_samples, apos=apos, priors=priors
        )

    @torch.inference_mode()
    def predict(
        self,
        query: Query,
        seed: int,
        *,
        num_apos: int = 1,
        num_samples: int = 5,
        num_recycles: int = 10,
        num_steps: int = 100,
        return_embeddings: bool = False,
        return_trajectory: bool = False,
        return_distogram: bool = False,
    ) -> FoldingResult:
        """Prepare apo structures, encode them, and predict one query.

        Provided structures are retained, and the caller's query is unchanged.

        Parameters
        ----------
        query : Query
            Parsed query to prepare and predict.
        seed : int
            Positive seed for preparation and model sampling.
        num_apos : int
            Number of apos per entry during automatic generation, from 1 to 5.
            Provided structures are used directly.
        num_samples : int
            Number of predictions to generate.
        num_recycles : int
            Number of model recycling iterations.
        num_steps : int
            Number of diffusion sampling steps.
        return_embeddings : bool
            Include shared single and pair embeddings.
        return_trajectory : bool
            Include diffusion trajectories for each sample.
        return_distogram : bool
            Include shared distance logits and bin metadata.

        Returns
        -------
        FoldingResult
            Predictions, confidence metrics, and prepared apo/prior candidates,
            plus the requested optional outputs.

        Raises
        ------
        ValueError
            seed or an inference count is nonpositive, num_apos exceeds 5,
            or query preparation fails validation.
        """
        # Validate prediction seeds and sampling counts.
        if seed <= 0 or min(num_apos, num_samples, num_recycles, num_steps) < 1:
            raise ValueError("seed and inference counts must be positive.")

        if num_apos > 5:
            raise ValueError("num_apos must be between 1 and 5.")

        prediction_start = perf_counter()
        query = copy.deepcopy(query)
        # Validate the query and load provided candidates.
        self._log("Loading provided apo and prior structures for %s.", query.name)
        apos, priors = self.load_apo_and_prior(query)
        # Generate apos automatically, keeping models cached for iterative prediction.
        # Output names use full query positions; candidate lists contain proteins only.
        protein_entries = (
            (sequence_index, entry)
            for sequence_index, entry in enumerate(query.sequences, start=1)
            if isinstance(entry, (ProteinSequence, ProteinPair))
        )
        for index, (sequence_index, entry) in enumerate(protein_entries):
            prefix = apo_output_prefix(
                sequence_index, multimer=isinstance(entry, ProteinPair)
            )
            if not entry.apo:
                sequence = (
                    (entry.sequence1, entry.sequence2)
                    if isinstance(entry, ProteinPair)
                    else entry.sequence
                )
                prediction = self.apo_runner.predict(
                    f"{query.name}_{prefix}",
                    sequence,
                    seeds=[seed * 10 + i for i in range(1, num_apos + 1)],
                )
                apos[index] = prediction.apos
                priors[index] = prediction.priors
                del prediction
        # Encode prepared candidates and build the complex model input.
        item = self.build_input(query, seed, num_samples, apos=apos, priors=priors)

        # Predict complex structures from the prepared input.
        self._log("Predicting query %s.", query.name)
        result = self.predict_from_input(
            item,
            seed,
            num_samples=num_samples,
            num_recycles=num_recycles,
            num_steps=num_steps,
            return_embeddings=return_embeddings,
            return_trajectory=return_trajectory,
            return_distogram=return_distogram,
        )
        result.apo_settings = {
            "version": version("atlasfold"),
            "seed": seed,
            "num_apos": num_apos,
            **asdict(self.apo_config),
        }
        self._log("Prediction complete in %.2f s.", perf_counter() - prediction_start)
        return result

    @torch.inference_mode()
    def predict_from_input(
        self,
        item: InferenceInput,
        seed: int,
        *,
        num_samples: int = 5,
        num_recycles: int = 10,
        num_steps: int = 100,
        return_embeddings: bool = False,
        return_distogram: bool = False,
        return_trajectory: bool = False,
    ) -> FoldingResult:
        """Generate predictions from a prepared input.

        Parameters
        ----------
        item : InferenceInput
            Prepared input containing an unbatched FoldingInput.
        seed : int
            Positive model sampling seed; use the input construction seed
            to match the behavior of predict.
        num_samples : int
            Number of diffusion samples. Use the input construction
            count to match predict; a different count reuses or truncates priors.
        num_recycles : int
            Number of model recycling iterations.
        num_steps : int
            Number of diffusion sampling steps.
        return_embeddings : bool
            Include shared single and pair embeddings.
        return_distogram : bool
            Include shared distance logits and bin metadata.
        return_trajectory : bool
            Include diffusion trajectories for each sample.

        Returns
        -------
        FoldingResult
            Predictions, confidence metrics, and prepared apo/prior candidates,
            plus the requested optional outputs.

        Raises
        ------
        ValueError
            The input is batched, or seed or an inference count is
            nonpositive.
        """
        # Validate runtime settings and the unbatched input contract.
        query, struct, f_input = item.query, item.ref_struct, item.f_input
        if seed <= 0:
            raise ValueError(f"Seed must be positive: {seed}.")
        if num_samples <= 0:
            raise ValueError(f"Num_samples must be positive: {num_samples}.")
        if num_recycles <= 0:
            raise ValueError(f"Num_recycles must be positive: {num_recycles}.")
        if num_steps <= 0:
            raise ValueError(f"Num_steps must be positive: {num_steps}.")
        if f_input.is_batched:
            raise ValueError("predict_from_input expects an unbatched FoldingInput.")

        # Transfer features to the model device, retaining unpadded output sizes.
        f_input = f_input.to(self.device)
        n_atoms = struct.num_atoms
        n_tokens = struct.num_tokens

        # Run inference with an isolated random state and mixed precision.
        with (
            torch.random.fork_rng(devices=[self.device]),
            torch.autocast(
                self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"
            ),
        ):
            torch.manual_seed(seed)
            out = self.model.inference(
                f_input,
                num_recycles=num_recycles,
                num_steps=num_steps,
                num_samples=num_samples,
                return_embeddings=return_embeddings,
                return_distogram=return_distogram,
                return_traj=return_trajectory,
            )

        # Remove coordinate padding and transfer predictions to CPU arrays.
        coords = list(out["diffusion"]["coordinates"][:, :n_atoms].float().cpu().numpy())

        # Compute confidence summaries and per-atom/token confidence arrays.
        summaries, scores = confidence_metrics.summarize_confidence_metrics(
            f_input, struct, out
        )

        # Collect requested auxiliary outputs, trimming padding before transfer.
        embeddings = None
        # Padding is trailing; slices avoid full pair-tensor copies on the GPU.
        if return_embeddings and "trunk" in out:
            embeddings = {
                "s_inputs": out["trunk"]["s_inputs"][:n_tokens].half().cpu().numpy(),
                "s_lm": out["trunk"]["s_lm"][:n_tokens].half().cpu().numpy(),
                "z": out["trunk"]["z"][:n_tokens, :n_tokens].half().cpu().numpy(),
            }

        if return_distogram and "distogram" in out:
            dgram_out = out["distogram"]
            distogram = {
                "logits": dgram_out["logits"][:n_tokens, :n_tokens].half().cpu().numpy(),
                "bin_edges": out["distogram"]["bin_boundaries"].float().cpu().numpy(),
                "asym_ids": f_input.token.asym_id[:n_tokens].int().cpu().numpy(),
                "res_ids": f_input.token.residue_index[:n_tokens].int().cpu().numpy(),
            }
        else:
            distogram = None

        trajectory = None
        if return_trajectory:
            trajectory = out["diffusion"]["traj"][:, :, :n_atoms].float().cpu().numpy()

        # Package predictions with their query, reference structure, and candidates.
        result = FoldingResult(
            query=query,
            seed=seed,
            structure=struct,
            coordinates=coords,
            confidence_summary=summaries,
            confidence_scores=scores,
            apos=item.apos,
            priors=item.priors,
            settings={
                "version": __version__,
                "seed": seed,
                "num_samples": num_samples,
                "num_recycles": num_recycles,
                "num_steps": num_steps,
                "use_struct_encoder": self.model.prot_struct_encoder is not None,
                "use_rna_encoder": self.model.rna_seq_encoder is not None,
                "cpu_offload": self.model.cpu_offload,
            },
            embeddings=embeddings,
            distogram=distogram,
            trajectory=trajectory,
        )
        return result
