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

"""Shared structure loading, conditioning, sampling, and cropping for training."""

import dataclasses
from pathlib import Path

import msgpack
import numpy as np

import kfold.constants as C
from kfold.data.pipelines import featurization, prior_sampling, tokenization
from kfold.data.types.ccd import CCD
from kfold.data.types.metadata import Metadata
from kfold.data.types.model_input import FoldingInput
from kfold.data.types.structure import RefStructure
from kfold.data.types.tokenized import TokenizedStructure
from kfold.training.dataset.cropper import BaseCropper
from kfold.training.dataset.sampler import BaseSampler, Sample
from kfold.training.dataset.utils import apo_perturbation, pre_crop
from kfold.training.dataset.utils.apo_io import (
    unpack_apo_multimer_record,
    unpack_apo_multimer_token_record,
    unpack_prior_multimer_stack_record,
)
from kfold.training.utils.permutation_alignment.symmetry import get_symmetries
from kfold.utils.registry import Registry

from .base import BaseLMDBDataset, DatasetConfig

StructInfo = dict


@dataclasses.dataclass(kw_only=True)
class TrainingDatasetConfig(DatasetConfig):
    """Configuration for training dataset.

    Attributes
    ----------
    type: str
        Type of the training dataset: "rcsb" or "distillation".
    weight : float
        Weight of the dataset during training.
    prob_drop_apo : float
        Probability of dropping apo structure for the entire complex.
    prob_drop_struct_token : float
        Probability of dropping structure tokens for the entire complex.
    sampler : BaseSampler.Config | None
        Sampler configuration for generating samples.
    cropper : BaseCropper.Config | None
        Cropper configuration for cropping structures.
    """

    # default
    name: str
    data_path: str | Path | None = None
    manifest_path: str | Path | None = None
    seed: int | None = None
    prob_perturbation: float = 0.9
    apo_perturb: apo_perturbation.ApoPerturbationConfig | None = None
    # train only
    type: str
    weight: float = 1.0
    prob_drop_apo: float = 0.0  # probability of dropping apo structure
    prob_drop_struct_token: float = 0.0  # probability of dropping structure tokens
    sampler: BaseSampler.Config | None
    cropper: BaseCropper.Config | None


# === Training Dataset === #
class TrainingDataset(BaseLMDBDataset):
    """Training dataset with AF3-style sampling and cropping."""

    def __init__(
        self,
        config: TrainingDatasetConfig,
        ccd: CCD,
        tokenizer: tokenization.Tokenizer,
        featurizer: featurization.InputFeaturizer,
        prior_sampler: prior_sampling.PriorSampler | None,
        safe_load: bool,
        max_chains: int,
        max_apo: int,
        max_tokens: int,
        max_atoms: int,
        max_sequence_tokens: int,
    ) -> None:
        """
        Parameters
        ----------
        config : TrainingDatasetConfig
            Dataset configuration.
        ccd: CCD
            CCD database
        tokenizer: tokenization.Tokenizer
            Tokenizer for tokenizing structures.
        featurizer: featurization.InputFeaturizer
            Featurizer for featurizing tokenized structures.
        prior_sampler: prior_sampling.PriorSampler | None
            Prior sampler for sampling prior coordinates (optional).
        max_chains : int
            Maximum number of chains per sample.
        max_tokens : int
            Maximum number of tokens per sample.
        max_atoms : int
            Maximum number of atoms per sample.
        max_sequence_tokens : int
            Maximum number of sequence tokens per sample,
            limiting the entire input size of PLM module.
        Notes
        -----
        This dataset implements AF3-style sampling (chain/interface-based).
        1. Samples are generated based on chains/interfaces in the structures.
        2. During data loading, samples are cropped to fit within `max_tokens`
           using the provided `cropper`.
        """
        super().__init__(
            config,
            ccd,
            tokenizer,
            featurizer,
            prior_sampler,
            safe_load=safe_load,
            train=True,
            max_apo=max_apo,
        )
        self.config: TrainingDatasetConfig = config
        if self.seed is not None:
            # Warn about fixed seed affecting randomness
            self.logger.warning(
                "Seed is set for TrainingDataset, which may affect randomness."
            )

        # For pre-cropping (RefStructure)
        self.max_chains: int = max_chains
        self.max_apo: int = max_apo
        # For main cropping (TokenizedStructure)
        self.max_tokens: int = max_tokens
        self.max_atoms: int = max_atoms
        self.max_sequence_tokens: int = max_sequence_tokens

        assert max_sequence_tokens >= max_tokens + (max_chains * 2), (
            f"max_sequence_tokens should be greater than max_tokens to accommodate "
            f"additional sequence tokens for PLM input."
            f" (max_sequence_tokens={max_sequence_tokens}, max_tokens={max_tokens}, "
            f"max_chains={max_chains})"
        )  # +2 tokens per chain for [CLS] and [SEP]

        assert config.cropper is not None, "Cropper config must be provided."
        self.cropper: BaseCropper = Registry.instantiate(config.cropper)

        if config.sampler is None:
            # Uniform sampling (complex-level)
            self.sampler: BaseSampler = BaseSampler()
        else:
            # AF3-style sampling (chain/interface-based)
            self.sampler: BaseSampler = Registry.instantiate(config.sampler)

        samples, weights = self.sampler.get_samples(self.metadatas)
        self.samples: list[Sample] = samples
        self.weights: np.ndarray = weights

    def load_lookup_table(self) -> dict:
        """Load optional external apo mappings for training."""
        if not (self.data_root / "apo_lookup.msgpack").exists():
            return {}
        return super().load_lookup_table()

    def setup(self) -> None:
        """Load multimer apo/prior lookup products."""
        self.prob_use_complex_apo = self.config.prob_use_complex_apo
        self.prob_use_complex_prior = self.config.prob_use_complex_prior
        self.apo_multimer_lookup_table = self.load_apo_multimer_lookup_table()

    def load_apo_multimer_lookup_table(self) -> dict:
        lookup_path = self.data_root / "apo_multimer_lookup.msgpack"
        if not lookup_path.exists():
            return {}
        with open(lookup_path, "rb") as f:
            return msgpack.unpack(f, raw=False)

    def _get_apo_multimer_source_lmdb_env(self, chain_type: str, source: str):
        assert chain_type == "protein"
        return self._get_source_lmdb_env(
            "_apo_source_lmdb_envs", "apo_lmdb", "protein-multimer", source
        )

    def _load_apo_multimer_info_from_lmdb(
        self,
        apo_info: dict,
        *,
        context: str,
    ) -> dict:
        """Attach multimer chain payloads from source-specific apo multimer LMDB."""
        loaded = apo_info.copy()
        source = loaded["source"]
        chain_type = loaded["chain_type"]
        lmdb_key = loaded["name"]

        env = self._get_apo_multimer_source_lmdb_env(chain_type, source)
        with env.begin(write=False) as txn:
            value_bytes = txn.get(lmdb_key.encode("utf-8"))
        if value_bytes is None:
            raise KeyError(f"Apo multimer '{source}:{lmdb_key}' not found for {context}")

        loaded["key"] = f"{source}:{lmdb_key}"
        loaded["lmdb_key"] = lmdb_key
        loaded["chains"] = unpack_apo_multimer_record(value_bytes)
        return loaded

    def _load_multimer_prior_candidates_from_lmdb(
        self,
        group: dict,
    ) -> dict[int, np.ndarray] | None:
        """Load all candidate coordinates for one protein multimer prior."""
        assert group["chain_type"] == "protein"
        name = group["name"]
        records = list(
            self._load_prior_records(
                "protein-multimer", name, unpack_prior_multimer_stack_record
            )
        )
        if not records:
            return None
        candidates_by_asym_id: dict[int, list[np.ndarray]] = {}
        for record in records:
            for asym_id, chain in record["chains"].items():
                coords = chain["coords"]
                if coords.ndim != 4 or coords.shape[0] == 0:
                    raise ValueError(
                        f"Prior multimer stack {name}/{asym_id} has shape "
                        f"{coords.shape}; expected (N, L, A, 3) with N > 0."
                    )
                candidates_by_asym_id.setdefault(int(asym_id), []).append(coords)

        return {
            asym_id: np.concatenate(candidates, axis=0)
            for asym_id, candidates in candidates_by_asym_id.items()
        }

    def get_apo_lookup(
        self, ref_struct: RefStructure, rng: np.random.Generator
    ) -> dict[int, list[dict | None]]:
        """Get monomer apo lookup, then replace selected entities with multimer apo."""
        num_apo = self.sample_num_apo(rng)
        apo_lookup = self.get_monomer_apo_lookup(ref_struct, rng, num_apo)
        entry_id = ref_struct.id
        multimer_groups: list[dict] = self.apo_multimer_lookup_table.get(entry_id, [])
        if not multimer_groups or rng.random() >= self.prob_use_complex_apo:
            return apo_lookup

        metadata_by_asym_id = {c.asym_id: c for c in ref_struct.metadata.chains}
        protein_asym_ids = {c.asym_id for c in ref_struct.chains if c.ctype.is_protein}

        # Multiple records with the same identity are alternative sources for one
        # multimer. Select one identity first so a chain never mixes monomer and
        # multimer coordinates across apo slots.
        candidates_by_identity: dict[tuple[str, int, tuple[int, ...]], list[dict]] = {}
        for group in multimer_groups:
            identity = (
                group["name"],
                int(group["apo_uid"]),
                tuple(int(asym_id) for asym_id in group["asym_ids"]),
            )
            candidates_by_identity.setdefault(identity, []).append(group)

        valid_candidates: list[tuple[tuple[int, ...], int, list[dict]]] = []
        for (_, apo_uid, group_asym_ids), candidates in candidates_by_identity.items():
            active_asym_ids = [
                asym_id for asym_id in group_asym_ids if asym_id in protein_asym_ids
            ]
            if not active_asym_ids:
                continue

            if len(set(active_asym_ids)) != len(active_asym_ids):
                self.logger.warning(
                    f"Skipping multimer apo group {entry_id}:{apo_uid} because it "
                    f"contains duplicate active asym_ids: {active_asym_ids}."
                )
                continue

            active_entity_ids = [
                metadata_by_asym_id[asym_id].entity_id for asym_id in active_asym_ids
            ]
            if len(set(active_entity_ids)) != len(active_entity_ids):
                self.logger.warning(
                    f"Skipping multimer apo group {entry_id}:{apo_uid} because its "
                    f"active chains share an entity: asym_ids={active_asym_ids}, "
                    f"entity_ids={active_entity_ids}."
                )
                continue

            valid_candidates.append((tuple(active_asym_ids), apo_uid, candidates))

        if not valid_candidates:
            return apo_lookup

        candidate_i = (
            int(rng.integers(len(valid_candidates))) if len(valid_candidates) > 1 else 0
        )
        active_asym_ids, apo_uid, candidates = valid_candidates[candidate_i]
        num_multimer = min(num_apo, len(candidates))
        selected_source_indices = rng.choice(
            len(candidates), size=num_multimer, replace=False
        )

        active_entity_ids = {
            metadata_by_asym_id[asym_id].entity_id for asym_id in active_asym_ids
        }
        # Reuse the selected coordinates across entity copies while preserving
        # each physical multimer pair's rigid-group ID from the lookup.
        apo_uid_by_asym_id = {asym_id: apo_uid for asym_id in active_asym_ids}
        for candidate_asym_ids, candidate_apo_uid, _ in valid_candidates:
            candidate_entity_ids = {
                metadata_by_asym_id[asym_id].entity_id for asym_id in candidate_asym_ids
            }
            if candidate_entity_ids != active_entity_ids:
                continue
            # Keep the selected identity when an alternative record assigns a
            # different UID to any of the same physical chains.
            if any(
                asym_id in apo_uid_by_asym_id
                and apo_uid_by_asym_id[asym_id] != candidate_apo_uid
                for asym_id in candidate_asym_ids
            ):
                continue
            apo_uid_by_asym_id.update(
                dict.fromkeys(candidate_asym_ids, candidate_apo_uid)
            )

        selected_by_entity: dict[int, list[dict | None]] = {
            entity_id: [None] * self.max_apo for entity_id in active_entity_ids
        }

        for apo_i, source_i in enumerate(selected_source_indices):
            group = candidates[int(source_i)]
            context = f"multimer group {entry_id}:{apo_uid}"
            multimer_info = self._load_apo_multimer_info_from_lmdb(group, context=context)
            multimer_chains: dict[int, dict] = multimer_info["chains"]
            for asym_id in active_asym_ids:
                if asym_id not in multimer_chains:
                    raise KeyError(
                        f"Apo multimer {multimer_info['key']} for entry {entry_id} "
                        f"does not contain asym_id {asym_id}."
                    )
                entity_id = metadata_by_asym_id[asym_id].entity_id
                loaded = group.copy()
                loaded.update(multimer_chains[asym_id])
                loaded["key"] = multimer_info["key"]
                loaded["lmdb_key"] = multimer_info["lmdb_key"]
                loaded["multimer_key"] = multimer_info["key"]
                loaded["source_asym_id"] = asym_id
                loaded["apo_uid"] = apo_uid
                loaded["is_multimer_apo"] = True
                selected_by_entity[entity_id][apo_i] = loaded

        for chain in ref_struct.chains:
            if not chain.ctype.is_protein or chain.entity_id not in selected_by_entity:
                continue
            chain_metadata = metadata_by_asym_id[chain.asym_id]
            chain_apo_uid = apo_uid_by_asym_id.get(
                chain.asym_id, int(chain_metadata.apo_uid)
            )
            apo_lookup[chain.asym_id] = [
                None
                if loaded is None
                else {
                    **loaded,
                    "chain_key": f"{entry_id}_{chain.asym_id}",
                    "apo_uid": chain_apo_uid,
                }
                for loaded in selected_by_entity[chain.entity_id]
            ]
            chain_metadata.apo_uid = chain_apo_uid

        return apo_lookup

    def get_prior_candidates(
        self,
        ref_struct: RefStructure,
        rng: np.random.Generator,
    ) -> dict[int, np.ndarray]:
        """Return monomer candidates with an optional multimer overlay."""
        prior_candidates = super().get_prior_candidates(ref_struct, rng)
        entry_id = ref_struct.id
        multimer_groups: list[dict] = self.apo_multimer_lookup_table.get(entry_id, [])
        if not multimer_groups or rng.random() >= self.prob_use_complex_prior:
            return prior_candidates

        multimer_candidates_by_name: dict[str, dict[int, np.ndarray] | None] = {}
        metadata_by_asym_id = {c.asym_id: c for c in ref_struct.metadata.chains}
        protein_asym_ids = {c.asym_id for c in ref_struct.chains if c.ctype.is_protein}

        for group in multimer_groups:
            name = group["name"]
            if name in multimer_candidates_by_name:
                continue
            prior_uid = int(group["apo_uid"])
            group_asym_ids = [int(aid) for aid in group["asym_ids"]]
            active_asym_ids = [
                asym_id for asym_id in group_asym_ids if asym_id in protein_asym_ids
            ]
            if not active_asym_ids:
                continue

            multimer_candidates_by_name[name] = (
                self._load_multimer_prior_candidates_from_lmdb(group)
            )
            multimer_candidates = multimer_candidates_by_name[name]
            if multimer_candidates is None:
                continue

            for asym_id in active_asym_ids:
                if asym_id not in multimer_candidates:
                    raise KeyError(
                        f"Prior multimer {name} for "
                        f"entry {entry_id} does not contain asym_id {asym_id}."
                    )
                prior_candidates[asym_id] = multimer_candidates[asym_id]
                if asym_id in metadata_by_asym_id:
                    metadata_by_asym_id[asym_id].prior_uid = prior_uid

        return prior_candidates

    def populate_structure_tokens(
        self,
        tokenized: TokenizedStructure,
        apo_lookup: dict[int, list[dict | None]],
    ) -> None:
        """Populate monomer and multimer apo structure tokens."""
        monomer_apo_lookup = {
            asym_id: [
                None
                if apo_info is not None and apo_info.get("is_multimer_apo", False)
                else apo_info
                for apo_info in apo_infos
            ]
            for asym_id, apo_infos in apo_lookup.items()
        }
        super().populate_structure_tokens(tokenized, monomer_apo_lookup)

        multimer_token_cache: dict[tuple[str, str], dict[int, np.ndarray]] = {}
        missing_token_sources: set[str] = set()
        missing_token_records: set[tuple[str, str]] = set()
        for c_i in range(tokenized.num_chains):
            if tokenized.chain.chain_type[c_i] != C.ChainType.PROTEIN.value:
                continue

            asym_id = int(tokenized.chain.asym_id[c_i])
            ek = f"{tokenized.id}:{asym_id}"
            for apo_i, apo_info in enumerate(apo_lookup.get(asym_id, [])):
                if apo_info is None or not apo_info.get("is_multimer_apo", False):
                    continue

                source = apo_info["source"]
                key = apo_info["name"]
                cache_key = (source, key)
                if cache_key not in multimer_token_cache:
                    if (
                        source in missing_token_sources
                        or cache_key in missing_token_records
                    ):
                        continue
                    token_lmdb_path = self._source_lmdb_path(
                        "apo_tok_lmdb", "protein-multimer", source
                    )
                    if not token_lmdb_path.exists():
                        self.logger.warning(
                            f"Apo multimer structure-token LMDB for {source} "
                            "does not exist. Skipping structure tokens for this source."
                        )
                        missing_token_sources.add(source)
                        continue
                    env = self._get_apo_tok_source_lmdb_env("protein-multimer", source)
                    with env.begin(write=False) as txn:
                        value = txn.get(key.encode("utf-8"))
                    if value is None:
                        self.logger.warning(
                            f"Apo multimer structure tokens {source}:{key} not found "
                            f"in LMDB for chain `{ek}`. Skipping this entry"
                        )
                        missing_token_records.add(cache_key)
                        continue
                    multimer_token_cache[cache_key] = unpack_apo_multimer_token_record(
                        value
                    )
                source_asym_id = int(apo_info["source_asym_id"])
                if source_asym_id not in multimer_token_cache[cache_key]:
                    self.logger.warning(
                        f"Apo multimer structure tokens {source}:{key} do not contain "
                        f"asym_id {source_asym_id}. Skipping chain `{ek}`."
                    )
                    continue
                self._insert_structure_tokens(
                    tokenized,
                    c_i,
                    apo_i,
                    multimer_token_cache[cache_key][source_asym_id],
                    key=f"{source}:{key}",
                )

    def sanity_check(self) -> None:
        """Perform sanity checks on the dataset."""
        cfg = self.config
        if cfg.apo_perturb is None:
            self.logger.warning("Apo perturbation is disabled for training set.")

    def __len__(self) -> int:
        return len(self.samples)

    def get_item_safe(
        self, index: int, num_trials: int = 100
    ) -> tuple[FoldingInput, StructInfo]:
        """Get the folding input for the given index, with retry on failure.
        NOTE: This is overridden to use `self.samples` instead of `self.metadatas`.
        """
        trials = []
        for _ in range(num_trials):
            sample = self.samples[index]
            metadata = Metadata.from_dict(sample.metadata)
            try:
                return self.get_item(metadata, asym_ids=sample.asym_id)
            except (KeyboardInterrupt, SystemExit) as e:
                raise e
            except Exception as e:
                sample_id = sample.metadata["id"]
                self.logger.error(
                    f"Error loading index {sample_id}({index}): {e}. Retrying..."
                )
                if not self.safe_load:
                    raise e
                index = np.random.randint(0, len(self))
                trials.append(sample)
        raise RuntimeError(
            f"Failed to load data after {num_trials} attempts. Tried: {trials}"
        )

    def get_item(self, metadata: Metadata, **kwargs) -> tuple[FoldingInput, StructInfo]:
        """Get the folding input for the given sample."""
        metadata_id: str = metadata.id

        # Initialize random number generator (create new rng based on metadata_id)
        rng = np.random.default_rng()

        # Load structure (NOTE: ref_struct.metadata == metadata)
        ref_struct: RefStructure = self.load_ref_structure(metadata)

        # Sub-complex structure extraction for large complex (>20 chains)
        # This is the on-the-fly pipeline of AlphaFold3 SI Section 2.5.4
        ref_struct = self.extract_substructure(ref_struct, rng=rng, **kwargs)
        metadata = ref_struct.metadata  # update metadata after extraction

        # Get apo lookup for the structure
        apo_lookup = self.get_apo_lookup(ref_struct, rng)

        # Fetch apo structure
        apo_dict = self.fetch_apo_structures(ref_struct, apo_lookup, rng)

        # Sample prior coordinates for diffusion bridge model.
        prior_coords = self.sample_prior_coords(ref_struct, rng)

        # Tokenization
        tokenized = self.tokenize(ref_struct, apo_dict, prior_coords, rng)

        # Populate structure tokens for apo structure (in-place)
        self.populate_structure_tokens(tokenized, apo_lookup)

        # Optionally drop apo structure for trunk input during training
        self.drop_apo_structure(tokenized, rng)

        # Cropping
        cropped = self.crop_structure(tokenized, metadata, rng=rng, **kwargs)

        # Featurization
        f_input = self.featurize(cropped)

        # Pad the features.
        f_input = self.pad_input(f_input)

        # Determine whether to train confidence head
        train_confidence = self.determine_confidence_train_data(metadata)

        struct_info = {
            "id": metadata_id,
            "structure": ref_struct,
            "symmetry": get_symmetries(ref_struct, self.ccd),
            "train_confidence_head": train_confidence,
        }
        return f_input, struct_info

    def pad_input(self, f_input: FoldingInput) -> FoldingInput:
        max_chains = self.max_chains
        max_tokens = self.max_tokens
        max_atoms = self.max_atoms
        max_sequence_tokens = self.max_sequence_tokens
        max_bonds = max_tokens * 10  # max 10 bonds per token
        return f_input.pad(
            max_tokens=max_tokens,
            max_chains=max_chains,
            max_atoms=max_atoms,
            max_bonds=max_bonds,
            max_sequence_tokens=max_sequence_tokens,
        )

    def tokenize(
        self,
        ref_struct: RefStructure,
        apo_dict: dict[int, np.ndarray],
        prior_coords: np.ndarray,
        rng: np.random.Generator,
    ) -> TokenizedStructure:
        """Tokenize the given structure."""
        return self.tokenizer(
            ref_struct,
            rng,
            apo_coords=apo_dict,
            num_apo=self.max_apo,
            prior_coords=prior_coords,
        )

    # === Utility methods for training dataset === #
    def extract_substructure(
        self,
        ref_struct: RefStructure,
        rng: np.random.Generator,
        **kwargs,
    ) -> RefStructure:
        assert "asym_ids" in kwargs, "asym_ids must be provided for cropping."
        asym_ids: int | tuple[int, int] | None = kwargs["asym_ids"]
        if self.max_chains < ref_struct.num_chains:
            # Get sub-complex with limited number of chains
            ref_struct = pre_crop.extract_substructure(
                ref_struct,
                max_chains=self.max_chains,
                bias_asym_id=asym_ids,
                rng=rng,
            )
        return ref_struct

    def crop_structure(
        self,
        tokenized: TokenizedStructure,
        metadata: Metadata,
        rng: np.random.Generator,
        **kwargs,
    ) -> TokenizedStructure:
        assert "asym_ids" in kwargs, "asym_ids must be provided for cropping."
        asym_ids: int | tuple[int, int] | None = kwargs["asym_ids"]
        if self.max_tokens < tokenized.num_tokens or self.max_atoms < tokenized.num_atoms:
            # Crop the tokenized structure
            tokenized = self.cropper.crop(
                tokenized,
                metadata,
                max_tokens=self.max_tokens,
                max_atoms=self.max_atoms,
                max_sequence_tokens=self.max_sequence_tokens,
                bias_asym_id=asym_ids,
                rng=rng,
            )
        return tokenized

    def drop_apo_structure(
        self, tokenized: TokenizedStructure, rng: np.random.Generator
    ) -> None:
        """Drop complex-level apo conditioning with hierarchical masking."""
        drop_apo = rng.random() < self.config.prob_drop_apo
        if drop_apo:
            tokenized.token.apo_center_coords.fill(np.nan)
            tokenized.token.apo_repr_coords.fill(np.nan)
            tokenized.token.apo_frame_coords.fill(np.nan)
            tokenized.token.apo_center_mask.fill(False)
            tokenized.token.apo_repr_mask.fill(False)
            tokenized.token.apo_frame_mask.fill(False)
            tokenized.atom.apo_coords.fill(np.nan)
            tokenized.atom.apo_mask.fill(False)

        drop_struct_token = drop_apo or (
            rng.random() < self.config.prob_drop_struct_token
        )
        if drop_struct_token:
            tokenized.sequence.bb_struct_token_id.fill(-1)
            tokenized.sequence.fa_struct_token_id.fill(-1)

    def determine_confidence_train_data(self, metadata: Metadata) -> bool:
        return False
