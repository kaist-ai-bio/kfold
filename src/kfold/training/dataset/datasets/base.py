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

"""
Dataset structures:

(Shared across datasets)
ccd-train.pkl

(For each dataset)
rcsb-train/
    manifest.json
    structure.lmdb
    apo_lmdb/{chain_type}/{source}.lmdb
    apo_tok_lmdb/protein/{source}.lmdb
    prior_lmdb/{chain_type}/{source}.lmdb
    apo_lookup.json     # mapping from each chain to apo structure(s).
af2-long/ ...           # simple dataset with AF2 structures.
    manifest.json
    structure.lmdb/
rcsb-val/ ...


* apo_lookup.json format
```json
{
  "6oim": {
    "1": [
      {
        "source": "esmfold"
        "name": "uniq_protein_000020-esmfold",
        "residue_map": "1:250->1:250",
      },
      {
        "source": "afdb"
        "name": "AF-P01116-F1-model_v6",
        "residue_map": "1:235->11:245",
      },
      {
        "source": "pdb"
        "name": "51d6-A",
        "residue_map": "5:250->5:250",
      }
    ],
    "2": [...]
  },
  "1a2c": {...}
}
```
"""

import dataclasses
import io
import logging
from pathlib import Path

import lmdb
import msgpack
import numpy as np
import torch

import kfold.constants as C
from kfold.data.pipelines import featurization, prior_sampling, tokenization
from kfold.data.types.ccd import CCD
from kfold.data.types.metadata import Metadata
from kfold.data.types.model_input import FoldingInput
from kfold.data.types.structure import Chain, RefStructure
from kfold.data.types.tokenized import TokenizedStructure
from kfold.training.dataset.utils import apo_io, apo_perturbation
from kfold.training.utils.permutation_alignment.symmetry import get_symmetries
from kfold.utils.geometry.random_augment import do_centering
from kfold.utils.misc import hash_seq

# Type alias
StructInfo = dict


# === Helper functions === #
def _open_lmdb(lmdb_path: str | Path) -> lmdb.Environment:
    if not Path(lmdb_path).exists():
        raise FileNotFoundError(f"LMDB file {lmdb_path} not found.")
    return lmdb.open(
        str(lmdb_path), readonly=True, lock=False, readahead=False, meminit=False
    )


def parse_residue_map(residue_map: str) -> tuple[int, int, int, int]:
    """Parse residue map string into start and end indices.
    Example: "1:100->5:104" -> (0, 100, 4, 104)
    """
    res_range, apo_range = residue_map.split("->")
    res_st, res_end = map(int, res_range.split(":"))
    apo_st, apo_end = map(int, apo_range.split(":"))
    if (res_end - res_st) != (apo_end - apo_st):
        return -1, -1, -1, -1  # invalid mapping
    # Convert to 0-based indexing
    # 1:100 means residues 1 to 100 inclusive -> coords[0:100]
    return res_st - 1, res_end, apo_st - 1, apo_end


# === Dataset Classes === #
@dataclasses.dataclass(kw_only=True)
class DatasetConfig:
    """Base configuration for dataset.

    Attributes
    ----------
    name : str
        Name of the dataset.
    data_path : str | Path
        Path to the dataset directory.
    manifest_path : str | Path | None
        Optional path to the custom manifest file.
    seed : int | None
        Random seed for data loading.
    prob_use_complex_apo : float
        Multimer apo selection probability for datasets that provide complex records.
    prob_use_complex_prior : float
        Multimer prior selection probability for datasets that provide complex records.
    """

    name: str
    data_path: str | Path | None = None
    manifest_path: str | Path | None = None
    seed: int | None = None
    prob_perturbation: float = 0.0
    apo_perturb: apo_perturbation.ApoPerturbationConfig | None = None
    prob_use_complex_apo: float = 0.0
    prob_use_complex_prior: float = 0.0


def next_multiple(n: int, divisor: int) -> int:
    """Return the next integer greater than or equal to n that is divisible by divisor."""
    return ((n + divisor - 1) // divisor) * divisor


class BaseLMDBDataset(torch.utils.data.Dataset):
    """A dataset that safely retries loading data on failure."""

    def __init__(
        self,
        config: DatasetConfig,
        ccd: CCD,
        tokenizer: tokenization.Tokenizer,
        featurizer: featurization.InputFeaturizer,
        prior_sampler: prior_sampling.PriorSampler | None,
        safe_load: bool,
        train: bool,
        max_apo: int = 1,
    ) -> None:
        """
        Parameters
        ----------
        config : DatasetConfig
            Dataset configuration.
        ccd: CCD
            CCD database
        tokenizer: tokenization.Tokenizer
            Tokenizer for tokenizing structures.
        featurizer: featurization.InputFeaturizer
            Featurizer for featurizing tokenized structures.
        prior_sampler: prior_sampling.PriorSampler | None
            Prior sampler for sampling prior coordinates (optional).
        safe_load : bool
            Whether to retry loading on failure.
        """
        # === Initialize parameters === #
        assert config.data_path is not None, "DatasetConfig.data_path must be specified."
        self.config: DatasetConfig = config
        self.name: str = config.name
        self.data_root: Path = Path(config.data_path)
        self.seed: int | None = config.seed
        self.safe_load: bool = safe_load
        self.train: bool = train
        assert max_apo >= 1, "max_apo must be positive."
        if not train:
            assert max_apo == 1, "Validation supports exactly one apo structure."
        self.max_apo: int = max_apo

        if train:
            self.logger = logging.getLogger(f"[Training Dataset:{self.name}]")
        else:
            self.logger = logging.getLogger(f"[Validation Dataset:{self.name}]")

        # Sanity check on dataset files and configurations
        self.sanity_check()

        # === Load dataset components === #
        # CCD (shared across datasets)
        self.ccd: CCD = ccd

        # Metadata list
        # NOTE: For AFDB distillation, which includes a lot of samples,
        # we keep a dict instead of Metadata obj to save memory.
        self.metadatas: list[dict] = self.load_manifest(
            custom_manifest=config.manifest_path  # optional custom manifest path
        )

        # Lookup table
        self.lookup_table: dict = self.load_lookup_table()

        # === Initialize modules === #
        self.tokenizer: tokenization.Tokenizer = tokenizer
        self.featurizer: featurization.InputFeaturizer = featurizer
        self.prior_sampler: prior_sampling.PriorSampler | None = prior_sampler
        self.num_priors: int = 5  # default number of prior samples

        # Data augmentation
        self.apo_perturb: apo_perturbation.ApoPerturbation | None = None
        if config.apo_perturb is not None:
            self.apo_perturb: apo_perturbation.ApoPerturbation = (
                apo_perturbation.ApoPerturbation(config.apo_perturb)
            )

        self.setup()

    def __len__(self) -> int:
        return len(self.metadatas)

    # === Setup === #
    def load_manifest(self, custom_manifest: str | Path | None = None) -> list[dict]:
        if custom_manifest is not None:
            manifest_path = Path(custom_manifest)
        else:
            manifest_path = self.data_root / "manifest.msgpack"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest file {manifest_path} not found.")
        if manifest_path.suffix != ".msgpack":
            raise ValueError(f"Manifest must be msgpack: {manifest_path}")
        with open(manifest_path, "rb") as f:
            return msgpack.unpack(f)

    def load_lookup_table(self) -> dict:
        lookup_path = self.data_root / "apo_lookup.msgpack"
        if not lookup_path.exists():
            raise FileNotFoundError(f"Apo lookup file {lookup_path} not found.")
        with open(lookup_path, "rb") as f:
            lookup_table: dict = msgpack.unpack(f, raw=False)

        # RCSB preprocessing writes entity IDs as msgpack/json-compatible strings.
        # Runtime code indexes them by integer entity_id from RefStructure chains.
        for entry_id, entry_lookup in lookup_table.items():
            lookup_table[entry_id] = {
                int(eid): list(infos) for eid, infos in entry_lookup.items()
            }
        return lookup_table

    def sanity_check(self) -> None:
        """Perform sanity checks on the dataset."""
        pass

    def setup(self) -> None:
        """Additional setup for subclasses."""
        pass

    @property
    def lmdb_env(self) -> lmdb.Environment:
        if not hasattr(self, "_lmdb_env"):
            self._lmdb_env = _open_lmdb(self.data_root / "structure.lmdb")
        return self._lmdb_env

    def _source_lmdb_path(self, root_name: str, chain_type: str, source: str) -> Path:
        """Prefer current stores, with exact aliases for archived multimers."""
        path = self.data_root / root_name / chain_type / f"{source}.lmdb"
        if not path.exists() and chain_type == "protein-multimer":
            legacy_roots = {
                "apo_lmdb": ("apo_multimer_lmdb", "protein"),
                "apo_tok_lmdb": ("apo_tok_lmdb", "protein_multimer"),
            }
            if root_name in legacy_roots:
                root, kind = legacy_roots[root_name]
                legacy = self.data_root / root / kind / f"{source}.lmdb"
                if legacy.exists():
                    return legacy
        return path

    def _flat_prior_lmdb_path(self, kind: str) -> Path:
        path = self.data_root / "prior_lmdb" / f"{kind}.lmdb"
        if not path.exists() and kind == "protein-multimer":
            legacy = self.data_root / "prior_multimer_lmdb" / "protein.lmdb"
            if legacy.exists():
                return legacy
        return path

    def _get_source_lmdb_env(
        self,
        cache_name: str,
        root_name: str,
        chain_type: str,
        source: str,
    ) -> lmdb.Environment:
        cache = getattr(self, cache_name, None)
        if cache is None:
            cache = {}
            setattr(self, cache_name, cache)
        cache_key = (chain_type, source)
        if cache_key not in cache:
            # Workers open LMDB handles lazily after fork; do not share handles
            # from the parent dataset construction path.
            cache[cache_key] = _open_lmdb(
                self._source_lmdb_path(root_name, chain_type, source)
            )
        return cache[cache_key]

    def _get_apo_source_lmdb_env(self, chain_type: str, source: str) -> lmdb.Environment:
        assert chain_type == "protein"
        return self._get_source_lmdb_env(
            "_apo_source_lmdb_envs", "apo_lmdb", chain_type, source
        )

    def _get_apo_tok_source_lmdb_env(
        self, chain_type: str, source: str
    ) -> lmdb.Environment:
        assert chain_type in ("protein", "protein-multimer")
        return self._get_source_lmdb_env(
            "_apo_tok_source_lmdb_envs", "apo_tok_lmdb", chain_type, source
        )

    def _load_prior_records(self, kind: str, key: str, unpack):
        """Read matching stacks from every source under a prepared input kind."""
        if not hasattr(self, "_prior_sources"):
            self._prior_sources = {}
        if kind not in self._prior_sources:
            self._prior_sources[kind] = [
                path.stem
                for path in sorted((self.data_root / "prior_lmdb" / kind).glob("*.lmdb"))
            ]
            # The shipped training archive stores one stacked LMDB per kind.
            # Prefer source-split stores when present; never count both layouts.
            legacy = self._flat_prior_lmdb_path(kind)
            if not self._prior_sources[kind] and legacy.exists():
                self._prior_sources[kind] = [None]
        for source in self._prior_sources[kind]:
            if source is None:
                if not hasattr(self, "_prior_stack_lmdb_envs"):
                    self._prior_stack_lmdb_envs = {}
                cache = self._prior_stack_lmdb_envs
                if (kind, None) not in cache:
                    cache[(kind, None)] = _open_lmdb(
                        self._flat_prior_lmdb_path(kind)
                    )
                env = cache[(kind, None)]
            else:
                env = self._get_source_lmdb_env(
                    "_prior_stack_lmdb_envs", "prior_lmdb", kind, source
                )
            with env.begin(write=False) as txn:
                value = txn.get(key.encode("utf-8"))
            if value is not None:
                yield unpack(value)

    def __del__(self):
        if hasattr(self, "_prior_stack_lmdb_envs"):
            for env in self._prior_stack_lmdb_envs.values():
                env.close()
        if hasattr(self, "_apo_tok_source_lmdb_envs"):
            for env in self._apo_tok_source_lmdb_envs.values():
                env.close()
        if hasattr(self, "_apo_source_lmdb_envs"):
            for env in self._apo_source_lmdb_envs.values():
                env.close()
        if hasattr(self, "_lmdb_env"):
            self._lmdb_env.close()

    # === Core dataset methods === #
    def __getitem__(self, index: int) -> tuple[FoldingInput, StructInfo]:
        """Load an input, retrying failures only during training."""
        return self.get_item_safe(index, num_trials=100)

    def get_item_safe(
        self, index: int, num_trials: int = 100
    ) -> tuple[FoldingInput, StructInfo]:
        """Load an input; validation never substitutes another sample."""
        if self.seed is not None:
            rng = np.random.default_rng(self.seed + index % (1 << 15))
        else:
            rng = np.random.default_rng()

        trials = []
        for _ in range(num_trials):
            sample: Metadata = Metadata.from_dict(self.metadatas[index])
            try:
                return self.get_item(sample)
            except (KeyboardInterrupt, SystemExit) as e:
                raise e
            except Exception as e:
                sample_id = sample.id
                if not self.train or not self.safe_load:
                    self.logger.error(f"Error loading index {sample_id}({index}): {e}.")
                    raise
                self.logger.error(
                    f"Error loading index {sample_id}({index}): {e}. Retrying..."
                )
                index = int(rng.integers(0, len(self)))
                trials.append(sample_id)
        raise RuntimeError(
            f"Failed to load data after {num_trials} attempts. Tried: {trials}"
        )

    def get_item(self, metadata: Metadata, **kwargs) -> tuple[FoldingInput, StructInfo]:
        """Get the folding input for the given sample."""
        metadata_id: str = metadata.id

        # Initialize random number generator (create new rng based on metadata_id)
        if self.seed is not None:
            offset = int(hash_seq(metadata.id), 16)
            rng = np.random.default_rng((self.seed + offset) % (1 << 32))
        else:
            rng = np.random.default_rng()

        # Load structure (NOTE: ref_struct.metadata == metadata)
        ref_struct: RefStructure = self.load_ref_structure(metadata)

        # Get apo lookup for the structure (asym_id to apo)
        apo_lookup = self.get_apo_lookup(ref_struct, rng)

        # Fetch apo structure
        apo_dict = self.fetch_apo_structures(ref_struct, apo_lookup, rng)

        # Sample prior coordinates for diffusion bridge model.
        prior_coords = self.sample_prior_coords(ref_struct, rng)

        # Tokenization
        tokenized = self.tokenize(ref_struct, apo_dict, prior_coords, rng)

        # Populate structure tokens for apo structure (in-place)
        # NOTE: For inference, this will be done on-the-fly.
        self.populate_structure_tokens(tokenized, apo_lookup)

        # Featurization
        f_input = self.featurize(tokenized)

        # Pad the features.
        f_input = self.pad_input(f_input)

        struct_info = {
            "id": metadata_id,
            "structure": ref_struct,
            "symmetry": get_symmetries(ref_struct, self.ccd),
            "train_confidence_head": False,
        }

        return f_input, struct_info

    def load_ref_structure(self, metadata: Metadata) -> RefStructure:
        """Get the structure for the given index."""
        name = metadata.id
        key_bytes = name.encode("utf-8")
        with self.lmdb_env.begin(write=False) as txn:
            value_bytes = txn.get(key_bytes)
            if value_bytes is None:
                raise KeyError(f"Record {name} not found in LMDB.")

        # Use io.BytesIO to wrap the raw bytes
        with io.BytesIO(value_bytes) as byte_stream:
            ref_struct = RefStructure.load_npz(byte_stream)

        # NOTE: Validate loaded record matches requested metadata
        # If there is no problem, the cluster ID (for training) and
        # low_homology flag (for validation) will be missing in npz
        ref_metadata = ref_struct.metadata
        assert ref_metadata.id == name, (
            f"Loaded ID {ref_metadata.id} does not match requested ID {name}."
        )
        assert ref_metadata.num_chains == metadata.num_chains, (
            f"Loaded num_chains {ref_metadata.num_chains} does not match "
            f"requested num_chains {metadata.num_chains}."
        )
        assert ref_metadata.num_residues == metadata.num_residues, (
            f"Loaded num_residues {ref_metadata.num_residues} does not match "
            f"requested num_residues {metadata.num_residues}."
        )
        assert ref_metadata.num_interfaces == metadata.num_interfaces, (
            f"Loaded num_interfaces {ref_metadata.num_interfaces} does not match "
            f"requested num_interfaces {metadata.num_interfaces}."
        )
        # Copy metadata (to update cluster_id if needed)
        ref_struct.metadata = metadata.copy()

        return ref_struct

    # === Helper methods for apo structure handling === #
    @staticmethod
    def _center_label_residue_coords(chain: Chain) -> np.ndarray:
        """Return centered label coordinates in residue-major atom order."""
        coords = chain.map_atom_coords_to_residue_coords(chain.atom.coords)
        mask = np.isfinite(coords).all(axis=-1)
        return do_centering(coords, mask=mask, mask_to_zero=False)

    def _load_apo_info_from_lmdb(self, apo_info: dict) -> bool:
        """Attach `seq` and `coords` from source-specific apo LMDB to a lookup record."""
        source = apo_info["source"]
        chain_type = apo_info["chain_type"]
        name = apo_info["name"]
        env = self._get_apo_source_lmdb_env(chain_type, source)
        with env.begin(write=False) as txn:
            v = txn.get(name.encode("utf-8"))
        if v is None:
            apo_key = apo_info["key"]
            entity_key = apo_info["entity_key"]
            self.logger.warning(
                f"Apo structure {apo_key} not found in LMDB for "
                f"entity `{entity_key}`. Skipping this entry"
            )
            return False
        else:
            apo_info.update(apo_io.unpack_apo_record(v))
            return True

    def _load_apo_tok_from_lmdb(self, apo_info: dict) -> bool:
        """Load the pre-computed structure tokens for the apo structure from LMDB."""
        source = apo_info["source"]
        chain_type = apo_info["chain_type"]
        name = apo_info["name"]
        if chain_type != "protein":
            # NOTE: Currently, we only use structure tokens for protein.
            return True

        env = self._get_apo_tok_source_lmdb_env(chain_type, source)
        with env.begin(write=False) as txn:
            v = txn.get(name.encode("utf-8"))
        if v is None:
            apo_key = apo_info["key"]
            entity_key = apo_info["entity_key"]
            self.logger.warning(
                f"Apo structure token {apo_key} not found in LMDB for "
                f"entity `{entity_key}`. Skipping this entry"
            )
            return False
        else:
            # Load pre-computed structure tokens for apo structure from LMDB
            apo_tok = np.frombuffer(v, dtype=np.int16).reshape(2, -1)
            apo_info["tokens"] = apo_tok
            return True

    def _load_prior_stack_info_from_lmdb(
        self,
        entry_id: str,
        entity_id: int,
        chain_type: str,
    ) -> np.ndarray | None:
        """Load an entity-level stacked prior LMDB record."""
        entity_key = f"{entry_id}_{entity_id}"
        stacks = []
        for record in self._load_prior_records(
            chain_type, entity_key, apo_io.unpack_prior_stack_record
        ):
            coords = record["coords"]
            if coords.ndim != 4 or coords.shape[0] == 0:
                raise ValueError(
                    f"Prior stack {entity_key} has shape {coords.shape}; "
                    "expected (N>0, L, A, 3)."
                )
            stacks.append(coords)
        return np.concatenate(stacks, axis=0) if stacks else None

    def sample_num_apo(self, rng: np.random.Generator) -> int:
        """Sample the requested number of apo structures for one example."""
        if not self.train:
            assert self.max_apo == 1
            return 1
        return int(rng.integers(1, self.max_apo + 1))

    def get_monomer_apo_lookup(
        self,
        ref_struct: RefStructure,
        rng: np.random.Generator,
        num_apo: int,
    ) -> dict[int, list[dict | None]]:
        """Sample monomer apo sources once for each protein entity."""
        entry_id: str = ref_struct.id
        if entry_id not in self.lookup_table:
            if any(c.is_protein for c in ref_struct.chains):
                self.logger.warning(
                    f"No apo info found for entry '{entry_id}' in lookup."
                )
            return {
                c.asym_id: [None] * self.max_apo
                for c in ref_struct.chains
                if c.is_protein
            }

        entry_lookup: dict[int, list[dict]] = self.lookup_table[entry_id]

        apo_lookup: dict[int, list[dict | None]] = {}
        selected_by_entity: dict[int, list[dict | None]] = {}
        for c in ref_struct.chains:
            if not c.is_protein:
                # Skip non-protein chains for apo lookup.
                continue

            eid: int = c.entity_id
            ek: str = f"{entry_id}:{eid}"  # For logging purpose

            if eid not in selected_by_entity:
                selected_infos: list[dict | None] = [None] * self.max_apo
                if eid not in entry_lookup:
                    self.logger.warning(
                        f"No apo info found for {c.ctype} entity '{ek}' in lookup."
                    )
                else:
                    # Validation deterministically uses the first source. Training
                    # samples without replacement and keeps empty slots when fewer
                    # sources exist.
                    entity_apo_infos: list[dict] = entry_lookup[eid]
                    num_available = len(entity_apo_infos)
                    if num_available == 0:
                        self.logger.warning(
                            f"No apo info found for {c.ctype} entity '{ek}' in lookup."
                        )
                        continue
                    num_selected = min(num_apo, num_available)
                    if self.train:
                        selected = rng.choice(
                            num_available, size=num_selected, replace=False
                        )
                    else:
                        assert num_apo == 1
                        selected = np.array([0], dtype=np.int64)

                    for apo_i, source_i in enumerate(selected):
                        apo_info = entity_apo_infos[int(source_i)].copy()
                        apo_info["key"] = f"{apo_info['source']}:{apo_info['name']}"
                        apo_info["entity_key"] = ek

                        if not self._load_apo_info_from_lmdb(apo_info):
                            continue
                        _ = self._load_apo_tok_from_lmdb(apo_info)
                        selected_infos[apo_i] = apo_info

                selected_by_entity[eid] = selected_infos

            apo_lookup[c.asym_id] = [
                None
                if apo_info is None
                else {
                    **apo_info,
                    "chain_key": f"{entry_id}_{c.asym_id}",
                    "apo_uid": c.asym_id,
                }
                for apo_info in selected_by_entity[eid]
            ]

        return apo_lookup

    def get_apo_lookup(
        self, ref_struct: RefStructure, rng: np.random.Generator
    ) -> dict[int, list[dict | None]]:
        """Get sampled apo sources for the given reference structure."""
        num_apo = self.sample_num_apo(rng)
        return self.get_monomer_apo_lookup(ref_struct, rng, num_apo)

    def _get_protein_apo_coords(
        self,
        chain: Chain,
        apo_info: dict | None,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Fetch the apo coordinates for a protein chain."""
        assert chain.is_protein

        def fallback() -> np.ndarray:
            return np.full((chain.num_residues, 37, 3), np.nan, dtype=np.float32)

        # If the apo structure is not found, mask the apo input for training.
        # Validation requires an explicit apo structure.
        if apo_info is None:
            if self.train:
                return fallback()
            else:
                raise KeyError(f"Apo structure not found for {chain.asym_id} in lookup.")

        # Get the apo coordinates from the lookup
        key = apo_info["key"]
        seq = apo_info["seq"]
        coords = apo_info["coords"]
        res_map = apo_info.get("residue_map", None)
        if not self.train:
            # NOTE: While res mapping is also allowed for inference,
            # we enforce that there is no missing residues in the apo for simplicity.
            assert res_map is None, (
                f"Residue map {res_map} found for {chain.asym_id} "
                f"in lookup. Residue mapping is only allowed for training."
            )

        # Perturb the apo coordinates
        # NOTE: We only perturb protein apo structures right now.
        if (
            chain.is_protein
            and self.apo_perturb is not None
            and rng.random() < self.config.prob_perturbation
        ):
            coords = self.apo_perturb.run_protein_perturbation(
                seq, coords, mask=None, rng=rng
            )

        # Map the apo coordinates to the chain's coordinates
        length = chain.num_residues
        if res_map is not None:
            res_st, res_end, apo_st, apo_end = parse_residue_map(res_map)
            if res_st == -1:
                self.logger.warning(
                    f"Invalid residue map {res_map} for {key}. "
                    "Use masked coordinates as apo."
                )
                return fallback()
            if res_st == 0 and res_end == length:
                # Simply crop apo_coords without padding
                coords = coords[apo_st:apo_end]
            else:
                # Need to pad coords to match the full sequence length
                padded_coords = np.full(
                    (length, coords.shape[1], 3), np.nan, dtype=np.float32
                )
                padded_coords[res_st:res_end] = coords[apo_st:apo_end]
                coords = padded_coords

        assert coords.shape[0] == chain.num_residues, (
            f"Apo coordinates length {coords.shape[0]} does not match "
            f"chain length {length} for chain {key}."
        )
        return coords

    def fetch_apo_structures(
        self,
        ref_struct: RefStructure,
        apo_lookup: dict[int, list[dict | None]],
        rng: np.random.Generator,
    ) -> dict[int, np.ndarray]:
        """Return protein apo coordinates for the given reference structure.
        Key: asymmetric chain ID (asym_id)
        Value: protein apo coordinates of shape [Napo, L, 37, 3].
        """
        chain_coords: dict[int, np.ndarray] = {}
        coords_by_entity: dict[int, np.ndarray] = {}
        for c in ref_struct.chains:
            if not c.is_protein:
                continue
            if c.entity_id not in coords_by_entity:
                coords_by_entity[c.entity_id] = np.stack(
                    [
                        self._get_protein_apo_coords(c, apo_info, rng)
                        for apo_info in apo_lookup.get(c.asym_id, [None] * self.max_apo)
                    ],
                    axis=0,
                )
            coords = coords_by_entity[c.entity_id].copy()
            assert coords.shape == (self.max_apo, c.num_residues, 37, 3)
            chain_coords[c.asym_id] = coords
        return chain_coords

    def sample_prior_coords(
        self,
        ref_struct: RefStructure,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample prior coordinates for the given structure."""
        if self.num_priors <= 0 or self.prior_sampler is None:
            return np.empty((0, ref_struct.num_atoms, 3), dtype=np.float32)

        prior_candidates = self.get_prior_candidates(ref_struct, rng)
        prior_coords = self.prior_sampler(
            ref_struct, prior_candidates, self.num_priors, rng
        )
        assert prior_coords.shape == (self.num_priors, ref_struct.num_atoms, 3)
        return prior_coords

    def get_prior_candidates(
        self,
        ref_struct: RefStructure,
        rng: np.random.Generator,
    ) -> dict[int, np.ndarray]:
        """Return protein prior candidate stacks keyed by asym_id."""
        entry_id: str = ref_struct.id
        prior_candidates: dict[int, np.ndarray] = {}
        metadata_by_asym_id = {c.asym_id: c for c in ref_struct.metadata.chains}

        for c in ref_struct.chains:
            if c.asym_id in metadata_by_asym_id:
                metadata_by_asym_id[c.asym_id].prior_uid = c.asym_id

            if not c.is_protein:
                continue

            loaded = self._load_prior_stack_info_from_lmdb(
                entry_id, c.entity_id, "protein"
            )
            if loaded is None:
                if self.train:
                    coords = self._center_label_residue_coords(c)
                    prior_candidates[c.asym_id] = coords[None]
                else:
                    raise KeyError(
                        f"Prior stack not found for chain {entry_id}:{c.asym_id}."
                    )
                continue

            expected_shape = (c.num_residues, 37, 3)
            assert loaded.shape[1:] == expected_shape, (
                f"Prior stack for chain {entry_id}:{c.asym_id} has shape "
                f"{loaded.shape}; expected (N, {c.num_residues}, 37, 3)."
            )
            assert loaded.shape[0] > 0, (
                f"Prior stack for chain {entry_id}:{c.asym_id} is empty."
            )
            prior_candidates[c.asym_id] = loaded

        return prior_candidates

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

    def featurize(self, tokenized: TokenizedStructure) -> FoldingInput:
        """Featurize the given tokenized structure."""
        return self.featurizer(tokenized)

    # === Optional to-override in subclasses === #
    def pad_input(self, f_input: FoldingInput) -> FoldingInput:
        """Pad the folding input to multiple of 64 for LocalAtomAttention."""
        # Pad inputs for CUDA efficiency.
        num_tokens = next_multiple(f_input.num_tokens, 32)
        num_sequence_tokens = next_multiple(f_input.num_sequence_tokens, 64)
        num_atoms = next_multiple(f_input.num_atoms, 64)
        return f_input.pad(
            max_tokens=num_tokens,
            max_atoms=num_atoms,
            max_sequence_tokens=num_sequence_tokens,
        )

    def populate_structure_tokens(
        self,
        tokenized: TokenizedStructure,
        apo_lookup: dict[int, list[dict | None]],
    ) -> None:
        """Populate the structure tokens for the given tokenized structure."""
        for c_i in range(tokenized.num_chains):
            if tokenized.chain.chain_type[c_i] != C.ChainType.PROTEIN.value:
                continue  # only populate structure tokens for protein chains

            asym_id = int(tokenized.chain.asym_id[c_i])
            k = f"{tokenized.id}-{asym_id}"  # For logging purpose

            apo_infos = apo_lookup.get(asym_id)
            if not apo_infos:
                self.logger.warning(
                    f"No apo info for chain `{k}` in apo lookup. Skipping."
                )
                continue

            for apo_i, apo_info in enumerate(apo_infos):
                if apo_info is None:
                    continue
                tok = apo_info.get("tokens", None)
                res_map = apo_info.get("residue_map", None)
                if tok is not None:
                    self._insert_structure_tokens(
                        tokenized,
                        c_i,
                        apo_i,
                        tok,
                        res_map,
                        key=apo_info["key"],
                    )

    def _insert_structure_tokens(
        self,
        tokenized: TokenizedStructure,
        c_i: int,
        apo_i: int,
        apo_tok: np.ndarray,
        res_map: str | None = None,
        *,
        key: str,
    ) -> None:
        """Insert one chain's apo structure tokens into a tokenized structure."""
        bb_struct_token_id = tokenized.sequence.bb_struct_token_id
        fa_struct_token_id = tokenized.sequence.fa_struct_token_id
        seq_lens = tokenized.chain.num_residues + 2
        seq_starts = np.cumsum(seq_lens) - seq_lens
        bb_tok, fa_tok = apo_tok
        toklen = len(bb_tok)

        seq_start = int(seq_starts[c_i])
        seq_end = seq_start + int(seq_lens[c_i])
        seq_token_len = int(tokenized.chain.num_residues[c_i])

        if res_map is None:
            if len(bb_tok) != seq_token_len:
                self.logger.warning(
                    f"Apo tokens ({key}, len={toklen}) cannot be "
                    f"aligned with sequence tokens (len={seq_token_len}) for "
                    f"chain {c_i} without residue map. Skipping."
                )
                return
            bb_struct_token_id[seq_start + 1 : seq_end - 1, apo_i] = bb_tok
            fa_struct_token_id[seq_start + 1 : seq_end - 1, apo_i] = fa_tok
        else:
            res_st, res_end, apo_st, apo_end = parse_residue_map(res_map)
            if res_st == -1:
                self.logger.warning(
                    f"Invalid residue map {res_map} for chain {c_i}. Skipping."
                )
                return
            if toklen < (apo_end - apo_st) or (seq_token_len < res_end):
                self.logger.warning(
                    f"Apo tokens ({key}, len={toklen}) cannot cover "
                    f"the residue mapping for chain {c_i}: {res_map}. Skipping."
                )
                return
            _st, _end = seq_start + 1 + res_st, seq_start + 1 + res_end
            bb_struct_token_id[_st:_end, apo_i] = bb_tok[apo_st:apo_end]
            fa_struct_token_id[_st:_end, apo_i] = fa_tok[apo_st:apo_end]
