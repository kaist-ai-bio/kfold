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

import copy
import dataclasses
import io
import pathlib
from functools import cached_property, lru_cache
from typing import Self

import numpy as np

import kfold.constants as C
from kfold.data.types.metadata import Metadata
from kfold.utils.misc import check_array

__all__ = ["RefStructure"]


# === Helper functions === #
@lru_cache(maxsize=128)
def _get_polymer_atom_order_in_residue(res_name: str, chain_type: int) -> tuple[int, ...]:
    """Get residue-major coordinate atom indices for atoms in a residue."""
    ctype = C.ChainType(chain_type)
    residue_name: C.ResidueName = C.ResidueName[res_name]
    if ctype.is_protein:
        atom_order = C.atom.protein_atom37_order
    elif ctype.is_rna or ctype.is_dna:
        atom_order = C.atom.nucleic_acid_atom29_order
    else:
        raise ValueError(f"Unsupported polymer chain type: {ctype}")
    return tuple(atom_order[an.value] for an in C.atom.RESIDUE_ATOMS[residue_name])


def pack_metadata(metadata: Metadata) -> np.ndarray:
    """Pack Metadata into a numpy bytes array."""
    import msgpack

    metadata_dict = metadata.to_dict()
    metadata_serialized = msgpack.packb(metadata_dict)
    return np.frombuffer(metadata_serialized, dtype=np.uint8)


def unpack_metadata(data: np.ndarray) -> Metadata:
    """Unpack Metadata from a numpy bytes array."""
    import msgpack

    if isinstance(data, np.ndarray) and data.ndim == 0:
        data = data.item()
    metadata_dict = msgpack.unpackb(data)
    return Metadata.from_dict(metadata_dict)


# === Data structures === #
@dataclasses.dataclass(frozen=True)
class Chain:
    """Chain information.

    Shape: [Nchain, ...]

    Attributes
    ----------
    chain_type: int
        Chain type.
    entity_id: int
        Entity ID.
    asym_id: int
        Asymmetric unit ID.
    sym_id: int
        Symmetry ID.
    residue: Residue
        Residue information.
    atom: Atom
        Atom information.
    bond: Bond
        Intra-chain bond information.
    """

    chain_type: int
    entity_id: int
    asym_id: int
    sym_id: int
    residue: "ResidueLayout"
    atom: "AtomLayout"
    bond: "BondLayout"
    smiles: str | None = None  # optional SMILES string for small molecule
    is_covalent_ligand: bool = False  # whether the chain is a covalent ligand

    # === Properties === #
    @cached_property
    def ctype(self) -> C.ChainType:
        """Chain type as enum."""
        return C.ChainType(self.chain_type)

    @cached_property
    def subtype(self) -> C.SubChainType:
        """Chain type as enum."""
        if self.is_protein:
            if self.is_peptide:
                return C.SubChainType.PEPTIDE
            else:
                return C.SubChainType.PROTEIN
        elif self.is_dna:
            return C.SubChainType.DNA
        elif self.is_rna:
            return C.SubChainType.RNA
        else:
            if self.is_glycan:
                return C.SubChainType.GLYCAN
            elif self.is_covalent_ligand:
                return C.SubChainType.COVALENT_LIGAND
            elif self.is_ion:
                return C.SubChainType.ION
            else:
                return C.SubChainType.SMALL_MOLECULE

    @property
    def is_protein(self) -> bool:
        """Whether the chain is a protein."""
        return self.ctype.is_protein

    @property
    def is_dna(self) -> bool:
        """Whether the chain is a dna."""
        return self.ctype.is_dna

    @property
    def is_rna(self) -> bool:
        """Whether the chain is a rna."""
        return self.ctype.is_rna

    @property
    def is_ligand(self) -> bool:
        """Whether the chain is a ligand."""
        return self.ctype.is_ligand

    @property
    def is_polymer(self) -> bool:
        """Whether the chain is a polymer."""
        return self.ctype.is_polymer

    @property
    def is_nonpolymer(self) -> bool:
        """Whether the chain is a non-polymer."""
        return self.ctype.is_nonpolymer

    @property
    def is_nucleic_acid(self) -> bool:
        """Whether the chain is a nucleic acid."""
        return self.ctype.is_nucleic_acid

    @property
    def is_peptide(self) -> bool:
        """Whether the chain is a peptide."""
        return self.ctype.is_protein and self.num_residues < 16

    @property
    def is_ion(self) -> bool:
        """Whether the chain is an ion."""
        if self.num_atoms > 1 or self.num_residues > 1:
            return False
        if self.is_polymer:
            return False
        return self.residue.name[0].item() in C.ccd.IONS

    @property
    def is_small_molecule(self) -> bool:
        """Whether the chain is a small molecule (non-polymer & non-ion)."""
        return self.is_nonpolymer and not self.is_ion

    @property
    def is_glycan(self) -> bool:
        """Whether the chain is a glycan."""
        return self.is_covalent_ligand and all(
            res in C.ccd.GLYCANS for res in self.residue.name.tolist()
        )

    @property
    def num_residues(self) -> int:
        """Number of residues in the chain."""
        return len(self.residue)

    @property
    def num_atoms(self) -> int:
        """Number of atoms in the chain."""
        return len(self.atom)

    @property
    def num_bonds(self) -> int:
        """Number of bonds in the chain."""
        return len(self.bond)

    @property
    def num_tokens(self) -> int:
        """Number of tokens in the structure."""
        num_tokens: int = 0
        num_atom_list = self.residue.num_atoms.tolist()
        for res_i in range(self.num_residues):
            if self.residue.is_standard[res_i]:
                num_tokens += 1
            else:
                num_tokens += num_atom_list[res_i]
        return num_tokens

    # === Methods === #
    def get_sequence(self, map_to_standard: bool = False) -> str:
        """Get the amino acid / nucleotide sequence of the chain.

        Parameters
        ----------
        map_to_standard: bool
            Whether to map ambiguous residues to standard ones.

        Returns
        -------
        sequence: str
            One-letter code sequence.
        """
        if self.ctype.is_nonpolymer:
            raise ValueError("Non-polymer chains do not have a sequence.")
        unk = "X" if self.ctype.is_protein else "N"

        tokens: list[str] = [
            C.residue.convert_ccd_name_to_one_letter(v, unk)
            for v in self.residue.name.tolist()
        ]

        if map_to_standard:
            if self.ctype.is_protein:
                tokens = [C.residue.PROTEIN_AMINO_ACID_MAPPING.get(t, t) for t in tokens]
                standard_set = C.residue.PROTEIN_AMINO_ACIDS_SET
            elif self.ctype.is_rna:
                standard_set = C.residue.RNA_BASES_SET
            elif self.ctype.is_dna:
                standard_set = C.residue.DNA_BASES_SET
            tokens = [t if t in standard_set else unk for t in tokens]

        return "".join(tokens)

    def get_ccd_sequence(self) -> list[str]:
        """Get the amino acid / nucleotide sequence of the chain."""
        return self.residue.name.tolist()

    def map_residue_coords_to_atom_coords(self, residue_coords: np.ndarray) -> np.ndarray:
        """Map atom37/atom29 residue coordinates to this chain's atom order."""
        if not self.is_polymer:
            raise ValueError("Only polymer chains have residue-major coordinates.")
        if self.is_protein:
            atom_order = C.atom.protein_atom37_order
            n_atoms_per_res = 37
        else:
            atom_order = C.atom.nucleic_acid_atom29_order
            n_atoms_per_res = 29
        if residue_coords.shape != (self.num_residues, n_atoms_per_res, 3):
            raise ValueError(
                f"Invalid residue_coords shape: {residue_coords.shape}, "
                f"expected ({self.num_residues}, {n_atoms_per_res}, 3)"
            )

        src_res_indices: list[int] = []
        src_atom_indices: list[int] = []
        dst_atom_indices: list[int] = []
        ccd_sequence = self.get_ccd_sequence()
        atom_names = self.atom.name.tolist()
        for res_i in range(self.num_residues):
            residue_index = res_i + 1
            if self.residue.is_standard[res_i]:
                atom_st = self.residue.atom_starts[res_i]
                atom_orders = _get_polymer_atom_order_in_residue(
                    ccd_sequence[res_i], self.chain_type
                )
                natoms = len(atom_orders)
                src_res_indices.extend([res_i] * natoms)
                src_atom_indices.extend(atom_orders)
                dst_atom_indices.extend(range(atom_st, atom_st + natoms))
            else:
                for atom_i in self.residue.iter_residue_atoms(residue_index):
                    atom_name = atom_names[atom_i]
                    if atom_name in atom_order:
                        src_res_indices.append(res_i)
                        src_atom_indices.append(atom_order[atom_name])
                        dst_atom_indices.append(atom_i)

        atom_coords = np.full((self.num_atoms, 3), np.nan, dtype=np.float32)
        atom_coords[dst_atom_indices] = residue_coords[src_res_indices, src_atom_indices]
        return atom_coords

    def map_atom_coords_to_residue_coords(self, atom_coords: np.ndarray) -> np.ndarray:
        """Map this chain's atom-order coordinates to atom37/atom29 residue order."""
        if not self.is_polymer:
            raise ValueError("Only polymer chains have residue-major coordinates.")
        if atom_coords.shape != (self.num_atoms, 3):
            raise ValueError(
                f"Invalid atom_coords shape: {atom_coords.shape}, "
                f"expected ({self.num_atoms}, 3)"
            )

        if self.is_protein:
            atom_order = C.atom.protein_atom37_order
            n_atoms_per_res = 37
        elif self.is_rna or self.is_dna:
            atom_order = C.atom.nucleic_acid_atom29_order
            n_atoms_per_res = 29
        else:
            raise ValueError(f"Unsupported polymer chain type: {self.ctype}")

        residue_coords = np.full(
            (self.num_residues, n_atoms_per_res, 3), np.nan, dtype=np.float32
        )
        atom_names = self.atom.name.tolist()
        for res_i in range(self.num_residues):
            residue_index = res_i + 1
            for atom_i in self.residue.iter_residue_atoms(residue_index):
                atom_name = atom_names[atom_i]
                atom_j = atom_order.get(atom_name)
                if atom_j is not None:
                    residue_coords[res_i, atom_j] = atom_coords[atom_i]
        return residue_coords

    def find_atom_index(self, residue_index: int, atom_name: str) -> int:
        """Find atom index given residue index and atom name.

        Parameters
        ----------
        residue_index: int
            1-based residue index.
        atom_name: str
            Atom name.

        Returns
        -------
        atom_index: int
            0-based atom index.

        Raises
        ------
        KeyError
            If the atom is not found.
        """
        # Get the range of atom indices for the given residue
        res_i = residue_index - 1  # convert to 0-based index
        st = self.residue.atom_starts[res_i]
        end = self.residue.atom_ends[res_i]

        # Assuming self.atom.name is a numpy array
        res_atoms = self.atom.name[st:end]

        # Vectorized strip and upper (Note: these return new arrays)
        # Only use this if the slice is very large
        mask = np.char.upper(res_atoms) == atom_name.upper()
        indices = np.where(mask)[0]
        if indices.size == 0:
            raise KeyError(
                f"Atom '{atom_name}' not found in residue index {residue_index}: "
                f"{res_atoms}"
            )
        elif indices.size > 1:
            raise KeyError(
                f"Multiple atoms '{atom_name}' found in residue index {residue_index}: "
                f"{res_atoms}"
            )
        return int(st + indices[0])

    def iter_residue_atoms(self, residue_index: int) -> range:
        """Get the range of atom indices for a given residue index."""
        # residue_index: 1-based index
        start = self.residue.atom_starts[residue_index - 1]
        end = start + self.residue.num_atoms[residue_index - 1]
        return range(start, end)

    def __repr__(self) -> str:
        """FoldingInput summary representation."""
        # Summary statistics
        return (
            "Chain(\n"
            + f"  chain_type: {self.ctype.name}\n"
            + f"  entity_id: {self.entity_id}\n"
            + f"  asym_id: {self.asym_id}\n"
            + f"  sym_id: {self.sym_id}\n"
            + f"  num_residues: {self.num_residues}\n"
            + f"  num_atoms: {self.num_atoms}\n"
            + f"  num_bonds: {self.num_bonds}\n"
            + (f"  smiles: {self.smiles}\n" if self.smiles is not None else "")
            + ")"
        )

    def clone(self) -> Self:
        """Create a copy of the Chain."""
        return copy.deepcopy(self)

    def copy_with(self, deepcopy: bool = False, **kwargs) -> Self:
        """Create a copy of the Chain with modified fields."""
        if deepcopy:
            out = copy.deepcopy(self)
        else:
            out = self
        return dataclasses.replace(out, **kwargs)

    # === Numpy serialization === #
    def to_npz_dict(self) -> dict[str, np.ndarray]:
        """Convert to a flat dictionary for NPZ storage.

        Returns a dictionary where tensor fields are converted to numpy arrays
        with hierarchical keys like 'chain.asym_id', 'token.token_type', etc.
        """
        result: dict[str, np.ndarray] = {
            "chain_type": np.array(self.chain_type, dtype=np.uint8),
            "entity_id": np.array(self.entity_id, dtype=np.uint16),
            "asym_id": np.array(self.asym_id, dtype=np.uint16),
            "sym_id": np.array(self.sym_id, dtype=np.uint16),
        }
        for prefix, struct in [
            ("residue.", self.residue),
            ("atom.", self.atom),
            ("bond.", self.bond),
        ]:
            for field in dataclasses.fields(struct):
                key = prefix + field.name
                value = getattr(struct, field.name)
                assert value is not None, f"{key} is None"
                result[key] = value
        if self.smiles is not None:
            result["smiles"] = np.array(self.smiles, dtype=np.dtype("U"))
        if self.is_covalent_ligand is not None:
            result["is_covalent_ligand"] = np.array(self.is_covalent_ligand, dtype=bool)
        return result

    @classmethod
    def from_npz_dict(cls, data: dict[str, np.ndarray]) -> Self:
        """Reconstruct from NPZ dictionary."""
        reconstructed = {}
        for prefix, struct_cls in [
            ("residue.", ResidueLayout),
            ("atom.", AtomLayout),
            ("bond.", BondLayout),
        ]:
            struct_data = {
                key[len(prefix) :]: value
                for key, value in data.items()
                if key.startswith(prefix)
            }
            reconstructed[prefix[:-1]] = struct_cls(**struct_data)
        if "smiles" in data:
            reconstructed["smiles"] = data["smiles"].item()
        if "is_covalent_ligand" in data:
            reconstructed["is_covalent_ligand"] = data["is_covalent_ligand"].item()
        return cls(
            chain_type=data["chain_type"].item(),
            entity_id=data["entity_id"].item(),
            asym_id=data["asym_id"].item(),
            sym_id=data["sym_id"].item(),
            **reconstructed,
        )


@dataclasses.dataclass(frozen=True)
class ResidueLayout:
    """Residue information.

    Attributes
    ----------
    names: np.ndarray (str, <U6)
        CCD IDs of shape [L,].
        Use up to 6-character names for custom CCDs.
    num_atoms: np.ndarray (int)
        Number of atoms per residue of shape [L,].
    is_standard: np.ndarray (bool)
        Whether the residue is a standard amino acid or nucleic acid of shape [L,].
        NOTE: UNK, DN, N are considered standard residues.
    """

    name: np.ndarray  # [L,], str
    num_atoms: np.ndarray  # [L,], int
    is_standard: np.ndarray  # [L,], bool

    def __len__(self) -> int:
        return len(self.name)

    def __post_init__(self):
        shape = (len(self),)
        check_array(self.name, name="names", dtype=np.str_, shape=shape)
        check_array(self.num_atoms, name="num_atoms", dtype=np.integer, shape=shape)
        check_array(self.is_standard, name="is_standard", dtype=bool, shape=shape)

    @cached_property
    def atom_starts(self) -> np.ndarray:
        """Starting indices of atoms for each residue."""
        return np.concatenate([np.array([0], dtype=np.int64), self.atom_ends[:-1]])

    @cached_property
    def atom_ends(self) -> np.ndarray:
        """Ending indices of atoms for each residue."""
        return np.cumsum(self.num_atoms, dtype=np.int64)

    def get_atom_slice(self, residue_index: int) -> slice:
        """Get slice objects for each residue's atoms."""
        res_i = residue_index - 1  # convert to 0-based index
        return slice(self.atom_starts[res_i], self.atom_ends[res_i])

    def iter_residue_atoms(self, residue_index: int) -> range:
        """Get the range of atom indices for a given residue index."""
        res_i = residue_index - 1  # convert to 0-based index
        return range(self.atom_starts[res_i], self.atom_ends[res_i])

    @classmethod
    def get_default_dtype(cls) -> dict[str, type | np.dtype]:
        """Get default dtypes for each field."""
        return {
            "name": np.dtype("<U6"),
            "num_atoms": np.uint8,  # max 255 atoms per residue
            "is_standard": bool,
            "is_linked": bool,
        }


@dataclasses.dataclass(frozen=True)
class AtomLayout:
    """Atom information.

    Attributes
    ----------
    name: np.ndarray (str, <U4)
        atom names of shape [Natom], str
    element: np.ndarray (int)
        element types of shape [Natom], int
    charge: np.ndarray (int)
        formal charge of each atom of shape [Natom], int
    coords: np.ndarray (float32)
        Holo (bound) state coordinates of shape [Natom, 3],
        This is for model training, so that this field is
        filled to 0 during inference.
    bfactor: np.ndarray (float16)
        B-factor values for each residue of shape [L,].
    """

    name: np.ndarray  # [Natom,], str
    element: np.ndarray  # [Natom,], int
    charge: np.ndarray  # [Natom,], int
    coords: np.ndarray  # [Natom, 3], float32
    bfactor: np.ndarray  # [L,], int

    def __len__(self) -> int:
        return len(self.name)

    def __post_init__(self):
        shape = self.name.shape
        check_array(self.name, name="name", dtype=np.str_, shape=shape)
        check_array(self.element, name="element", dtype=np.integer, shape=shape)
        check_array(self.charge, name="charge", dtype=np.integer, shape=shape)
        check_array(self.coords, name="coords", dtype=np.floating, shape=(*shape, 3))
        check_array(self.bfactor, name="bfactor", dtype=np.floating, shape=shape)

    @classmethod
    def get_default_dtype(cls) -> dict[str, type | np.dtype]:
        """Get default dtypes for each field."""
        return {
            "name": np.dtype("<U4"),
            "element": np.uint8,
            "charge": np.int8,
            "coords": np.float32,
            "bfactor": np.float16,
        }

    @property
    def is_resolved(self) -> np.ndarray:
        """Get mask of resolved atoms in holo structure.

        Returns
        -------
        holo_mask: np.ndarray (bool)
            Shape [Natom,], bool
        """
        return np.isfinite(self.coords).all(axis=-1)


@dataclasses.dataclass(frozen=True)
class BondLayout:
    """Intra-chain Bond information.

    Shape: [Nbond, ...]

    Attributes
    ----------
    asym_id: np.ndarray
        Chain asym indices of the connecting atoms in the bond of shape [Nbond, 2].
    residue_index: np.ndarray
        Residue indices of the connecting atoms in the bond of shape [Nbond, 2].
    atom_name: np.ndarray
        Atom names of the connecting atoms in the bond of shape [Nbond, 2].
    bond_type: np.ndarray
        RDKit BondType values of shape [Nbond,], indicating the type of each bond.

    # TODO: is there any inter-residue bond in RCSB?
    """

    residue_index: np.ndarray  # [Nbond, 2], int
    atom_name: np.ndarray  # [Nbond, 2], int
    bond_type: np.ndarray  # [Nbond,], int

    def __len__(self) -> int:
        return len(self.bond_type)

    def __post_init__(self):
        shape = self.bond_type.shape
        check_array(
            self.residue_index, name="residue_index", dtype=np.integer, shape=(*shape, 2)
        )
        check_array(
            self.atom_name, name="atom_name", dtype=np.dtype("<U4"), shape=(*shape, 2)
        )
        check_array(self.bond_type, name="bond_type", dtype=np.integer, shape=shape)

    @classmethod
    def get_default_dtype(cls) -> dict[str, type | np.dtype]:
        """Get default dtypes for each field."""
        return {
            "residue_index": np.uint32,
            "atom_name": np.dtype("<U4"),
            "bond_type": np.uint8,  # 0-5, RDKit BondType values
        }


@dataclasses.dataclass(frozen=True)
class CovalentConnection:
    """Inter-chain & inter-molecule covalent connection information."""

    asym_id: tuple[int, int]
    residue_index: tuple[int, int]
    atom_names: tuple[str, str]

    # === Numpy serialization === #
    def to_npz_dict(self) -> dict[str, np.ndarray]:
        """Convert to a flat dictionary for NPZ storage."""
        result: dict[str, np.ndarray] = {
            "asym_id": np.array(self.asym_id, dtype=np.uint16),
            "residue_index": np.array(self.residue_index, dtype=np.uint16),
            "atom_names": np.array(self.atom_names, dtype=np.dtype("<U4")),
        }
        return result

    @classmethod
    def from_npz_dict(cls, data: dict[str, np.ndarray]) -> Self:
        """Reconstruct from NPZ dictionary."""
        return cls(
            asym_id=tuple(data["asym_id"].tolist()),
            residue_index=tuple(data["residue_index"].tolist()),
            atom_names=tuple(data["atom_names"].tolist()),
        )


@dataclasses.dataclass(kw_only=True)
class RefStructure:
    """Reference Structure information.

    Attributes
    ----------
    chains: list[Chain]
        Chain information.
    connections: list[CovalentConnection]
        Covalent connection information.
    metadata: Metadata
        Metadata information.
        # NOTE: Metadata might not include cluster info.
    """

    chains: list[Chain]
    connections: list[CovalentConnection]
    metadata: Metadata

    @property
    def id(self) -> str:
        """Structure ID."""
        return self.metadata.id

    @property
    def entity_ids(self) -> list[int]:
        """List of entity IDs in the structure."""
        return [chain.entity_id for chain in self.chains]

    @property
    def asym_ids(self) -> list[int]:
        """List of asym IDs in the structure."""
        return [chain.asym_id for chain in self.chains]

    @property
    def num_chains(self) -> int:
        """Number of chains in the structure."""
        return len(self.chains)

    @property
    def num_polymer_chains(self) -> int:
        """Number of polymer chains in the structure."""
        return sum(chain.ctype.is_polymer for chain in self.chains)

    @property
    def num_nonpolymer_chains(self) -> int:
        """Number of non-polymer chains in the structure."""
        return sum(chain.ctype.is_nonpolymer for chain in self.chains)

    @property
    def num_residues(self) -> int:
        """Number of residues in the structure."""
        return sum(chain.num_residues for chain in self.chains)

    @cached_property
    def num_atoms(self) -> int:
        """Number of atoms in the structure."""
        return sum(chain.num_atoms for chain in self.chains)

    @cached_property
    def num_tokens(self) -> int:
        """Number of tokens in the structure."""
        return sum(chain.num_tokens for chain in self.chains)

    @property
    def num_bonds(self) -> int:
        """Number of bonds in the structure."""
        return sum(chain.num_bonds for chain in self.chains)

    @property
    def num_connections(self) -> int:
        """Number of covalent connections in the structure."""
        return len(self.connections)

    def __repr__(self) -> str:
        """FoldingInput summary representation."""
        # Summary statistics
        num_chains = self.num_chains
        num_connections = self.num_connections

        return (
            "RefStructure(\n"
            + f"  num_chains: {num_chains}\n"
            + f"  num_interfaces: {len(self.metadata.interfaces)}\n"
            + f"  num_connections: {num_connections}\n"
            + "  chains: ["
            + ", ".join(f"{chain.ctype.name}" for chain in self.chains)
            + "]\n"
            "  interfaces: ["
            + ", ".join(
                f"{iface.asym_ids[0]}-{iface.asym_ids[1]}"
                for iface in self.metadata.interfaces
            )
            + "]\n"
            + ")"
        )

    def get_chain_by_asym_id(self, asym_id: int) -> Chain:
        """Get chain by asym_id."""
        for chain in self.chains:
            if chain.asym_id == asym_id:
                return chain
        raise KeyError(f"Chain with asym_id {asym_id} not found.")

    def get_atom_coords(self) -> np.ndarray:
        """Get atom coordinates of the structure.

        Returns
        -------
        coords: np.ndarray
            Shape [Natom, 3], float32
        """
        return np.concatenate([c.atom.coords for c in self.chains], axis=0)

    def clone(self) -> Self:
        """Create a deep copy of the RefStructure."""
        return copy.deepcopy(self)

    def copy_with_new_coords(
        self,
        coords: np.ndarray,
        b_factors: np.ndarray | None = None,
    ) -> Self:
        """Create a deep copy of the RefStructure."""
        assert coords.shape == (self.num_atoms, 3), (
            f"Invalid coords shape: {coords.shape}, expected ({self.num_atoms}, 3)"
        )
        if b_factors is None:
            b_factors = np.full((self.num_atoms,), fill_value=0, dtype=np.float32)

        new_chains = []
        atom_start = 0
        for chain in self.chains:
            atom_end = atom_start + chain.num_atoms
            new_atom = dataclasses.replace(
                chain.atom,
                coords=coords[atom_start:atom_end],
                bfactor=b_factors[atom_start:atom_end],
            )
            new_chain = chain.copy_with(deepcopy=False, atom=new_atom)
            new_chains.append(new_chain)
            atom_start = atom_end
        return dataclasses.replace(self, chains=new_chains)

    # === Writer === #
    def write(self, filename: str | pathlib.Path):
        """Write the structure to a file."""
        from kfold.data.utils.writer import KFoldWriter

        writer = KFoldWriter()
        writer.write(self, filename)

    # === Helper functions === #
    def validate(self) -> None:
        """Validate the consistency of the RefStructure."""
        # Check that asym_ids are unique
        asym_ids = [chain.asym_id for chain in self.chains]
        if len(asym_ids) != len(set(asym_ids)):
            raise ValueError("Duplicate asym_ids found in chains.")

        # Check that connections refer to valid asym_ids
        valid_asym_ids = set(asym_ids)
        for conn in self.connections:
            for asym_id in conn.asym_id:
                if asym_id not in valid_asym_ids:
                    raise ValueError(f"Connection refers to invalid asym_id {asym_id}.")

        # Check consistency between structure and metadata
        assert self.num_chains == self.metadata.num_chains
        meta_asym_ids = set(self.metadata.asym_ids)
        struct_asym_ids = set(asym_ids)
        if meta_asym_ids != struct_asym_ids:
            raise ValueError("Mismatch between metadata asym_ids and structure asym_ids.")

        for c, cm in zip(self.chains, self.metadata.chains, strict=True):
            assert c.entity_id == cm.entity_id
            assert c.asym_id == cm.asym_id
            assert c.sym_id == cm.sym_id
            assert c.num_residues == cm.num_residues

        for iface in self.metadata.interfaces:
            for asym_id in iface.asym_ids:
                if asym_id not in valid_asym_ids:
                    raise ValueError(f"Interface refers to invalid asym_id {asym_id}.")

    def to(self, *args, **kwargs) -> Self:
        """No-op for device/dtype movement for pytorch lightning compatibility."""
        return self

    # === Numpy serialization for model training === #
    def to_npz_dict(self) -> dict[str, np.ndarray]:
        """Convert to a flat dictionary for NPZ storage.

        Returns a dictionary where tensor fields are converted to numpy arrays
        with hierarchical keys like 'chain.0.asym_id', ...
        """
        result: dict[str, np.ndarray] = {}

        # Chains
        for i, chain in enumerate(self.chains):
            prefix = f"chain.{i}."
            chain_dict = chain.to_npz_dict()
            for key, value in chain_dict.items():
                result[prefix + key] = value

        # Connections
        for i, conn in enumerate(self.connections):
            prefix = f"connect.{i}."
            conn_dict = conn.to_npz_dict()
            for key, value in conn_dict.items():
                result[prefix + key] = value

        # Metadata
        result["_metadata"] = pack_metadata(self.metadata)

        return result

    @classmethod
    def from_npz_dict(cls, data: dict[str, np.ndarray]) -> Self:
        """Reconstruct from NPZ dictionary."""
        # Chains
        chains: list[Chain] = []
        i = 0
        while True:
            prefix = f"chain.{i}."
            chain_keys = [key for key in data.keys() if key.startswith(prefix)]
            if not chain_keys:
                break
            chain_data = {key[len(prefix) :]: data[key] for key in chain_keys}
            chains.append(Chain.from_npz_dict(chain_data))
            i += 1

        connections: list[CovalentConnection] = []
        j = 0
        while True:
            prefix = f"connect.{j}."
            connection_keys = [key for key in data.keys() if key.startswith(prefix)]
            if not connection_keys:
                break
            connection_data = {key[len(prefix) :]: data[key] for key in connection_keys}
            connections.append(CovalentConnection.from_npz_dict(connection_data))
            j += 1

        metadata = unpack_metadata(data["_metadata"])

        return cls(
            chains=chains,
            connections=connections,
            metadata=metadata,
        )

    def dump_npz(self, path: pathlib.Path | str | io.BytesIO) -> None:
        """Save to compressed NPZ file."""
        np.savez_compressed(
            path,
            **self.to_npz_dict(),
        )

    def save_npz(self, path: pathlib.Path | str | io.BytesIO) -> None:
        """Save to compressed NPZ file."""
        self.dump_npz(path)

    @classmethod
    def load_npz(cls, path: pathlib.Path | str | io.BytesIO) -> Self:
        """Load from NPZ file."""
        with np.load(path) as data:
            return cls.from_npz_dict(dict(data))
