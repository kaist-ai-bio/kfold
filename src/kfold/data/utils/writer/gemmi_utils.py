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

"""MMCIF writer utilities."""

import logging
from collections import defaultdict

import gemmi
import numpy as np

import kfold.constants as C
from kfold.data.types.structure import RefStructure

# Set up logging
logger = logging.getLogger(__name__)


# === Core implementation === #
def make_mmcif_block(
    struct: RefStructure,
    *,
    block: gemmi.cif.Block | None = None,
) -> gemmi.cif.Block:
    """Create a Gemmi MMCIF block from a reference structure.

    Parameters
    ----------
    struct : RefStructure
        Structure containing coordinates and ligand atom and bond definitions.
    block : gemmi.cif.Block | None, optional
        Block containing metadata to retain before the structure categories.

    Returns
    -------
    gemmi.cif.Block
        The constructed MMCIF Block.
    """
    gemmi_struct = create_gemmi_structure(struct)
    if block is None:
        block = gemmi.cif.Block(gemmi_struct.name)
    gemmi_struct.update_mmcif_block(block)
    _add_pdbx_nonpoly_scheme(block, gemmi_struct)
    _add_pdbx_poly_seq_scheme(block, gemmi_struct)
    _update_entity_poly(block, gemmi_struct)
    _update_entity_poly_seq(block, gemmi_struct)
    _update_chem_comp(block)
    _add_ligand_chem_comp(block, struct)
    return block


def _add_ligand_chem_comp(block: gemmi.cif.Block, struct: RefStructure) -> None:
    """Write ligand atom and bond definitions once per chemical component."""
    atoms: dict[tuple[str, str], tuple[int, int]] = {}  # (element, charge)
    bonds: dict[tuple[str, str, str], int] = {}  # bond-type
    for chain in struct.chains:
        if not chain.is_ligand:
            continue
        for residue_index, comp_id in enumerate(chain.residue.name, start=1):
            for atom_index in chain.iter_residue_atoms(residue_index):
                key = (str(comp_id), str(chain.atom.name[atom_index]))
                atoms.setdefault(
                    key,
                    (
                        int(chain.atom.element[atom_index]),
                        int(chain.atom.charge[atom_index]),
                    ),
                )
        for residue_indices, atom_names, bond_type in zip(
            chain.bond.residue_index,
            chain.bond.atom_name,
            chain.bond.bond_type,
            strict=True,
        ):
            residue_index_1, residue_index_2 = residue_indices
            if residue_index_1 != residue_index_2:
                continue  # Chemical component bonds only describe one residue.
            comp_id = str(chain.residue.name[residue_index_1 - 1])
            atom_id_1, atom_id_2 = sorted(str(name) for name in atom_names)
            bonds.setdefault((comp_id, atom_id_1, atom_id_2), int(bond_type))

    if atoms:
        atom_loop = block.init_loop(
            "_chem_comp_atom.", ["comp_id", "atom_id", "type_symbol", "charge"]
        )
        for (comp_id, atom_id), (element, charge) in sorted(atoms.items()):
            atom_loop.add_row(
                [
                    gemmi.cif.quote(comp_id),
                    gemmi.cif.quote(atom_id),
                    gemmi.Element(element).name,
                    str(charge),
                ]
            )
    if bonds:
        bond_loop = block.init_loop(
            "_chem_comp_bond.",
            ["comp_id", "atom_id_1", "atom_id_2", "value_order", "pdbx_aromatic_flag"],
        )
        # Keys are RDKit bond type values; aromatic bonds use 12.
        bond_orders = {1: "SING", 2: "DOUB", 3: "TRIP", 4: "QUAD", 12: "AROM"}
        for (comp_id, atom_id_1, atom_id_2), bond_type in sorted(bonds.items()):
            bond_loop.add_row(
                [
                    gemmi.cif.quote(comp_id),
                    gemmi.cif.quote(atom_id_1),
                    gemmi.cif.quote(atom_id_2),
                    bond_orders.get(bond_type, "?"),
                    "Y" if bond_type == 12 else "N",
                ]
            )


def create_gemmi_structure(
    struct: RefStructure,
    pdb_compatible: bool = False,
) -> gemmi.Structure:
    """Convert a RefStructure to a Gemmi Structure

    Parameters
    ----------
    struct : RefStructure
        The input structure containing chain metadata and coordinates.
    pdb_compatible : bool, optional
        Whether to ensure PDB compatibility (default is False).

    Returns
    ----
    gemmi.Structure
        The constructed Gemmi Structure object.
    """
    metadata = struct.metadata

    gemmi_struct = gemmi.Structure()
    gemmi_struct.name = metadata.id

    # === Make chain name unique if PDB compatible === #
    if pdb_compatible:
        # Re-assign chain names to be unique single characters
        if len(metadata.chains) > 62:
            logger.warning(
                f"Number of chains ({len(metadata.chains)}) exceeds "
                "PDB format limit (62). Chain names will be truncated "
                "to unique single characters, which may cause confusion."
            )

        def alphabet_generator():
            yield from (chr(i) for i in range(ord("A"), ord("Z") + 1))
            yield from (chr(i) for i in range(ord("a"), ord("z") + 1))
            yield from (str(i) for i in range(0, 10))

        gen = alphabet_generator()

        metadata = metadata.copy()
        for chain in metadata.chains:
            chain.name = next(gen)

    # === Create entity lists === #
    entity_ctypes: dict[int, C.ChainType] = {}
    entity_sequences: dict[int, list[str]] = {}
    entity_asym_ids: dict[int, list[str]] = defaultdict(list)
    for chain_i in range(len(struct.chains)):
        chain_meta = metadata.chains[chain_i]
        ref_chain = struct.chains[chain_i]
        entity_id = ref_chain.entity_id
        if entity_id not in entity_sequences:
            # First time seeing this entity, store its type and sequence
            entity_ctypes[entity_id] = ref_chain.ctype
            entity_sequences[entity_id] = ref_chain.get_ccd_sequence()
        # Append asym_id (chain name) to the entity's list
        entity_asym_ids[entity_id].append(chain_meta.name)

    entity_list: list[gemmi.Entity] = []
    for entity_id in sorted(entity_sequences.keys()):
        entity = gemmi.Entity(str(entity_id))
        ctype = entity_ctypes[entity_id]
        if ctype.is_polymer:
            entity.entity_type = gemmi.EntityType.Polymer
            match ctype:
                case C.ChainType.PROTEIN:
                    entity.polymer_type = gemmi.PolymerType.PeptideL
                case C.ChainType.RNA:
                    entity.polymer_type = gemmi.PolymerType.Rna
                case C.ChainType.DNA:
                    entity.polymer_type = gemmi.PolymerType.Dna
                case _:
                    raise ValueError(f"Unsupported polymer chain type: {ctype}")
        else:
            # FIXME: add glycan support later
            entity.entity_type = gemmi.EntityType.NonPolymer
        entity.full_sequence = entity_sequences[entity_id]
        entity.subchains = entity_asym_ids[entity_id]
        entity_list.append(entity)

    entities: gemmi.EntityList = gemmi.EntityList(entity_list)
    del entity_list  # free memory
    gemmi_struct.entities = entities

    # === Build Model === #
    try:
        model = gemmi.Model(1)
    except Exception:
        # Fallback for older Gemmi versions
        model = gemmi.Model("1")

    for chain_i in range(len(struct.chains)):
        ref_chain = struct.chains[chain_i]
        chain_meta = metadata.chains[chain_i]
        ctype = ref_chain.ctype

        # Retrieve layout data
        res_layout = ref_chain.residue
        atom_layout = ref_chain.atom

        atom_names: list[str] = atom_layout.name.tolist()
        atom_elements: list[int] = atom_layout.element.tolist()
        atom_charges: list[int] = atom_layout.charge.tolist()

        atom_coords = atom_layout.coords
        b_factor = atom_layout.bfactor

        chain_id = chain_meta.name  # e.g., "A", "B", etc.
        entity_id = chain_meta.entity_id

        # Determine if it is a polymer (ATOM) or non-polymer/ligand (HETATM)
        is_polymer = chain_meta.ctype.is_polymer
        het_flag = "A" if is_polymer else "H"

        # Create gemmi chain
        # Note: In Gemmi, chain.name usually maps to auth_asym_id
        chain = gemmi.Chain(chain_id)

        # Iterate over residues
        for res_i in range(ref_chain.num_residues):
            residue_index = res_i + 1  # 1-based indexing

            # Iterate over atoms
            atoms: list[gemmi.Atom] = []
            for atom_i in ref_chain.iter_residue_atoms(residue_index):
                # Check for valid coordinates (skip NaNs or Infs)
                xyz = atom_coords[atom_i]
                if not np.isfinite(xyz).all():
                    continue
                x, y, z = xyz.tolist()

                atom = gemmi.Atom()
                atom.name = atom_names[atom_i]
                atom.element = gemmi.Element(atom_elements[atom_i])
                atom.charge = atom_charges[atom_i]
                atom.pos = gemmi.Position(round(x, 3), round(y, 3), round(z, 3))
                atom.b_iso = round(float(b_factor[atom_i]), 2)
                atoms.append(atom)

            # Only add residue if it has atoms
            if len(atoms) > 0:
                residue = gemmi.Residue()
                if ctype.is_polymer:
                    residue.label_seq = residue_index  # 1-based indexing
                residue.name = str(res_layout.name[res_i])
                residue.seqid.num = residue_index
                residue.het_flag = het_flag  # 'A' for polymer, 'H' for non-polymer
                residue.entity_id = str(entity_id)  # Link to _entity category
                residue.subchain = chain_id  # Maps to _atom_site.label_asym_id
                for atom in atoms:
                    residue.add_atom(atom)
                chain.add_residue(residue)

        model.add_chain(chain)

    gemmi_struct.add_model(model)
    gemmi_struct.setup_entities()

    # Preserve explicit query bonds using the chain names written to the file.
    chains_by_asym = {chain.asym_id: chain for chain in struct.chains}
    names_by_asym = {chain.asym_id: chain.name for chain in metadata.chains}
    for index, connection in enumerate(struct.connections, start=1):
        partners = []
        for asym_id, residue_index, atom_name in zip(
            connection.asym_id,
            connection.residue_index,
            connection.atom_names,
            strict=True,
        ):
            chain = chains_by_asym[asym_id]
            address = gemmi.AtomAddress(
                names_by_asym[asym_id],
                gemmi.SeqId(residue_index, " "),
                str(chain.residue.name[residue_index - 1]),
                atom_name,
            )
            if gemmi_struct[0].find_cra(address).atom is None:
                raise ValueError(
                    f"Covalent bond endpoint is absent from output: {address}"
                )
            partners.append(address)

        bond = gemmi.Connection()
        bond.name = f"covale{index}"
        bond.type = gemmi.ConnectionType.Covale
        bond.asu = gemmi.Asu.Same
        bond.partner1, bond.partner2 = partners
        gemmi_struct.connections.append(bond)
    return gemmi_struct


def _add_pdbx_poly_seq_scheme(block: gemmi.cif.Block, structure: gemmi.Structure):
    """
    Manually add the _pdbx_poly_seq_scheme category to the CIF block.
    This is required for OST compatibility and proper polymer parsing.
    """
    # Columns required for _pdbx_poly_seq_scheme
    columns = [
        "asym_id",  # label_asym_id (residue.subchain)
        "entity_id",  # entity_id
        "mon_id",  # residue name
        "seq_id",  # residue sequence number
        "pdb_strand_id",  # auth_asym_id (chain.name)
        "pdb_seq_num",  # auth_seq_id
        "pdb_ins_code",  # PDB insertion code
    ]
    loop = block.init_loop("_pdbx_poly_seq_scheme.", columns)
    # Iterate strictly over the first model (assuming single model structure for AF3)
    model = structure[0]
    for chain in model:
        for res in chain:
            # Check if residue is part of a polymer ('A' het_flag)
            if res.het_flag == "A":
                # Map values
                asym_id = res.subchain if res.subchain else chain.name
                entity_id = res.entity_id
                mon_id = res.name
                seq_num = str(res.seqid.num)
                strand_id = chain.name  # auth_asym_id
                ins_code = "." if res.seqid.icode == " " else res.seqid.icode
                loop.add_row(
                    [
                        asym_id,  # asym_id
                        entity_id,  # entity_id
                        mon_id,  # mon_id
                        seq_num,  # seq_id
                        strand_id,  # pdb_strand_id
                        seq_num,  # pdb_seq_num
                        ins_code,  # pdb_ins_code
                    ]
                )


def _add_pdbx_nonpoly_scheme(block: gemmi.cif.Block, structure: gemmi.Structure):
    """
    Manually add the _pdbx_nonpoly_scheme category to the CIF block.
    This is required for OST compatibility and proper ligand parsing.
    """
    # Columns required for _pdbx_nonpoly_scheme
    columns = [
        "asym_id",  # label_asym_id (residue.subchain)
        "entity_id",  # entity_id
        "mon_id",  # residue name
        "ndb_seq_num",  # label_seq_id
        "pdb_seq_num",  # auth_seq_id
        "auth_seq_num",  # auth_seq_id
        "pdb_mon_id",  # auth_comp_id
        "auth_mon_id",  # auth_comp_id
        "pdb_strand_id",  # auth_asym_id (chain.name)
        "pdb_ins_code",  # PDB insertion code
    ]

    loop = block.init_loop("_pdbx_nonpoly_scheme.", columns)

    # Iterate strictly over the first model (assuming single model structure for AF3)
    model = structure[0]

    for chain in model:
        for res in chain:
            # Check if residue is explicitly marked as non-polymer ('H')
            # or if the entity it belongs to is non-polymer
            if res.het_flag == "H":
                # Map values
                asym_id = res.subchain if res.subchain else chain.name
                entity_id = res.entity_id
                mon_id = res.name
                seq_num = str(res.seqid.num)
                strand_id = chain.name  # auth_asym_id
                ins_code = "." if res.seqid.icode == " " else res.seqid.icode

                loop.add_row(
                    [
                        asym_id,  # asym_id
                        entity_id,  # entity_id
                        mon_id,  # mon_id
                        seq_num,  # ndb_seq_num
                        seq_num,  # pdb_seq_num
                        seq_num,  # auth_seq_num
                        mon_id,  # pdb_mon_id
                        mon_id,  # auth_mon_id
                        strand_id,  # pdb_strand_id
                        ins_code,  # pdb_ins_code
                    ]
                )


def _update_entity_poly(block: gemmi.cif.Block, structure: gemmi.Structure):
    """Update the _entity_poly_seq category in the CIF block to reflect sequences."""
    table: gemmi.cif.Table = block.find_mmcif_category("_entity_poly.")

    rows = []
    for row in table:
        rows.append(
            [
                row["entity_id"],
                row["type"],
                row["pdbx_strand_id"],
                row["pdbx_seq_one_letter_code"],
                row["pdbx_seq_one_letter_code"],
            ],
        )
    loop: gemmi.cif.Loop = block.init_mmcif_loop(
        "_entity_poly.",
        [
            "entity_id",
            "type",
            "pdbx_strand_id",
            "pdbx_seq_one_letter_code",
            "pdbx_seq_one_letter_code_can",
        ],
    )
    for row in rows:
        loop.add_row(row)


def _update_entity_poly_seq(block: gemmi.cif.Block, structure: gemmi.Structure):
    """Update the _entity_poly_seq category in the CIF block to reflect sequences."""
    table: gemmi.cif.Table = block.find_mmcif_category("_entity_poly_seq.")

    rows = []
    for row in table:
        rows.append([row["entity_id"], row["num"], row["mon_id"], "n"])
    loop: gemmi.cif.Loop = block.init_mmcif_loop(
        "_entity_poly_seq.",
        [
            "entity_id",
            "num",
            "mon_id",
            "hetero",
        ],
    )
    for row in rows:
        loop.add_row(row)


def _update_chem_comp(block: gemmi.cif.Block):
    """Add or modify the _chem_comp category in the CIF block to include residue types."""
    table: gemmi.cif.Table = block.find_mmcif_category("_chem_comp.")

    rows = []
    for row in table:
        res_id = row["id"]
        res: gemmi.ResidueInfo = gemmi.find_tabulated_residue(res_id)
        if res is not None:
            is_standard = res.is_standard()
            if res.kind == gemmi.ResidueKind.AA:
                res_type = "'L-peptide linking'"
            elif res.kind == gemmi.ResidueKind.RNA:
                res_type = "'RNA linking'"
            elif res.kind == gemmi.ResidueKind.DNA:
                res_type = "'DNA linking'"
            else:
                res_type = "non-polymer"
            res_weight = f"{res.weight:.3f}"
        else:
            is_standard = False
            res_type = "."
            res_weight = "."

        rows.append(
            [
                res_id,
                res_type,
                ".",
                ".",
                res_weight,
                "y" if not is_standard else "n",
            ]
        )
    loop: gemmi.cif.Loop = block.init_mmcif_loop(
        "_chem_comp.",
        [
            "id",
            "type",
            "name",
            "formula",
            "formula_weight",
            "mon_nstd_flag",
        ],
    )
    for row in rows:
        loop.add_row(row)
