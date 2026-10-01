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

from pathlib import Path
from textwrap import wrap

import gemmi
import numpy as np

from kfold import __version__
from kfold.data.types.structure import RefStructure
from kfold.utils.geometry.rigid_align import rigid_align

from .gemmi_utils import create_gemmi_structure, make_mmcif_block

MODEL_NAME = "K-Fold"
ARTICLE_TITLE = (
    "Generative modeling of binding-induced structural change in biomolecular complexes"
)
AUTHOR_NAME = "Team KAIST"


def _add_pdb_header(struct: gemmi.Structure) -> None:
    """Add prediction and reference metadata to PDB output."""
    remarks = [
        "HEADER",
        f"TITLE     {MODEL_NAME} prediction",
        "REMARK   1 REFERENCE 1",
        f"REMARK   1  AUTH   {AUTHOR_NAME}",
    ]
    for index, line in enumerate(wrap(ARTICLE_TITLE, width=61), start=1):
        continuation = "  " if index == 1 else f"{index:2d}"
        remarks.append(f"REMARK   1  TITL{continuation} {line}")
    struct.raw_remarks = remarks


def _make_prediction_mmcif_block(struct: RefStructure) -> gemmi.cif.Block:
    """Create prediction metadata before adding structure categories."""
    block = gemmi.cif.Block(struct.metadata.id)
    block.set_pair("_entry.id", gemmi.cif.quote(struct.metadata.id))
    audit_loop = block.init_loop("_audit_author.", ["name", "pdbx_ordinal"])
    audit_loop.add_row([gemmi.cif.quote(AUTHOR_NAME), "1"])
    # block.set_pair("_citation.id", "primary")
    # block.set_pair("_citation.title", gemmi.cif.quote(ARTICLE_TITLE))
    # author_loop = block.init_loop(
    #     "_citation_author.", ["citation_id", "ordinal", "name"]
    # )
    software_loop = block.init_loop(
        "_software.",
        ["pdbx_ordinal", "name", "type", "description", "classification", "version"],
    )
    software_loop.add_row(
        [
            "1",
            MODEL_NAME,
            "package",
            gemmi.cif.quote(f"{MODEL_NAME} prediction pipeline"),
            gemmi.cif.quote("model building"),
            gemmi.cif.quote(__version__),
        ]
    )
    software_loop.add_row(
        [
            "2",
            "AtlasFold",
            "package",
            gemmi.cif.quote("Apo structure and prior candidate generation"),
            gemmi.cif.quote("model building"),
            "1.0.2",
        ]
    )
    return make_mmcif_block(struct, block=block)


class KFoldWriter:
    # =========================================================
    # Single structure write methods
    # =========================================================
    @classmethod
    def write_new_coords(
        cls,
        struct: RefStructure,
        filename: str | Path,
        coordinates: np.ndarray,
        b_factors: np.ndarray | None = None,
    ):
        if coordinates.shape != (struct.num_atoms, 3):
            raise ValueError(
                f"Coordinates shape {coordinates.shape} does not match shape "
                f"({struct.num_atoms}, 3)"
            )
        struct = struct.copy_with_new_coords(coordinates, b_factors=b_factors)
        cls.write(struct, filename)

    @classmethod
    def write(
        cls,
        struct: RefStructure,
        filename: str | Path,
    ):
        format = Path(filename).suffix.lower()
        if format not in {".pdb", ".cif"}:
            raise ValueError(f"Unsupported file format: {format}")
        try:
            if format == ".pdb":
                cls.write_pdb(struct, filename)
            else:
                cls.write_mmcif(struct, filename)
        except Exception as e:
            print(f"Failed to write structure to {filename}: {e}")

    @staticmethod
    def write_mmcif(
        struct: RefStructure,
        filename: str | Path,
    ) -> None:
        block = _make_prediction_mmcif_block(struct)
        block.write_file(str(filename))

    @staticmethod
    def write_pdb(
        struct: RefStructure,
        filename: str | Path,
    ) -> None:
        gemmi_struct: gemmi.Structure = create_gemmi_structure(
            struct, pdb_compatible=True
        )
        _add_pdb_header(gemmi_struct)
        gemmi_struct.write_pdb(str(filename))

    @staticmethod
    def write_mmcifstring(struct: RefStructure) -> str:
        block = _make_prediction_mmcif_block(struct)
        return block.as_string()

    @staticmethod
    def write_pdbstring(
        struct: RefStructure,
    ) -> str:
        gemmi_struct: gemmi.Structure = create_gemmi_structure(
            struct, pdb_compatible=True
        )
        _add_pdb_header(gemmi_struct)
        return gemmi_struct.make_pdb_string()

    # =========================================================
    # Trajectory write methods
    # =========================================================
    @classmethod
    def write_trajectory(
        cls,
        struct: RefStructure,
        trajectory: np.ndarray,
        filename: str | Path,
        align: bool = True,
    ):
        if trajectory.shape[1:] != (struct.num_atoms, 3):
            raise ValueError(
                f"Trajectory shape {trajectory.shape} does not match shape "
                f"(*, {struct.num_atoms}, 3)"
            )

        format = Path(filename).suffix.lower()
        if format not in {".pdb", ".cif"}:
            raise ValueError(f"Unsupported file format: {format}")

        pdb_compatible = format == ".pdb"

        try:
            n_frames = trajectory.shape[0]
            traj_structures: gemmi.Structure = gemmi.Structure()
            traj_structures.name = struct.metadata.id
            # Add models
            prev_coords = None
            for i in range(n_frames):
                frame_coords = trajectory[i]
                if align and prev_coords is not None:
                    frame_coords = rigid_align(frame_coords, prev_coords, mask=None)
                prev_coords = frame_coords
                frame_struct = struct.copy_with_new_coords(frame_coords)
                if pdb_compatible:
                    _struct = create_gemmi_structure(frame_struct, pdb_compatible=True)
                else:
                    frame_block = _make_prediction_mmcif_block(frame_struct)
                    if i == 0:
                        block = frame_block
                    _struct = gemmi.make_structure_from_block(frame_block)
                if i == 0:
                    traj_structures.connections = _struct.connections
                model = _struct[0]
                if hasattr(model, "name"):
                    model.name = str(i + 1)
                else:
                    model.num = i + 1
                traj_structures.add_model(model, pos=-1)

            # Write to file
            if format == ".pdb":
                _add_pdb_header(traj_structures)
                traj_structures.write_pdb(str(filename))
            else:
                # Keep shared component definitions and replace all frame coordinates.
                groups = gemmi.MmcifOutputGroups(False)
                groups.atoms = True
                traj_structures.update_mmcif_block(block, groups)
                block.write_file(str(filename))
        except Exception as e:
            raise OSError(f"Failed to write trajectory to {filename}") from e
