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

import contextlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download

import kfold.constants as C
from kfold.data.types.model_input import FoldingInput
from kfold.model.layers.struct_enc import (
    BackboneTokenizer,
    FullAtomTokenizer,
    ProteinNetEncoder,
)
from kfold.model.layers.struct_enc.bb_vqvae.rotary import RotaryEmbedding
from kfold.utils.config import configurable

# AF2 residue types
restypes = [
    "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I",
    "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V",
]  # fmt: skip
restype_order = {restype: i for i, restype in enumerate(restypes)}

HF_REPO_ID = "kaist-ai-bio/kfold-assets"
HF_ENCODER_FILENAME = "weights/prot_struct_encoder_3b.pth"
HF_BB_TOKENIZER_FILENAME = "weights/prot_struct_bb_tokenizer.pth"
HF_FA_TOKENIZER_FILENAME = "weights/prot_struct_fa_tokenizer.pth"


@configurable
class StructureEncoder(torch.nn.Module):
    bb_tok: BackboneTokenizer
    fa_tok: FullAtomTokenizer
    encoder: ProteinNetEncoder

    @dataclass(kw_only=True)
    class Config:
        """TriProRep 3B configuration.

        Attributes
        ----------
        model_path: str | None
            Directory containing the three local structure encoder checkpoints. If
            unset, download them from Hugging Face.
        cache_dir: str | None
            Directory used to cache weights downloaded from Hugging Face.
        """

        model_path: str | None = None
        cache_dir: str | None = None
        d_model: int = 2560
        n_layers: int = 33
        n_heads: int = 40

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg: StructureEncoder.Config = cfg

        with torch.device("meta"):
            self.bb_tok = BackboneTokenizer()
            self.fa_tok = FullAtomTokenizer()
            self.encoder = ProteinNetEncoder(cfg.d_model, cfg.n_layers, cfg.n_heads)
        self._load_weights()

        # Freeze parameters.
        self.eval()
        self.requires_grad_(False)

        # Backbone token offset
        self.offset = 4  # number of special tokens

        seq_to_restype = torch.full((64,), -1, dtype=torch.long)
        for aa_i, aa in enumerate(restypes):
            seq_i = C.sequence.encode_protein_amino_acid(aa)
            seq_to_restype[seq_i] = aa_i
        self.register_buffer("seq_to_restype", seq_to_restype, persistent=False)

    def _load_weights(self) -> None:
        if self.cfg.model_path is None:
            repo_dir = Path(snapshot_download(HF_REPO_ID, cache_dir=self.cfg.cache_dir))
        else:
            repo_dir = Path(self.cfg.model_path)
        self.encoder.load_pretrained_weights(repo_dir / HF_ENCODER_FILENAME)
        for module, filename in (
            (self.bb_tok, HF_BB_TOKENIZER_FILENAME),
            (self.fa_tok, HF_FA_TOKENIZER_FILENAME),
        ):
            path = repo_dir / filename
            state_dict = torch.load(
                path, map_location="cpu", mmap=True, weights_only=True
            )
            module.load_state_dict(state_dict, strict=True, assign=True)
            del state_dict

        # Rotary frequencies are nonpersistent and absent from the checkpoint.
        for module in self.bb_tok.modules():
            if isinstance(module, RotaryEmbedding):
                module.inv_freq = module._compute_inv_freq(device="cpu")

        meta_tensors = [
            name
            for name, tensor in (*self.named_parameters(), *self.named_buffers())
            if tensor.is_meta
        ]
        if meta_tensors:
            raise RuntimeError(
                "Checkpoint loading left tensors on the meta device: "
                + ", ".join(meta_tensors[:5])
            )

    @property
    def d_model(self) -> int:
        return self.cfg.d_model

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def tokenize(
        self,
        sequence: str,
        atom37_coords: np.ndarray | torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Tokenize apo structure, masking residues with incomplete backbone coordinates.

        Parameters
        ----------
        sequence : str
            Amino acid sequence of the protein.
        atom37_coords : np.ndarray | torch.Tensor
            Full-atom coordinates of shape (L, 37, 3), with NaN for missing atoms.

        Returns
        -------
        dict[str, torch.Tensor]
            seq_token_id, bb_token_id, and fa_token_id arrays of shape (L,).
            Structure token IDs are -1 where backbone coordinates are incomplete.
        """
        device = self.device
        if isinstance(atom37_coords, np.ndarray):
            atom37_coords = torch.from_numpy(atom37_coords)
        atom37_coords = atom37_coords.to(device)

        length = len(sequence)
        if atom37_coords.shape != (length, 37, 3):
            raise ValueError(
                f"Expected atom37_coords to have shape ({length}, 37, 3), "
                f"but got {atom37_coords.shape}"
            )

        seq_tok_id = torch.tensor(
            C.sequence.encode_protein_sequence(sequence), dtype=torch.long, device=device
        )
        aatypes = self.seq_to_restype[seq_tok_id]
        backbone_mask = torch.isfinite(atom37_coords[:, :3]).all(dim=(-1, -2))
        bb_tok_id = self.bb_tok.tokenize(atom37_coords[..., :3, :])
        fa_tok_id = self.fa_tok.tokenize(aatypes, atom37_coords, attn_mask=backbone_mask)
        return {
            "seq_token_id": seq_tok_id,
            "bb_token_id": bb_tok_id.masked_fill(~backbone_mask, -1),
            "fa_token_id": fa_tok_id.masked_fill(~backbone_mask, -1),
        }

    def tokenize_batch(
        self,
        batch: list[tuple[str, np.ndarray | torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """Tokenize apo structure with the structure encoder's tokenizer.

        Parameters
        ----------
        batch: list[tuple[str, torch.Tensor]]
            A batch of (sequence, atom37_coords) pairs, where:
            - sequence: str, amino acid sequence of the protein.
            - atom37_coords: torch.Tensor, full-atom coordinates of shape (L, 37, 3).

        Returns
        -------
        token_ids: dict[str, torch.Tensor]
            - "seq_token_id": Tensor of shape (B, L) containing sequence token IDs.
            - "bb_struct_token_id": Tensor of shape (B, L) containing backbone
                structure token IDs.
            - "fa_struct_token_id": Tensor of shape (B, L) containing full-atom
                structure token IDs.
        """
        # Convert all coordinates to tensors and move to the correct device
        device = self.device
        to_tensor = lambda x: (  # noqa
            torch.as_tensor(x, device=device) if isinstance(x, np.ndarray) else x
        )
        batch: list[tuple[str, torch.Tensor]] = [
            (seq, to_tensor(coords)) for seq, coords in batch
        ]

        # Validate inputs
        if len(batch) == 0:
            raise ValueError("Batch cannot be empty.")

        for sequence, atom37_coords in batch:
            length = len(sequence)
            if atom37_coords.shape != (length, 37, 3):
                raise ValueError(
                    f"Expected atom37_coords to have shape ({length}, 37, 3), "
                    f"but got {atom37_coords.shape}"
                )

        # Stack inputs into tensors
        B = len(batch)
        L = max(len(sequence) for sequence, _ in batch)

        pad_idx = C.sequence.PAD_TOKEN_INDEX
        seq_tok_ids = torch.full((B, L), pad_idx, dtype=torch.long)
        coords = torch.full((B, L, 37, 3), float("nan"), dtype=torch.float)
        mask = torch.zeros((B, L), dtype=torch.bool)
        for i, (sequence, atom37_coords) in enumerate(batch):
            length = len(sequence)
            seq_tok_ids[i, :length] = torch.tensor(
                C.sequence.encode_protein_sequence(sequence), dtype=torch.long
            )
            coords[i, :length] = atom37_coords
            mask[i, :length] = True

        seq_tok_ids, coords, mask = (
            seq_tok_ids.to(device),
            coords.to(device),
            mask.to(device),
        )

        aatypes = self.seq_to_restype[seq_tok_ids]

        # Tokenize backbone and full-atom structures
        backbone_mask = mask & torch.isfinite(coords[..., :3, :]).all(dim=(-1, -2))
        bb_tok_ids = self.bb_tok.tokenize_batch(coords[..., :3, :])
        fa_tok_ids = self.fa_tok.tokenize_batch(aatypes, coords, attn_mask=backbone_mask)

        seq_tok_ids.masked_fill_(~mask, pad_idx)
        bb_tok_ids.masked_fill_(~backbone_mask, -1)
        fa_tok_ids.masked_fill_(~backbone_mask, -1)
        return {
            "seq_token_id": seq_tok_ids,
            "bb_token_id": bb_tok_ids,
            "fa_token_id": fa_tok_ids,
        }

    def forward(self, f_input: FoldingInput) -> torch.Tensor:
        """Forward pass of sequence representation module.

        Parameters
        ----------
        f_input: FoldingInput
            The input features

        Returns
        -------
        x_token: torch.Tensor
            Tensor of shape (B, Ntoken, D) containing sequence representations,
            where Ntoken is the number of tokens and D is the model dimension.
        """
        device_type = f_input.device.type
        with (
            torch.no_grad(),
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device_type == "cuda"
            else contextlib.nullcontext(),
        ):
            return self._forward(f_input)

    def _forward(self, f_input: FoldingInput) -> torch.Tensor:
        """Forward pass of sequence representation module.

        Parameters
        ----------
        f_input: FoldingInput
            The input features

        Returns
        -------
        x_token: torch.Tensor
            Tensor of shape (B, Ntoken, D) containing sequence representations.
        """
        # NOTE: padding tokens have bb_token_ids of -1, which will be masked out
        # in the attention computation.
        seq_token_ids = f_input.sequence.seq_token_id  # [B, L]
        bb_token_ids = f_input.sequence.bb_struct_token_id  # [B, L, Napo]
        fa_token_ids = f_input.sequence.fa_struct_token_id  # [B, L, Napo]
        batch_size, seq_len, num_apo = bb_token_ids.shape

        # HACK: (Seonghwan) Since we use the shared sequence vocab for both sequence
        # and structure encoder, structure encoder does not have vocab ids for
        # dna and rna tokens. We set those to 0 to prevent out-of-vocab errors.
        seq_mask = f_input.sequence.pad_mask & f_input.sequence.is_protein
        seq_token_ids = seq_token_ids.masked_fill(~seq_mask, 0)

        seq_id = f_input.sequence.asym_id
        pos_id = f_input.sequence.pos_id

        def expand_over_apo(x: torch.Tensor) -> torch.Tensor:
            return (
                x[:, None, :]
                .expand(batch_size, num_apo, seq_len)
                .reshape(batch_size * num_apo, seq_len)
            )

        seq_token_ids = expand_over_apo(seq_token_ids)
        seq_id = expand_over_apo(seq_id)
        pos_id = expand_over_apo(pos_id)
        bb_token_ids = bb_token_ids.transpose(1, 2).reshape(batch_size * num_apo, seq_len)
        fa_token_ids = fa_token_ids.transpose(1, 2).reshape(batch_size * num_apo, seq_len)

        # Mask out unallowed tokens
        allow_mask = bb_token_ids != -1  # we set bb_token_id to -1 for invalid tokens.
        seq_id = seq_id.masked_fill(~allow_mask, -1)  # entity id >= 1 for valid tokens

        x = self.encoder(
            seq_token_ids + self.offset,
            bb_token_ids + self.offset,
            fa_token_ids + self.offset,
            seq_id=seq_id,
            pos_id=pos_id,
        )
        x = x * allow_mask[..., None]  # mask out invalid tokens
        x = x.reshape(batch_size, num_apo, seq_len, -1)
        allow_mask = allow_mask.reshape(batch_size, num_apo, seq_len)
        num_valid = allow_mask.sum(dim=1).clamp(min=1)
        x = x.sum(dim=1) / num_valid[..., None]

        # sequence -> token index mapping
        batch_index = torch.arange(x.shape[0], device=x.device)[:, None]
        seq_token_index = f_input.token.seq_token_index
        x = x[batch_index, seq_token_index]  # [B, Ntoken, D]

        # mask out non-protein tokens
        token_mask = f_input.token.pad_mask & f_input.token.is_protein
        x.masked_fill_(~token_mask[..., None], 0.0)
        return x
