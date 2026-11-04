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

from dataclasses import dataclass
from functools import partial

import torch
import torch.nn.functional as F

from kfold.data.types.model_input import FoldingInput
from kfold.model.layers.folding.attention_pair_bias import SelfAttentionPairBias
from kfold.model.layers.folding.transition import Transition
from kfold.model.modules.tri_stack import TriangularBlock
from kfold.model.primitives import LayerNorm, LinearNoBias
from kfold.model.primitives.utils import add, gather_dim, get_context_dtype
from kfold.utils.checkpointing import checkpoint_blocks
from kfold.utils.config import configurable


def to_atom_layout(
    x: torch.Tensor,
    num_token_atoms: torch.Tensor,
    max_total_atoms: int,
) -> torch.Tensor:
    """Convert a tensor from token representation to dense representation.

    Parameters
    ----------
    x : torch.Tensor
        Tensor of shape (*, Ntoken, 24, C).
    num_token_atoms : torch.Tensor
        Tensor of shape (*, Ntoken) containing the number of atoms
        for each token.
    max_total_atoms : int
        The maximum total number of atoms across all tokens. This is used to
        determine the output shape.

    Returns
    -------
    x_atom: torch.Tensor
        Tensor of shape (B, Natom, C).
    """
    *batch_dims, L, _, C = x.shape

    arange = torch.arange(24, device=x.device)
    mask = arange < num_token_atoms.unsqueeze(-1)
    offsets = torch.cumsum(num_token_atoms, dim=-1) - num_token_atoms
    target_idx = offsets.unsqueeze(-1) + arange
    target_idx = target_idx.masked_fill(~mask, 0)
    x_flat = x.reshape(*batch_dims, -1, C)
    target_idx_flat = target_idx.reshape(*batch_dims, -1, 1).expand_as(x_flat)
    x_flat_masked = x_flat.masked_fill(~mask.reshape(*batch_dims, -1, 1), 0)

    # Scatter add the valid atoms into the dense atom representation
    out = torch.zeros(
        *batch_dims, max_total_atoms, C, device=x.device, dtype=x.dtype
    ).scatter_add_(dim=-2, index=target_idx_flat, src=x_flat_masked)

    return out


class ConfidencePairSingleStack(torch.nn.Module):
    """Joint confidence stack with pair updates and z-to-s message passing."""

    def __init__(
        self,
        channel_s: int,
        channel_z: int,
        num_heads: int,
        num_blocks: int,
        dropout: float,
        blocks_per_ckpt: int | None = None,
        kernel_backend: str = "torch",
    ) -> None:
        super().__init__()
        self.kernel_backend = kernel_backend
        self.blocks_per_ckpt = blocks_per_ckpt
        self.blocks = torch.nn.ModuleList(
            [
                ConfidencePairSingleBlock(
                    channel_s=channel_s,
                    channel_z=channel_z,
                    num_heads=num_heads,
                    dropout=dropout,
                    kernel_backend=self.kernel_backend,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(
        self,
        z: torch.Tensor,
        s: torch.Tensor,
        pair_mask: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        blocks = [
            partial(
                b,
                pair_mask=pair_mask,
                mask=mask,
            )
            for b in self.blocks
        ]
        return checkpoint_blocks(
            blocks,
            (z, s),
            self.blocks_per_ckpt,
            use_reentrant=False,
        )


class ConfidencePairSingleBlock(torch.nn.Module):
    def __init__(
        self,
        channel_s: int,
        channel_z: int,
        num_heads: int,
        dropout: float,
        kernel_backend: str = "torch",
    ) -> None:
        super().__init__()
        self.kernel_backend = kernel_backend
        self.pair_block = TriangularBlock(
            channel_z,
            dropout,
            kernel_backend=self.kernel_backend,
        )
        self.layernorm_z = LayerNorm(channel_z)
        self.linear_pair_bias = LinearNoBias(channel_z, num_heads)
        self.attention = SelfAttentionPairBias(
            channel_a=channel_s,
            num_heads=num_heads,
            channel_s=None,
            call_site="confidence",
            kernel_backend="sdpa",
        )
        self.transition = Transition(channel_s, expansion_factor=4)

    def forward(
        self,
        z: torch.Tensor,
        s: torch.Tensor,
        pair_mask: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _add = partial(add, inplace=not self.training)

        z = self.pair_block(z, pair_mask)
        pair_bias = self.linear_pair_bias(self.layernorm_z(z))
        pair_bias = pair_bias.movedim(-1, -3)  # [B, H, L, L]
        attention_update = self.attention(
            s,
            None,
            pair_bias,
            mask,
        )
        s = _add(
            s,
            attention_update,
        )
        s = _add(s, self.transition(s))
        return z, s


@configurable
class ConfidenceHead(torch.nn.Module):
    """Base class for confidence head modules.
    See Section 4.3.5 Algorithm 31 Confidence head
    """

    @dataclass(kw_only=True)
    class Config:
        """Base configuration class for confidence head modules.

        Parameters
        ----------
        channel_s : int
            The channel of single representation.
        channel_z : int
            The channel of pair representation.
        """

        channel_s: int = 384
        channel_z: int = 256
        num_heads_attn: int = 16
        num_blocks: int = 4
        dropout: float = 0.25
        num_bins: int = 39
        min_dist: float = 3.25
        max_dist: float = 50.75

        # head dimensions
        min_pae_dist: float = 0.0
        max_pae_dist: float = 32.0
        num_pae_bins: int = 64
        min_pde_dist: float = 0.0
        max_pde_dist: float = 32.0
        num_pde_bins: int = 64
        num_plddt_bins: int = 50

        # For training
        blocks_per_ckpt: int | None = None

    def __init__(
        self,
        cfg: Config,
        kernel_backend: str = "torch",
    ):
        super().__init__()
        self.kernel_backend = kernel_backend
        self.num_pae_bins = cfg.num_pae_bins
        self.num_pde_bins = cfg.num_pde_bins
        self.num_plddt_bins = cfg.num_plddt_bins
        self.is_compiled = False

        def create_bin_centers(d_min: float, d_max: float, num_bins: int) -> torch.Tensor:
            bin_size = (d_max - d_min) / num_bins
            return torch.linspace(
                d_min + bin_size / 2,
                d_max - bin_size / 2,
                num_bins,
            )

        self.register_buffer(
            "pae_bin_centers",
            create_bin_centers(cfg.min_pae_dist, cfg.max_pae_dist, cfg.num_pae_bins),
            persistent=False,
        )
        self.register_buffer(
            "pde_bin_centers",
            create_bin_centers(cfg.min_pde_dist, cfg.max_pde_dist, cfg.num_pde_bins),
            persistent=False,
        )
        self.register_buffer(
            "plddt_bin_centers",
            create_bin_centers(0.0, 1.0, cfg.num_plddt_bins),
            persistent=False,
        )

        self.linear_s1 = LinearNoBias(cfg.channel_s, cfg.channel_z)
        self.linear_s2 = LinearNoBias(cfg.channel_s, cfg.channel_z)
        self.s_lm_to_s = torch.nn.Sequential(
            LayerNorm(cfg.channel_s),
            LinearNoBias(cfg.channel_s, cfg.channel_s),
        )

        self.num_bins = cfg.num_bins
        boundaries = torch.linspace(cfg.min_dist, cfg.max_dist, self.num_bins - 1)
        self.register_buffer("boundaries", boundaries, persistent=False)
        self.linear_distogram = LinearNoBias(self.num_bins, cfg.channel_z)

        self.stack = ConfidencePairSingleStack(
            channel_s=cfg.channel_s,
            channel_z=cfg.channel_z,
            num_heads=cfg.num_heads_attn,
            num_blocks=cfg.num_blocks,
            dropout=cfg.dropout,
            blocks_per_ckpt=cfg.blocks_per_ckpt,
            kernel_backend=self.kernel_backend,
        )

        self.pae_head = torch.nn.Sequential(
            LayerNorm(cfg.channel_z),
            LinearNoBias(cfg.channel_z, self.num_pae_bins, init="final"),
        )
        self.pde_head = torch.nn.Sequential(
            LayerNorm(cfg.channel_z),
            LinearNoBias(cfg.channel_z, self.num_pde_bins, init="final"),
        )
        self.plddt_head = torch.nn.Sequential(
            LayerNorm(cfg.channel_s),
            LinearNoBias(cfg.channel_s, 24 * self.num_plddt_bins, init="final"),
        )
        self.resolved_head = torch.nn.Sequential(
            LayerNorm(cfg.channel_s),
            LinearNoBias(cfg.channel_s, 24 * 2, init="final"),
        )

    def do_compile(self, **kwargs):
        """Compile the score model module."""
        self._compile(**kwargs)
        self.is_compiled = True

    def _compile(self, **kwargs):
        """Compile the triangular stack."""
        self.stack = torch.compile(self.stack, **kwargs)

    def get_stack(self) -> ConfidencePairSingleStack:
        """Get the triangular stack"""
        if self.is_compiled and not self.training:
            return self.stack
        return self.stack

    def forward(
        self,
        f_input: FoldingInput,
        s_inputs: torch.Tensor,
        s_lm: torch.Tensor,
        z: torch.Tensor,
        x_pred: torch.Tensor,
        *,
        apo_relation: torch.nn.Module | None = None,
        apo_context: dict | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass of confidence head module.

        Parameters
        ----------
        f_input : FoldingInput
            The folding input features.
        s_inputs : torch.Tensor
            Tensor of shape (B, L, C_s) containing input single representation.
        s_lm : torch.Tensor
            Tensor of shape (B, L, C_s_lm) containing LM single representation.
        z: torch.Tensor
            Tensor of shape (B, L, L, C_s) containing pair representation.
        x_pred: torch.Tensor
            Tensor of shape (B, N, Latom, 3) containing predicted coordinates.

        Returns
        -------
        confidence_out: dict[str, torch.Tensor]
            Confidence logits and their corresponding bin centers.
        """
        # Get the device and dtype for computations
        device = s_inputs.device
        dtype = get_context_dtype(device.type)

        # Extract the representative atom coordinates
        # [B, Natom, 3] -> [B, L, 3]
        B, N, Natom, _ = x_pred.shape
        L = s_inputs.shape[1]
        repr_idx = f_input.token.repr_index.unsqueeze(-2)  # [B, 1, Ntoken]
        x_repr = gather_dim(x_pred, dim=-2, index=repr_idx[..., None])

        if apo_relation is not None and apo_context is None:
            raise ValueError("Candidate-conditioned apo confidence requires apo context.")

        s_inputs, s_lm, z = s_inputs.to(dtype), s_lm.to(dtype), z.to(dtype)
        trunk_z = z
        s = self.s_lm_to_s(s_lm)

        z = (
            z
            + self.linear_s1(s_inputs)[..., None, :, :]
            + self.linear_s2(s_inputs)[..., :, None, :]
        )

        # Prepare output tensors
        pae_logits = torch.zeros(
            (B, N, L, L, self.num_pae_bins), device=device, dtype=dtype
        )
        pde_logits = torch.zeros(
            (B, N, L, L, self.num_pde_bins), device=device, dtype=dtype
        )
        plddt_logits = torch.zeros(
            (B, N, L, 24, self.num_plddt_bins), device=device, dtype=torch.float32
        )
        resolved_logits = torch.zeros(
            (B, N, L, 24, 2), device=device, dtype=torch.float32
        )
        # Process each sample in the batch separately to save memory
        mask = f_input.token.pad_mask
        for i in range(N):
            candidate_z, candidate_s = z, s
            if apo_relation is not None:
                # Keep the apo axis until the particular candidate queries it.
                # The relation is passed in, rather than registered twice under
                # both the confidence head and the mixer checkpoint namespace.
                delta_z, delta_lm = apo_relation(
                    f_input, apo_context, trunk_z, s_inputs, x_repr[:, i]
                )
                candidate_z = z + delta_z.to(dtype)
                candidate_s = self.s_lm_to_s(s_lm + delta_lm.to(dtype))
            _pae_logits, _pde_logits, _plddt_logits, _resolved_logits = (
                self.forward_single(candidate_z, candidate_s, x_repr[:, i], mask=mask)
            )
            pae_logits[:, i] = _pae_logits
            pde_logits[:, i] = _pde_logits
            plddt_logits[:, i] = _plddt_logits
            resolved_logits[:, i] = _resolved_logits

        with torch.autocast(device.type, enabled=False):
            # Reshape plddt and resolved logits to (B, N, Natom, ...)
            num_token_atoms = f_input.token.num_atoms.unsqueeze(-2).expand(-1, N, -1)
            plddt_logits = to_atom_layout(
                plddt_logits, num_token_atoms, f_input.num_atoms
            )
            resolved_logits = to_atom_layout(
                resolved_logits, num_token_atoms, f_input.num_atoms
            )

            # Mask out padding
            token_mask = f_input.token.pad_mask[..., None, :]  # [B, 1, L]
            pair_mask = (
                token_mask[..., :, None] & token_mask[..., None, :]
            )  # [B, 1, L, L]
            pair_mask = pair_mask.to(dtype)
            pae_logits = pae_logits * pair_mask[..., None]
            pde_logits = pde_logits * pair_mask[..., None]

            atom_mask = f_input.atom.pad_mask[..., None, :]  # [B, 1, Natom]
            atom_mask = atom_mask.to(torch.float32)
            plddt_logits = plddt_logits * atom_mask[..., None]
            resolved_logits = resolved_logits * atom_mask[..., None]

        return {
            "pae_logits": pae_logits,
            "pae_bin_centers": self.pae_bin_centers,
            "pde_logits": pde_logits,
            "pde_bin_centers": self.pde_bin_centers,
            "plddt_logits": plddt_logits,
            "plddt_bin_centers": self.plddt_bin_centers,
            "resolved_logits": resolved_logits,
        }

    def forward_single(
        self,
        z: torch.Tensor,
        s: torch.Tensor,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of confidence head module.

        Parameters
        ----------
        z: torch.Tensor
            Tensor of shape (B, L, L, C_s) containing pair representation.
        s: torch.Tensor
            Tensor of shape (B, L, C_s) containing projected LM single representation.
        x: torch.Tensor
            Tensor of shape (B, N, L, 3) containing predicted coordinates of
            representative atoms.
        mask: torch.Tensor
            Tensor of shape (B, L) containing the mask for valid tokens.

        Returns
        -------
        pae_logits: torch.Tensor
            Tensor of shape (B, L, L, num_pae_bins) containing predicted aligned
            error logits.
        pde_logits: torch.Tensor
            Tensor of shape (B, L, L, num_pde_bins) containing predicted distance
            error logits.
        lddt_logits: torch.Tensor
            Tensor of shape (B, L, 24, num_lddt_bins) containing predicted lddt logits.
        resolved_logits: torch.Tensor
            Tensor of shape (B, L, 24, 2) containing predicted resolved atom logits.
        """
        z, s = z.clone(), s.clone()  # Clone to avoid in-place modifications
        # Compute distogram of sampled coordinates.
        with torch.autocast(x.device.type, dtype=torch.float32), torch.no_grad():
            d = (x[..., :, None, :] - x[..., None, :, :]).norm(dim=-1)
        dgram = F.one_hot((d[..., None] > self.boundaries).sum(dim=-1), self.num_bins)

        # Jointly update pair and LM single representations.
        stack = self.get_stack()

        z = z + self.linear_distogram(dgram.to(z.dtype))
        pair_mask = mask[..., :, None] & mask[..., None, :]
        z, s = stack(z, s, pair_mask, mask)
        z, s = z.to(torch.float32), s.to(torch.float32)

        # Confidence heads.
        with torch.autocast(s.device.type, enabled=False):
            pae_logits = self.pae_head(z)  # [B, L, L, num_pae_bins]
            pde_logits = self.pde_head(z)  # [B, L, L, num_pde_bins]
            pde_logits = pde_logits + pde_logits.transpose(-2, -3)  # symmetrize

            plddt_logits = self.plddt_head(s).unflatten(-1, (24, self.num_plddt_bins))
            resolved_logits = self.resolved_head(s).unflatten(-1, (24, 2))

        return pae_logits, pde_logits, plddt_logits, resolved_logits
