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

"""Apo structure embedding module for multimer trunk inputs."""

from dataclasses import dataclass
from functools import partial

import torch

from kfold.data.types.model_input import FoldingInput
from kfold.model.layers.folding.embeddings import RelativePositionEncoding
from kfold.model.layers.folding.transition import Transition
from kfold.model.primitives import (
    DropoutColumnwise,
    DropoutRowwise,
    LayerNorm,
    LinearNoBias,
    TriangleAttentionEndingNode,
    TriangleAttentionStartingNode,
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from kfold.model.primitives.utils import add
from kfold.utils.checkpointing import checkpoint_blocks
from kfold.utils.config import configurable


class PairformerStack(torch.nn.Module):
    """Pairformer stack."""

    def __init__(
        self,
        channel_z: int = 64,
        num_tri_heads: int = 4,
        num_blocks: int = 48,
        dropout: float = 0.1,
        blocks_per_ckpt: int | None = None,
        kernel_backend: str = "torch",
    ):
        """Initialize the Pairformer module."""
        super().__init__()
        self.kernel_backend = kernel_backend
        self.channel_z: int = channel_z
        self.num_blocks: int = num_blocks
        self.dropout: float = dropout

        self.blocks_per_ckpt: int | None = blocks_per_ckpt

        self.blocks = torch.nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(
                PairformerBlock(
                    self.channel_z,
                    num_tri_heads,
                    self.dropout,
                    kernel_backend=self.kernel_backend,
                )
            )

    def forward(
        self,
        z: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Perform the forward pass.

        Parameters
        ----------
        z : torch.Tensor
            The pairwise embeddings
        pair mask : torch.Tensor
            The pair token mask
        Returns
        -------
        torch.Tensor
            The updated sequence embeddings.
        """
        blocks = [
            partial(
                b,
                pair_mask=pair_mask,
            )
            for b in self.blocks
        ]
        z = checkpoint_blocks(
            blocks,
            (z,),
            self.blocks_per_ckpt,
            use_reentrant=False,
        )[0]

        return z


class PairformerBlock(torch.nn.Module):
    """Pairformer block."""

    def __init__(
        self,
        channel_z: int = 64,
        num_tri_heads: int = 4,
        dropout: float = 0.1,
        kernel_backend: str = "torch",
    ):
        """Initialize the Pairformer module.

        Parameters
        ----------
        channel_z : int
            The token pairwise embedding size.
        dropout : float, optional
            The dropout rate, by default 0.1
        """
        super().__init__()
        self.kernel_backend = kernel_backend
        self.channel_z: int = channel_z
        self.tri_mul_out = TriangleMultiplicationOutgoing(
            channel_z, kernel_backend=self.kernel_backend
        )
        self.tri_mul_in = TriangleMultiplicationIncoming(
            channel_z, kernel_backend=self.kernel_backend
        )
        self.tri_att_start = TriangleAttentionStartingNode(
            channel_z, num_tri_heads, kernel_backend=self.kernel_backend
        )
        self.tri_att_end = TriangleAttentionEndingNode(
            channel_z, num_tri_heads, kernel_backend=self.kernel_backend
        )
        self.transition_z = Transition(channel_z, expansion_factor=2)
        self.dropout_rowwise_z = DropoutRowwise(dropout)
        self.dropout_columnwise_z = DropoutColumnwise(dropout)

    def forward(
        self,
        z: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Perform the forward pass.
        See Section 3.6 Algorithm 20 Pairformer Stack
        """
        _add = partial(add, inplace=not self.training)

        z = _add(
            z,
            self.dropout_rowwise_z(self.tri_mul_out(z, pair_mask)),
        )
        z = _add(
            z,
            self.dropout_rowwise_z(self.tri_mul_in(z, pair_mask)),
        )
        z = _add(
            z,
            self.dropout_rowwise_z(self.tri_att_start(z, pair_mask)),
        )
        z = _add(
            z,
            self.dropout_columnwise_z(self.tri_att_end(z, pair_mask)),
        )
        z = _add(z, self.transition_z(z))
        return z * pair_mask[..., None]


def compute_distogram(
    coords: torch.Tensor,
    mask: torch.Tensor,
    boundaries: torch.Tensor,
) -> torch.Tensor:
    """Compute one-hot pseudo-beta distogram bins.

    Parameters
    ----------
    coords
        Tensor of shape [B, T, L, 3].
    mask
        Boolean tensor of shape [B, T, L].
    boundaries
        Distogram lower breaks of shape [num_bins].


    Returns
    -------
    distogram
        Tensor of shape [B, T, L, L, num_bins]
    """
    with torch.autocast(coords.device.type, enabled=False):
        coords, boundaries = coords.float(), boundaries.float()
        lower_breaks = torch.square(boundaries)
        upper_breaks = torch.cat(
            [lower_breaks[1:], lower_breaks.new_tensor([1e8])],
            dim=-1,
        )
        diff = coords[..., None, :, :] - coords[..., :, None, :]
        dist2 = torch.sum(torch.square(diff), dim=-1, keepdim=True)
        distogram = (dist2 > lower_breaks) * (dist2 < upper_breaks)
        pair_mask = mask[..., :, None] & mask[..., None, :]
        distogram = distogram * pair_mask[..., None]
    return distogram


def compute_unit_vector(
    frame_coords: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute local-frame backbone unit vectors.

    Parameters
    ----------
    frame_coords
        Tensor of shape [B, T, L, 3, 3] containing backbone
        coordinates in the order of N, CA, C.
    mask
        Boolean tensor of shape [B, T, L] indicating valid residues.
    eps
        Small value to avoid division by zero in normalization.

    Returns
    -------
    unit_vector
        Tensor of shape [B, T, L, L, 3] containing unit vectors
    """
    with torch.autocast(frame_coords.device.type, enabled=False):
        x_n, x_ca, x_c = frame_coords.float().unbind(dim=-2)

        def normalize(x: torch.Tensor) -> torch.Tensor:
            norm2 = torch.sum(torch.square(x), dim=-1, keepdim=True)
            norm = torch.sqrt(torch.clamp(norm2, min=eps**2))
            return x / norm

        e0 = normalize(x_n - x_ca)
        e1 = x_c - x_ca
        e1 = normalize(e1 - torch.sum(e1 * e0, dim=-1, keepdim=True) * e0)
        e2 = torch.linalg.cross(e0, e1, dim=-1)
        basis = torch.stack([e0, e1, e2], dim=-2)

        ca_vec = x_ca.unsqueeze(-3) - x_ca.unsqueeze(-2)
        local_vec = (basis.unsqueeze(-3) @ ca_vec.unsqueeze(-1)).squeeze(-1)
        unit_vector = normalize(local_vec)
        pair_mask = mask[..., :, None] & mask[..., None, :]
        unit_vector = unit_vector * pair_mask[..., None]
    return unit_vector


@configurable
class ApoModule(torch.nn.Module):
    """Embed apo structure features and return a pair-representation update."""

    @dataclass(kw_only=True)
    class Config:
        channel_z: int = 256
        channel_apo: int = 64
        num_blocks: int = 2
        num_tri_heads: int = 4
        dropout: float = 0.1
        num_distogram_bins: int = 39
        min_dist: float = 3.25
        max_dist: float = 50.75
        blocks_per_ckpt: int | None = None

    def __init__(
        self,
        cfg: Config,
        kernel_backend: str = "torch",
    ) -> None:
        super().__init__()
        self.kernel_backend = kernel_backend
        self.channel_z = cfg.channel_z
        self.channel_apo = cfg.channel_apo

        boundaries = torch.linspace(cfg.min_dist, cfg.max_dist, cfg.num_distogram_bins)
        self.register_buffer("distogram_boundaries", boundaries, persistent=False)

        # NOTE: 32 is entire number of restypes.
        input_dim = (
            cfg.num_distogram_bins + 1 + 3 + 1 + 2 * 32
        )  # distogram, pseudo_beta_mask, restype_i, restype_j, unit_vector, backbone_mask
        self.linear_apo = LinearNoBias(input_dim, self.channel_apo, init="relu")
        self.rel_pos_encoding = RelativePositionEncoding(r_max=32, s_max=2)
        self.linear_rel_pos = LinearNoBias(
            self.rel_pos_encoding.dimension, self.channel_apo
        )
        self.stack = PairformerStack(
            channel_z=self.channel_apo,
            num_tri_heads=cfg.num_tri_heads,
            num_blocks=cfg.num_blocks,
            dropout=cfg.dropout,
            blocks_per_ckpt=cfg.blocks_per_ckpt,
            kernel_backend=self.kernel_backend,
        )
        self.layernorm_out = LayerNorm(self.channel_apo)
        self.linear_out = LinearNoBias(self.channel_apo, self.channel_z, init="relu")

    def forward(
        self,
        f_input: FoldingInput,
        *,
        feature_taps: dict | None = None,
    ) -> torch.Tensor:
        """Return an apo pair update of shape ``[B, L, L, C_z]``."""
        pseudo_beta = f_input.token.apo_repr_coords.transpose(1, 2)
        pseudo_beta_mask = f_input.token.apo_repr_mask.transpose(1, 2)
        backbone_coords = f_input.token.apo_frame_coords.permute(0, 2, 1, 3, 4)
        backbone_frame_mask = f_input.token.apo_frame_mask.transpose(1, 2)
        apo_uid = f_input.token.apo_uid[:, None, :]

        B, T, L, _ = pseudo_beta.shape
        if T == 0:
            if feature_taps is not None:
                feature_taps["apo_pair"] = pseudo_beta.new_zeros(
                    B, 0, L, L, self.channel_apo
                ).detach()
                feature_taps["apo_pair_mask"] = torch.zeros(
                    B, 0, L, L, device=pseudo_beta.device, dtype=torch.bool
                )
            return pseudo_beta.new_zeros(B, L, L, self.channel_z)

        token_mask = f_input.token.pad_mask[:, None, :]
        source_token_mask = token_mask & (pseudo_beta_mask | backbone_frame_mask)
        valid_uid = apo_uid >= 0
        same_apo = apo_uid[..., :, None] == apo_uid[..., None, :]
        pair_mask = (
            same_apo
            & valid_uid[..., :, None]
            & valid_uid[..., None, :]
            & source_token_mask[..., :, None]
            & source_token_mask[..., None, :]
        )
        pair_mask_f = pair_mask.float()

        b_backbone_frame_mask = (
            backbone_frame_mask[..., :, None] & backbone_frame_mask[..., None, :]
        )
        b_backbone_frame_mask &= pair_mask
        b_backbone_frame_mask = b_backbone_frame_mask.float()

        b_pseudo_beta_mask = (
            pseudo_beta_mask[..., :, None] & pseudo_beta_mask[..., None, :]
        )
        b_pseudo_beta_mask &= pair_mask
        b_pseudo_beta_mask = b_pseudo_beta_mask.float()

        f_distogram = compute_distogram(
            pseudo_beta,
            pseudo_beta_mask,
            self.distogram_boundaries,
        ).float()
        f_distogram *= pair_mask_f[..., None]

        f_unit_vector = compute_unit_vector(
            backbone_coords,
            backbone_frame_mask,
        ).float()
        f_unit_vector *= pair_mask_f[..., None]

        restype = f_input.token.res_type
        restype = restype.unsqueeze(-3)  #  [B, L, 32] -> [B, 1, L, 32]
        restype_i = restype[..., :, None, :].expand(B, T, L, L, -1)
        restype_j = restype[..., None, :, :].expand(B, T, L, L, -1)

        a = torch.cat(
            [
                f_distogram,
                b_pseudo_beta_mask[..., None],
                f_unit_vector,
                b_backbone_frame_mask[..., None],
                restype_i,
                restype_j,
            ],
            dim=-1,
        )
        a *= pair_mask_f[..., None]

        # Apo structures are independent and share the same stack weights, so
        # process the apo axis as part of the batch dimension.
        v = self.linear_apo(a)
        rel_pos = self.linear_rel_pos(self.rel_pos_encoding(f_input, v.dtype))
        v = v + rel_pos[:, None] * pair_mask[..., None]
        v = self.stack(
            v.flatten(0, 1),
            pair_mask.flatten(0, 1),
        ).unflatten(0, (B, T))
        v = self.layernorm_out(v)

        # Average over apo structures
        v = v * pair_mask[..., None]
        if feature_taps is not None:
            feature_taps["apo_pair"] = v.detach()
            feature_taps["apo_pair_mask"] = pair_mask.detach()
        u = v.sum(dim=1)
        num_valid_apo = pair_mask.sum(dim=1).clamp(min=1)
        u = u / num_valid_apo[..., None]

        return self.linear_out(torch.relu(u))
