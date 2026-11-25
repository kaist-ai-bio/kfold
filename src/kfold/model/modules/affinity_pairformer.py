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

"""Shared cross-pair affinity encoder and the two Ratio745 readouts."""

from copy import deepcopy

import torch
from torch import Tensor, nn

from kfold.model.modules.confidence_head import ConfidencePairSingleStack
from kfold.model.primitives import LayerNorm, LinearNoBias


class AffinityPairformer(nn.Module):
    """Pool ligand-related pair and single representations for affinity."""

    def __init__(
        self, *, channel_s: int, channel_z: int, blocks: int, heads: int, dropout: float
    ) -> None:
        super().__init__()
        self.s_lm_to_s = nn.Sequential(
            LayerNorm(channel_s), LinearNoBias(channel_s, channel_s)
        )
        self.s_to_z_left = LinearNoBias(channel_s, channel_z)
        self.s_to_z_right = LinearNoBias(channel_s, channel_z)
        self.distogram_to_z = LinearNoBias(3, channel_z)
        self.stack = ConfidencePairSingleStack(
            channel_s=channel_s,
            channel_z=channel_z,
            num_heads=heads,
            num_blocks=blocks,
            dropout=dropout,
            kernel_backend="torch",
        )
        self.single_to_pair = LinearNoBias(channel_s, channel_z)

    def forward(
        self,
        *,
        s_inputs: Tensor,
        s_lm: Tensor,
        z: Tensor,
        distogram_features: Tensor,
        token_mask: Tensor,
        protein_mask: Tensor,
        ligand_mask: Tensor,
    ) -> Tensor:
        dtype = z.dtype
        s = self.s_lm_to_s(s_lm.to(dtype))
        z = (
            z
            + self.s_to_z_left(s_inputs.to(dtype))[:, :, None, :]
            + self.s_to_z_right(s_inputs.to(dtype))[:, None, :, :]
            + self.distogram_to_z(distogram_features.to(dtype))
        )
        protein_ligand = (protein_mask[:, :, None] & ligand_mask[:, None, :]) | (
            ligand_mask[:, :, None] & protein_mask[:, None, :]
        )
        ligand_ligand = ligand_mask[:, :, None] & ligand_mask[:, None, :]
        all_pairs = token_mask[:, :, None] & token_mask[:, None, :]
        pair_mask = all_pairs & (protein_ligand | ligand_ligand)
        z = z * pair_mask[..., None]

        for block in self.stack.blocks:
            z = block.pair_block(z, pair_mask) * pair_mask[..., None]
            pair_bias = block.linear_pair_bias(block.layernorm_z(z)).movedim(-1, -3)
            pair_bias = pair_bias.masked_fill(~pair_mask[:, None], -1e9)
            s = s + block.attention(s, None, pair_bias, token_mask)
            s = s + block.transition(s)

        diagonal = torch.eye(z.shape[1], device=z.device, dtype=torch.bool)[None]
        pool_mask = pair_mask & (protein_ligand | (ligand_ligand & ~diagonal))
        if not pool_mask.any():
            raise ValueError("An affinity input must include a protein–ligand pair.")
        pair_weight = pool_mask.to(z.dtype)
        pooled_z = (z * pair_weight[..., None]).sum(dim=(-3, -2))
        pooled_z = pooled_z / pair_weight.sum(dim=(-2, -1)).clamp_min(1)[..., None]
        single_weight = token_mask.to(s.dtype)
        pooled_s = (s * single_weight[..., None]).sum(dim=-2)
        pooled_s = pooled_s / single_weight.sum(dim=-1, keepdim=True).clamp_min(1)
        return pooled_z + self.single_to_pair(pooled_s)


class JointAffinityHead(nn.Module):
    """Predict pActivity and a binary binding logit from a shared encoder."""

    def __init__(
        self, *, channel_s: int, channel_z: int, blocks: int, heads: int, dropout: float
    ) -> None:
        super().__init__()
        # Keep the shared.base.* parameter paths in the training checkpoint.
        self.shared = nn.ModuleDict(
            {
                "base": AffinityPairformer(
                    channel_s=channel_s,
                    channel_z=channel_z,
                    blocks=blocks,
                    heads=heads,
                    dropout=dropout,
                )
            }
        )
        self.value_head = nn.Sequential(
            LayerNorm(channel_z),
            LinearNoBias(channel_z, channel_z),
            nn.GELU(),
            LinearNoBias(channel_z, channel_z),
            nn.GELU(),
            LinearNoBias(channel_z, 1, init="final"),
        )
        self.binary_head = deepcopy(self.value_head)

    def forward(self, **features: Tensor) -> dict[str, Tensor]:
        pooled = self.shared["base"](**features)
        return {
            "value": self.value_head(pooled).squeeze(-1),
            "logits": self.binary_head(pooled).squeeze(-1),
        }
