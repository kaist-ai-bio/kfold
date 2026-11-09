"""Pairformer readout for affinity inference from frozen KFold features."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import torch

from kfold.model.modules.confidence_head import ConfidencePairSingleStack
from kfold.model.primitives import LayerNorm, LinearNoBias
from kfold.utils.checkpointing import checkpoint_blocks


class AffinityPairformer(torch.nn.Module):
    """Map final frozen trunk features to one protein--ligand affinity scalar."""

    @dataclass(kw_only=True)
    class Config:
        channel_s: int = 384
        channel_z: int = 256
        num_distogram_features: int = 3
        num_heads_attn: int = 16
        num_blocks: int = 6
        dropout: float = 0.1
        blocks_per_ckpt: int | None = None
        cross_pair_only: bool = True

    def __init__(self, config: Config, *, use_kernels: bool = False,
                 kernel_backend: str | None = None) -> None:
        super().__init__()
        self.config = config
        self.use_kernels = use_kernels
        self.s_lm_to_s = torch.nn.Sequential(
            LayerNorm(config.channel_s),
            LinearNoBias(config.channel_s, config.channel_s),
        )
        self.s_to_z_left = LinearNoBias(config.channel_s, config.channel_z)
        self.s_to_z_right = LinearNoBias(config.channel_s, config.channel_z)
        self.distogram_to_z = LinearNoBias(
            config.num_distogram_features,
            config.channel_z,
        )
        self.stack = ConfidencePairSingleStack(
            channel_s=config.channel_s,
            channel_z=config.channel_z,
            num_heads=config.num_heads_attn,
            num_blocks=config.num_blocks,
            dropout=config.dropout,
            blocks_per_ckpt=config.blocks_per_ckpt,
            kernel_backend=kernel_backend or ("cuequiv" if use_kernels else "torch"),
        )
        self.single_to_pair = LinearNoBias(config.channel_s, config.channel_z)
        self.readout = torch.nn.Sequential(
            LayerNorm(config.channel_z),
            LinearNoBias(config.channel_z, config.channel_z),
            torch.nn.GELU(),
            LinearNoBias(config.channel_z, config.channel_z),
            torch.nn.GELU(),
            LinearNoBias(config.channel_z, 1, init="final"),
        )

    def forward(
        self,
        *,
        s_inputs: torch.Tensor,
        s_lm: torch.Tensor,
        z: torch.Tensor,
        distogram_features: torch.Tensor,
        token_mask: torch.Tensor,
        protein_mask: torch.Tensor,
        ligand_mask: torch.Tensor,
        pair_residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return one scalar prediction for each cropped PL complex."""
        if z.ndim != 4:
            raise ValueError(f"Expected z [B, L, L, C], received {tuple(z.shape)}.")
        if distogram_features.shape[:3] != z.shape[:3]:
            raise ValueError("Distogram features must align with z's [B, L, L] axes.")
        if distogram_features.shape[-1] != self.config.num_distogram_features:
            raise ValueError("Unexpected number of distogram feature channels.")
        if s_inputs.shape[:2] != z.shape[:2] or s_lm.shape[:2] != z.shape[:2]:
            raise ValueError("Single and pair features must share the [B, L] axes.")
        if token_mask.shape != z.shape[:2]:
            raise ValueError("token_mask must have shape [B, L].")
        if (
            protein_mask.shape != token_mask.shape
            or ligand_mask.shape != token_mask.shape
        ):
            raise ValueError("Entity masks must have shape [B, L].")

        dtype = z.dtype
        s = self.s_lm_to_s(s_lm.to(dtype))
        z = (
            z
            + self.s_to_z_left(s_inputs.to(dtype))[:, :, None, :]
            + self.s_to_z_right(s_inputs.to(dtype))[:, None, :, :]
            + self.distogram_to_z(distogram_features.to(dtype))
        )
        if pair_residual is not None:
            if pair_residual.shape != z.shape:
                raise ValueError("Pair residual must match the complete pair layout")
            z = z + pair_residual.to(z.dtype)
        protein_ligand = (protein_mask[:, :, None] & ligand_mask[:, None, :]) | (
            ligand_mask[:, :, None] & protein_mask[:, None, :]
        )
        ligand_ligand = ligand_mask[:, :, None] & ligand_mask[:, None, :]
        all_pairs = token_mask[:, :, None] & token_mask[:, None, :]
        affinity_pairs = all_pairs & (protein_ligand | ligand_ligand)
        pair_mask = affinity_pairs if self.config.cross_pair_only else all_pairs
        if self.config.cross_pair_only:
            z = z * pair_mask[..., None]
            z, s = self._cross_only_stack(z, s, pair_mask, token_mask)
        else:
            z, s = self.stack(
                z,
                s,
                pair_mask=pair_mask,
                mask=token_mask,
            )

        diagonal = torch.eye(z.shape[1], device=z.device, dtype=torch.bool)[None]
        pool_mask = affinity_pairs & (protein_ligand | (ligand_ligand & ~diagonal))
        if not pool_mask.any():
            raise ValueError("An affinity batch must include a protein--ligand pair.")

        pair_weight = pool_mask.to(z.dtype)
        pooled_z = (z * pair_weight[..., None]).sum(dim=(-3, -2))
        pooled_z = pooled_z / pair_weight.sum(dim=(-2, -1)).clamp_min(1)[..., None]
        single_weight = token_mask.to(s.dtype)
        pooled_s = (s * single_weight[..., None]).sum(dim=-2)
        pooled_s = pooled_s / single_weight.sum(dim=-1, keepdim=True).clamp_min(1)
        return self.readout(pooled_z + self.single_to_pair(pooled_s)).squeeze(-1)

    def _cross_only_stack(
        self,
        z: torch.Tensor,
        s: torch.Tensor,
        pair_mask: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        blocks = [
            partial(
                self._cross_only_block,
                block=block,
                pair_mask=pair_mask,
                token_mask=token_mask,
            )
            for block in self.stack.blocks
        ]
        return checkpoint_blocks(
            blocks,
            (z, s),
            self.config.blocks_per_ckpt,
            use_reentrant=False,
        )

    def _cross_only_block(
        self,
        z: torch.Tensor,
        s: torch.Tensor,
        *,
        block: torch.nn.Module,
        pair_mask: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = block.pair_block(
            z,
            pair_mask,
        )
        z = z * pair_mask[..., None]
        pair_bias = block.linear_pair_bias(block.layernorm_z(z)).movedim(-1, -3)
        pair_bias = pair_bias.masked_fill(~pair_mask[:, None], -1e9)
        s = s + block.attention(s, None, pair_bias, token_mask)
        s = s + block.transition(s)
        return z, s
