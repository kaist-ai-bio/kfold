"""Colleague Pairformer baseline and an explicit mixed-s_lm outer-product ablation."""
from dataclasses import dataclass

import torch
from torch import nn

from kfold.model.modules.affinity_pairformer import AffinityPairformer


class SingleOuterProduct(nn.Module):
    """Learn a directed channel outer product, retaining only protein pairs.

    s_lm mixes sequence and projected apo information; this is not a pure apo
    embedding or a recovered backbone PP tensor. Chunking avoids [B,N,N,R,R].
    """
    def __init__(self, single_channels=384, pair_channels=256, rank=16, chunk=32):
        super().__init__()
        self.rank, self.chunk = rank, chunk
        self.norm = nn.LayerNorm(single_channels)
        self.left = nn.Linear(single_channels, rank, bias=False)
        self.right = nn.Linear(single_channels, rank, bias=False)
        self.output = nn.Linear(rank * rank, pair_channels, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, single, protein_mask):
        x = self.norm(single)
        a, b = self.left(x), self.right(x)
        result = []
        for start in range(0, single.shape[1], self.chunk):
            outer = torch.einsum('bir,bjs->bijrs', a[:, start:start+self.chunk], b)
            value = self.output(outer.flatten(-2))
            mask = protein_mask[:, start:start+self.chunk, None] & protein_mask[:, None, :]
            result.append(value * mask[..., None])
        return torch.cat(result, dim=1)


@dataclass(frozen=True)
class HeadConfig:
    variant: str = 'colleague_cross'
    channel_s: int = 384
    channel_z: int = 256
    blocks: int = 6
    heads: int = 16
    dropout: float = .1
    outer_rank: int = 16
    kernel_backend: str = 'torch'


class AffinityHead(nn.Module):
    """Same scalar architecture serves pActivity regression or a binary logit.

    colleague_cross: exact original mask and no outer module.
    full_pair_control: PP allowed, capacity-matched null-input outer adapter.
    slm_outer: same full-pair topology, real s_lm outer-product PP residual.
    """
    def __init__(self, config: HeadConfig):
        super().__init__()
        if config.variant not in ('colleague_cross', 'full_pair_control', 'slm_outer'):
            raise ValueError('Unknown controlled affinity-head variant')
        self.config = config
        self.base = AffinityPairformer(AffinityPairformer.Config(
            channel_s=config.channel_s, channel_z=config.channel_z,
            num_blocks=config.blocks, num_heads_attn=config.heads,
            dropout=config.dropout, blocks_per_ckpt=1,
            cross_pair_only=config.variant == 'colleague_cross'),
            kernel_backend=config.kernel_backend)
        self.outer = None if config.variant == 'colleague_cross' else SingleOuterProduct(
            config.channel_s, config.channel_z, config.outer_rank)

    def forward(self, **features):
        if self.outer is None:
            return self.base(**features)
        single = features['s_lm']
        if self.config.variant == 'full_pair_control':
            single = single * 0  # Same trainable capacity/compute, no new information.
        residual = self.outer(single, features['protein_mask'] & features['token_mask'])
        return self.base(**features, pair_residual=residual)
