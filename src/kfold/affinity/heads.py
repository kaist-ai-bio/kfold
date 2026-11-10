"""Colleague baseline and versioned mixed-s_lm pair adapter ablations."""
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

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


class SingleProductDifference(nn.Module):
    """384-channel Hadamard product + signed difference ->768->256 PP residual.

    Token-pair axes are broadcast; channel pairs are never expanded to C*C.
    Chunk checkpointing bounds concat activation storage during training. The
    old low-rank outer adapter remains available only for historical replay.
    """
    def __init__(self, single_channels=384, pair_channels=256, chunk=32):
        super().__init__()
        if chunk <= 0:
            raise ValueError('Positive token chunk required')
        self.chunk = chunk
        self.norm = nn.LayerNorm(single_channels)
        self.output = nn.Linear(2*single_channels, pair_channels, bias=False)
        nn.init.zeros_(self.output.weight)

    def project_chunk(self, left, right):
        product = left[:, :, None, :] * right[:, None, :, :]
        difference = left[:, :, None, :] - right[:, None, :, :]
        return self.output(torch.cat((product,difference),dim=-1))

    def forward(self, single, protein_mask):
        if single.ndim != 3 or protein_mask.shape != single.shape[:2]:
            raise ValueError('Expected singles[B,N,C] and protein_mask[B,N]')
        x = self.norm(single)
        result=[]
        for start in range(0,x.shape[1],self.chunk):
            left=x[:,start:start+self.chunk]
            if self.training and torch.is_grad_enabled():
                value=checkpoint(self.project_chunk,left,x,use_reentrant=False)
            else:
                value=self.project_chunk(left,x)
            mask=protein_mask[:,start:start+self.chunk,None] & protein_mask[:,None,:]
            result.append(value * mask[...,None])
        return torch.cat(result,dim=1)


@dataclass(frozen=True)
class HeadConfig:
    variant: str = 'colleague_cross'
    channel_s: int = 384
    channel_z: int = 256
    blocks: int = 6
    heads: int = 16
    dropout: float = .1
    outer_rank: int = 16
    proddiff_chunk: int = 32
    kernel_backend: str = 'torch'


class AffinityHead(nn.Module):
    """Same scalar architecture serves pActivity regression or a binary logit.

    colleague_cross: exact original mask and no outer module.
    full_pair_control: PP allowed, capacity-matched null-input outer adapter.
    slm_outer: same full-pair topology, real s_lm outer-product PP residual.
    slm_proddiff: full-pair topology, product+signed-difference PP residual.
    full_pair_proddiff_control: same proddiff parameters/topology, zeroed input.
    """
    def __init__(self, config: HeadConfig):
        super().__init__()
        if config.variant not in ('colleague_cross', 'full_pair_control', 'slm_outer',
                                  'slm_proddiff', 'full_pair_proddiff_control'):
            raise ValueError('Unknown controlled affinity-head variant')
        self.config = config
        self.base = AffinityPairformer(AffinityPairformer.Config(
            channel_s=config.channel_s, channel_z=config.channel_z,
            num_blocks=config.blocks, num_heads_attn=config.heads,
            dropout=config.dropout, blocks_per_ckpt=1,
            cross_pair_only=config.variant == 'colleague_cross'),
            kernel_backend=config.kernel_backend)
        self.outer = SingleOuterProduct(config.channel_s,config.channel_z,config.outer_rank) if (
            config.variant in ('full_pair_control','slm_outer')) else None
        self.proddiff = SingleProductDifference(config.channel_s,config.channel_z,chunk=config.proddiff_chunk) if (
            config.variant in ('slm_proddiff','full_pair_proddiff_control')) else None

    def forward(self, **features):
        if self.outer is None and self.proddiff is None:
            return self.base(**features)
        single = features['s_lm']
        if self.config.variant in ('full_pair_control','full_pair_proddiff_control'):
            single = single * 0  # Same trainable capacity/compute, no new information.
        adapter=self.proddiff if self.proddiff is not None else self.outer
        residual = adapter(single, features['protein_mask'] & features['token_mask'])
        return self.base(**features, pair_residual=residual)
