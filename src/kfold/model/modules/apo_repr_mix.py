"""Learned residual mixers and configuration for apo-joint models."""

from dataclasses import dataclass, replace

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


@dataclass(kw_only=True)
class ApoReprMixConfig:
    enabled: bool = False
    hidden: int = 512
    chunk_size: int = 1024
    max_combinations: int = 3
    train_max_apo: int = 3
    weighted_structure: bool = False
    relation_depth: int = 0
    relation_hidden: int = 64
    utility_rollout_fraction: float = 0.25
    utility_loss_weight: float = 0.01

    def __post_init__(self):
        if (
            min(self.hidden, self.chunk_size, self.max_combinations, self.train_max_apo)
            < 1
        ):
            raise ValueError("Apo mixer sizes and combination limits must be positive")
        if self.relation_depth < 0 or self.relation_hidden < 1:
            raise ValueError("Invalid apo relation depth/hidden size")
        if not 0 <= self.utility_rollout_fraction <= 1 or self.utility_loss_weight < 0:
            raise ValueError("Invalid utility rollout fraction/loss weight")


class ResidualSetMixer(nn.Module):
    """Pool apo branch residuals in chunks and add a learned update to the base."""

    def __init__(self, channels, hidden, chunk_size):
        super().__init__()
        self.chunk_size = chunk_size
        self.chunk_is_compiled = False
        self.phi = nn.Sequential(
            nn.LayerNorm(2 * channels), nn.Linear(2 * channels, hidden), nn.SiLU()
        )
        self.rho = nn.Sequential(nn.Linear(hidden + channels, hidden), nn.SiLU())
        self.out = nn.Linear(hidden, channels)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def do_compile(self, **kwargs):
        # The full pair axis has hundreds of checkpointed chunks. Compile only
        # their shared work so Dynamo does not unroll the outer Python loop.
        self._chunk = torch.compile(self._chunk, **kwargs)
        self.chunk_is_compiled = True

    def _chunk(self, base, peers, valid):
        h = self.phi(torch.cat((peers, peers - base[:, None]), dim=-1))
        weights = valid[:, :, None, None].to(h.dtype)
        pooled = (h * weights).sum(1) / weights.sum(1).clamp_min(1)
        delta = self.out(self.rho(torch.cat((base, pooled), dim=-1)))
        return delta * valid.any(1)[:, None, None]

    def forward(self, base, peers, valid):
        # base [B,...,C], peers [B,K,...,C], valid [B,K].
        shape = base.shape
        base_flat = base.reshape(shape[0], -1, shape[-1])
        peers_flat = peers.reshape(shape[0], peers.shape[1], -1, shape[-1])
        deltas = []
        for start in range(0, base_flat.shape[1], self.chunk_size):
            args = (
                base_flat[:, start : start + self.chunk_size],
                peers_flat[:, :, start : start + self.chunk_size],
                valid,
            )
            if torch.is_grad_enabled() and self.training:
                delta = checkpoint(self._chunk, *args, use_reentrant=False)
            else:
                delta = self._chunk(*args)
            deltas.append(delta)
        return base + torch.cat(deltas, dim=1).reshape(shape).to(base.dtype)


class WeightedResidualSetMixer(ResidualSetMixer):
    """Learn a separate K-axis weight for each full-complex post-trunk pair."""

    def __init__(self, channels, hidden, chunk_size):
        super().__init__(channels, hidden, chunk_size)
        self.score = nn.Linear(hidden, 1)
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    def _chunk(self, base, peers, valid):
        h = self.phi(torch.cat((peers, peers - base[:, None]), dim=-1))
        mask = valid[:, :, None].expand(*h.shape[:-1])
        logits = self.score(h).squeeze(-1).float().masked_fill(~mask, -torch.inf)
        # A null reference lets the model suppress all unhelpful branches.
        logits = torch.cat((logits, torch.zeros_like(logits[:, :1])), 1)
        weights = logits.softmax(1)[:, :-1].to(h.dtype)
        pooled = (h * weights[..., None]).sum(1)
        delta = self.out(self.rho(torch.cat((base, pooled), -1)))
        return delta * valid.any(1)[:, None, None]


class ApoRepresentationMixers(nn.Module):
    """Own the structure mixer and independently trained confidence adapters."""

    def __init__(self, channel_z, channel_s, config):
        super().__init__()
        initial_rng = torch.random.get_rng_state()
        mixer_cls = (
            WeightedResidualSetMixer if config.weighted_structure else ResidualSetMixer
        )
        self.structure_z = mixer_cls(channel_z, config.hidden, config.chunk_size)
        self.confidence_relation = None
        self.apo_selector = None
        if config.relation_depth:
            from kfold.model.modules.apo_relation import (
                ApoCandidateRelation,
                ApoUtilityHead,
            )

            self.confidence_z = self.confidence_s = None
            self.confidence_relation = ApoCandidateRelation(channel_z, channel_s, config)
            self.apo_selector = ApoUtilityHead(channel_s, config.relation_hidden)
        else:
            self.confidence_z = ResidualSetMixer(
                channel_z, config.hidden, config.chunk_size
            )
            self.confidence_s = ResidualSetMixer(
                channel_s, config.hidden, config.chunk_size
            )
        if config.relation_depth > 1:
            self._initialize_shared_weights(channel_z, channel_s, config, initial_rng)

    def _initialize_shared_weights(self, channel_z, channel_s, config, initial_rng):
        """Match depth-one initialization for weights shared across depth variants.

        Extra relation blocks keep their own initialization. Recreating the
        shallow model does not advance the caller's RNG state.
        """
        with torch.random.fork_rng(devices=[]):
            torch.random.set_rng_state(initial_rng)
            shallow = ApoRepresentationMixers(
                channel_z, channel_s, replace(config, relation_depth=1)
            )
        state = self.state_dict()
        state.update(shallow.state_dict())
        self.load_state_dict(state, strict=True)

    def forward(self, z, s, branch_z, branch_s, valid):
        # Never share trainable parameters or pass confidence gradients to structure_z.
        if self.confidence_relation is not None:
            return self.structure_z(z, branch_z, valid), z, s
        return (
            self.structure_z(z, branch_z, valid),
            self.confidence_z(z, branch_z, valid),
            self.confidence_s(s, branch_s, valid),
        )
