"""Candidate-conditioned, permutation-invariant use of pre-trunk apo evidence.

No native coordinates enter this module. Independent apo coordinate frames are
compared only within their observed apo_uid group. Partner geometry comes from
the complete predicted candidate, never from relative placement of separate apos.
"""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from kfold.model.modules.apo_features import APO_SINGLE_CHANNELS, ApoContext, ApoEvidence


def masked_apo_pool(values, logits, mask):
    """Local K-axis attention, with a zero-valued null reference available."""
    logits = logits.float().masked_fill(~mask, -torch.inf)
    null = torch.zeros_like(logits[:, :1])
    weights = torch.softmax(torch.cat((logits, null), dim=1), dim=1)[:, :-1]
    return (values * weights[..., None].to(values.dtype)).sum(1)


def partner_mean(values, asym_id, pad_mask):
    """Pool all other chains, including ligand/nucleic-acid input features."""
    other = asym_id[:, :, None] != asym_id[:, None, :]
    other &= pad_mask[:, :, None] & pad_mask[:, None, :]
    weights = other.to(values.dtype)
    denominator = weights.sum(-1).clamp_min(1)
    if values.ndim == 4:
        return (
            torch.einsum("bij,bkjh->bkih", weights, values)
            / denominator[:, None, :, None]
        )
    return torch.bmm(weights, values) / denominator[..., None]


@torch.no_grad()
def candidate_geometry(f, context: ApoContext, x):
    """Build distance features and contact weights shared by both confidence heads.

    Apo distances are masked outside observed groups. Candidate distances describe
    the complete predicted complex, including pairs from different chains.
    """
    distance = torch.cdist(x.float(), x.float())
    apo_distance = context["apo_distance"].float()
    observed = context["apo_distance_mask"]
    same_chain = f.token.asym_id[:, :, None] == f.token.asym_id[:, None, :]
    candidate_distance = distance[:, None].expand_as(apo_distance)
    geometry = torch.stack(
        (
            candidate_distance.clamp_max(50) / 50,
            torch.exp(-(candidate_distance / 8).square()),
            apo_distance.clamp_max(50) / 50 * observed,
            ((candidate_distance - apo_distance).clamp(-30, 30) / 30) * observed,
            observed.to(candidate_distance.dtype),
            (~same_chain[:, None])
            .expand_as(candidate_distance)
            .to(candidate_distance.dtype),
        ),
        -1,
    )
    # Both contacting and distant pairs remain visible to the relation.
    contact = 0.05 + 0.95 * torch.sigmoid((12 - distance) / 2)
    return geometry, contact


class PreApoProjection(nn.Module):
    """Combine observed apo features with the complete complex's input features."""

    def __init__(self, single_channels, input_channels, hidden, pair_channels=64):
        super().__init__()
        self.single = nn.Sequential(
            nn.LayerNorm(single_channels), nn.Linear(single_channels, hidden)
        )
        self.system = nn.Sequential(
            nn.LayerNorm(input_channels), nn.Linear(input_channels, hidden)
        )
        self.pair = nn.Sequential(
            nn.LayerNorm(pair_channels), nn.Linear(pair_channels, hidden)
        )
        self.partner = nn.Linear(hidden, hidden, bias=False)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, f, context: ApoEvidence, s_inputs):
        mask = context["apo_single_mask"]
        raw = context["apo_single"].detach().masked_fill(~mask[..., None], 0)
        apo = self.single(raw) * mask[..., None]
        pair_mask = context["apo_pair_mask"]
        pair = context["apo_pair"].detach().masked_fill(~pair_mask[..., None], 0)
        pair = pair.sum(-2) / pair_mask.sum(-1).clamp_min(1)[..., None]
        apo = apo + self.pair(pair) * pair_mask.any(-1)[..., None]
        system = self.system(s_inputs.detach())
        complete = apo + system[:, None]
        partner = partner_mean(complete, f.token.asym_id, f.token.pad_mask)
        h = self.norm(complete + self.partner(partner))
        observed = mask | context["apo_pair_mask"].any(-1)
        valid = context["valid"] & observed.any(-1)
        token_mask = valid[..., None] & f.token.pad_mask[:, None]
        return h * token_mask[..., None], token_mask


class RelationUpdate(nn.Module):
    """Pair -> candidate-contact token -> pair update; quadratic, not cubic."""

    def __init__(self, hidden, chunk_size):
        super().__init__()
        self.chunk_size = chunk_size
        self.token = nn.Sequential(
            nn.LayerNorm(2 * hidden),
            nn.Linear(2 * hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.left = nn.Linear(hidden, hidden, bias=False)
        self.right = nn.Linear(hidden, hidden, bias=False)
        self.pair = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 2 * hidden),
            nn.SiLU(),
            nn.Linear(2 * hidden, hidden),
        )

    def forward(self, pair, token, contact, pair_mask, token_mask):
        w = contact[:, None] * pair_mask
        message = (pair * w[..., None]).sum(-2) / w.sum(-1).clamp_min(1)[..., None]
        token = (
            token + self.token(torch.cat((token, message), -1)) * token_mask[..., None]
        )
        updates = []
        # Chunk rows to avoid materializing repeated high-width concatenations.
        for start in range(0, pair.shape[2], self.chunk_size):
            value = pair[:, :, start : start + self.chunk_size]
            value = (
                value
                + self.left(token[:, :, start : start + self.chunk_size])[:, :, :, None]
            )
            value = value + self.right(token)[:, :, None, :]
            updates.append(self.pair(value))
        pair = pair + torch.cat(updates, 2) * pair_mask[..., None]
        return pair, token


class CandidateRelationBranch(nn.Module):
    """An independently parameterized relation encoder for one head input."""

    def __init__(
        self,
        output_channels,
        channel_z,
        channel_s,
        hidden,
        depth,
        chunk_size,
        single_channels=APO_SINGLE_CHANNELS,
        pair_channels=64,
    ):
        super().__init__()
        self.pre_apo = PreApoProjection(single_channels, channel_s, hidden, pair_channels)
        self.pair_apo = nn.Sequential(
            nn.LayerNorm(pair_channels), nn.Linear(pair_channels, hidden)
        )
        self.query = nn.Sequential(nn.LayerNorm(channel_z), nn.Linear(channel_z, hidden))
        self.geometry = nn.Linear(6, hidden)
        self.left = nn.Linear(hidden, hidden, bias=False)
        self.right = nn.Linear(hidden, hidden, bias=False)
        self.pair_norm = nn.LayerNorm(hidden)
        self.pair_transition = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU())
        self.token_transition = nn.Sequential(
            nn.LayerNorm(2 * hidden), nn.Linear(2 * hidden, hidden), nn.SiLU()
        )
        self.updates = nn.ModuleList(
            RelationUpdate(hidden, max(1, chunk_size // 32)) for _ in range(depth - 1)
        )
        self.score = nn.Linear(hidden, 1)
        self.out = nn.Linear(hidden, output_channels)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(
        self, f, context: ApoContext, z, s_inputs, geometry, contact, *, pair_output
    ):
        """Update candidate-conditioned apo representations and pool across apos."""
        token, token_mask = self.pre_apo(f, context, s_inputs)
        pair_mask = token_mask[..., :, None] & token_mask[..., None, :]
        observed = context["apo_pair_mask"]
        raw_pair = context["apo_pair"].detach().masked_fill(~observed[..., None], 0)
        pair = self.pair_apo(raw_pair) * observed[..., None]
        pair = (
            pair
            + self.query(z.detach())[:, None]
            + self.geometry(geometry.to(pair.dtype))
        )
        pair = pair + self.left(token)[:, :, :, None] + self.right(token)[:, :, None, :]
        pair = self.pair_transition(self.pair_norm(pair)) * pair_mask[..., None]
        summary = pair.sum(-2) / pair_mask.sum(-1).clamp_min(1)[..., None]
        token = (
            self.token_transition(torch.cat((token, summary), -1)) * token_mask[..., None]
        )
        for block in self.updates:
            if self.training and torch.is_grad_enabled():
                pair, token = checkpoint(
                    block,
                    pair,
                    token,
                    contact,
                    pair_mask,
                    token_mask,
                    use_reentrant=False,
                )
            else:
                pair, token = block(pair, token, contact, pair_mask, token_mask)
        if pair_output:
            return self._pool_pair(pair, token, pair_mask)
        return self._pool_single(pair, token, pair_mask, token_mask)

    def _pool_pair(self, pair, token, pair_mask):
        """Read token updates into pairs and pool over valid apo combinations."""
        pair = pair + self.left(token)[:, :, :, None] + self.right(token)[:, :, None, :]
        pooled = masked_apo_pool(pair, self.score(pair).squeeze(-1), pair_mask)
        # Output bias must not update positions without usable apo evidence.
        return self.out(pooled) * pair_mask.any(1)[..., None]

    def _pool_single(self, pair, token, pair_mask, token_mask):
        """Read the final pair update into tokens and pool over apo combinations."""
        # Read the final pair update back into single features as well. This
        # keeps every deep relation block supervised in the single-only branch.
        summary = pair.sum(-2) / pair_mask.sum(-1).clamp_min(1)[..., None]
        token = (
            token
            + self.token_transition(torch.cat((token, summary), -1))
            * token_mask[..., None]
        )
        pooled = masked_apo_pool(token, self.score(token).squeeze(-1), token_mask)
        return self.out(pooled) * token_mask.any(1)[..., None]


class ApoCandidateRelation(nn.Module):
    """Independent pair and single adapters using the same candidate geometry."""

    def __init__(
        self,
        channel_z,
        channel_s,
        config,
        single_channels=APO_SINGLE_CHANNELS,
        pair_channels=64,
    ):
        super().__init__()
        args = (
            channel_z,
            channel_s,
            config.relation_hidden,
            config.relation_depth,
            config.chunk_size,
            single_channels,
            pair_channels,
        )
        self.confidence_z = CandidateRelationBranch(channel_z, *args)
        self.confidence_s = CandidateRelationBranch(channel_s, *args)

    def forward(self, f_input, context: ApoContext, z, s_inputs, x_repr):
        geometry, contact = candidate_geometry(f_input, context, x_repr)
        # These branches intentionally share no trainable parameters.
        return (
            self.confidence_z(
                f_input, context, z, s_inputs, geometry, contact, pair_output=True
            ),
            self.confidence_s(
                f_input, context, z, s_inputs, geometry, contact, pair_output=False
            ),
        )


class ApoUtilityHead(nn.Module):
    """Pre-candidate branch utility; separate parameters and stopped input grads.

    Scores are trained against frozen-generator interface quality. A branch is a
    complete apo combination, not an identified contribution of each molecule.
    """

    def __init__(
        self, channel_s, hidden, single_channels=APO_SINGLE_CHANNELS, pair_channels=64
    ):
        super().__init__()
        self.pre_apo = PreApoProjection(single_channels, channel_s, hidden, pair_channels)
        self.score = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, f_input, context: ApoEvidence, s_inputs):
        # Raw producers are frozen; this utility-specific adapter is learned only
        # by utility supervision and cannot modify confidence/generation layers.
        token, mask = self.pre_apo(f_input, context, s_inputs.detach())
        pooled = (token * mask[..., None]).sum(-2) / mask.sum(-1).clamp_min(1)[..., None]
        return self.score(pooled).squeeze(-1)
