"""Contracts for candidate-specific apo confidence and sampled utility labels."""

from types import SimpleNamespace

import torch

from kfold.model.modules.confidence_head import ConfidenceHead
from kfold.training.apo_utility import apo_utility_quality, compute_apo_utility_loss


def features(chains=(1, 1, 2, 2)):
    length = len(chains)
    mask = torch.ones(1, length, dtype=torch.bool)
    token = SimpleNamespace(
        repr_index=torch.arange(length)[None],
        num_atoms=torch.ones(1, length, dtype=torch.long),
        pad_mask=mask,
        asym_id=torch.tensor([chains]),
        is_rna=torch.zeros_like(mask),
        is_dna=torch.zeros_like(mask),
    )
    return SimpleNamespace(
        token=token, atom=SimpleNamespace(pad_mask=mask), num_atoms=length
    )


def geometry():
    return torch.tensor(
        [[[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [4.0, 0.0, 0.0], [5.0, 0.0, 0.0]]]]
    )


def test_utility_target_measures_generated_quality_and_not_apo_similarity():
    f = features()
    native = geometry()
    mask = torch.ones(1, 1, 4, dtype=torch.bool)
    perfect, observed = apo_utility_quality(f, native + 7, native, mask)
    assert observed.all() and torch.equal(perfect, torch.ones_like(perfect))
    wrong = native.clone()
    wrong[:, :, 2:, 0] += 10
    quality, observed = apo_utility_quality(f, wrong, native, mask)
    assert observed.all() and torch.equal(quality, torch.zeros_like(quality))
    # No native interchain contacts means no label, not a failed prediction.
    far_native = native.clone()
    far_native[:, :, 2:, 0] += 100
    _, observed = apo_utility_quality(f, far_native, far_native, mask)
    assert not observed.any()


def test_single_chain_fallback_and_unresolved_targets():
    native = geometry()
    mask = torch.ones(1, 1, 4, dtype=torch.bool)
    quality, observed = apo_utility_quality(features((1, 1, 1, 1)), native, native, mask)
    assert observed.all() and torch.equal(quality, torch.ones_like(quality))
    mask.zero_()
    quality, observed = apo_utility_quality(features(), native, native, mask)
    assert not observed.any() and torch.equal(quality, torch.zeros_like(quality))


def test_only_observed_branch_gets_utility_supervision():
    scores = torch.zeros(1, 3, requires_grad=True)
    native = geometry()
    mask = torch.ones(1, 1, 4, dtype=torch.bool)
    outputs = {
        "scores": scores,
        "branch_index": 1,
        "valid": torch.ones(1, 3, dtype=torch.bool),
    }
    loss, metrics = compute_apo_utility_loss(
        outputs,
        features(),
        native,
        native,
        mask,
        torch.ones(1, dtype=torch.bool),
    )
    loss.backward()
    assert scores.grad[0, 1] != 0
    assert scores.grad[0, 0] == scores.grad[0, 2] == 0
    assert metrics["apo_utility_target"] == 1
    scores.grad = None
    outputs["branch_index"] = -1
    loss, _ = compute_apo_utility_loss(
        outputs,
        features(),
        native,
        native,
        mask,
        torch.ones(1, dtype=torch.bool),
    )
    loss.backward()
    assert torch.equal(scores.grad, torch.zeros_like(scores))


def test_relation_is_candidate_specific_and_zero_update_preserves_head():
    torch.manual_seed(3)
    head = ConfidenceHead(
        ConfidenceHead.Config(
            channel_s=16,
            channel_z=16,
            num_heads_attn=4,
            num_blocks=1,
            dropout=0,
        )
    ).eval()
    with torch.no_grad():
        for name, parameter in head.named_parameters():
            if name.endswith("weight") and parameter.ndim == 2:
                parameter.normal_(0, 0.1)
    f = features()
    args = (
        torch.randn(1, 4, 16),
        torch.randn(1, 4, 16),
        torch.randn(1, 4, 4, 16),
        torch.randn(1, 2, 4, 3),
    )

    class Relation(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = []
            self.scale = torch.nn.Parameter(torch.zeros(()))

        def forward(self, f_input, context, z, single, candidate):
            self.seen.append(candidate.clone())
            distances = torch.cdist(candidate, candidate)
            pair_pattern = torch.arange(z.shape[-1], dtype=z.dtype)
            single_pattern = torch.arange(single.shape[-1], dtype=single.dtype)
            return (
                self.scale * distances[..., None] * pair_pattern,
                self.scale * candidate.norm(dim=-1)[..., None] * single_pattern,
            )

    relation = Relation()
    with torch.no_grad():
        baseline = head(f, *args)
        upgraded = head(f, *args, apo_relation=relation, apo_context={})
    assert len(relation.seen) == 2
    torch.testing.assert_close(relation.seen[0], args[-1][:, 0], rtol=0, atol=0)
    torch.testing.assert_close(relation.seen[1], args[-1][:, 1], rtol=0, atol=0)
    for name in baseline:
        torch.testing.assert_close(upgraded[name], baseline[name], rtol=0, atol=0)
    assert baseline["pde_logits"].abs().sum() > 0
    assert not any(name.startswith("apo_relation.") for name in head.state_dict())
    # Frozen decoder weights still use its training path so input gradients
    # avoid the inference-only inplace updates.
    head.requires_grad_(False).train()
    result = head(f, *args, apo_relation=relation, apo_context={})
    result["pae_logits"].square().sum().backward()
    assert relation.scale.grad is not None and relation.scale.grad.abs() > 0
    assert all(parameter.grad is None for parameter in head.parameters())


def test_utility_teacher_draw_preserves_diffusion_rng(monkeypatch):
    from kfold.model import model_train

    f = SimpleNamespace(is_batched=True, batch_size=1, device=torch.device("cpu"))
    single, lm, pair = torch.zeros(1, 4, 3), torch.zeros(1, 4, 3), torch.zeros(1, 4, 4, 2)
    context = {
        "valid": torch.ones(1, 2, dtype=torch.bool),
        "branches": [object(), object()],
        "branch_z": torch.ones(1, 2, 4, 4, 2),
    }

    def mix(model, *args, **kwargs):
        model.last_apo_mix_context = context
        return pair, pair, lm

    monkeypatch.setattr(model_train, "mix_representations", mix)

    class Diffusion:
        def training_step(self, *args):
            return {"probe": torch.rand(3)}

        def sample_structure(self, **kwargs):
            self.input = kwargs["f_input"]
            return {"coordinates": torch.randn(1, 1, 4, 3)}

    diffusion = Diffusion()
    model = SimpleNamespace(
        training=True,
        run_trunk=lambda *args: (single, lm, pair),
        apo_repr_mix=SimpleNamespace(
            confidence_relation=object(),
            apo_selector=lambda *args: torch.zeros(1, 2, requires_grad=True),
        ),
        apo_repr_mix_config=SimpleNamespace(utility_rollout_fraction=0.0),
        config=SimpleNamespace(
            diffusion_conditioning_drop_rate=0,
            confidence_conditioning_drop_rate=0,
        ),
        diffusion_head=diffusion,
        confidence_head=lambda *args, **kwargs: {},
    )
    outputs, states = [], []
    for fraction in (0.0, 1.0):
        torch.manual_seed(15)
        model.apo_repr_mix_config.utility_rollout_fraction = fraction
        outputs.append(
            model_train.KFoldForTrain.forward_train(
                model,
                f,
                None,
                train_trunk=False,
                train_diffusion_head=False,
                train_confidence_module=False,
            )
        )
        states.append(torch.get_rng_state())
    assert outputs[0]["apo_utility"]["branch_index"] == -1
    branch = outputs[1]["apo_utility"]["branch_index"]
    assert branch in (0, 1) and diffusion.input is context["branches"][branch]
    assert torch.equal(states[0], states[1])
    assert torch.equal(
        outputs[0]["sample"]["coordinates"], outputs[1]["sample"]["coordinates"]
    )
