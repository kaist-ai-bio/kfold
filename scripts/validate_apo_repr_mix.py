"""Bounded real-data Lightning/DDP validation of mixer-only training."""

import importlib.util
import json
import os
import socket
import time
from pathlib import Path

import lightning.pytorch as pl
import torch


class MixerContract(pl.Callback):
    def __init__(self, root):
        self.root = Path(root)
        self.rows = []

    def on_train_start(self, trainer, module):
        assert module.train_apo_repr_mix
        assert not any(
            (
                module.train_trunk,
                module.train_diffusion_head,
                module.train_confidence_head,
            )
        )
        self.rank = trainer.global_rank
        trainer.fit_loop.epoch_loop.max_steps = trainer.global_step + int(
            os.environ.get("MIX_VALIDATION_STEPS", "10")
        )
        self.initial_step = trainer.global_step
        trainable = {n: p for n, p in module.model.named_parameters() if p.requires_grad}
        assert trainable and all(n.startswith("apo_repr_mix.") for n in trainable)
        opt_ids = {
            id(p) for group in trainer.optimizers[0].param_groups for p in group["params"]
        }
        assert opt_ids == {id(p) for p in trainable.values()}
        self.initial = {n: p.detach().clone() for n, p in trainable.items()}
        self.frozen = {
            n: (p, p._version)
            for n, p in module.model.named_parameters()
            if not p.requires_grad
        }
        self.grad_seen = {
            name: False for name in ("structure_z", "confidence_z", "confidence_s")
        }
        self.root.mkdir(parents=True, exist_ok=True)
        self.start = time.monotonic()

    def on_train_batch_start(self, trainer, module, batch, batch_idx):
        if not self.rows:
            # Compare full legacy and zero-residual forward paths with identical RNG.
            model = module.model
            mixers = model.apo_repr_mix
            kwargs = dict(
                num_recycles=1,
                grad_recurrence_steps=0,
                num_mini_rollout_steps=30,
                num_mini_rollout_samples=1,
                diffusion_batch_size=1,
                soar_config=module.soar_config,
                train_trunk=False,
                train_diffusion_head=True,
                train_confidence_module=True,
            )
            with (
                torch.random.fork_rng(devices=[module.device.index]),
                torch.no_grad(),
                torch.autocast("cuda", dtype=torch.bfloat16),
            ):
                model.apo_repr_mix = None
                try:
                    with torch.random.fork_rng(devices=[module.device.index]):
                        legacy = model.forward_train(batch[0], **kwargs)
                finally:
                    model.apo_repr_mix = mixers
                mixed = model.forward_train(batch[0], **kwargs)
            for group in ("diffusion", "sample", "confidence"):
                for name, tensor in legacy[group].items():
                    if torch.is_tensor(tensor):
                        torch.testing.assert_close(
                            tensor, mixed[group][name], rtol=0, atol=0, equal_nan=True
                        )
            self.initial_identity = True
            del legacy, mixed
        self.batch_start = time.monotonic()
        self.ids = [x["id"] for x in batch[1]]
        self.supervised = [bool(x["train_confidence_head"]) for x in batch[1]]
        self.tokens = int(batch[0].token.pad_mask.shape[-1])
        torch.cuda.reset_peak_memory_stats()

    def on_before_optimizer_step(self, trainer, module, optimizer):
        assert all(p.grad is None and p._version == v for p, v in self.frozen.values())
        self.norms = {}
        for group in self.grad_seen:
            params = getattr(module.model.apo_repr_mix, group).parameters()
            grads = [p.grad for p in params if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads)
            norm = sum(g.float().square().sum() for g in grads).sqrt().item()
            self.norms[group] = norm
            self.grad_seen[group] |= norm > 0

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        assert torch.isfinite(outputs["loss"]).all()
        self.rows.append(
            dict(
                step=trainer.global_step,
                loss=float(outputs["loss"]),
                ids=self.ids,
                confidence_supervision=self.supervised,
                tokens=self.tokens,
                gradients=self.norms,
                combinations=module.model.last_apo_mix_assignments,
                seconds=time.monotonic() - self.batch_start,
                peak_gib=torch.cuda.max_memory_allocated() / 2**30,
            )
        )
        self.write("running", trainer)

    def write(self, status, trainer, **extra):
        result = dict(
            status=status,
            rank=self.rank,
            host=socket.gethostname(),
            world_size=trainer.world_size,
            initial_step=self.initial_step,
            initial_identity=getattr(self, "initial_identity", False),
            rows=self.rows,
            nonzero_gradients=self.grad_seen,
            **extra,
        )
        (self.root / f"rank-{self.rank}.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )

    def on_train_end(self, trainer, module):
        assert all(p.grad is None and p._version == v for p, v in self.frozen.values())
        changed = {g: False for g in self.grad_seen}
        for n, p in module.model.named_parameters():
            if n in self.initial and not torch.equal(p, self.initial[n]):
                changed[n.split(".")[1]] = True
        assert all(changed.values()), changed
        assert all(self.grad_seen.values()), self.grad_seen
        path = self.root / "validated.ckpt"
        trainer.save_checkpoint(path)
        trainer.strategy.barrier()
        if trainer.is_global_zero:
            ck = torch.load(path, map_location="cpu", weights_only=False)
            for n, p in module.model.apo_repr_mix.state_dict().items():
                torch.testing.assert_close(
                    ck["state_dict"]["model.apo_repr_mix." + n], p.cpu(), rtol=0, atol=0
                )
            assert ck["optimizer_states"] and ck["ema"]
            del ck
        self.write(
            "passed",
            trainer,
            changed=changed,
            frozen_unchanged=True,
            elapsed_seconds=time.monotonic() - self.start,
        )


def main():
    spec = importlib.util.spec_from_file_location(
        "kfold_train_script", Path(__file__).with_name("train.py")
    )
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    original = train.build_trainer

    def build(cfg, debug=False):
        trainer = original(cfg, debug)
        trainer.callbacks.append(
            MixerContract(Path(os.environ["RUN_ROOT"]) / "validation")
        )
        return trainer

    train.build_trainer = build
    train.train(train.parse_args())


if __name__ == "__main__":
    main()
