"""Run mixer training with frozen-parameter audits and a resumable final checkpoint."""

import importlib.util
import json
import os
import socket
from pathlib import Path

import lightning.pytorch as pl
import torch


class MixerTrainingAudit(pl.Callback):
    groups = ("structure_z", "confidence_z", "confidence_s")

    def on_train_start(self, trainer, module):
        assert module.train_apo_repr_mix
        assert not any(
            (module.train_trunk, module.train_diffusion_head, module.train_confidence_head)
        )
        trainable = {n: p for n, p in module.model.named_parameters() if p.requires_grad}
        assert trainable and all(n.startswith("apo_repr_mix.") for n in trainable)
        actual = {id(p) for g in trainer.optimizers[0].param_groups for p in g["params"]}
        assert actual == {id(p) for p in trainable.values()}
        self.frozen = [
            (p, p._version) for p in module.model.parameters() if not p.requires_grad
        ]
        self.initial = {n: p.detach().clone() for n, p in trainable.items()}
        self.grad_seen = dict.fromkeys(self.groups, False)
        self.initial_step = trainer.global_step
        self.parameters = sum(p.numel() for p in trainable.values())
        self.root = Path(os.environ["MIX_EXPERIMENT_ROOT"])
        (self.root / "progress").mkdir(exist_ok=True)
        torch.cuda.reset_peak_memory_stats()
        self.write(trainer, "running")

    def on_before_optimizer_step(self, trainer, module, optimizer):
        self.norms = {}
        for group in self.groups:
            grads = [
                p.grad for p in getattr(module.model.apo_repr_mix, group).parameters()
                if p.grad is not None
            ]
            assert grads and all(torch.isfinite(g).all() for g in grads), group
            norm = sum(g.float().square().sum() for g in grads).sqrt().item()
            self.norms[group] = norm
            self.grad_seen[group] |= norm > 0

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        assert torch.isfinite(outputs["loss"]).all(), "Nonfinite training loss"
        if trainer.global_step <= self.initial_step + 10 or trainer.global_step % 100 == 0:
            assert all(p.grad is None and p._version == v for p, v in self.frozen)
            self.write(
                trainer, "running", loss=float(outputs["loss"]),
                gradients=self.norms,
                combinations=module.model.last_apo_mix_assignments,
            )

    def write(self, trainer, status, **extra):
        result = dict(
            status=status, rank=trainer.global_rank, host=socket.gethostname(),
            world_size=trainer.world_size, initial_step=self.initial_step,
            global_step=trainer.global_step, target_steps=trainer.max_steps,
            trainable_parameters=self.parameters, nonzero_gradients=self.grad_seen,
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30, **extra,
        )
        path = self.root / "progress" / f"rank-{trainer.global_rank}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(path)
        if trainer.is_global_zero:
            print("MIX_TRAINING_PROGRESS", json.dumps(result), flush=True)

    def on_train_end(self, trainer, module):
        assert all(p.grad is None and p._version == v for p, v in self.frozen)
        changed = dict.fromkeys(self.groups, False)
        for n, p in module.model.named_parameters():
            if n in self.initial and not torch.equal(p, self.initial[n]):
                changed[n.split(".")[1]] = True
        assert all(changed.values()) and all(self.grad_seen.values())
        checkpoint = self.root / "checkpoints" / "progress.ckpt"
        trainer.save_checkpoint(checkpoint)
        trainer.strategy.barrier()
        self.write(
            trainer, "complete" if trainer.global_step >= trainer.max_steps else "paused",
            frozen_unchanged=True, changed=changed, checkpoint=str(checkpoint),
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
        trainer.callbacks.append(MixerTrainingAudit())
        return trainer

    train.build_trainer = build
    train.train(train.parse_args())


if __name__ == "__main__":
    main()
