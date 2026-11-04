"""Train the new apo adapters, auditing frozen weights and optimizer membership."""

import importlib.util
import json
import math
import os
import socket
from pathlib import Path

import lightning.pytorch as pl
import torch
import torch.distributed as dist
from lightning.pytorch.callbacks import Timer
from lightning.pytorch.loggers import CSVLogger


class SegmentTimer(Timer):
    """Reset the allocation walltime budget on an explicit full-state resume."""

    def load_state_dict(self, state_dict):
        pass


class JointTrainingAudit(pl.Callback):
    groups = (
        "structure_z",
        "confidence_relation.confidence_z",
        "confidence_relation.confidence_s",
        "apo_selector",
    )

    def on_train_start(self, trainer, module):
        if not module.train_apo_repr_mix or any(
            (
                module.train_trunk,
                module.train_diffusion_head,
                module.train_confidence_head,
            )
        ):
            raise AssertionError("Only the new apo adapters may be trained")
        trainable = {n: p for n, p in module.model.named_parameters() if p.requires_grad}
        if not trainable or not all(n.startswith("apo_repr_mix.") for n in trainable):
            raise AssertionError("Unexpected trainable parameters outside apo_repr_mix")
        self.group_parameters = {
            group: {
                n: p
                for n, p in trainable.items()
                if n.startswith(f"apo_repr_mix.{group}.")
            }
            for group in self.groups
        }
        if not all(self.group_parameters.values()):
            raise AssertionError("Every adapter group must contain trainable parameters")
        assigned = {n for members in self.group_parameters.values() for n in members}
        if assigned != set(trainable):
            raise AssertionError(
                f"Unaudited adapter parameters: {set(trainable) - assigned}"
            )
        if len(trainer.optimizers) != 1:
            raise AssertionError("Expected one optimizer for the adapter allowlist")
        optimizer_parameters = [
            p for group in trainer.optimizers[0].param_groups for p in group["params"]
        ]
        if len(optimizer_parameters) != len(trainable) or {
            id(p) for p in optimizer_parameters
        } != {id(p) for p in trainable.values()}:
            raise AssertionError(
                "Optimizer parameters differ from the trainable allowlist"
            )
        self.frozen = [
            (n, p, p._version)
            for n, p in module.model.named_parameters()
            if not p.requires_grad
        ]
        # Only adapter snapshots are copied; the entire frozen model is version-audited.
        self.initial = {n: p.detach().cpu().clone() for n, p in trainable.items()}
        self.grad_seen = dict.fromkeys(self.groups, False)
        self.norms = dict.fromkeys(self.groups, 0.0)
        self.initial_step = int(trainer.global_step)
        if not 0 <= self.initial_step < trainer.max_steps:
            raise ValueError("Resume checkpoint must precede the requested update target")
        self.parameters = {
            g: sum(p.numel() for p in ps.values())
            for g, ps in self.group_parameters.items()
        }
        self.root = Path(os.environ["JOINT_EXPERIMENT_ROOT"])
        (self.root / "progress").mkdir(parents=True, exist_ok=True)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.assert_frozen()
        self.write(trainer, "running")

    def assert_frozen(self):
        altered = [
            n
            for n, p, version in self.frozen
            if p.grad is not None or p._version != version or p.requires_grad
        ]
        if altered:
            raise AssertionError(f"Frozen parameter audit failed: {altered[:8]}")

    def on_before_optimizer_step(self, trainer, module, optimizer):
        self.norms = {}
        for group, members in self.group_parameters.items():
            grads = [p.grad.detach() for p in members.values() if p.grad is not None]
            norm = (
                sum(g.float().square().sum() for g in grads).sqrt().item()
                if grads
                else 0.0
            )
            if not math.isfinite(norm):
                raise AssertionError(f"Nonfinite gradient in {group}")
            self.norms[group] = norm
            self.grad_seen[group] |= norm > 0
        # Utility sampling is sparse: a rank/batch without utility gradients is valid.
        self.assert_frozen()

    @staticmethod
    def scalar_metrics(trainer):
        metrics = {}
        for name, value in trainer.callback_metrics.items():
            if not (
                name.startswith("train/") and ("loss" in name or "apo_utility" in name)
            ):
                continue
            if torch.is_tensor(value) and value.numel() == 1:
                value = value.detach().float().item()
            if isinstance(value, (float, int)):
                metrics[name] = value
        return metrics

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        if loss is not None and not torch.isfinite(torch.as_tensor(loss)).all():
            raise AssertionError("Nonfinite training loss")
        if trainer.global_step > trainer.max_steps:
            raise AssertionError("Training exceeded the fixed update target")
        if (
            trainer.global_step <= self.initial_step + 10
            or trainer.global_step % 100 == 0
        ):
            self.write(
                trainer,
                "running",
                gradients=self.norms,
                losses=self.scalar_metrics(trainer),
                combinations=getattr(module.model, "last_apo_mix_assignments", None),
            )

    def write(self, trainer, status, **extra):
        result = dict(
            status=status,
            rank=trainer.global_rank,
            host=socket.gethostname(),
            variant=os.environ.get("JOINT_VARIANT"),
            world_size=trainer.world_size,
            initial_step=self.initial_step,
            global_step=int(trainer.global_step),
            target_steps=trainer.max_steps,
            validation_run=bool(os.environ.get("JOINT_VALIDATION_STEPS")),
            trainable_parameters=self.parameters,
            nonzero_gradients=self.grad_seen,
            frozen_parameter_tensors=len(self.frozen),
            peak_allocated_gib=(
                torch.cuda.max_memory_allocated() / 2**30
                if torch.cuda.is_available()
                else 0.0
            ),
            **extra,
        )
        path = self.root / "progress" / f"rank-{trainer.global_rank}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(path)
        if trainer.is_global_zero:
            print("JOINT_TRAINING_PROGRESS", json.dumps(result), flush=True)

    def on_train_end(self, trainer, module):
        self.assert_frozen()
        changed = {
            group: any(
                not torch.equal(p.detach().cpu(), self.initial[n])
                for n, p in members.items()
            )
            for group, members in self.group_parameters.items()
        }
        device = module.device
        evidence = torch.tensor(
            [[int(self.grad_seen[g]), int(changed[g])] for g in self.groups],
            device=device,
            dtype=torch.int32,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(evidence, op=dist.ReduceOp.MAX)
        evidence = evidence.cpu().tolist()
        global_gradients = {
            g: bool(row[0]) for g, row in zip(self.groups, evidence, strict=True)
        }
        global_changed = {
            g: bool(row[1]) for g, row in zip(self.groups, evidence, strict=True)
        }
        checkpoint = self.root / "checkpoints" / "progress.ckpt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        pending = checkpoint.with_name("progress.pending.ckpt")
        trainer.save_checkpoint(pending)
        trainer.strategy.barrier()
        if trainer.is_global_zero:
            pending.replace(checkpoint)
        trainer.strategy.barrier()
        all_exercised = all(global_gradients.values()) and all(global_changed.values())
        self.write(
            trainer,
            "complete" if trainer.global_step == trainer.max_steps else "paused",
            frozen_unchanged=True,
            changed=changed,
            global_nonzero_gradients=global_gradients,
            global_changed=global_changed,
            all_groups_exercised=all_exercised,
            checkpoint=str(checkpoint),
        )
        required = self.groups[:-1]
        if not all(global_gradients[g] and global_changed[g] for g in required):
            raise AssertionError("Generation/confidence adapters were not exercised")
        require_utility = os.environ.get("JOINT_REQUIRE_UTILITY_GRAD", "1") == "1"
        if require_utility and not all_exercised:
            raise AssertionError(
                "Utility gradients/updates were not observed across ranks"
            )

    def on_exception(self, trainer, module, exception):
        if hasattr(self, "root"):
            self.write(
                trainer,
                "failed",
                error=f"{type(exception).__name__}: {str(exception)[:500]}",
            )


def main():
    spec = importlib.util.spec_from_file_location(
        "kfold_joint_train_script", Path(__file__).with_name("train.py")
    )
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    original = train.build_trainer
    original_datamodule = train.TrainingDataModule

    def seeded_datamodule(config):
        datamodule = original_datamodule(config)
        # Shallow/deep initialization consumes different numbers of random draws.
        # Reset once after model and data-module construction, immediately before
        # fit. Do not reset per epoch or overwrite RNG during a training step.
        pl.seed_everything(int(config.sampling_seed), workers=True, verbose=False)
        return datamodule

    def build(cfg, debug=False):
        trainer = original(cfg, debug)
        trainer.callbacks = [
            SegmentTimer(duration=cfg.train.trainer.max_time)
            if isinstance(callback, Timer)
            else callback
            for callback in trainer.callbacks
        ]
        trainer.callbacks.append(JointTrainingAudit())
        log_root = Path(os.environ.get("RUN_ROOT", cfg.train.out_dir))
        trainer.logger = CSVLogger(save_dir=str(log_root), name="metrics", version=0)
        if trainer.is_global_zero:
            log_root.mkdir(parents=True, exist_ok=True)
            train.save_config(cfg, log_root / "resolved-config.yaml")
        return trainer

    train.build_trainer = build
    train.TrainingDataModule = seeded_datamodule
    train.train(train.parse_args())


if __name__ == "__main__":
    main()
