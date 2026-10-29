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

import argparse
import logging
from pathlib import Path

import lightning.pytorch as pl
import lightning.pytorch.callbacks as pl_callbacks
import torch
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, ListConfig

from kfold.training.dataset.datamodule import TrainingDataModule
from kfold.training.optim.ema import initialize_parameter_groups_from_ema
from kfold.training.training_module import KFoldTrainingModule
from kfold.utils.config import load_config, print_config, save_config, to_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Co-Folding model.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the yaml configuration file.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="Name of the experiment for logging purposes.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        help="Output root directory for saving logs and checkpoints.",
    )
    # Easy overrides
    parser.add_argument(
        "--num_gpus",
        type=int,
        help="Number of GPUs to use for training.",
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        help="Number of nodes to use for training.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Training batch size per GPU.",
    )
    parser.add_argument(
        "--global_batch_size",
        type=int,
        help="Effective batch size across all GPUs; sets gradient accumulation.",
    )
    parser.add_argument(
        "--num_batches_per_epoch",
        type=int,
        help="Number of training batches per GPU per epoch.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        help="Number of workers to use for dataloader.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        help="Path to a checkpoint file to resume training from.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    parser.add_argument(
        "--override",
        type=str,
        nargs="+",
        help="Override configuration options using 'key=value' format.",
    )
    return parser.parse_args()


def parse_config(args) -> DictConfig:
    cfg = load_config(args.config, override_args=args.override)

    # Override some config options with command line args
    if args.out_dir is not None:
        cfg.train.out_dir = args.out_dir
    if args.experiment_name is not None:
        cfg.train.name = args.experiment_name
    if args.num_gpus is not None:
        cfg.train.trainer.devices = args.num_gpus
    if args.num_nodes is not None:
        cfg.train.trainer.num_nodes = args.num_nodes
    if args.batch_size is not None:
        cfg.train.data.train_batch_size = args.batch_size
    if args.num_batches_per_epoch is not None:
        cfg.train.trainer.limit_train_batches = args.num_batches_per_epoch
    if args.num_workers is not None:
        cfg.train.data.num_workers = args.num_workers
    if args.wandb:
        cfg.train.wandb.use = True

    # Use the training seed for weighted data sampling.
    cfg.train.data.sampling_seed = cfg.train.seed

    if args.debug:
        # Enable debug mode settings
        print("Debug mode is enabled: Single GPU, 0 workers, no wandb.")
        cfg.train.trainer.devices = 1
        cfg.train.trainer.num_nodes = 1
        cfg.train.trainer.accumulate_grad_batches = 1
        cfg.train.trainer.log_every_n_steps = 1
        cfg.train.trainer.limit_train_batches = 100
        if cfg.train.trainer.limit_val_batches != 0:
            cfg.train.trainer.limit_val_batches = 10
        cfg.train.trainer.enable_checkpointing = False
        cfg.train.data.num_workers = 0
        cfg.train.data.safe_load = False
        cfg.train.wandb.use = False

    if args.global_batch_size is not None:
        devices = cfg.train.trainer.devices
        if devices == "auto" or devices == -1:
            num_gpus = torch.cuda.device_count()
        elif isinstance(devices, (list, ListConfig)):
            num_gpus = len(devices)
        else:
            num_gpus = int(devices)
        batch_size = cfg.train.data.train_batch_size
        num_nodes = cfg.train.trainer.num_nodes
        if min(batch_size, num_gpus, num_nodes, args.global_batch_size) <= 0:
            raise ValueError("Batch sizes and GPU/node counts must be positive.")
        batch_per_step = batch_size * num_gpus * num_nodes
        if args.global_batch_size % batch_per_step:
            raise ValueError(
                f"Global batch size {args.global_batch_size} must be divisible by "
                f"batch size × GPUs × nodes ({batch_per_step})."
            )
        cfg.train.trainer.accumulate_grad_batches = (
            args.global_batch_size // batch_per_step
        )

    return cfg


def build_trainer(cfg, debug: bool = False) -> pl.Trainer:
    train_cfg = cfg.train
    pl_trainer_cfg = train_cfg.trainer
    save_dir = Path(train_cfg.out_dir) / train_cfg.name

    callbacks = []
    if train_cfg.wandb.use:
        from lightning.pytorch.loggers import WandbLogger

        wandb_logger = WandbLogger(
            name=train_cfg.name,
            project=train_cfg.wandb.project,
            group=train_cfg.wandb.group,
            entity=train_cfg.wandb.entity,
            tags=train_cfg.wandb.tags,
            config=to_dict(cfg),
            save_dir=save_dir,
        )
        loggers = [wandb_logger]

        @rank_zero_only
        def _save_config() -> None:
            config_out = Path(wandb_logger.experiment.dir) / "train_config.yaml"
            save_config(cfg, config_out)
            wandb_logger.experiment.save("train_config.yaml")

        _save_config()

    else:
        loggers = None  # use default logger

    # Learning rate monitor
    lr_monitor = pl_callbacks.LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    # Model summary
    model_summary = pl_callbacks.ModelSummary(max_depth=2)
    callbacks.append(model_summary)

    # TQDM
    tqdm_refresh_rate = 1 if debug else cfg.train.trainer.log_every_n_steps
    tqdm_callback = pl_callbacks.TQDMProgressBar(refresh_rate=tqdm_refresh_rate)
    callbacks.append(tqdm_callback)

    if not debug and pl_trainer_cfg.enable_checkpointing:
        checkpoint_options = dict(
            every_n_train_steps=pl_trainer_cfg.get("checkpoint_every_n_train_steps"),
            save_last=pl_trainer_cfg.get("checkpoint_save_last", False),
            dirpath=pl_trainer_cfg.get("checkpoint_dir"),
        )
        if pl_trainer_cfg.limit_val_batches == 0:
            # Save checkpoint only based on training loss
            checkpoint_callback = pl_callbacks.ModelCheckpoint(
                monitor="train/loss",
                save_top_k=-1,
                filename="epoch{epoch:04d}_step{step:08d}_loss{train/loss:.4f}",
                mode="min",
                auto_insert_metric_name=False,
                **checkpoint_options,
            )
        else:
            checkpoint_callback = pl_callbacks.ModelCheckpoint(
                monitor="rcsb-val/monitor/weighted_lddt",
                save_top_k=-1,
                filename="epoch{epoch:04d}_step{step:08d}_wlddt{rcsb-val/monitor/weighted_lddt:.4f}",
                mode="max",
                auto_insert_metric_name=False,
                **checkpoint_options,
            )
        callbacks.append(checkpoint_callback)

    trainer = pl.Trainer(
        default_root_dir=save_dir,
        logger=loggers,
        callbacks=callbacks,
        accelerator=pl_trainer_cfg.accelerator,
        strategy=pl_trainer_cfg.strategy,
        devices=pl_trainer_cfg.devices,
        num_nodes=pl_trainer_cfg.num_nodes,
        precision=pl_trainer_cfg.precision,
        max_epochs=pl_trainer_cfg.max_epochs,
        max_steps=pl_trainer_cfg.get("max_steps", -1),
        max_time=pl_trainer_cfg.get("max_time"),
        limit_train_batches=pl_trainer_cfg.limit_train_batches,
        limit_val_batches=pl_trainer_cfg.limit_val_batches,
        num_sanity_val_steps=(
            0
            if pl_trainer_cfg.limit_val_batches == 0
            else pl_trainer_cfg.get("num_sanity_val_steps", 2)
        ),
        log_every_n_steps=pl_trainer_cfg.log_every_n_steps,
        enable_checkpointing=pl_trainer_cfg.enable_checkpointing,
        accumulate_grad_batches=pl_trainer_cfg.accumulate_grad_batches,
        gradient_clip_val=pl_trainer_cfg.gradient_clip_val,
        use_distributed_sampler=False,
        benchmark=True,
        # reload_dataloaders_every_n_epochs=1,
    )
    return trainer


def fit_with_initialized_optimizer_state(
    trainer: pl.Trainer,
    model_module: KFoldTrainingModule,
    data_module: TrainingDataModule,
    checkpoint_path: str,
    load_global_step: bool,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    state_dict = checkpoint["state_dict"]
    init_from_ema = tuple(model_module.config.init_from_ema)
    if init_from_ema:
        if "ema" not in checkpoint:
            raise KeyError(
                "Checkpoint does not contain EMA parameters required by "
                f"init_from_ema={list(init_from_ema)}."
            )
        state_dict, initialized_keys = initialize_parameter_groups_from_ema(
            state_dict=state_dict,
            ema_params=checkpoint["ema"]["shadow_params"],
            parameter_groups=model_module.model.get_parameter_group_names(),
            groups_to_initialize=init_from_ema,
        )
        if trainer.is_global_zero:
            logging.info(
                "Initialized %d model parameters from EMA for groups %s.",
                len(initialized_keys),
                list(init_from_ema),
            )

    if model_module.train_apo_repr_mix:
        prefix = "model.apo_repr_mix."
        current = model_module.state_dict()
        expected = {k for k in current if k.startswith(prefix)}
        present = {k for k in state_dict if k.startswith(prefix)}
        if present and present != expected:
            raise ValueError(
                "Partial apo mixer checkpoint; refusing silent initialization"
            )
        if not present:
            # Only brand-new mixer tensors may be absent from a base checkpoint.
            state_dict = dict(state_dict)
            state_dict.update({k: current[k] for k in expected})
        model_module.load_state_dict(state_dict, strict=True)
        # A new optimizer must start EMA from the actual initialized weights,
        # not from stale pre-load values or differently selected legacy EMA groups.
        from kfold.training.optim.ema import ExponentialMovingAverage

        model_module.ema = ExponentialMovingAverage(
            model_module.model,
            model_module.optimizer_config.ema_decay,
            model_module.submodules_to_ignore_for_ema,
        )
    else:
        model_module.load_state_dict(state_dict, strict=True)
        model_module.on_load_checkpoint(checkpoint)
    if load_global_step:
        model_module.last_lr_step = checkpoint["global_step"]
        trainer.fit_loop.load_state_dict(checkpoint["loops"]["fit_loop"])
    del checkpoint  # Free memory

    trainer.fit(model_module, datamodule=data_module)


def train(args) -> None:
    # To ignore warning
    torch.set_float32_matmul_precision("high")

    cfg = parse_config(args)

    trainer = build_trainer(cfg, args.debug)

    # Set random seed
    pl.seed_everything(cfg.train.seed, workers=True, verbose=False)

    model_module = KFoldTrainingModule(cfg)
    data_module = TrainingDataModule(cfg.train.data)

    # Print config
    if trainer.is_global_zero:
        print_config(cfg)

    if cfg.train.load_opt_state:
        trainer.fit(
            model_module,
            datamodule=data_module,
            ckpt_path=args.resume_from_checkpoint,
        )
    else:
        if args.resume_from_checkpoint is None:
            raise ValueError(
                "--resume_from_checkpoint is required when load_opt_state is false."
            )
        fit_with_initialized_optimizer_state(
            trainer,
            model_module,
            data_module,
            args.resume_from_checkpoint,
            load_global_step=cfg.train.load_global_step,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    args = parse_args()
    train(args)
