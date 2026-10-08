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

"""Define training modules for k-fold"""

import dataclasses
import gc
from typing import Any, Self

import lightning.pytorch as pl
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torchmetrics import MeanMetric, MetricCollection

from kfold.data.types.model_input import FoldingInput
from kfold.data.types.structure import RefStructure
from kfold.model.model_train import KFoldConfig, KFoldForTrain
from kfold.model.modules.ecsi import ECSISOARConfig
from kfold.training.utils.gradient_logging import gradient_norm, parameter_norm
from kfold.utils import confidence_metrics
from kfold.utils.config import to_dict
from kfold.utils.geometry.rigid_align import compute_rmsd

from . import loss as loss_fn
from .metrics import structure_metrics as validation_metrics
from .optim.ema import ExponentialMovingAverage
from .optim.lr_scheduler import AF3LRScheduler

_PARCAE_RECURRENCE_BASE_SEED = 42
_PARCAE_RECURRENCE_SCHEDULE_SIZE = 100_000
_PARCAE_RECURRENCE_SAMPLING_MODES = {"shared", "rank_independent"}


class _Config:
    @classmethod
    def from_dict(cls, config) -> Self:
        merged = OmegaConf.merge(OmegaConf.create(cls), OmegaConf.create(config))
        return OmegaConf.to_object(merged)


@dataclasses.dataclass(kw_only=True)
class TrainConfig:
    """Configuration for training and validation steps."""

    name: str
    out_dir: str
    seed: int
    compile: "CompileConfig"
    training: "TrainingConfig"
    validation: "ValidationConfig"
    optimizer: "OptimizerConfig"
    loss: "LossConfig"
    # Multi-stage training
    load_opt_state: bool = True
    load_global_step: bool = True
    init_from_ema: tuple[str, ...] = ()


@dataclasses.dataclass(kw_only=True)
class OptimizerConfig(_Config):
    """Optimizer configuration.
    See Section 5.4 of the AlphaFold3 paper.
    """

    # optimizer
    opt: str = "adam"
    beta_1: float = 0.9
    beta_2: float = 0.95
    eps: float = 1e-8
    # lr scheduler
    lr_scheduler: str = "af3"  # or "none"
    base_lr: float = 0
    max_lr: float = 0.0018
    lr_warmup_no_steps: int = 1000
    lr_start_decay_after_n_steps: int = 50000
    lr_decay_every_n_steps: int = 50000
    lr_decay_factor: float = 0.95
    # ema
    ema_decay: float = 0.999
    validate_with_ema_after_n_steps: int = 10000


@dataclasses.dataclass(kw_only=True)
class ParcaeTrainConfig(_Config):
    """Training-time Parcae recycle-count sampling configuration."""

    max_recycles: int = 5
    min_recycles: int = 0
    poisson_mean: float = 2.0
    grad_recurrence_steps: int = 2
    recurrence_sampling_mode: str = "shared"


def _build_clamped_poisson_recycle_schedule(
    config: ParcaeTrainConfig,
    seed: int,
    size: int = _PARCAE_RECURRENCE_SCHEDULE_SIZE,
) -> np.ndarray:
    rng = np.random.default_rng(seed=seed)
    sampled_recycles = rng.poisson(
        lam=config.poisson_mean,
        size=size,
    )
    return np.clip(
        sampled_recycles,
        config.min_recycles,
        config.max_recycles,
    ).astype(np.int64)


def _select_recycle_count(schedule: np.ndarray, global_step: int) -> int:
    idx = global_step % len(schedule)
    return int(schedule[idx])


@dataclasses.dataclass(kw_only=True)
class TrainingConfig(_Config):
    """Training step configuration."""

    # Whether to train each submodules
    train_trunk: bool = True
    train_diffusion_head: bool = True
    train_confidence_head: bool = False

    parcae: ParcaeTrainConfig
    # for structure model training
    diffusion_batch_size: int = 48
    soar: dict[str, Any]
    # for confidence module training
    num_mini_rollout_steps: int = 20
    num_mini_rollout_samples: int = 1


def _training_metric_log_name(metric_name: str) -> str:
    if metric_name.startswith("x_0_perturb_"):
        return f"x_0_perturb/{metric_name.removeprefix('x_0_perturb_')}"
    return f"train/{metric_name}"


@dataclasses.dataclass(kw_only=True)
class ValidationConfig(_Config):
    """Validation step configuration."""

    num_recycles: int = 3
    num_steps: int = 20
    num_diffusion_samples: int = 5


@dataclasses.dataclass(kw_only=True)
class LossConfig(_Config):
    """Loss configuration."""

    weights: dict[str, float]
    diffusion_loss: Any
    patch_geometry_loss: Any


@dataclasses.dataclass(kw_only=True)
class CompileConfig(_Config):
    enabled: bool = False
    mode: str = "default"
    dynamic: bool = False


class KFoldTrainingModule(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.global_config: DictConfig = config
        self.config: TrainConfig = config.train
        self.training_config: TrainingConfig = TrainingConfig.from_dict(
            self.config.training
        )
        self.soar_config = ECSISOARConfig(**self.training_config.soar)
        self.parcae_train_config: ParcaeTrainConfig = ParcaeTrainConfig.from_dict(
            self.training_config.parcae
        )
        self.validation_config: ValidationConfig = ValidationConfig.from_dict(
            self.config.validation
        )
        self.optimizer_config: OptimizerConfig = OptimizerConfig.from_dict(
            self.config.optimizer
        )
        self.loss_config: LossConfig = LossConfig.from_dict(self.config.loss)
        self.compile_config: CompileConfig = CompileConfig.from_dict(self.config.compile)

        # Save hyperparameters
        self.save_hyperparameters(to_dict(self.global_config))

        # Whether to train structure and confidence modules
        self.train_trunk: bool = self.training_config.train_trunk
        self.train_diffusion_head: bool = self.training_config.train_diffusion_head
        self.train_confidence_head: bool = self.training_config.train_confidence_head

        # Initialize model here
        model_config: KFoldConfig = self.global_config.model
        self.model = KFoldForTrain(model_config)

        # Compile
        if self.compile_config.enabled:
            self.model.do_compile(
                mode=self.compile_config.mode, dynamic=self.compile_config.dynamic
            )

        # Freeze parts of the model if needed
        self.freeze_submodules()

        # Setup EMA
        self.submodules_to_ignore_for_ema = (
            "prot_seq_encoder",
            "rna_seq_encoder",
            "prot_struct_encoder",
        )
        self.ema: ExponentialMovingAverage = ExponentialMovingAverage(
            model=self.model,
            decay=self.optimizer_config.ema_decay,
            submodules_to_ignore=self.submodules_to_ignore_for_ema,
        )
        self.stored_weights: dict[str, torch.Tensor] | None = None
        self.last_lr_step = -1

        # Setup losses and metrics
        self.setup_losses()
        self.setup_metrics()

        # Parcae methodology: pre-sample a clamped-Poisson recycle schedule for
        # training. The number of recurrent trunk steps saved for backprop is
        # controlled by parcae.grad_recurrence_steps. "shared" preserves the
        # previous cross-rank lockstep schedule; "rank_independent" lazily builds
        # a deterministic rank-specific schedule once Lightning rank is known.
        self._shared_recycles_per_step: np.ndarray = (
            _build_clamped_poisson_recycle_schedule(
                self.parcae_train_config,
                seed=_PARCAE_RECURRENCE_BASE_SEED,
            )
        )
        self._rank_independent_recycles_per_step: np.ndarray | None = None
        self._rank_independent_recycles_rank: int | None = None

    def _get_recurrence_schedule_rank(self) -> int:
        trainer = getattr(self, "_trainer", None)
        if trainer is None:
            return 0
        return int(getattr(trainer, "global_rank", 0))

    def _get_active_recycles_per_step(self) -> np.ndarray:
        if self.parcae_train_config.recurrence_sampling_mode == "shared":
            return self._shared_recycles_per_step

        rank = self._get_recurrence_schedule_rank()
        if (
            self._rank_independent_recycles_per_step is None
            or self._rank_independent_recycles_rank != rank
        ):
            self._rank_independent_recycles_per_step = (
                _build_clamped_poisson_recycle_schedule(
                    self.parcae_train_config,
                    seed=_PARCAE_RECURRENCE_BASE_SEED + rank,
                )
            )
            self._rank_independent_recycles_rank = rank

        assert self._rank_independent_recycles_per_step is not None
        return self._rank_independent_recycles_per_step

    def _get_num_recycles_for_current_step(self) -> int:
        schedule = self._get_active_recycles_per_step()
        return _select_recycle_count(schedule, int(self.global_step))

    def freeze_submodules(self):
        """Freeze submodules based on the training configuration."""
        # FIXME: (SeonghwanSeo) I did not test this function yet.
        # This is required when only selected model components are trained.

        self.frozen_modules = []
        self.frozen_modules += self.model.get_pretrained_module_names()

        if self.train_trunk is False:
            self.frozen_modules += self.model.get_trunk_module_names()
            self.frozen_modules += self.model.get_distogram_head_module_names()

            # freeze trunk Parcae params directly, as they are not treated as modules
            for param_name in self.model.get_trunk_parameter_names():
                param = getattr(self.model, param_name)
                param.requires_grad_(False)

        if self.train_diffusion_head is False:
            self.frozen_modules += self.model.get_diffusion_head_module_names()

        if self.train_confidence_head is False:
            self.frozen_modules += self.model.get_confidence_head_module_names()

        for module_name in self.frozen_modules:
            module = getattr(self.model, module_name)
            if module is None:
                continue
            for param in module.parameters():
                param.requires_grad_(False)

    def train(self, mode: bool = True):
        """Override train() to set sub-modules to eval mode if frozen."""
        out = super().train(mode)
        for module_name in self.frozen_modules:
            module = getattr(self.model, module_name)
            if module is not None:
                module.eval()
        return out

    def setup_losses(self):
        """Setup loss functions for training"""
        loss_config = self.loss_config
        self.loss_weights: dict[str, float] = loss_config.weights

        # Distogram loss
        self.distogram_loss = loss_fn.distogram.DistogramLoss()
        self.patch_geometry_loss = loss_fn.patch_geometry.PatchPairGeometryLoss(
            **loss_config.patch_geometry_loss
        )

        # Diffusion loss
        diffusion_loss_config = loss_config.diffusion_loss
        self.weighted_mse_loss = loss_fn.diffusion.WeightedMSELoss(
            **diffusion_loss_config["mse_loss"]
        )
        # Only used in fine-tuning stage
        self.bond_loss = loss_fn.diffusion.BondLoss(**diffusion_loss_config["bond_loss"])
        # Only used in regular training stage
        self.smooth_lddt_loss = loss_fn.diffusion.SmoothLDDTLoss(
            **diffusion_loss_config["smooth_lddt_loss"]
        )

        self.plddt_loss = loss_fn.confidence.PLDDTLoss()
        self.pde_loss = loss_fn.confidence.PDELoss()
        self.exp_res_loss = loss_fn.confidence.ExperimentallyResolvedPredictionLoss()
        self.pae_loss = loss_fn.confidence.PAELoss()

    def setup_metrics(self):
        """Setup metrics for validation"""
        # NOTE (Seonghwan): MeanMetric is required since the number of values
        # per each metric key are different for each batch during validation.
        # self.log() raises deadlock error when aggregating metrics in DDP.
        self.val_dataset_names: list[str] = [
            ds.name for ds in self.global_config.train.data.val_datasets
        ]
        val_metrics = []
        for name in self.val_dataset_names:
            dataset_metrics = {}
            for prefix in ["top1", "top5"]:
                for k in validation_metrics.main_metric_names:
                    dataset_metrics[f"{prefix}/{k}"] = MeanMetric()
            for k in validation_metrics.monitor_metric_names:
                dataset_metrics[f"monitor/{k}"] = MeanMetric()
            val_metrics.append(MetricCollection(dataset_metrics, prefix=f"{name}/"))
        self.val_metrics = torch.nn.ModuleList(val_metrics)

    def configure_optimizers(self):  # type: ignore
        config = self.optimizer_config
        parameters = [p for p in self.parameters() if p.requires_grad]

        if config.opt.lower() == "adam":
            optimizer = torch.optim.Adam(
                parameters,
                betas=(config.beta_1, config.beta_2),
                eps=config.eps,
                lr=config.base_lr,
            )
        else:
            raise NotImplementedError(f"Optimizer {config.opt} not implemented yet.")

        if self.last_lr_step != -1:
            for param_group in optimizer.param_groups:
                param_group.setdefault("initial_lr", config.base_lr)

        if config.lr_scheduler == "af3":
            scheduler = AF3LRScheduler(
                optimizer,
                last_epoch=self.last_lr_step,
                base_lr=config.base_lr,
                max_lr=config.max_lr,
                warmup_no_steps=config.lr_warmup_no_steps,
                start_decay_after_n_steps=config.lr_start_decay_after_n_steps,
                decay_every_n_steps=config.lr_decay_every_n_steps,
                decay_factor=config.lr_decay_factor,
            )
        else:
            raise NotImplementedError(
                f"LR scheduler {config.lr_scheduler} not implemented yet."
            )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward(
        self, f_input: FoldingInput, mode: str
    ) -> dict[str, dict[str, torch.Tensor]]:
        if mode == "train":
            training_config = self.training_config
            num_recycles = self._get_num_recycles_for_current_step()
            return self.model.forward_train(
                f_input,
                num_recycles=num_recycles,
                grad_recurrence_steps=self.parcae_train_config.grad_recurrence_steps,
                num_mini_rollout_steps=training_config.num_mini_rollout_steps,
                num_mini_rollout_samples=training_config.num_mini_rollout_samples,
                diffusion_batch_size=training_config.diffusion_batch_size,
                soar_config=self.soar_config,
                train_trunk=self.train_trunk,
                train_diffusion_head=self.train_diffusion_head,
                train_confidence_module=self.train_confidence_head,
            )
        elif mode == "validation":
            val_config = self.validation_config
            dict_out = self.model.sample_validation(
                f_input,
                num_recycles=val_config.num_recycles,
                num_steps=val_config.num_steps,
                num_samples=val_config.num_diffusion_samples,
            )
            return dict_out
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def training_step(
        self,
        batch: tuple[FoldingInput, list[dict]],
        batch_idx: int,
    ) -> torch.Tensor:
        f_input, _ = batch  # second one is full_structure_dict, not used in training step

        # Compute the forward pass
        out: dict[str, torch.Tensor] = self(f_input=f_input, mode="train")
        with torch.autocast("cuda", dtype=torch.float32):
            loss, metrics = self.compute_losses(batch, out)

        for k, v in metrics.items():
            self.log(
                _training_metric_log_name(k),
                v,
                prog_bar=(k == "loss"),
                sync_dist=False,
            )

        return loss

    def compute_losses(
        self, batch: tuple[FoldingInput, list[dict]], model_output: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute losses of given the model output."""
        f_input, struct_info = batch

        # NOTE: Compute the losses in float32 for better numerical stability
        # Compute losses
        if self.train_trunk:
            distogram_loss = self.distogram_loss(
                distogram_out=model_output["distogram"],
                f_input=f_input,
            ).mean()
            patch_weight = self.loss_weights["patch_geometry"]
            if patch_weight > 0:
                patch_geometry_loss, patch_geometry_metrics = self.patch_geometry_loss(
                    model_output["patch_geometry"]
                )
            else:
                patch_geometry_loss, patch_geometry_metrics = 0.0, {}
        else:
            distogram_loss = 0.0
            patch_geometry_loss, patch_geometry_metrics = 0.0, {}

        if self.train_diffusion_head:
            diffusion_out = model_output["diffusion"]
            diffusion_loss, diffusion_metrics = self.compute_diffusion_loss(
                x_pred=diffusion_out["x_0_hat"],
                x_true=diffusion_out["x_gt"],
                f_input=f_input,
                per_sample_weights=model_output["diffusion"]["loss_weights"],
                supervision_weights=diffusion_out.get("soar_supervision_weights"),
            )
            if "x_0_perturb_applied_mask" in diffusion_out:
                self._add_x_0_perturb_metrics(
                    diffusion_metrics=diffusion_metrics,
                    diffusion_out=diffusion_out,
                )

        else:
            diffusion_loss, diffusion_metrics = 0.0, {}

        if self.train_confidence_head:
            x_pred = model_output["sample"]["coordinates"]
            confidence_loss_mask = torch.tensor(
                [info["train_confidence_head"] for info in struct_info],
                device=x_pred.device,
                dtype=torch.bool,
            )
            x_gt, mask_gt = loss_fn.confidence.get_aligned_gt_structure(
                x_pred=x_pred,
                f_input=f_input,
                struct_info=struct_info,
            )  # [B, Nsample, Latom, 3]
            confidence_loss, confidence_metrics = self.compute_confidence_loss(
                confidence_out=model_output["confidence"],
                x_pred=x_pred,
                x_gt=x_gt,
                mask=mask_gt,
                f_input=f_input,
                loss_mask=confidence_loss_mask,
            )
            # Log the rmsd between mini-rollout sample and GT.
            rmsd = compute_rmsd(x_pred, x_gt, mask_gt, align=True)  # [B, Nsample]
            sample_metrics = {"mini_rollout_rmsd": rmsd.mean()}

        else:
            confidence_loss, confidence_metrics = 0.0, {}
            sample_metrics = {}

        # Aggregate losses
        # See Section 5.3 Equation 15
        loss_weights = self.loss_weights
        loss = (
            loss_weights["diffusion"] * diffusion_loss
            + loss_weights["distogram"] * distogram_loss
            + loss_weights["confidence"] * confidence_loss
            + loss_weights["patch_geometry"] * patch_geometry_loss
        )  # [B,]
        assert torch.is_tensor(loss), "Loss must be a torch.Tensor."

        # Log loss and metrics
        all_metrics = (
            patch_geometry_metrics
            | diffusion_metrics
            | confidence_metrics
            | sample_metrics
        )
        all_metrics["loss"] = loss.detach()

        return loss, all_metrics

    @staticmethod
    def _add_x_0_perturb_metrics(
        *,
        diffusion_metrics: dict[str, torch.Tensor],
        diffusion_out: dict[str, torch.Tensor],
    ) -> None:
        """Add actual high-time x0-perturb exposure and displacement telemetry."""
        t = diffusion_out["t"][:, : diffusion_out["x_0_perturb_applied_mask"].shape[1]]
        time_eligible = diffusion_out["x_0_perturb_time_eligible_mask"].bool()
        eligible = diffusion_out["x_0_perturb_eligible_mask"].bool()
        requested = diffusion_out["x_0_perturb_requested_mask"].bool()
        applied = diffusion_out["x_0_perturb_applied_mask"].bool()
        x_0_rmsd = diffusion_out["x_0_perturb_x_0_rmsd"].detach().float()
        x_t_rmsd = diffusion_out["x_0_perturb_x_t_rmsd"].detach().float()
        chain_count = diffusion_out["x_0_perturb_resolved_chain_count"].detach().float()

        def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            weights = mask.to(values.dtype)
            return (values * weights).sum() / weights.sum().clamp(min=1.0)

        prefix = "x_0_perturb_"
        diffusion_metrics[f"{prefix}time_eligible_fraction"] = (
            time_eligible.float().mean()
        )
        diffusion_metrics[f"{prefix}eligible_fraction"] = eligible.float().mean()
        diffusion_metrics[f"{prefix}requested_fraction"] = requested.float().mean()
        diffusion_metrics[f"{prefix}applied_fraction"] = applied.float().mean()
        diffusion_metrics[f"{prefix}applied_given_eligible"] = (
            applied.float().sum() / eligible.float().sum().clamp(min=1.0)
        )
        diffusion_metrics[f"{prefix}applied_t_mean"] = masked_mean(t.float(), applied)
        diffusion_metrics[f"{prefix}x_0_rmsd_mean"] = masked_mean(x_0_rmsd, applied)
        diffusion_metrics[f"{prefix}x_t_rmsd_mean"] = masked_mean(x_t_rmsd, applied)
        diffusion_metrics[f"{prefix}x_t_rmsd_max"] = x_t_rmsd.max()
        diffusion_metrics[f"{prefix}resolved_chain_count_mean"] = chain_count.mean()

        for bin_index in range(10):
            lower = bin_index / 10.0
            upper = (bin_index + 1) / 10.0
            in_bin = (t >= lower) & (t < upper)
            bin_name = f"t{bin_index:02d}_{bin_index + 1:02d}"
            diffusion_metrics[f"{prefix}{bin_name}_applied_fraction"] = (
                applied & in_bin
            ).float().sum() / in_bin.float().sum().clamp(min=1.0)
            diffusion_metrics[f"{prefix}{bin_name}_x_t_rmsd_mean"] = masked_mean(
                x_t_rmsd, applied & in_bin
            )

    def validation_step(
        self,
        batch: tuple[FoldingInput, dict],
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        val_config = self.validation_config
        num_samples = val_config.num_diffusion_samples

        f_input, struct_info = batch
        assert not f_input.is_batched, "Validation input should not be batched."
        ref_struct: RefStructure = struct_info["structure"]
        symmetry_dict: dict = struct_info["symmetry"]

        try:
            model_out: dict[str, dict[str, torch.Tensor]] = self(
                f_input=f_input, mode="validation"
            )
        except RuntimeError as e:  # catch out of memory exceptions
            if "out of memory" in str(e):
                print("**WARNING**: ran out of memory, skipping batch")
                gc.collect()
                torch.cuda.empty_cache()
                return
            else:
                raise e
        diffusion_out = model_out["diffusion"]
        distogram_out = model_out["distogram"]
        confidence_out = model_out["confidence"]

        token_mask = f_input.token.pad_mask  # [L,]
        atom_mask = f_input.atom.pad_mask  # [Natom,]
        n_tokens: int = int(token_mask.sum().item())
        n_atoms: int = int(atom_mask.sum().item())
        assert n_atoms == ref_struct.num_atoms

        # Compute validation metrics
        sample_metrics: list[dict[str, Any]] = []
        with torch.autocast("cuda", torch.float32):
            # Permute predicted and true coordinates to align
            for i in range(num_samples):
                pred_coords_i = diffusion_out["coordinates"][i, :n_atoms]
                struct_i = validation_metrics.get_aligned_gt_structure(
                    ref_struct,
                    pred_coords_i,
                    symmetry_dict=symmetry_dict,
                )
                metric_i = validation_metrics.compute_validation_metric(
                    struct_i, pred_coords_i
                )
                sample_metrics.append(metric_i)

            # Select the best sample based on global PDE score.
            top1_index = None  # Use oracle sample.
            if self.train_confidence_head:
                pde = confidence_metrics.compute_pde(
                    confidence_out["pde_logits"],
                    confidence_out["pde_bin_centers"],
                    mask=token_mask,
                )  # [Nsample, Ntoken, Ntoken]
                prob_contact = distogram_out["prob_contact"]
                gpde: torch.Tensor = validation_metrics.compute_global_pde(
                    pde[:, :n_tokens, :n_tokens],
                    prob_contact[:n_tokens, :n_tokens],
                )  # [Nsample,]
                assert gpde.shape == (num_samples,)
                top1_index = int(gpde.argmin().item())

            # Aggregate metrics
            aggr_metrics = validation_metrics.aggregate_validation_metrics(
                sample_metrics, top1_index
            )

        # Update validation metrics
        metrics: MetricCollection = self.val_metrics[dataloader_idx]
        for prefix in ["top1", "top5"]:
            _m = aggr_metrics[prefix]
            for k in validation_metrics.main_metric_names:
                if k in _m:
                    metrics[f"{prefix}/{k}"].update(_m[k])
        for k in validation_metrics.monitor_metric_names:
            _m = aggr_metrics["monitor"]
            if k in _m:
                metrics[f"monitor/{k}"].update(_m[k])

    def on_validation_epoch_start(self):
        torch.backends.cudnn.benchmark = False

    def on_validation_epoch_end(self):
        torch.backends.cudnn.benchmark = True
        for metrics in self.val_metrics:
            if not self.trainer.sanity_checking:
                avg_values = metrics.compute()
                # NOTE: do not filter out NaN values to avoid deadlock in DDP
                self.log_dict(
                    avg_values,
                    on_step=False,
                    on_epoch=True,
                    # Already synced in compute(), but keep to avoid warning...
                    sync_dist=True,
                )
            metrics.reset()

        # Clear cache after validation
        # NOTE: is this necessary?
        gc.collect()
        torch.cuda.empty_cache()

    # === Loss functions === #
    def compute_diffusion_loss(
        self,
        x_pred: torch.Tensor,
        x_true: torch.Tensor,
        f_input: FoldingInput,
        per_sample_weights: torch.Tensor,
        supervision_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute diffusion structure loss.
        See Section 3.7.1 Diffusion Training.

        Parameters
        ----------
        x_pred : torch.Tensor
            The predicted atom coordinates of shape (B, Nsample, Latom, 3).
        x_true : torch.Tensor
            The ground truth atom coordinates of shape (B, Nsample, Latom, 3).
        f_input : FoldingInput
            The input features containing the target distogram and masks.
        per_sample_weights : torch.Tensor
            The per-sample loss weights of shape (B, Nsample),
            which is computed from the diffusion noise scale.

        Returns
        -------
        diffusion_loss : torch.Tensor
            The computed diffusion loss (scalar).
        metrics : dict[str, torch.Tensor]
            A dictionary containing loss metrics.
        """
        metrics: dict[str, torch.Tensor] = {}
        alpha_bond = self.loss_weights["bond"]
        alpha_smooth_lddt = self.loss_weights["smooth_lddt"]
        if supervision_weights is None:
            supervision_weights = torch.ones_like(per_sample_weights)
        if supervision_weights.shape != per_sample_weights.shape:
            raise ValueError(
                "supervision_weights and per_sample_weights must have the same shape."
            )

        def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            return (values * weights).sum() / weights.sum().clamp(min=1.0)

        # Equations 3-4
        L_mse = self.weighted_mse_loss(x_pred, x_true, f_input)
        L_mse_weighted = L_mse * per_sample_weights  # [B, Nsample]
        metrics["mse_loss"] = weighted_mean(L_mse_weighted.detach(), supervision_weights)

        # Equation 5
        if alpha_bond > 0:
            L_bond = self.bond_loss(x_pred, x_true, f_input)
            L_bond_weighted = L_bond * per_sample_weights  # [B, Nsample]
            metrics["bond_loss"] = weighted_mean(
                L_bond_weighted.detach(), supervision_weights
            )
        else:
            L_bond_weighted = None

        # Algorithm 27
        if alpha_smooth_lddt > 0:
            L_smooth_lddt = self.smooth_lddt_loss(x_pred, x_true, f_input)  # [B, Nsample]
            metrics["smooth_lddt_loss"] = weighted_mean(
                L_smooth_lddt.detach(), supervision_weights
            )
        else:
            L_smooth_lddt = None

        # Equation 6
        # NOTE: per-sample weights are already applied in L_mse and L_bond
        # L_diff = loss_weights(L_mse + α_com * L_com + α_bond * L_bond) + L_smooth_lddt
        L_diffusion_per_sample = L_mse_weighted
        if L_bond_weighted is not None:
            L_diffusion_per_sample = L_diffusion_per_sample + alpha_bond * L_bond_weighted
        if L_smooth_lddt is not None:
            L_diffusion_per_sample = (
                L_diffusion_per_sample + alpha_smooth_lddt * L_smooth_lddt
            )

        # Normalize once over base and auxiliary objective mass.
        L_diffusion = weighted_mean(L_diffusion_per_sample, supervision_weights)
        metrics["diffusion_loss"] = L_diffusion.detach()

        return L_diffusion, metrics

    def compute_confidence_loss(
        self,
        confidence_out: dict[str, torch.Tensor],
        x_pred: torch.Tensor,
        x_gt: torch.Tensor,
        mask: torch.Tensor,
        f_input: FoldingInput,
        loss_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute confidence head loss.

        Parameters
        ----------
        confidence_out : dict[str, torch.Tensor]
            Confidence logits and their corresponding bin centers.
        x_pred : torch.Tensor
            The mini-rollout sample coordinates of shape (B, Nsample, Latom, 3).
        x_gt : torch.Tensor
            The GT coordinates aligned to x_pred of shape (B, Nsample, Latom, 3).
        mask : torch.Tensor
            The mask indicating which residues to include in the loss computation,
            of shape (B, Nsample, Latom).
        f_input : FoldingInput
            The input features containing the target distogram and masks.
        loss_mask : torch.Tensor
            A boolean tensor of shape (B,) indicating which samples in the batch should
            contribute to the confidence loss.

        Returns
        -------
        confidence_loss : torch.Tensor
            The computed confidence loss (scalar).
        metrics : dict[str, torch.Tensor]
            A dictionary containing loss metrics.
        """
        metrics: dict[str, torch.Tensor] = {}
        num_samples = x_pred.shape[1]
        loss_mask = loss_mask.float()[:, None]  # [B, 1]
        num_valid_samples = (loss_mask.sum() * num_samples).clamp(1)

        L_pde = self.pde_loss(
            confidence_out["pde_logits"],
            confidence_out["pde_bin_centers"],
            x_pred,
            x_gt,
            mask,
            f_input,
        )
        L_pde = L_pde * loss_mask  # [B, Nsample]
        metrics["pde_loss"] = L_pde.detach().sum() / num_valid_samples

        L_plddt = self.plddt_loss(
            confidence_out["plddt_logits"],
            confidence_out["plddt_bin_centers"],
            x_pred,
            x_gt,
            mask,
            f_input,
        )
        L_plddt = L_plddt * loss_mask  # [B, Nsample]
        metrics["plddt_loss"] = L_plddt.detach().sum() / num_valid_samples

        is_resolved = mask
        pad_mask = f_input.atom.pad_mask
        L_resolved = self.exp_res_loss(
            confidence_out["resolved_logits"],
            is_resolved,
            pad_mask,
        )
        L_resolved = L_resolved * loss_mask  # [B, Nsample]
        metrics["resolved_loss"] = L_resolved.detach().sum() / num_valid_samples

        # NOTE: PAE loss return 0.0 when alpha_pae is 0.
        L_pae = self.pae_loss(
            confidence_out["pae_logits"],
            confidence_out["pae_bin_centers"],
            x_pred,
            x_gt,
            mask,
            f_input,
        )
        L_pae = L_pae * loss_mask  # [B, Nsample]
        metrics["pae_loss"] = L_pae.detach().sum() / num_valid_samples

        L_confidence_per_sample = L_pde + L_plddt + L_resolved + L_pae
        L_confidence = L_confidence_per_sample.sum() / num_valid_samples

        metrics["confidence_loss"] = L_confidence.detach()

        return L_confidence, metrics

    # === Training logs === #
    def on_before_optimizer_step(self, optimizer) -> None:
        if self.trainer.global_step % 10 == 0:
            self.log_model_state()

    def log_model_state(self):
        """Log model parameter and gradient norms."""

        model = self.model
        self.log("monitor/grad_norm", gradient_norm(model), prog_bar=False)
        self.log("monitor/param_norm", parameter_norm(model), prog_bar=False)

        if self.train_trunk:
            self.log(
                "monitor/grad_norm_lm_stack",
                gradient_norm(model.lm_stack),
                sync_dist=False,
                prog_bar=False,
            )
            self.log(
                "monitor/param_norm_lm_stack",
                parameter_norm(model.lm_stack),
                sync_dist=False,
                prog_bar=False,
            )
            self.log(
                "monitor/grad_norm_main_stack",
                gradient_norm(model.main_stack),
                sync_dist=False,
                prog_bar=False,
            )
            self.log(
                "monitor/param_norm_refine_stack",
                parameter_norm(model.refine_stack),
                sync_dist=False,
                prog_bar=False,
            )
            self.log(
                "monitor/grad_norm_refine_stack",
                gradient_norm(model.refine_stack),
                sync_dist=False,
                prog_bar=False,
            )
            self.log(
                "monitor/param_norm_main_stack",
                parameter_norm(model.main_stack),
                sync_dist=False,
                prog_bar=False,
            )

        if self.train_diffusion_head:
            self.log(
                "monitor/grad_norm_score_model",
                gradient_norm(model.score_model),
                sync_dist=False,
                prog_bar=False,
            )
            self.log(
                "monitor/param_norm_score_model",
                parameter_norm(model.score_model),
                sync_dist=False,
                prog_bar=False,
            )

        if self.train_confidence_head:
            self.log(
                "monitor/grad_norm_confidence_head",
                gradient_norm(model.confidence_head),
                sync_dist=False,
                prog_bar=False,
            )
            self.log(
                "monitor/param_norm_confidence_head",
                parameter_norm(model.confidence_head),
                sync_dist=False,
                prog_bar=False,
            )

        pass

    def _remove_orig_mod_from_state_dict(
        self, state_dict: dict[str, Any]
    ) -> dict[str, Any]:
        """Remove '._orig_mod.' from state dict keys if present."""
        return {
            k.replace("._orig_mod.", ".") if "._orig_mod." in k else k: v
            for k, v in state_dict.items()
        }

    def _add_orig_mod_to_state_dict(
        self, state_dict: dict[str, Any], model_state_dict: dict[str, Any]
    ) -> dict[str, Any]:
        """Add '._orig_mod.' to state dict keys if required"""
        model_keys = set(model_state_dict.keys())
        state_keys = set(state_dict.keys())

        # Keys expected by the compiled model but missing in the checkpoint
        remaining_keys = model_keys - state_keys
        if len(remaining_keys) == 0:
            return state_dict  # No modification needed

        new_state_dict = dict(state_dict)
        for rk in remaining_keys:
            k = rk.replace("._orig_mod.", ".")
            if k in state_dict:
                new_state_dict[rk] = new_state_dict.pop(k)
        return new_state_dict

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        # remove pretrained model keys
        checkpoint["state_dict"] = {
            k: v
            for k, v in checkpoint["state_dict"].items()
            if "prot_seq_encoder." not in k
            and "rna_seq_encoder." not in k
            and "prot_struct_encoder." not in k
        }

        # Remove '._orig_mod.' from checkpoint keys
        checkpoint["state_dict"] = self._remove_orig_mod_from_state_dict(
            checkpoint["state_dict"]
        )

        # Add EMA state dict
        # Remove '._orig_mod.' from EMA state dict keys
        ema_state_dict = self.ema.state_dict()
        ema_state_dict["shadow_params"] = self._remove_orig_mod_from_state_dict(
            ema_state_dict["shadow_params"]
        )
        checkpoint["ema"] = ema_state_dict

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        # Load EMA state dict
        self.load_ema_state_dict(checkpoint["ema"])

    def load_state_dict(
        self, state_dict: dict[str, Any], strict: bool = True, assign: bool = False
    ):  # type: ignore
        """Override load_state_dict to handle EMA state dict."""
        # Remove '._orig_mod.' from state dict keys if present
        state_dict = self._remove_orig_mod_from_state_dict(state_dict)
        # Then, add '._orig_mod.' to state dict keys if required by the model
        state_dict = self._add_orig_mod_to_state_dict(state_dict, self.state_dict())
        # Remove 'model.' prefix from state dict keys if present
        state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}
        out = self.model.load_state_dict(state_dict, strict=strict)
        return out

    # === EMA === #
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):  # type: ignore
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)

        if self.ema.device != self.device:
            self.ema.to(self.device)
        self.ema.update(self.model)

    def _replace_ema_weights(self):
        if self.stored_weights is not None:
            # NOTE: To avoid accidentally replacing the weights multiple times,
            # we force only one replacement.
            raise ValueError("EMA weights have already been replaced.")
        self.stored_weights = {
            name: param.clone()
            for name, param in self.model.named_parameters()
            if not name.startswith(self.submodules_to_ignore_for_ema)
        }
        ema_params = self.ema.shadow_params
        for name, param in self.model.named_parameters():
            if name in ema_params:
                param.data.copy_(ema_params[name].data)

    def _restore_weights(self):
        if self.stored_weights is None:
            raise ValueError("No stored weights to restore.")
        for name, param in self.model.named_parameters():
            if name in self.stored_weights:
                param.data.copy_(self.stored_weights[name].data)
        self.stored_weights = None

    def on_validation_start(self):
        if self.ema.device != self.device:
            self.ema.to(self.device)
        if self.global_step >= self.config.optimizer.validate_with_ema_after_n_steps:
            self._replace_ema_weights()

    def on_validation_end(self) -> None:
        if self.stored_weights is not None:
            self._restore_weights()

    def load_ema_state_dict(self, state_dict: dict[str, Any]):
        """Load EMA state dict."""
        # Remove 'model.' prefix from EMA state dict keys if present.
        state_dict["shadow_params"] = {
            k.removeprefix("model."): v for k, v in state_dict["shadow_params"].items()
        }
        # Remove '._orig_mod.' from EMA state dict keys if present.
        state_dict["shadow_params"] = self._remove_orig_mod_from_state_dict(
            state_dict["shadow_params"]
        )
        # Add '._orig_mod.' to EMA state dict keys if required by the model.
        state_dict["shadow_params"] = self._add_orig_mod_to_state_dict(
            state_dict["shadow_params"], self.ema.shadow_params
        )
        assert self.ema.compatible(state_dict), (
            "EMA state dict is not compatible with the model."
        )
        self.ema.load_state_dict(state_dict, device=torch.device("cpu"))
