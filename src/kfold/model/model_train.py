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

import pathlib
from collections.abc import Mapping
from typing import Self

import torch
import torch.nn.functional as F

from kfold.data.types.model_input import FoldingInput
from kfold.model.model import KFold, KFoldConfig
from kfold.model.modules.ecsi import ECSISOARConfig
from kfold.utils.runtime import is_cuequivariance_installed


class KFoldForTrain(KFold):
    def __init__(self, config: KFoldConfig, *, kernel_backend: str = "auto"):
        # Triton kernels support inference only.
        if kernel_backend == "auto":
            kernel_backend = "cuequiv" if is_cuequivariance_installed() else "torch"
        if kernel_backend == "triton":
            raise ValueError(
                "The Triton backend is inference-only and cannot be used for training."
            )
        super().__init__(config, kernel_backend=kernel_backend)
        self.is_compiled = False

    def get_pretrained_module_names(self) -> list[str]:
        """Get the names of pretrained modules."""
        return [
            "prot_seq_encoder",
            "rna_seq_encoder",
            "prot_struct_encoder",
        ]

    def get_trunk_module_names(self) -> list[str]:
        """Get the names of trunk modules."""
        module_names = [
            "input_embedder",
            "prot_seq_to_s_lm",
            "lm_to_pair",
            "apo_module",
            "layernorm_z",
            "lm_stack",
            "main_stack",
            "linear_refine",
            "refine_stack",
            "patch_pair_geometry_head",
        ]
        if self.rna_seq_encoder is not None:
            module_names.append("rna_seq_to_s_lm")
        if self.prot_struct_encoder is not None:
            module_names.append("prot_struct_to_s_lm")
        return module_names

    def get_trunk_parameter_names(self) -> list[str]:
        """Get standalone trunk parameter names (Parcae)."""
        return [
            "parcae_log_a",
            "parcae_log_delta",
            "parcae_b_cont",
        ]

    def get_distogram_head_module_names(self) -> list[str]:
        """Get the names of distogram head modules."""
        return ["distogram_head"]

    def get_diffusion_head_module_names(self) -> list[str]:
        """Get the names of diffusion head modules."""
        return ["score_model"]

    def get_confidence_head_module_names(self) -> list[str]:
        """Get the names of confidence head modules."""
        return ["confidence_head"]

    def get_parameter_group_names(self) -> dict[str, list[str]]:
        """Get model parameter prefixes grouped by training component."""
        return {
            "trunk": [
                *self.get_trunk_module_names(),
                *self.get_trunk_parameter_names(),
            ],
            "distogram_head": self.get_distogram_head_module_names(),
            "diffusion_head": self.get_diffusion_head_module_names(),
            "confidence_head": self.get_confidence_head_module_names(),
        }

    def do_compile(self, mode: str = "default", dynamic: bool = False):
        """Compile the trunk and score model."""
        opts = {"mode": mode, "dynamic": dynamic}
        self.is_compiled = True
        self.prot_seq_encoder = torch.compile(self.prot_seq_encoder, **opts)
        if self.rna_seq_encoder is not None:
            self.rna_seq_encoder = torch.compile(self.rna_seq_encoder, **opts)
        if self.prot_struct_encoder is not None:
            self.prot_struct_encoder = torch.compile(self.prot_struct_encoder, **opts)
        self.apo_module = torch.compile(self.apo_module, **opts)

        self.lm_stack = torch.compile(self.lm_stack, **opts)
        self.main_stack = torch.compile(self.main_stack, **opts)
        self.refine_stack = torch.compile(self.refine_stack, **opts)
        self.score_model.do_compile(**opts)
        self.confidence_head.do_compile(**opts)

    def _get_model_module(self, module: torch.nn.Module | None) -> torch.nn.Module | None:
        """Return the underlying module when a compiled wrapper is not used."""
        if self.is_compiled and not self.training:
            return getattr(module, "_orig_mod", module)
        return module

    def _encode_lm_single(self, f_input: FoldingInput) -> torch.Tensor:
        """Merge the enabled pretrained encoders into the shared LM single."""
        prot_seq_encoder = self._get_model_module(self.prot_seq_encoder)
        rna_seq_encoder = self._get_model_module(self.rna_seq_encoder)
        prot_struct_encoder = self._get_model_module(self.prot_struct_encoder)

        assert prot_seq_encoder is not None
        s_lm = self.prot_seq_to_s_lm(prot_seq_encoder(f_input))

        if rna_seq_encoder is not None:
            s_lm = s_lm + self.rna_seq_to_s_lm(rna_seq_encoder(f_input))

        if prot_struct_encoder is not None:
            s_lm = s_lm + self.prot_struct_to_s_lm(prot_struct_encoder(f_input))

        return s_lm

    def run_trunk(
        self,
        f_input: FoldingInput,
        num_recycles: int,
        grad_recurrence_steps: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform the forward pass.

        Parameters
        ----------
        f_input : FoldingInput
            The input features.
        num_recycles : int
            The number of recycling steps.
        grad_recurrence_steps : int, optional
            Number of final recurrent trunk steps to track with autograd during
            training, by default 0.

        Returns
        -------
        s_inputs: torch.Tensor
            The input single representation of shape (B, L, c_s).
        s_lm: torch.Tensor
            The LM single representation of shape (B, L, c_s_lm).
        z: torch.Tensor
            The updated tensor of shape (B, L, L, c_z).
        """
        train = self.training and grad_recurrence_steps > 0

        # Get the underlying modules for compiled models
        apo_stack = self._get_model_module(self.apo_module)
        lm_stack = self._get_model_module(self.lm_stack)
        main_stack = self._get_model_module(self.main_stack)
        refine_stack = self._get_model_module(self.refine_stack)

        # Parcae theory: stable channel-wise state decay (a) and
        # Euler-discretized normalized input injection (b).
        a, b = self._parcae_discretized_dynamics()  # [C_z], [C_z, C_z]
        a, b = a.float(), b.float()

        # Input embedding
        s_inputs, z_inputs = self.input_embedder(f_input)

        # Trunk with recycling
        z_inputs = z_inputs.float()  # cast to float32 for numerical stability

        # Embedding of the apo state into the pair representation.
        z_inputs = z_inputs + apo_stack(f_input)

        # Initialize an independent pair-state z_0 instead of recycling from zeros.
        z = self._init_parcae_pair_state(z_inputs)
        token_mask = f_input.token.pad_mask
        pair_mask = token_mask[..., None] & token_mask[..., None, :]

        # Extract LM representation
        s_lm = self._encode_lm_single(f_input)
        z_lm = self.lm_to_pair(s_lm)

        # Main trunk iteration with Parcae recurrence
        grad_start = max(0, num_recycles + 1 - grad_recurrence_steps)
        for i in range(0, num_recycles + 1):
            enable_grad = train and i >= grad_start
            with torch.set_grad_enabled(enable_grad):
                if enable_grad and torch.is_autocast_enabled():
                    torch.clear_autocast_cache()
                _z_lm = F.dropout(z_lm, p=self.dropout, training=True)
                u_t = z_inputs + lm_stack(_z_lm, pair_mask)
                # Parcae recurrence: z_in = a * z_t + B_bar LN(u_t), followed
                # by the pair folding trunk as the nonlinear recurrent update.
                z = a * z + F.linear(self.layernorm_z(u_t), b)
                z = main_stack(z, pair_mask)

        # Refinement iteration
        z = refine_stack(self.linear_refine(z), pair_mask)

        return s_inputs, s_lm, z

    def forward_train(
        self,
        f_input: FoldingInput,
        soar_config: ECSISOARConfig,
        num_recycles: int = 3,
        diffusion_batch_size: int = 48,
        num_mini_rollout_steps: int = 20,
        num_mini_rollout_samples: int = 1,
        train_trunk: bool = True,
        train_diffusion_head: bool = True,
        train_confidence_module: bool = True,
        grad_recurrence_steps: int = 1,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Forward pass of KFold for model training.
        See Figure 2c in the main article of AlphaFold3.

        Parameters
        ----------
        f_input : FoldingInput
            Input data for folding model. Preferred to be batched.

        # For trunk with recycling:
        num_recycles : int
            Number of recycling cycles in trunk.
        grad_recurrence_steps : int, optional
            Number of final recurrent trunk steps to track with autograd during
            training.

        # For structure module training:
        diffusion_batch_size : int
            Batch size for diffusion training step.
        soar_config : ECSISOARConfig
            Sampler-matched Exact-Markov ECSI SOAR configuration.

        # For confidence module training with diffusion mini-rollout:
        num_mini_rollout_steps : int
            Number of diffusion steps to sample structures:
            Used for validation and confidence module training.
        num_mini_rollout_samples : int
            Number of diffusion samples to sample structures for
            confidence module training.

        train_trunk : bool, optional
            Whether to train trunk, by default True
        train_diffusion_head : bool, optional
            Whether to train diffusion head, by default True
        train_confidence_module : bool, optional
            Whether to train confidence module, by default True

        Returns
        -------
        model_out : dict[str, torch.Tensor]

            # For structure module training (distogram, diffusion)
            - distogram:
                - logits: [B, Ltoken, Ltoken, Dd]
                    Distogram logits
            - diffusion:
                - loss_weights: [B, N_noise]
                    Weights for diffusion noise scale
                - prior_atom_coords: [B, N_noise, Latom, 3]
                    Prior atom coordinates
                - noised_atom_coords: [B, N_noise, Latom, 3]
                    Noised atom coordinates
                - denoised_atom_coords: [B, N_noise, Latom, 3]
                    Denoised atom coordinates
                - true_atom_coords: [B, N_noise, Latom, 3]
                    Ground truth atom coordinates

            # For confidence module training
            - sample:
                - coordinates: [B, N_samples, Ltoken, 3]
                    Sampled atom coordinates
            - confidence:
                - pae_logits: [B, Ltoken, Ltoken, Dp]
                    Predicted aligned error logits
                - pde_logits: [B, Ltoken, Dp]
                    Predicted distance error logits
                - plddt_logits: [B, Latom, Dp]
                    Predicted lDDT logits
                - experimental_resolved_logits: [B, Latom, 2]
                    Predicted experimental resolved logits
        """
        # Ensure batched input
        assert f_input.is_batched, "Input must be batched for training.."
        batch_size: int = f_input.batch_size
        device: torch.device = f_input.device

        if train_trunk:
            assert grad_recurrence_steps > 0, (
                "grad_recurrence_steps must be > 0 for trunk training."
            )
        else:
            grad_recurrence_steps = 0  # No gradient tracking for trunk if not training

        # Output dictionary
        dict_out: dict[str, dict[str, torch.Tensor]] = {}

        s_inputs, s_lm, z = self.run_trunk(f_input, num_recycles, grad_recurrence_steps)
        z = z.float()

        if train_trunk:
            # Distogram head
            dict_out["distogram"] = self.distogram_head(z)
            if self.patch_pair_geometry_head is not None:
                patch_geometry_out = self.patch_pair_geometry_head(f_input, z)
                if patch_geometry_out:
                    dict_out["patch_geometry"] = patch_geometry_out

        if train_diffusion_head:
            # Diffusion head
            _z = z
            drop_rate = self.config.diffusion_conditioning_drop_rate
            if drop_rate > 0.0:
                drop_conditioning = torch.rand(batch_size, device=device) < drop_rate
                mask = ~drop_conditioning
                _z = z * mask[:, None, None, None]

            # Forward pass through diffusion head for training.
            with torch.autocast(device.type, enabled=False):
                dict_out["diffusion"] = self.diffusion_head.training_step(
                    f_input,
                    s_inputs,
                    _z,
                    diffusion_batch_size,
                    soar_config,
                )

        if train_confidence_module:
            # Stop gradients to input features and trunk outputs.
            # Sample structures with diffusion mini-rollout.
            with torch.no_grad(), torch.autocast(device.type, enabled=False):
                coordinates = self.diffusion_head.sample_structure(
                    f_input=f_input,
                    s_inputs=s_inputs,
                    z=z,
                    num_steps=num_mini_rollout_steps,
                    num_samples=num_mini_rollout_samples,
                )["coordinates"]  # [B, N_samples, Latom, 3]
            dict_out["sample"] = {
                "coordinates": coordinates,
            }
            _s_inputs = s_inputs.detach()
            _s_lm = s_lm.detach()
            _z = z.detach()

            # Randomly drop conditioning information for confidence head.
            drop_rate = self.config.confidence_conditioning_drop_rate
            if drop_rate > 0.0:
                drop_conditioning = torch.rand(batch_size, device=device) < drop_rate
                mask = ~drop_conditioning
                _z = _z * mask[:, None, None, None]

            # Forward pass through confidence head
            dict_out["confidence"] = self.confidence_head(
                f_input,
                _s_inputs,
                _s_lm,
                _z,
                coordinates,
            )

        return dict_out

    @torch.inference_mode()
    def sample_validation(
        self,
        f_input: FoldingInput,
        num_recycles: int = 4,
        num_steps: int = 200,
        num_samples: int = 5,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Forward pass of KFold model for model training.

        Parameters
        ----------
        f_input : FoldingInput
            Input data for folding model.
        num_recycles : int
            Number of recycling cycles in trunk.
        num_steps : int
            Number of diffusion steps for training.
        num_samples : int
            Number of diffusion samples for training.
        Returns
        -------
        model_out : dict[str, dict[str, torch.Tensor]]
            Output dictionary containing sampled structures and intermediate features:
            - trunk: intermediate trunk outputs. (optional)
            - distogram: predicted distogram logits.
            - diffusion: sampled structures from diffusion head.
            - confidence: predicted confidence metrics from confidence head.
        """
        dict_out: dict[str, dict[str, torch.Tensor]] = {}

        assert not f_input.is_batched, "Input must be unbatched for validation sampling."
        f_input = f_input.add_batch_dim()

        if f_input.batch_size != 1:
            # TODO: Support batched inference.
            raise NotImplementedError(
                "Batched input with batch_size > 1 is not supported for inference yet."
            )

        # Trunk with recycling
        s_inputs, s_lm, z = self.run_trunk(f_input, num_recycles)
        z = z.float()

        # Distogram head
        dict_out["distogram"] = self.distogram_head.forward_inference(f_input, z)

        # Diffusion head
        # pred_atom_coords: [B, Nsample, La, 3]
        with torch.autocast(f_input.device.type, enabled=False):
            dict_out["diffusion"] = self.diffusion_head.sample_structure(
                f_input,
                s_inputs,
                z,
                num_steps,
                num_samples,
                chunk_size=None,
            )

        coords = dict_out["diffusion"]["coordinates"]
        dict_out["confidence"] = self.confidence_head(
            f_input,
            s_inputs,
            s_lm,
            z,
            coords,
        )
        # Remove batch dimension from outputs for validation
        dict_out = {
            k: {kk: vv.squeeze(0) for kk, vv in v.items()} for k, v in dict_out.items()
        }

        return dict_out

    # ============================================================
    # Utility Methods
    # ============================================================
    @classmethod
    def from_checkpoint(
        cls,
        config_path: str | pathlib.Path,
        ckpt_path: str | pathlib.Path,
        override_args: list[str] | None = None,
        use_ema: bool = True,
        strict: bool = True,
    ) -> Self:
        """Load model from checkpoint."""
        from kfold.utils.config import load_config

        # Load model config
        config = load_config(config_path, override_args=override_args)

        # Initialize model
        model = cls(config)

        # Load checkpoint
        ckpt = torch.load(ckpt_path, map_location="cpu")

        if "state_dict" not in ckpt:
            # Assume the checkpoint is a state_dict itself
            state_dict = ckpt
        elif use_ema:
            # Load EMA weights
            if "ema" not in ckpt:
                raise KeyError(
                    "EMA weights not found in checkpoint. "
                    "Please set use_ema=False to load regular weights."
                )
            else:
                state_dict = ckpt["ema"]["shadow_params"]
        else:
            # Load regular weights
            state_dict = ckpt["state_dict"]

        state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict, strict=strict)
        del ckpt, state_dict

        return model

    def load_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        strict: bool = True,
        assign: bool = False,
    ):
        """Load state dict without pretrained sequence encoder"""
        # Add '._orig_mod.' to state dict keys if required for compiled models
        state_dict = self._add_orig_mod_to_state_dict(state_dict)

        # If strict is False, it is fine to have missing keys (e.g., pretrained model)
        incompatible_keys = super().load_state_dict(state_dict, strict=False)
        if strict:
            missing_keys = incompatible_keys.missing_keys
            unexpected_keys = incompatible_keys.unexpected_keys
            # If the sequence encoder is pretrained and not included in the state dict,
            # missing keys starting with "sequence_encoder." or "structure_encoder." are
            # allowed.
            missing_keys = {
                k
                for k in missing_keys
                if not k.startswith(
                    (
                        "prot_seq_encoder.",
                        "rna_seq_encoder.",
                        "prot_struct_encoder.",
                    )
                )
            }
            if missing_keys:
                raise KeyError(f"Missing keys in state_dict: {missing_keys}")
            if unexpected_keys:
                raise KeyError(f"Unexpected keys in state_dict: {unexpected_keys}")
        return incompatible_keys

    def _add_orig_mod_to_state_dict(
        self, state_dict: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Add '._orig_mod.' to state dict keys if required"""
        model_keys = set(self.state_dict().keys())
        state_keys = set(state_dict.keys())

        # Keys expected by the compiled model but missing in the checkpoint
        remaining_keys = model_keys - state_keys
        if len(remaining_keys) == 0:
            return dict(state_dict)  # No modification needed

        new_state_dict = dict(state_dict)
        for rk in remaining_keys:
            k = rk.replace("._orig_mod.", ".")
            if k in state_dict:
                new_state_dict[rk] = new_state_dict.pop(k)
        return new_state_dict
