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

"""Implementation of Diffusion-bridge sampler for biomolecular structure prediction

Models the transition from source (apo) to target (holo) conformations.
- t=1 (Prior): Apo chain structures with random translation and rotation.
- t=0 (Data): Ground-truth assembled holo complexes.
"""

import dataclasses
import math
from typing import Literal, TypeVar

import numpy as np
import torch

from kfold.data.types.model_input import FoldingInput
from kfold.model.primitives.utils import expand_dim
from kfold.utils.config import configurable
from kfold.utils.geometry.random_augment import (
    CenterRandomAugmentation,
    do_centering,
    random_rotations_torch,
)
from kfold.utils.geometry.rigid_align import get_rigid_transform_torch

from .score_model import DiffusionModule

_T = TypeVar("_T", float, torch.Tensor)


# === Utility functions with type flexibility and numerical stability handling === #
def _clip(t: _T, eps: float = 1e-10) -> _T:
    return t.clip(min=eps) if isinstance(t, torch.Tensor) else max(t, eps)  # type: ignore


def _sqrt(t: _T) -> _T:
    return _clip(t, eps=0) ** 0.5  # type: ignore


def _log(t: _T) -> _T:
    log = torch.log if isinstance(t, torch.Tensor) else math.log
    return log(_clip(t, eps=1e-10))  # type: ignore


# === Custom rigid align function === #
def custom_rigid_align(
    coords: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
    rotation_only: bool = False,
    output_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Torch implementation of weighted rigid alignment.

    `mask` selects the atoms that drive the fit; `output_mask` selects the atoms
    that survive in the result, and defaults to `mask`.
    """
    output_mask_was_none = output_mask is None
    if mask is None:
        mask = torch.ones(coords.shape[:-1], dtype=torch.bool, device=coords.device)
    if not mask.any():
        if output_mask_was_none:
            return coords
        keep_mask = output_mask.bool().unsqueeze(-1)
        return coords.masked_fill(~keep_mask, 0.0)
    if output_mask is None:
        output_mask = mask

    original_dtype = coords.dtype
    fit_mask = mask.bool().unsqueeze(-1)
    keep_mask = output_mask.bool().unsqueeze(-1)

    with torch.autocast(device_type=coords.device.type, enabled=False):
        coords, target = coords.float(), target.float()

        # Fit on the alignment overlap only.
        fit_coords = coords.masked_fill(~fit_mask, 0.0)
        fit_target = target.masked_fill(~fit_mask, 0.0)
        weights = mask.to(dtype=fit_coords.dtype)
        RT, T = get_rigid_transform_torch(fit_coords, fit_target, weights)

        # Apply it to every atom the caller considers valid.
        aligned_coords = coords.masked_fill(~keep_mask, 0.0) @ RT
        if not rotation_only:
            aligned_coords = (aligned_coords + T.unsqueeze(-2)).masked_fill(
                ~keep_mask, 0.0
            )

    return aligned_coords.to(original_dtype)


# === Main ECSI module implementation === #
@dataclasses.dataclass
class SICoeffs:
    """Stochastic interpolant coefficient helper for ECSI."""

    gamma_max: float

    def alpha(self, t: _T) -> _T:
        return 1.0 - t

    def beta(self, t: _T) -> _T:
        return t

    def gamma(self, t: _T) -> _T:
        return 2 * self.gamma_max * _sqrt(t * (1 - t))

    def gamma_deriv(self, t: _T) -> _T:
        denom = _sqrt(t * (1 - t))
        return self.gamma_max * (1 - 2 * t) / _clip(denom)

    def sigma_eff(self, t: _T) -> _T:
        r"""Return the analytical effective noise scale ``gamma(t) / alpha(t)``.

        This value selects a sigma-matched churn time only. It does not transform
        the coordinates supplied to the score model.
        """
        return self.gamma(t) / _clip(self.alpha(t))


@dataclasses.dataclass(frozen=True, kw_only=True)
class ECSISOARConfig:
    """Configuration for ECSI SDE rollouts and exact forward SOAR transitions.

    Parameters
    ----------
    enabled : bool
        Whether to add SOAR auxiliary samples to base ECSI training.
    root_time_min : float
        Lower bound of the rollout root-time band. Roots are sampled uniformly
        within uniformly selected sampling-schedule cells intersecting the band.
    root_time_max : float
        Upper bound of the rollout root-time band.
    rollout_schedule_num_steps : int
        Number of steps in the sampling schedule used to select rollout times.
    rollout_step_size : float
        Distance from the root in schedule-index units, allowing fractional steps.
        One ECSI SDE update spans this distance; this is not an update count.
    auxiliary_samples_per_root : int
        Number of exact forward-transition samples drawn from each rollout state.
    forward_retention_min : float
        Lower bound of alpha(t2) / alpha(t1) for forward transitions from t1 to t2.
        The bound is raised when necessary to keep t2 at or below time_max.
    aux_loss_weight : float
        Relative supervision weight of each auxiliary sample; base samples have
        weight one. The combined loss is normalized by the total weight.

    Notes
    -----
    Rollouts always use ECSI SDE without churn, including below sde_end_time.
    They share the inference time schedule but not its SI ODE switching policy.
    """

    enabled: bool = False
    root_time_min: float = 0.2
    root_time_max: float = 0.8
    rollout_schedule_num_steps: int = 100
    rollout_step_size: float = 1.0
    auxiliary_samples_per_root: int = 4
    forward_retention_min: float = 0.5
    aux_loss_weight: float = 1.0

    def num_auxiliary_samples(self, num_roots: int) -> int:
        return num_roots * self.auxiliary_samples_per_root


@configurable
class KFoldECSI:
    r"""Endpoint-Conditioned Stochastic Interpolant module for structure prediction.

    Implements the ECSI framework from "Exploring the Design Space of Diffusion Bridge
    Models" for biomolecular structure prediction (apo -> holo translation).

    Key features:
    - Linear route:
      \alpha_t=1-t and \beta_t=t
    - Tunable base gamma:
      \gamma_t^2=4\gamma_{\max}^2 \cdot t(1-t)
    - Stochasticity control via \eta during sampling
    - ECSI Preconditioning

    Reference:
    - ECSI: Zhang et al., "Exploring the Design Space of Diffusion Bridge Models"

    NOTE: K-Fold uses forward-pinned churn with ECSI SDE updates and a low-time
    SI ODE phase for all atoms.
    """

    @dataclasses.dataclass(kw_only=True)
    class Config:
        """Configuration for the ECSI structure module.

        Parameters
        ----------
        # ECSI preconditioning & coefficients
        sigma_0 : float
            Effective scale of target/holo coordinates x_0 for preconditioning.
        sigma_T : float
            Effective scale of prior/apo coordinates x_T for preconditioning.
        cov_0T : float
            Cross-covariance between target x_0 and prior x_T coordinates used by
            the bridge preconditioning formulas.
        time_min : float
            Lower bound for training times and the positive sampling schedule.
            Inference appends a final time of zero.
        time_max : float
            Maximum time value.
        gamma_max : float
            Peak of the base bridge noise schedule gamma(t), reached at t = 0.5.
        # Inference time schedule
        time_schedule_boundary : float
            Absolute time separating the high- and low-time schedule intervals.
            Must lie strictly between time_min and time_max.
        sample_step_fraction : tuple[float, float]
            Fractions of schedule progress allocated to the same intervals.
            Entries must be positive and sum to one. Discrete counts depend on
            num_steps; the final update to zero is appended separately.
        sample_time_power : tuple[float, float]
            Positive exponent for each interval's remaining progress. Values
            below one concentrate points near its start; values above one near
            its end. Each tuple contains exactly two values: high time, then low time.

        # Inference updates
        sde_end_time : float
            Time at or below which inference switches from ECSI SDE to SI ODE.
        eta : float
            Stochasticity control for SDE updates and SOAR forward transitions.
        step_scale : float
            Multiplier for ODE displacement and SDE drift displacement.

        # Churn
        churn_end_time : float
            Time at or below which churn is disabled.
        churn_factor_range : tuple[float, float]
            Fractional effective-noise increases near churn_end_time and time_max,
            in that order. Quadratically interpolated in time, not randomly sampled.
            For example, (0.1, 0.4) targets 1.1 to 1.4 times the current scale,
            subject to the time_max cap. Both endpoints must be non-negative.

        # Training time scheduling
        train_time_distribution : str
            Training-time sampling distribution. Supported values are "logistic" and
            "uniform".
        train_time_distribution_params : tuple[float, float]
            A tuple of (mu, std) for the logistic time sampling distribution.
            Time values are sampled from sigmoid(N(mu, std)), then scaled to
            [time_min, time_max].
        train_x_0_perturb_time_min : float
            Apply chain-rigid x0 perturbation only strictly above this time.
        train_x_0_perturb_prob : float
            Probability of perturbing an eligible high-time training sample.
        train_x_0_perturb_translation_std : float
            Per-axis standard deviation of each chain's Gaussian translation,
            in Angstrom. Applied chains also receive a random SO(3) rotation.
        """

        # ECSI preconditioning & coefficients
        sigma_0: float = 16.0
        sigma_T: float = 27.0
        cov_0T: float = 175.0
        time_min: float = 1e-8
        time_max: float = 0.9999
        gamma_max: float = 6.0

        # Inference time schedule
        time_schedule_boundary: float = 0.5
        sample_step_fraction: tuple[float, float] = (0.4, 0.6)
        sample_time_power: tuple[float, float] = (0.5, 2.0)

        # Inference updates
        sde_end_time: float = 0.1
        eta: float = 1.0
        step_scale: float = 1.0

        # Churn
        churn_end_time: float = 0.5
        churn_factor_range: tuple[float, float] = (0.1, 0.4)

        # Train time scheduling
        train_time_distribution: str = "logistic"
        train_time_distribution_params: tuple[float, float] = (-2.15, 2.25)

        # Optional high-time chain-wise x0 perturbation for base bridge samples.
        train_x_0_perturb_prob: float = 0.0
        train_x_0_perturb_time_min: float = 0.7
        train_x_0_perturb_translation_std: float = 4.0

        def __post_init__(self) -> None:
            schedules = (
                self.sample_step_fraction,
                self.sample_time_power,
            )
            if any(len(values) != 2 for values in schedules):
                raise ValueError(
                    "Sampling schedule tuples must each contain exactly two values."
                )
            if any(
                not math.isfinite(value) or value <= 0
                for values in schedules
                for value in values
            ):
                raise ValueError("Sampling schedule values must be finite and positive.")
            if not math.isclose(
                sum(self.sample_step_fraction), 1.0, rel_tol=0.0, abs_tol=1e-8
            ):
                raise ValueError("Sampling step fractions must sum to one.")
            if not self.time_min < self.time_schedule_boundary < self.time_max:
                raise ValueError(
                    "time_schedule_boundary must lie between time_min and time_max."
                )

            if len(self.churn_factor_range) != 2 or any(
                not math.isfinite(value) or value < 0 for value in self.churn_factor_range
            ):
                raise ValueError(
                    "churn_factor_range must contain two finite non-negative values."
                )

    def __init__(self, cfg: Config, score_model: DiffusionModule):
        """Initialize bridge coefficients, runtime settings, and augmentation."""
        self.cfg = cfg
        self.score_model: DiffusionModule = score_model
        self.random_augmentation = CenterRandomAugmentation()

        # ECSI preconditioning & coefficients
        self.sigma_0: float = cfg.sigma_0
        self.sigma_T: float = cfg.sigma_T
        self.cov_0T: float = cfg.cov_0T
        self.coeff = SICoeffs(cfg.gamma_max)
        self.time_min: float = cfg.time_min
        self.time_max: float = cfg.time_max

        # Inference time sampling
        self.time_schedule_boundary = cfg.time_schedule_boundary
        self.sample_step_fraction = cfg.sample_step_fraction
        self.sample_time_power = cfg.sample_time_power

        # Inference updates
        self.sde_end_time: float = cfg.sde_end_time
        self.eta: float = cfg.eta
        self.step_scale: float = cfg.step_scale

        # Churn
        self.churn_end_time: float = cfg.churn_end_time
        self.churn_factor_range = cfg.churn_factor_range

        # Train time scheduling
        self.train_time_distribution: str = cfg.train_time_distribution
        self.train_time_distribution_params: tuple[float, float] = (
            cfg.train_time_distribution_params
        )
        self.train_x_0_perturb_time_min = cfg.train_x_0_perturb_time_min
        self.train_x_0_perturb_prob = cfg.train_x_0_perturb_prob
        self.train_x_0_perturb_translation_std = cfg.train_x_0_perturb_translation_std

    # === Bridge Preconditioning Coefficients === #
    def _get_bridge_scalings(self, t: _T) -> tuple[_T, _T, _T]:
        """Compute bridge diffusion scalings for ECSI.

        Parameters
        ----------
        t : float | torch.Tensor
            Time values, in range [0, 1].

        Returns
        -------
        c_in : float | torch.Tensor
            Input scaling coefficient.
        c_skip : float | torch.Tensor
            Skip connection coefficient.
        c_out : float | torch.Tensor
            Output scaling coefficient.
        """
        C = self.coeff
        alpha_t, beta_t, gamma_t = C.alpha(t), C.beta(t), C.gamma(t)
        sigma_0, sigma_T, cov_0T = self.sigma_0, self.sigma_T, self.cov_0T

        c_in = 1 / _clip(
            _sqrt(
                (alpha_t * sigma_0) ** 2
                + (beta_t * sigma_T) ** 2
                + 2 * alpha_t * beta_t * cov_0T
                + gamma_t**2
            )
        )
        c_skip = (alpha_t * sigma_0**2 + beta_t * cov_0T) * (c_in**2)
        c_out = (
            _sqrt(
                (beta_t * sigma_0 * sigma_T) ** 2
                - (beta_t * cov_0T) ** 2
                + gamma_t**2 * sigma_0**2
            )
            * c_in
        )
        if isinstance(c_out, torch.Tensor):
            c_in, c_skip, c_out = c_in.float(), c_skip.float(), c_out.float()
        return c_in, c_skip, c_out

    def c_in(self, t: _T) -> _T:
        """Input scaling coefficient for ECSI preconditioning."""
        return self._get_bridge_scalings(t)[0]

    def c_skip(self, t: _T) -> _T:
        """Skip connection coefficient for ECSI preconditioning."""
        return self._get_bridge_scalings(t)[1]

    def c_out(self, t: _T) -> _T:
        """Output scaling coefficient for ECSI preconditioning."""
        return self._get_bridge_scalings(t)[2]

    def c_noise(self, t: _T) -> _T:
        """Noise level conditioning coefficient."""
        return 0.25 * _log(t)

    # ============================================================
    # For inference
    # ============================================================
    def get_sampling_schedule(self, num_steps: int) -> list[float]:
        """Build a two-interval power schedule joined at time_schedule_boundary.

        Parameters
        ----------
        num_steps : int
            Number of sampling updates.

        Returns
        -------
        times : list[float]
            Strictly decreasing time points, ending at zero. Length num_steps + 1.
        """
        if num_steps < 1:
            raise ValueError("num_steps must be positive.")
        progress = np.linspace(0.0, 1.0, num_steps)
        high_steps, low_steps = self.sample_step_fraction
        high_power, low_power = self.sample_time_power
        high_mask = progress <= high_steps

        times = np.empty_like(progress)
        high_remaining = (1.0 - progress[high_mask] / high_steps).clip(0.0, 1.0)
        low_remaining = ((1.0 - progress[~high_mask]) / low_steps).clip(0.0, 1.0)
        boundary = self.time_schedule_boundary
        times[high_mask] = (
            boundary + (self.time_max - boundary) * high_remaining**high_power
        )
        times[~high_mask] = (
            self.time_min + (boundary - self.time_min) * low_remaining**low_power
        )

        times = np.append(times, 0.0)
        if np.any(times[:-1] <= times[1:]):
            raise RuntimeError("ECSI sampling schedule must decrease.")
        return times.tolist()

    def sample_structure(
        self,
        f_input: FoldingInput,
        s_inputs: torch.Tensor,
        z: torch.Tensor,
        num_steps: int = 100,
        num_samples: int = 1,
        chunk_size: int | None = None,
        return_traj: bool = False,
    ) -> dict[str, torch.Tensor]:
        r"""Sample structures via ECSI sampling.

        Applies forward-pinned churn and Euler-Maruyama ECSI SDE updates, then
        switches to SI ODE at sde_end_time. The eta parameter controls
        SDE stochasticity.

        The high-time sampling SDE is:
        dX_t = b(t, X_t, x_T) dt + \sqrt{2\epsilon_t} dW_t

        where:
        b(t, x_t, x_T) = -\hat{x}_0 + x_T
                       + (\dot{\gamma}_t + \epsilon_t/\gamma_t) \hat{z}_t
        \hat{z}_t = (x_t - \alpha_t \hat{x}_0 - \beta_t x_T) / \gamma_t
        \epsilon_t = \eta (\gamma_t \dot{\gamma}_t + 1/\alpha_t \gamma_t^2)
        """
        model = self.score_model

        # Get the exact production schedule (from t_max toward t_min).
        times = self.get_sampling_schedule(num_steps)

        # Sample x_T from prior (apo structures)
        x_T = self.sample_prior(f_input, num_samples)  # (B, N, Natom, 3)
        x_t = x_T.clone()
        mask = f_input.atom.pad_mask[..., None, :]  # (B, 1, Natom)

        # Compute time-independent variables
        z = model.get_pair_conditioning(f_input, z)
        q, c, p = model.get_atom_embeddings(f_input, z)
        pair_bias = model.get_pair_bias(z)

        def run_step(x_t: torch.Tensor, t: float) -> torch.Tensor:
            c_noise = torch.tensor(self.c_noise(t), device=s_inputs.device)
            s = model.get_single_conditioning(s_inputs, c_noise.view(1, 1))
            return self.inference_step(
                f_input, x_t, x_T, t, q, c, p, s, pair_bias, chunk_size
            )

        traj: list[torch.Tensor] = []

        def append_traj(x_t: torch.Tensor):
            if return_traj:
                traj.append(x_t.cpu())

        # Sampling loop
        append_traj(x_t)
        for step_idx in range(num_steps):
            # Apply random augmentation without centering to preserve x_T/x_t translation.
            x_t, x_T = self.random_augmentation(x_t, x_T, mask=mask, centering=False)

            t = times[step_idx]
            t_next = times[step_idx + 1]

            x_noisy, t = self._apply_forward_pinned_churn(x_t, x_T, mask, t)

            # Get denoised prediction \hat{x}_0
            x_0_hat = run_step(x_noisy, t)

            # Rotate x_0_hat toward x_t before centering.
            x_0_hat = custom_rigid_align(x_0_hat, x_noisy, mask, rotation_only=True)

            # Centering the predicted x_0_hat
            x_0_hat = do_centering(x_0_hat, mask=mask)

            use_ode = t <= self.sde_end_time
            x_t = self._update_step(
                x_noisy,
                x_0_hat,
                x_T,
                mask,
                t,
                t_next,
                mode="ode" if use_ode else "sde",
                equation="si" if use_ode else "ecsi",
            )
            append_traj(x_t)

        sample_out: dict[str, torch.Tensor] = {}
        sample_out["init_coordinates"] = x_T
        sample_out["coordinates"] = x_t
        if return_traj:
            sample_out["traj"] = torch.stack(traj, dim=-3)  # (B, N, num_steps, Natom, 3)

        return sample_out

    def sample_prior(self, f_input: FoldingInput, num_samples: int) -> torch.Tensor:
        """Sample xT (prior) coordinates for ECSI sampling.

        Parameters
        ----------
        f_input : FoldingInput
            FoldingInput object containing model inputs.
        num_samples : int
            Number of diffusion samples

        Returns
        -------
        x_T : torch.Tensor
            prior coordinates. Shape (B, N, Natom, 3).
        """
        x_apo = f_input.atom.prior_coords.permute(0, 2, 1, 3)  # [B, Nprior, L, 3]
        apo_mask = f_input.atom.pad_mask  # [B, L]

        # If num_diffusion_samples > num_prior, cycle through prior coords
        num_prior = x_apo.shape[-3]
        idx = [i % num_prior for i in range(num_samples)]
        x_T = x_apo[:, idx, :, :]  # [B, N, L, 3]
        x_T_mask = apo_mask.unsqueeze(-2)  # [B, 1, L]

        # Apply random augmentation without centering to preserve the prior distribution.
        x_T = self.random_augmentation(x_T, mask=x_T_mask, centering=False)
        return x_T

    def inference_step(
        self,
        f_input: FoldingInput,
        x_t: torch.Tensor,
        x_T: torch.Tensor,
        t: float,
        q: torch.Tensor,
        c: torch.Tensor,
        p: torch.Tensor,
        s: torch.Tensor,
        pair_bias: torch.Tensor,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        """Forward pass through the score model.
        See Section 3.7: Diffusion Module, Algorithm 20 of AlphaFold3 paper.

        Parameters
        ----------
        f_input : FoldingInput
            FoldingInput object containing model inputs.
        x_t : torch.Tensor
            Noisy atom coordinates. Shape (B, N, L, 3).
        x_T : torch.Tensor
            Prior (apo) coordinates. Shape (B, N, L, 3).
        t : float
            Diffusion time value for the current step, in range [0, 1].
        q : torch.Tensor
            The atom single representation, shape [B, Natom, c_atom].
        c : torch.Tensor
            The atom single conditioning, shape [B, Natom, c_atom].
        p : torch.Tensor
            The atom pair representation, shape [B, Natom, Natom, c_atompair].
        s : torch.Tensor
            Single conditioning. Shape (B, 1, L, c_s), broadcast to (B, N, L, c_s).
        pair_bias : torch.Tensor
            The pair bias for the token transformer, shape [B, Nblock, H, Lt, Lt].

        Returns
        -------
        x_out : torch.Tensor
            Denoised atom coordinates. Shape (B, N, L, 3).
        """
        token_index = f_input.atom.token_index  # [B, Natom]
        atom_mask = f_input.atom.pad_mask  # [B, Natom]
        token_mask = f_input.token.pad_mask  # [B, L]

        c_in, c_skip, c_out = self._get_bridge_scalings(t)  # [B, N]

        # Input preconditioning: r_noisy = c_in * x_t
        r_noisy = c_in * x_t

        # End-point conditioning: r_T = x_T / sigma_T
        r_T = x_T / self.sigma_T  # [B, N, L, 3]
        r_noisy = torch.cat([r_noisy, r_T], dim=-1)

        def _step(r: torch.Tensor) -> torch.Tensor:
            return self.score_model.step(
                r,  # [B, N, L, 6]
                q,  # [B, Natom, c_atom]
                c,  # [B, Natom, c_atom]
                p,  # [B, Natom, Natom, c_atompair]
                token_index,  # [B, Natom]
                atom_mask,  # [B, Natom]
                s,  # [B, 1, Ntoken, c_s]
                pair_bias,  # [B, Nblock, H, Lt, Lt]
                token_mask,  # [B, L]
            )

        if chunk_size is None:
            r_update = _step(r_noisy)
        else:
            r_update = torch.zeros_like(x_t)
            for st in range(0, x_t.shape[1], chunk_size):
                end = st + chunk_size
                r_update[:, st:end] = _step(r_noisy[:, st:end])

        # Output preconditioning: \hat{x}_0 = c_{skip} * x_t + c_{out} * F_\theta
        x_out = c_skip * x_t + c_out * r_update
        return x_out

    def _apply_forward_pinned_churn(
        self,
        x_t: torch.Tensor,
        x_T: torch.Tensor,
        mask: torch.Tensor,
        t: float,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, float]:
        """Apply forward-pinned churn with a time-dependent noise inflation.

        The effective noise scale selects t_hat within the configured churn band;
        the model still receives x-space coordinates. Targets beyond the band
        are capped at time_max.
        """
        # Keep both bounds open so the initial state consumes no churn RNG.
        if not self.churn_end_time < t < self.time_max:
            return x_t, t

        span = self.time_max - self.churn_end_time
        weight = ((t - self.churn_end_time) / span) ** 2
        low_factor, high_factor = self.churn_factor_range
        chi = low_factor + (high_factor - low_factor) * weight
        if chi <= 0.0:
            return x_t, t

        coeff = self.coeff
        sigma = float(coeff.sigma_eff(t))
        target = (1.0 + chi) * sigma
        if target <= sigma:
            return x_t, t
        if target >= float(coeff.sigma_eff(self.time_max)):
            t_hat = self.time_max
        else:
            # Invert the monotonic effective-noise schedule within the churn band.
            low, high = t, self.time_max
            for _ in range(60):
                mid = 0.5 * (low + high)
                if float(coeff.sigma_eff(mid)) < target:
                    low = mid
                else:
                    high = mid
            t_hat = high
        if t_hat <= t:
            return x_t, t

        alpha_ratio = float(coeff.alpha(t_hat)) / _clip(float(coeff.alpha(t)))
        variance = (
            float(coeff.gamma(t_hat)) ** 2 - (alpha_ratio**2) * float(coeff.gamma(t)) ** 2
        )
        if variance <= 0.0:
            return x_t, t

        if noise is None:
            noise = torch.randn_like(x_t)
        elif noise.shape != x_t.shape:
            raise ValueError("Sigma-matched churn noise must match x_t shape.")
        noise = noise.masked_fill(~mask[..., None], 0.0)
        x_hat = (
            alpha_ratio * x_t
            + (float(coeff.beta(t_hat)) - alpha_ratio * float(coeff.beta(t))) * x_T
            + math.sqrt(variance) * noise
        )
        return x_hat, t_hat

    def _update_step(
        self,
        x_t: torch.Tensor,
        x_0_hat: torch.Tensor,
        x_T: torch.Tensor,
        mask: torch.Tensor,
        t: float,
        t_next: float,
        mode: Literal["ode", "sde"] = "ode",
        equation: Literal["si", "ecsi"] = "si",
    ) -> torch.Tensor:
        """Apply one SI or ECSI update using the specified ODE or SDE mode.

        Parameters
        ----------
        x_t : torch.Tensor
            Current coordinates at time t. Shape (*, Natom, 3).
        x_0_hat : torch.Tensor
            Denoised coordinates predicted by the score model. Shape (*, Natom, 3).
        x_T : torch.Tensor
            Prior (apo) coordinates. Shape (*, Natom, 3).
        mask : torch.Tensor
            Atom mask broadcastable to the coordinate batch dimensions.
        t : float
            Current time value.
        t_next : float
            Next time value after the update step.
        mode : {"ode", "sde"}, optional
            Update mode. Defaults to ODE.
        equation : {"si", "ecsi"}, optional
            Equation for the update. Defaults to SI.
        """
        if mode not in {"ode", "sde"}:
            raise ValueError(f"Unknown sampler mode: {mode!r}.")
        if equation not in {"si", "ecsi"}:
            raise ValueError(f"Unknown sampler equation: {equation!r}.")
        C = self.coeff
        alpha_t = C.alpha(t)
        beta_t = C.beta(t)

        if mode == "ode":
            # Deterministic ODE update.
            alpha_tm, beta_tm = C.alpha(t_next), C.beta(t_next)
            if equation == "si":
                c_skip = beta_tm / beta_t
                c_update = alpha_tm - alpha_t * c_skip
                x_target = c_skip * x_t + c_update * x_0_hat
            else:
                gamma_t, gamma_tm = C.gamma(t), C.gamma(t_next)
                z_hat = (x_t - alpha_t * x_0_hat - beta_t * x_T) / _clip(gamma_t)
                x_target = alpha_tm * x_0_hat + beta_tm * x_T + gamma_tm * z_hat
            drift = x_target - x_t
            return x_t + self.step_scale * drift
        else:
            # SDE update for all valid atoms.
            gamma_t = C.gamma(t)
            gamma_dot = C.gamma_deriv(t)
            eps = self.eta * (gamma_t * gamma_dot + gamma_t**2 / _clip(alpha_t))
            noise = torch.randn_like(x_t)
            noise = noise.masked_fill(~mask[..., None], 0.0)

            if equation == "si":
                # SI drift pinned to the denoised endpoint.
                f_t = 1 / _clip(beta_t)
                s_t = -1 - alpha_t / _clip(beta_t)
                drift = f_t * x_t + s_t * x_0_hat
            else:
                # Reconstruct ECSI bridge noise and compute the reverse-time drift.
                z_hat = (x_t - alpha_t * x_0_hat - beta_t * x_T) / _clip(gamma_t)
                drift = -x_0_hat + x_T + (gamma_dot + eps / _clip(gamma_t)) * z_hat

            # Euler-Maruyama update with decreasing time.
            dt = t - t_next
            return x_t - self.step_scale * drift * dt + _sqrt(2 * eps * dt) * noise

    # ============================================================
    # For training
    # ============================================================
    def loss_weights(self, t: torch.Tensor) -> torch.Tensor:
        """ECSI training loss weights"""
        return 1 / self.c_out(t).pow(2).clamp(min=1e-8)

    def training_step(
        self,
        f_input: FoldingInput,
        s_inputs: torch.Tensor,
        z: torch.Tensor,
        diffusion_batch_size: int,
        soar_config: ECSISOARConfig,
    ) -> dict[str, torch.Tensor]:
        """Perform base ECSI training plus optional Exact-Markov SOAR."""
        with torch.autocast(f_input.device.type, enabled=False):
            train_input = self.sample_train_input(f_input, diffusion_batch_size)

        t = train_input["t"]  # [B, N]
        x_0 = train_input["x_0"]  # [B, N, Natom, 3]
        x_t = train_input["x_t"]  # [B, N, Natom, 3]
        x_T = train_input["x_T"]  # [B, N, Natom, 3]
        atom_mask = train_input["atom_mask"]

        x_0_hat = self._forward_train(
            x_t=x_t,  # [B, N, Natom, 3]
            t=t,  # [B, N]
            f_input=f_input,
            s_inputs=s_inputs,  # [B, Lt, c_s]
            z=z,  # [B, Lt, Lt, c_z]
            x_T=x_T,  # [B, N, Natom, 3]
            atom_mask=atom_mask,  # [B, Natom]
        )  # [B, N, Natom, 3]

        loss_weights = self.loss_weights(t)  # [B, N]

        output = {
            "t": t,
            "x_t": x_t,
            "x_T": x_T,
            "x_0_hat": x_0_hat,
            "x_gt": x_0,
            "loss_weights": loss_weights,
        }
        for name in (
            "time_eligible_mask",
            "eligible_mask",
            "requested_mask",
            "applied_mask",
            "x_0_rmsd",
            "x_t_rmsd",
            "resolved_chain_count",
        ):
            output[f"x_0_perturb_{name}"] = train_input[f"x_0_perturb_{name}"]
        if not soar_config.enabled:
            return output

        auxiliary = self._build_soar_training_batch(
            f_input=f_input,
            s_inputs=s_inputs,
            z=z,
            config=soar_config,
            x_0=x_0,
            x_T=x_T,
            atom_mask=atom_mask,
            bridge_noise=train_input["noise"],
            base_t0=t,
        )
        output["t"] = torch.cat((t, auxiliary["t_aux"]), dim=1)
        output["x_t"] = torch.cat((x_t, auxiliary["x_aux"]), dim=1)
        output["x_T"] = torch.cat((x_T, auxiliary["x_T"]), dim=1)
        output["x_0_hat"] = torch.cat((x_0_hat, auxiliary["x_0_hat_aux"]), dim=1)
        output["x_gt"] = torch.cat((x_0, auxiliary["x_0"]), dim=1)
        output["loss_weights"] = torch.cat(
            (loss_weights, self.loss_weights(auxiliary["t_aux"])), dim=1
        )

        output["soar_supervision_weights"] = torch.cat(
            (torch.ones_like(t), auxiliary["supervision_weights"]), dim=1
        )
        return output

    def _forward_train(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        f_input: FoldingInput,
        s_inputs: torch.Tensor,
        z: torch.Tensor,
        x_T: torch.Tensor | None = None,
        atom_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass through the score model with ECSI preconditioning.

        Parameters
        ----------
        x_t : torch.Tensor
            Noisy atom coordinates. Shape (B, N, L, 3).
        t : torch.Tensor
            Time values. Shape (B, N).
        f_input : FoldingInput
            FoldingInput object containing model inputs.
        s_inputs : torch.Tensor
            Input sequence embeddings. Shape (B, L, c_s).
        z : torch.Tensor
            Trunk pairwise embeddings. Shape (B, L, L, c_z).
        x_T : torch.Tensor | None
            Source (apo) coordinates x_T. Shape (B, N, L, 3).
        atom_mask : torch.Tensor | None
            Atoms the coordinate stack may attend to. Shape (B, L).

        Returns
        -------
        x_0_hat : torch.Tensor
            Denoised atom coordinates. Shape (B, N, Natom, 3).
        """
        assert x_T is not None, "x_T must be provided for ECSI"
        c_in, c_skip, c_out = self._get_bridge_scalings(t)  # [B, N]
        c_noise = self.c_noise(t)  # [B, N]

        # Input preconditioning: r_noisy = c_in * x_t, r_T = x_T / sigma_T
        r_noisy = c_in[:, :, None, None] * x_t  # [B, N, Natom, 3]

        # End-point conditioning: r_T = x_T / sigma_T
        r_T = x_T / self.sigma_T  # [B, N, Natom, 3]
        r_noisy = torch.cat([r_noisy, r_T], dim=-1)

        # Call score model
        r_update = self.score_model.train_step(
            f_input=f_input,
            r_noisy=r_noisy,  # [B, N, Natom, 6]
            c_noise=c_noise,  # [B, N]
            s_inputs=s_inputs,  # [B, Lt, c_s]
            z=z,  # [B, Lt, Lt, c_z]
            atom_mask=atom_mask,  # [B, Natom]
        )

        # Output preconditioning: \hat{x}_0 = c_{skip} * x_t + c_{out} * F_\theta
        x_0_hat = c_skip[..., None, None] * x_t + c_out[..., None, None] * r_update
        return x_0_hat

    def sample_noise_level(
        self, shape: tuple[int, ...], device: torch.device
    ) -> torch.Tensor:
        r"""Sample time values for training.

        Returns samples in [time_min, time_max] which represents
        the time interval [t_{min}, t_{max}] \subset [0, 1].

        Returns
        -------
        t : torch.Tensor
            Time values. Shape (B, N).
        """
        if self.train_time_distribution == "uniform":
            t = torch.rand(shape, device=device)
        else:
            mu, std = self.train_time_distribution_params
            z = torch.randn(shape, device=device)
            x = mu + std * z
            t = torch.sigmoid(x)

        # Scale to [time_min, time_max]
        t = self.time_min + (self.time_max - self.time_min) * t
        return t

    def _sample_x_0_perturb_masks(
        self,
        *,
        t: torch.Tensor,
        resolved_chain_count: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Select high-time perturbations without touching disabled RNG."""
        time_eligible = t > self.train_x_0_perturb_time_min
        multichain = resolved_chain_count[:, None] > 1
        eligible = time_eligible & multichain
        if self.train_x_0_perturb_prob == 0.0:
            requested = torch.zeros_like(time_eligible)
        else:
            requested = time_eligible & (torch.rand_like(t) < self.train_x_0_perturb_prob)
        applied = requested & multichain
        return {
            "time_eligible": time_eligible,
            "eligible": eligible,
            "requested": requested,
            "applied": applied,
        }

    def _perturb_x_0(
        self,
        *,
        x_0: torch.Tensor,
        x_T: torch.Tensor,
        t: torch.Tensor,
        f_input: FoldingInput,
        x_0_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Apply the original high-time chain-rigid x0 corruption contract."""
        batch_size, num_samples, num_atoms, _ = x_0.shape
        if self.train_x_0_perturb_prob == 0.0:
            time_eligible = t > self.train_x_0_perturb_time_min
            false_mask = torch.zeros_like(time_eligible)
            zero = torch.zeros_like(t)
            return {
                "x_0_bridge": x_0,
                "time_eligible_mask": time_eligible,
                "eligible_mask": false_mask,
                "requested_mask": false_mask,
                "applied_mask": false_mask,
                "x_0_rmsd": zero,
                "resolved_chain_count": torch.zeros_like(t, dtype=torch.long),
            }
        num_chains = f_input.chain.asym_id.shape[1]
        token_index = f_input.atom.token_index.clamp(
            min=0, max=f_input.token.asym_id.shape[-1] - 1
        )
        atom_asym_id = torch.gather(f_input.token.asym_id, 1, token_index)
        resolved_mask = x_0_mask[:, 0].bool()
        atom_in_chain = atom_asym_id[:, :, None] == f_input.chain.asym_id[:, None, :]
        atom_in_chain &= f_input.chain.pad_mask[:, None, :]
        atom_chain_index = atom_in_chain.to(torch.int64).argmax(dim=-1)
        atom_is_valid = resolved_mask & atom_in_chain.any(dim=-1)
        chain_has_atoms = (atom_in_chain & resolved_mask[:, :, None]).any(dim=1)
        resolved_chain_count = chain_has_atoms.sum(dim=-1)
        selection = self._sample_x_0_perturb_masks(
            t=t, resolved_chain_count=resolved_chain_count
        )
        applied = selection["applied"]
        chain_index_xyz = atom_chain_index[:, None, :, None].expand(
            batch_size, num_samples, num_atoms, 3
        )
        chain_sums = torch.zeros(
            (batch_size, num_samples, num_chains, 3),
            dtype=x_0.dtype,
            device=x_0.device,
        )
        chain_sums.scatter_add_(
            dim=2,
            index=chain_index_xyz,
            src=x_0 * atom_is_valid[:, None, :, None],
        )
        chain_counts = (atom_in_chain & resolved_mask[:, :, None]).sum(dim=1)
        chain_centers = chain_sums / chain_counts[:, None, :, None].clamp_min(1)
        chain_rotations = random_rotations_torch(
            (batch_size, num_samples, num_chains),
            dtype=x_0.dtype,
            device=x_0.device,
        )
        chain_translations = torch.randn_like(chain_centers)
        chain_translations *= self.train_x_0_perturb_translation_std
        atom_centers = torch.gather(chain_centers, dim=2, index=chain_index_xyz)
        atom_translations = torch.gather(chain_translations, dim=2, index=chain_index_xyz)
        rotation_index = atom_chain_index[:, None, :, None, None].expand(
            batch_size, num_samples, num_atoms, 3, 3
        )
        atom_rotations = torch.gather(chain_rotations, dim=2, index=rotation_index)
        transformed = (
            torch.einsum("bnad,bnads->bnas", x_0 - atom_centers, atom_rotations)
            + atom_centers
            + atom_translations
        )
        apply_atom_mask = applied[..., None, None] & atom_is_valid[:, None, :, None]
        x_0_perturbed = torch.where(apply_atom_mask, transformed, x_0)
        expanded_mask = x_0_mask.expand(-1, num_samples, -1)
        aligned = custom_rigid_align(
            x_0_perturbed,
            x_T,
            expanded_mask,
            rotation_only=True,
        )
        aligned = do_centering(aligned, mask=expanded_mask)
        apply_mask = applied[..., None, None] & expanded_mask[..., None]
        x_0_bridge = torch.where(apply_mask, aligned, x_0)
        x_0_rmsd = self._masked_coordinate_rmsd(x_0_bridge - x_0, expanded_mask)
        return {
            "x_0_bridge": x_0_bridge,
            "time_eligible_mask": selection["time_eligible"],
            "eligible_mask": selection["eligible"],
            "requested_mask": selection["requested"],
            "applied_mask": applied,
            "x_0_rmsd": x_0_rmsd,
            "resolved_chain_count": resolved_chain_count[:, None].expand_as(t),
        }

    def sample_train_input(
        self,
        f_input: FoldingInput,
        diffusion_batch_size: int,
    ) -> dict[str, torch.Tensor]:
        """Sample training inputs for the structure module.

        Parameters
        ----------
        f_input : FoldingInput
            FoldingInput object containing model inputs.
        diffusion_batch_size : int
            The number of samples to generate for training.

        Returns
        -------
        dict[str, torch.Tensor]
            A dictionary containing the x_0, x_t, and related representations.
        """
        batch_size = f_input.batch_size
        num_samples = diffusion_batch_size
        device = f_input.device
        t = self.sample_noise_level((batch_size, num_samples), device)

        # === Prepare x_0 and x_T === #
        x_holo = f_input.atom.label_coords  # [B, Natom, 3]
        holo_mask = f_input.atom.resolved_mask  # [B, Natom]
        x_apo = f_input.atom.prior_coords.permute(0, 2, 1, 3)  # [B, Nprior, Natom, 3]
        apo_mask = f_input.atom.pad_mask  # [B, Natom]

        # Repeat holo coords
        x_0 = expand_dim(x_holo, num_samples, dim=-3).clone()  # [B, N, Natom, 3]
        x_0_mask = holo_mask.unsqueeze(-2)  # [B, 1, Natom]

        # Sample from prior coordinates
        # If num_diffusion_samples > num_prior, cycle through prior coords
        num_prior = x_apo.shape[-3]
        idx = [i % num_prior for i in range(num_samples)]
        x_T = x_apo[:, idx, :, :]  # [B, N, Natom, 3]

        # Atoms the coordinate stack may see. Unresolved atoms have no holo target,
        # so x_0 is 0 there and any interpolant through it is meaningless. Hiding
        # them keeps that meaningless coordinate out of every geometric operation
        # below and out of the score model's attention. Inference has no
        # `resolved_mask`, so it keeps using the full `pad_mask`.
        train_mask = apo_mask & holo_mask  # [B, Natom]
        x_T_mask = train_mask.unsqueeze(-2)  # [B, 1, Natom]

        # Apply centering/coordinate augmentation
        x_0 = self.random_augmentation(x_0, mask=x_0_mask)

        # Rotate x_T toward x_0 while preserving the prior translation distribution.
        x_T = custom_rigid_align(
            x_T, x_0, x_0_mask, rotation_only=True, output_mask=x_T_mask
        )

        # Perturb only the endpoint used by the base bridge. SOAR reconstructs
        # its roots from the clean x_0 returned below.
        perturb = self._perturb_x_0(
            x_0=x_0,
            x_T=x_T,
            t=t,
            f_input=f_input,
            x_0_mask=x_0_mask,
        )

        # ECSI interpolation with atom-wise shared bridge noise.
        noise = torch.randn_like(x_0).masked_fill_(~x_T_mask[..., None], 0.0)
        x_t_clean = self._interpolate_bridge(x_0, x_T, noise, t)
        x_t = self._interpolate_bridge(perturb["x_0_bridge"], x_T, noise, t)
        expanded_mask = x_T_mask.expand(-1, num_samples, -1)
        x_t_rmsd = self._masked_coordinate_rmsd(x_t - x_t_clean, expanded_mask)

        # Zero the hidden atoms as well as the padding, so that a code path which
        # forgets `train_mask` sees the origin rather than a prior-scale offset.
        x_0.masked_fill_(~x_0_mask[..., None], 0.0)
        x_T.masked_fill_(~x_T_mask[..., None], 0.0)
        x_t.masked_fill_(~x_T_mask[..., None], 0.0)

        return {
            "t": t,
            "x_0": x_0,
            "x_t": x_t,
            "x_T": x_T,
            "atom_mask": train_mask,
            "noise": noise,
            "x_0_perturb_time_eligible_mask": perturb["time_eligible_mask"],
            "x_0_perturb_eligible_mask": perturb["eligible_mask"],
            "x_0_perturb_requested_mask": perturb["requested_mask"],
            "x_0_perturb_applied_mask": perturb["applied_mask"],
            "x_0_perturb_x_0_rmsd": perturb["x_0_rmsd"],
            "x_0_perturb_x_t_rmsd": x_t_rmsd,
            "x_0_perturb_resolved_chain_count": perturb["resolved_chain_count"],
        }

    def _interpolate_bridge(
        self,
        x_0: torch.Tensor,
        x_T: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the ECSI bridge with caller-supplied shared noise."""
        expanded_t = t[..., None, None]
        return (
            self.coeff.alpha(expanded_t) * x_0
            + self.coeff.beta(expanded_t) * x_T
            + self.coeff.gamma(expanded_t) * noise
        )

    def _build_soar_training_batch(
        self,
        *,
        f_input: FoldingInput,
        s_inputs: torch.Tensor,
        z: torch.Tensor,
        config: ECSISOARConfig,
        x_0: torch.Tensor,
        x_T: torch.Tensor,
        atom_mask: torch.Tensor,
        bridge_noise: torch.Tensor,
        base_t0: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Build detached Exact-Markov auxiliaries from model-generated states."""
        if x_0.shape != x_T.shape or x_0.shape != bridge_noise.shape:
            raise ValueError("SOAR endpoint and bridge-noise shapes must match.")
        if base_t0.shape != x_0.shape[:2]:
            raise ValueError("SOAR base times must match the base sample dimensions.")
        if atom_mask.shape != (x_0.shape[0], x_0.shape[-2]):
            raise ValueError("SOAR atom_mask must have shape [B, Natom].")

        batch_size, num_roots = base_t0.shape
        if num_roots == 0:
            raise ValueError("Active ECSI SOAR requires at least one base root.")
        num_auxiliary = config.num_auxiliary_samples(num_roots)
        root_mask = atom_mask.unsqueeze(1).expand(-1, num_roots, -1)

        with torch.no_grad(), torch.autocast(f_input.device.type, enabled=False):
            t0 = self.construct_soar_root_times(base_t0=base_t0, config=config)
            t1, t2 = self.sample_soar_auxiliary_times(t0=t0, config=config)
            x_t0 = self._interpolate_bridge(x_0, x_T, bridge_noise, t0)
            x_t0 = x_t0.masked_fill(~root_mask[..., None], 0.0)

        with torch.no_grad():
            x_0_hat_t0 = self._forward_train(
                x_t=x_t0,
                t=t0,
                f_input=f_input,
                s_inputs=s_inputs,
                z=z,
                x_T=x_T,
                atom_mask=atom_mask,
            ).detach()

        with torch.no_grad(), torch.autocast(f_input.device.type, enabled=False):
            model_endpoint = self._postprocess_soar_endpoint(
                endpoint=x_0_hat_t0,
                x_t=x_t0,
                mask=root_mask,
            )
            x_t1_model = self._apply_soar_sampler_update(
                x_t=x_t0,
                x_0_hat=model_endpoint,
                x_T=x_T,
                mask=root_mask,
                t=t0,
                t_next=t1,
            )
            x_aux_branched = self._exact_ecsi_forward_transition(
                x_t1=x_t1_model,
                x_T=x_T,
                mask=root_mask,
                t1=t1,
                t2=t2,
            )
            x_aux = x_aux_branched.reshape(
                batch_size, num_auxiliary, *x_0.shape[-2:]
            ).detach()
            x_T_aux = (
                x_T.unsqueeze(2)
                .expand(-1, -1, config.auxiliary_samples_per_root, -1, -1)
                .reshape_as(x_aux)
            )
            x_0_aux = (
                x_0.unsqueeze(2)
                .expand(-1, -1, config.auxiliary_samples_per_root, -1, -1)
                .reshape_as(x_aux)
            )
            t_aux = t2.reshape(batch_size, num_auxiliary)

        x_0_hat_aux = self._forward_train(
            x_t=x_aux,
            t=t_aux,
            f_input=f_input,
            s_inputs=s_inputs,
            z=z,
            x_T=x_T_aux,
            atom_mask=atom_mask,
        )

        supervision_weights = torch.full(
            (batch_size, num_auxiliary),
            config.aux_loss_weight,
            dtype=t_aux.dtype,
            device=t_aux.device,
        )
        return {
            "supervision_weights": supervision_weights,
            "t_aux": t_aux,
            "x_0": x_0_aux,
            "x_T": x_T_aux,
            "x_aux": x_aux,
            "x_0_hat_aux": x_0_hat_aux,
        }

    def construct_soar_root_times(
        self, *, base_t0: torch.Tensor, config: ECSISOARConfig
    ) -> torch.Tensor:
        """Sample SOAR root times from schedule cells in the configured time band."""
        if base_t0.ndim != 2:
            raise ValueError("SOAR base time must have shape [B, Nroot].")
        schedule = torch.tensor(
            self.get_sampling_schedule(config.rollout_schedule_num_steps),
            dtype=base_t0.dtype,
            device=base_t0.device,
        )
        return self._sample_soar_schedule_cell_band(
            shape=base_t0.shape,
            schedule=schedule,
            lower=max(self.time_min, config.root_time_min),
            upper=min(self.time_max, config.root_time_max),
        )

    @staticmethod
    def _sample_soar_schedule_cell_band(
        *,
        shape: torch.Size | tuple[int, ...],
        schedule: torch.Tensor,
        lower: float,
        upper: float,
    ) -> torch.Tensor:
        """Sample uniformly over schedule cells intersecting one time band."""
        cell_high = torch.minimum(schedule[:-1], torch.full_like(schedule[:-1], upper))
        cell_low = torch.maximum(schedule[1:], torch.full_like(schedule[1:], lower))
        valid_indices = torch.nonzero(cell_high > cell_low, as_tuple=False).flatten()
        if valid_indices.numel() == 0:
            raise ValueError(
                f"No production-schedule cell intersects SOAR band [{lower}, {upper}]."
            )
        sampled_offset = torch.randint(
            valid_indices.numel(), shape, device=schedule.device
        )
        sampled_index = valid_indices[sampled_offset]
        sampled_low = cell_low[sampled_index]
        sampled_high = cell_high[sampled_index]
        return sampled_low + (sampled_high - sampled_low) * torch.rand(
            shape, dtype=schedule.dtype, device=schedule.device
        )

    def sample_soar_auxiliary_times(
        self, *, t0: torch.Tensor, config: ECSISOARConfig
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select rollout and forward-transition times from each root.

        Move rollout_step_size schedule-index units toward lower time
        to obtain t1, clamped to time_min. Then sample t2 at or above t1 through
        the configured forward-retention range.
        """
        schedule_values = self.get_sampling_schedule(config.rollout_schedule_num_steps)
        work_dtype = torch.float64 if t0.dtype == torch.float64 else torch.float32
        schedule = torch.tensor(schedule_values, dtype=work_dtype, device=t0.device)
        t0_work = t0.to(dtype=work_dtype)
        cell_index = ((schedule[:-1] >= t0_work.unsqueeze(-1)).sum(dim=-1) - 1).clamp(
            min=0, max=config.rollout_schedule_num_steps - 1
        )
        cell_start = schedule[cell_index]
        cell_end = schedule[cell_index + 1]
        cell_fraction = ((cell_start - t0_work) / (cell_start - cell_end)).clamp(0, 1)
        u0 = cell_index.to(dtype=work_dtype) + cell_fraction
        u1 = (u0 + config.rollout_step_size).clamp_max(config.rollout_schedule_num_steps)
        next_index = (
            torch.floor(u1)
            .to(dtype=torch.long)
            .clamp_max(config.rollout_schedule_num_steps - 1)
        )
        next_fraction = u1 - next_index.to(dtype=work_dtype)
        t1 = torch.lerp(
            schedule[next_index], schedule[next_index + 1], next_fraction
        ).clamp_min(self.time_min)
        t1 = t1.to(dtype=t0.dtype)

        uniform = torch.rand(
            (*t1.shape, config.auxiliary_samples_per_root),
            dtype=t1.dtype,
            device=t1.device,
        )
        alpha_t1 = self.coeff.alpha(t1)
        feasible_min = self.coeff.alpha(torch.full_like(t1, self.time_max)) / (
            alpha_t1.clamp_min(torch.finfo(alpha_t1.dtype).eps)
        )
        retention_min = torch.maximum(
            torch.full_like(t1, config.forward_retention_min), feasible_min
        ).clamp_max(1.0)
        retention = (
            retention_min.unsqueeze(-1) + (1.0 - retention_min.unsqueeze(-1)) * uniform
        )
        t2 = 1.0 - retention * alpha_t1.unsqueeze(-1)
        t2 = torch.maximum(t2, t1.unsqueeze(-1))
        t2 = torch.minimum(t2, torch.full_like(t2, self.time_max))
        return t1, t2

    def _apply_soar_sampler_update(
        self,
        *,
        x_t: torch.Tensor,
        x_0_hat: torch.Tensor,
        x_T: torch.Tensor,
        mask: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
    ) -> torch.Tensor:
        """Apply one ECSI SDE update per root, without the inference ODE switch."""
        output = torch.empty_like(x_t)
        for b_i in range(t.shape[0]):
            for s_i in range(t.shape[1]):
                state = self._update_step(
                    x_t[b_i, s_i][None, None, ...],
                    x_0_hat[b_i, s_i][None, None, ...],
                    x_T[b_i, s_i][None, None, ...],
                    mask[b_i, s_i][None, None, ...],
                    float(t[b_i, s_i]),
                    float(t_next[b_i, s_i]),
                    mode="sde",
                    equation="ecsi",
                )
                output[b_i, s_i] = state[0, 0]
        return output.detach()

    @staticmethod
    def _masked_coordinate_rmsd(
        displacement: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        if displacement.shape[:-1] != mask.shape:
            raise ValueError("Coordinate displacement and atom-mask shapes must match.")
        weights = mask.to(displacement.dtype)
        squared = displacement.square().sum(dim=-1)
        return torch.sqrt(
            (squared * weights).sum(dim=-1) / weights.sum(dim=-1).clamp(min=1.0)
        )

    def _exact_ecsi_forward_transition(
        self,
        *,
        x_t1: torch.Tensor,
        x_T: torch.Tensor,
        mask: torch.Tensor,
        t1: torch.Tensor,
        t2: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample the exact ECSI bridge transition from t1 to later t2."""
        alpha_t1 = self.coeff.alpha(t1).unsqueeze(-1)
        retention = self.coeff.alpha(t2) / alpha_t1.clamp_min(torch.finfo(t1.dtype).eps)
        gamma_t1 = self.coeff.gamma(t1).unsqueeze(-1)
        gamma_t2 = self.coeff.gamma(t2)
        variance = self.eta * (gamma_t2.square() - retention.square() * gamma_t1.square())
        tolerance = (
            32.0
            * torch.finfo(variance.dtype).eps
            * (gamma_t2.square() + retention.square() * gamma_t1.square())
            .abs()
            .clamp_min(1.0)
        )
        if torch.any(variance < -tolerance):
            raise RuntimeError("Exact ECSI forward-transition variance is negative.")
        variance = variance.clamp_min(0.0)
        branch_shape = (*t2.shape, *x_t1.shape[-2:])
        x_t1_branches = x_t1.unsqueeze(-3).expand(branch_shape)
        x_T_branches = x_T.unsqueeze(-3).expand(branch_shape)
        branch_mask = mask.unsqueeze(-2).expand(*t2.shape, mask.shape[-1])
        mean = (
            retention[..., None, None] * x_t1_branches
            + (self.coeff.beta(t2) - retention * self.coeff.beta(t1).unsqueeze(-1))[
                ..., None, None
            ]
            * x_T_branches
        )
        if noise is None:
            noise = torch.randn_like(mean)
        elif noise.shape != mean.shape:
            raise ValueError("ECSI forward-transition noise shape is incompatible.")
        noise = noise.masked_fill(~branch_mask[..., None], 0.0)
        x_t2 = mean + torch.sqrt(variance)[..., None, None] * noise
        x_t2 = x_t2.masked_fill(~branch_mask[..., None], 0.0)
        return x_t2

    def _postprocess_soar_endpoint(
        self,
        *,
        endpoint: torch.Tensor,
        x_t: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the endpoint transform used by production ECSI sampling."""
        endpoint = custom_rigid_align(
            endpoint, x_t, mask, rotation_only=True, output_mask=mask
        )
        return do_centering(endpoint, mask=mask)
