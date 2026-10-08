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

from dataclasses import dataclass

import torch

from kfold.data.types.model_input import FoldingInput
from kfold.model.layers.folding.diffusion import DiffusionStack
from kfold.utils.config import configurable


@configurable
class DiffusionModule(torch.nn.Module):
    """Diffusion module
    Section 3.7 Algorithm 20: Diffusion Module in the AF3 paper.
    """

    @dataclass(kw_only=True)
    class Config:
        """Initialize the diffusion module.

        Parameters
        ----------
        channel_a : int
            The token transformer dimension.
        channel_s : int
            The single representation dimension.
        channel_z : int
            The pair representation dimension.
        channel_atom : int
            The atom single representation dimension.
        channel_atompair : int
            The atom pair representation dimension.
        channel_coords : int
            The coordinate dimension, default to 3 for (x, y, z).
        separate_endpoint_atom_encoder : bool, optional
            Split 6-channel ECSI coordinates into separate current-state and
            endpoint atom encoders before shared token-level attention.
        atom_encoder_blocks : int, optional
            The number of blocks of the atom encoder, by default 3.
        atom_encoder_heads : int, optional
            The number of heads in the atom encoder, by default 4.
        token_transformer_blocks : int, optional
            The number of blocks of the token transformer, by default 24.
        token_transformer_heads : int, optional
            The number of heads in the token transformer, by default 8.
        atom_decoder_blocks : int, optional
            The number of blocks of the atom decoder, by default 3.
        atom_decoder_heads : int, optional
            The number of heads in the atom decoder, by default 4.
        blocks_per_ckpt : int | None, optional
            The number of blocks per checkpoint, by default None.
        ckpt_atom_stack : bool, optional
            Whether to checkpoint each complete atom transformer stack.
        """

        channel_a: int = 768
        channel_s: int = 384
        channel_z: int = 256
        channel_atom: int = 128
        channel_atompair: int = 16
        channel_coords: int = 3
        separate_endpoint_atom_encoder: bool = False
        atom_encoder_blocks: int = 3
        atom_encoder_heads: int = 4
        token_transformer_blocks: int = 12
        token_transformer_heads: int = 16
        atom_decoder_blocks: int = 3
        atom_decoder_heads: int = 4
        blocks_per_ckpt: int | None = None
        ckpt_atom_stack: bool = False

    def __init__(
        self,
        cfg,
        kernel_backend: str = "torch",
    ):
        super().__init__()
        self.kernel_backend = kernel_backend
        self.cfg = cfg
        self.is_compiled: bool = False
        self.diffusion_stack = DiffusionStack(
            channel_a=cfg.channel_a,
            channel_s=cfg.channel_s,
            channel_z=cfg.channel_z,
            channel_atom=cfg.channel_atom,
            channel_atompair=cfg.channel_atompair,
            channel_coords=cfg.channel_coords,
            separate_endpoint_atom_encoder=cfg.separate_endpoint_atom_encoder,
            atom_encoder_blocks=cfg.atom_encoder_blocks,
            atom_encoder_heads=cfg.atom_encoder_heads,
            token_transformer_blocks=cfg.token_transformer_blocks,
            token_transformer_heads=cfg.token_transformer_heads,
            atom_decoder_blocks=cfg.atom_decoder_blocks,
            atom_decoder_heads=cfg.atom_decoder_heads,
            blocks_per_ckpt=cfg.blocks_per_ckpt,
            ckpt_atom_stack=cfg.ckpt_atom_stack,
            kernel_backend=self.kernel_backend,
        )

    def do_compile(self, **kwargs):
        """Compile the score model module."""
        self._compile(**kwargs)
        self.is_compiled = True

    def _compile(self, **kwargs):
        """Compile the diffusion stack."""
        self.diffusion_stack = torch.compile(self.diffusion_stack, **kwargs)

    @property
    def _diffusion_stack(self) -> DiffusionStack:
        """Get the uncompiled diffusion stack."""
        if self.is_compiled and not self.training:
            return self.diffusion_stack._orig_mod
        return self.diffusion_stack

    def train_step(
        self,
        f_input: FoldingInput,
        r_noisy: torch.Tensor,
        c_noise: torch.Tensor,
        s_inputs: torch.Tensor,
        z: torch.Tensor,
        atom_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Training forward pass of the AF3 diffusion module.
        See Section 3.7 Algorithm 20: Diffusion Module in the AF3 paper.

        Parameters
        ----------
        f_input : FoldingInput
            The folding input.
        r_noisy : torch.Tensor
            The noisy atom positions, shape [B, N, La, 3],
            where N is number of diffusion samples and La is number of atoms.
        c_noise : torch.Tensor
            The diffusion noise level (or sigmas), shape [B, N].
            c_noise = 1/4 log(t_hat / sigma_data) (See Algorithm 21.)
            c_noise is computed outside of this class (See StructureModule).
        s_inputs : torch.Tensor
            The input single representation, shape [B, Lt, c_s].
        z : torch.Tensor
            The trunk pair representation, shape [B, Lt, c_z].
        atom_mask : torch.Tensor | None
            Atoms the coordinate stack may attend to, shape [B, La].
            Defaults to `f_input.atom.pad_mask`.

        Returns
        -------
        r_update : torch.Tensor
            The denoised atom positions, shape [B, N, La, 3].
        """
        # NOTE (SeonghwanSeo): cuEquiv uses pytorch fallback for short sequences.
        return self.diffusion_stack(
            f_input,
            r_noisy,
            c_noise,
            s_inputs,
            z,
            atom_mask=atom_mask,
        )

    # === Inference step ===
    def get_pair_conditioning(
        self, f_input: FoldingInput, z: torch.Tensor
    ) -> torch.Tensor:
        """Get the pair conditioning for the diffusion module.
        This is time-independent and can be pre-computed before the diffusion steps.

        Parameters
        ----------
        f_input : FoldingInput
            The folding input.
        z : torch.Tensor
            The trunk pair representation, shape [B, Lt, Lt, c_z].

        Returns
        -------
        z : torch.Tensor
            The conditioned pair representation, shape [B, Lt, Lt, c_z].
        """
        return self._diffusion_stack.get_pair_conditioning(f_input, z)

    def get_single_conditioning(
        self, s_inputs: torch.Tensor, c_noise: torch.Tensor
    ) -> torch.Tensor:
        """Get the single conditioning for the diffusion module.
        This is time-dependent and needs to be computed at each diffusion step.

        Parameters
        ----------
        s_inputs : torch.Tensor
            The input single representation, shape [B, Lt, c_s].
        c_noise : torch.Tensor
            Tensor of shape (B, N) containing diffusion noise level (or sigma).
            c_noise = 1/4 log(t_hat / sigma_data) (See Algorithm.)

        Returns
        -------
        s : torch.Tensor
            The single conditioning, shape [B, N, Lt, c_s].
        """
        return self._diffusion_stack.get_single_conditioning(s_inputs, c_noise)

    def get_atom_embeddings(
        self,
        f_input: FoldingInput,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare the inputs which are static across diffusion steps.
        # Algorithm 5 Line 1-10, 13-14.

        Parameters
        ----------
        f_input : FoldingInput
            The folding input.
        z : torch.Tensor
            The trunk pair conditioning, shape [B, Lt, Lt, c_z].

        Returns
        -------
        q : torch.Tensor
            The atom single representation, shape [B, La, c_atom].
        c : torch.Tensor
            The atom single conditioning, shape [B, La, c_atom].
        p : torch.Tensor
            The atom pair representation, shape [B, La, La, c_atompair].
        """
        return self._diffusion_stack.get_atom_embeddings(f_input, z)

    def get_pair_bias(self, z: torch.Tensor) -> torch.Tensor:
        """Get the pair bias for the token transformer.
        This is time-independent and can be pre-computed before the diffusion steps.

        Parameters
        ----------
        z : torch.Tensor
            The pair conditioning, shape [B, Lt, Lt, c_z].

        Returns
        -------
        pair_bias : torch.Tensor
            The pair bias for the token transformer, shape [B, Nblock, H, Lt, Lt].
        """
        return self._diffusion_stack.get_pair_bias(z)

    def step(
        self,
        # atom-level inputs
        r_noisy: torch.Tensor,
        q: torch.Tensor,
        c: torch.Tensor,
        p: torch.Tensor,
        token_index: torch.Tensor,
        atom_mask: torch.Tensor,
        # token-level inputs
        s: torch.Tensor,
        pair_bias: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the AF3 diffusion module (Time-dependent part only)
        See Section 3.7 Algorithm 20: Diffusion Module in the AF3 paper.

        Parameters
        ----------
        r_noisy: torch.Tensor
            The noisy atom positions, shape [B, N, La, 3],
            where N is number of diffusion samples and La is number of atoms.
        q: torch.Tensor
            The atom single representation, shape [B, La, c_atom].
        c: torch.Tensor
            The atom conditioning, shape [B, La, c_atom].
        p: torch.Tensor
            The atom pair representation, shape [B, W, Lq, Lk, c_atompair],
            where W is the number of attention windows.
        token_index: torch.Tensor
            The token index for each atom, shape [B, La].
        atom_mask: torch.Tensor
            The atom padding mask, shape [B, La].
        s: torch.Tensor
            The single conditioning, shape [B, N, Lt, c_s].
        pair_bias: torch.Tensor
            The pair bias for the token transformer, shape [B, Nblock, H, Lt, Lt].
        token_mask: torch.Tensor
            The token padding mask, shape [B, Lt].

        Returns
        -------
        r_update : torch.Tensor
            The scaled updated atom positions, shape [B, N, La, 3].
        """
        return self._diffusion_stack.step(
            r_noisy, q, c, p, token_index, atom_mask, s, pair_bias, token_mask
        )
