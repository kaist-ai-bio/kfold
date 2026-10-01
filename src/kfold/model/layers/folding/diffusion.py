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

"""Section 3.7 Diffusion Module in the AF3 paper."""

from functools import partial

import torch
import torch.nn as nn

from kfold.data.types.model_input import FoldingInput
from kfold.model.primitives import LayerNorm, LinearNoBias
from kfold.model.primitives.utils import add

from .atom_transformer import AtomAttentionDecoder, AtomAttentionEncoder, AtomEmbedder
from .diffusion_transformer import CachedGlobalTransformerStack
from .embeddings import FourierEmbedding, RelativePositionEncoding
from .transition import Transition

# === Diffusion Conditioning Layers === #
"""Diffusion conditioning layer.

NOTE(Seonghwan Seo):
According to AlphaFold3 paper, s is time-dependent single representation,
while z is time-independent pair representation.

For model efficiency, I separate the diffusion conditioning into two classes,
PairConditioning and SingleConditioning, which can be called separately in the
diffusion module. The PairConditioning can be pre-computed before the diffusion
steps, while the SingleConditioning needs to be computed at each diffusion step.
"""


class PairConditioning(nn.Module):
    """Diffusion conditioning layer for pair representations.
    See Section 3.7 Algorithm 21 Diffusion Conditioning in the AF3 paper.
    """

    def __init__(self, channel_z: int = 256):
        """Initialize the single conditioning layer.

        Parameters
        ----------
        channel_z : int
            The pair representation dimension, by default 128.
        """
        super().__init__()
        # Pair representation conditioning
        self.rel_pos_encoding = RelativePositionEncoding(32, 2)
        rel_pos_dim = self.rel_pos_encoding.dimension

        in_channel = channel_z + rel_pos_dim
        self.layernorm = LayerNorm(in_channel, create_offset=False)
        self.linear = LinearNoBias(in_channel, channel_z, init="default", precision=32)
        self.transitions = nn.ModuleList(
            [Transition(channel_z, expansion_factor=2) for _ in range(2)]
        )

    def forward(self, f_input: FoldingInput, z: torch.Tensor) -> torch.Tensor:
        """See Section 3.7 Algorithm 21 Diffusion Conditioning in the AF3 paper.

        Parameters
        ----------
        f_input : FoldingInput
            The folding input.
        z : torch.Tensor
            Tensor of shape (B, Lt, Lt, c_z) containing trunk pair embeddings.

        Returns
        -------
        z : torch.Tensor
            Tensor of shape (B, Lt, Lt, c_z) containing conditioned pair embeddings.
        """
        _add = partial(add, inplace=not self.training)

        z = z.float()
        rel_pos_feats = self.rel_pos_encoding(f_input, z.dtype)
        z = torch.cat((z, rel_pos_feats), dim=-1)
        z = self.linear(self.layernorm(z))  # [B, Lt, Lt, c_z]
        for transition in self.transitions:
            z = _add(z, transition(z))

        return z


class SingleConditioning(nn.Module):
    """Diffusion conditioning layer for single representations.
    See Section 3.7 Algorithm 21 Diffusion Conditioning in the AF3 paper.
    """

    def __init__(self, channel_s: int = 384, dim_fourier: int = 256):
        """Initialize the single conditioning layer.

        Parameters
        ----------
        channel_s : int
            The single representation dimension, by default 384.
        dim_fourier : int
            The fourier embeddings dimension, by default 256.
        """
        super().__init__()
        self.fourier_embed = FourierEmbedding(dim_fourier)
        self.layernorm_fourier = LayerNorm(dim_fourier, create_offset=False)
        self.linear_fourier = LinearNoBias(
            dim_fourier, channel_s, init="default", precision=32
        )
        self.linear = LinearNoBias(channel_s, channel_s, init="default", precision=32)
        self.transitions = nn.ModuleList(
            [Transition(channel_s, expansion_factor=2) for _ in range(2)]
        )

    def forward(self, s_inputs: torch.Tensor, c_noise: torch.Tensor) -> torch.Tensor:
        """See Section 3.7 Algorithm 21 Diffusion Conditioning in the AF3 paper.

        Parameters
        ----------
        s_inputs : torch.Tensor
            Tensor of shape (B, Lt, c_s) containing input single embeddings.
        c_noise : torch.Tensor
            Tensor of shape (B, N) containing diffusion noise level (or sigma).
            c_noise = 1/4 log(t_hat / sigma_data) (See Algorithm.)

        Returns
        -------
        s : torch.Tensor
            Tensor of shape (B, N, Lt, c_s) containing conditioned single embeddings.
        """
        _add = partial(add, inplace=not self.training)

        s = self.linear(s_inputs)  # [B, Lt, c_s]

        # NOTE: 1/4 log(t_hat / sigma_data) is computed outside of this class.
        # See StructureModule for more details.
        fourier_embed = self.fourier_embed(c_noise.float())  # [B, N, d_fourier]

        fourier_embed = self.linear_fourier(self.layernorm_fourier(fourier_embed))
        s = s[:, None, :, :] + fourier_embed[:, :, None, :]  # [B, N, Lt, c_s]

        for transition in self.transitions:
            s = _add(s, transition(s))

        return s


# === Main Diffusion Score Model === #
class DiffusionStack(nn.Module):
    """AF3 Diffusion module
    Section 3.7 Algorithm 20: Diffusion Module in the AF3 paper.
    """

    def __init__(
        self,
        channel_a: int = 768,
        channel_s: int = 384,
        channel_z: int = 256,
        channel_atom: int = 128,
        channel_atompair: int = 16,
        channel_coords: int = 3,
        separate_endpoint_atom_encoder: bool = False,
        atom_encoder_blocks: int = 3,
        atom_encoder_heads: int = 4,
        token_transformer_blocks: int = 24,
        token_transformer_heads: int = 16,
        atom_decoder_blocks: int = 3,
        atom_decoder_heads: int = 4,
        blocks_per_ckpt: int | None = None,
        ckpt_atom_stack: bool = False,
        kernel_backend: str = "torch",
    ) -> None:
        """Initialize the diffusion module.

        Parameters
        ----------
        channel_a : int
            The token representation dimension.
        channel_s : int
            The single representation dimension.
        channel_z : int
            The pair representation dimension.
        channel_atom : int
            The atom single representation dimension.
        channel_atompair : int
            The atom pair representation dimension.
        channel_coords : int
            The atom coordinates dimension, by default 3.
        separate_endpoint_atom_encoder : bool, optional
            Whether to split a 6-channel ECSI coordinate input into separate
            current-state and endpoint atom encoders, by default False.
        atom_encoder_blocks : int, optional
            The number of blocks in the atom encoder, by default 3.
        atom_encoder_heads : int, optional
            The number of heads in the atom encoder, by default 4.
        token_transformer_blocks : int, optional
            The number of blocks in the token transformer, by default 24.
        token_transformer_heads : int, optional
            The number of heads in the token transformer, by default 16.
        atom_decoder_blocks : int, optional
            The number of blocks in the atom decoder, by default 3.
        atom_decoder_heads : int, optional
            The number of heads in the atom decoder, by default 4.
        blocks_per_ckpt : int | None, optional
            The number of blocks per checkpoint for gradient checkpointing,
            by default None.
        ckpt_atom_stack : bool, optional
            Whether to checkpoint each complete atom transformer stack.

        """
        super().__init__()
        self.kernel_backend = kernel_backend
        self.channel_s: int = channel_s
        self.channel_z: int = channel_z
        self.channel_atom: int = channel_atom
        self.channel_atompair: int = channel_atompair
        self.channel_coords: int = channel_coords
        self.separate_endpoint_atom_encoder: bool = separate_endpoint_atom_encoder
        self.atom_encoder_blocks: int = atom_encoder_blocks
        self.atom_encoder_heads: int = atom_encoder_heads
        self.token_transformer_blocks: int = token_transformer_blocks
        self.token_transformer_heads: int = token_transformer_heads
        self.atom_decoder_blocks: int = atom_decoder_blocks
        self.atom_decoder_heads: int = atom_decoder_heads

        # === Diffusion conditioning === #
        self.pair_conditioning = PairConditioning(channel_z)
        self.single_conditioning = SingleConditioning(channel_s, dim_fourier=256)

        # === Local atom-level attention encoder === #
        if separate_endpoint_atom_encoder and channel_coords != 6:
            raise ValueError(
                "separate_endpoint_atom_encoder expects channel_coords=6 "
                "for [r_t, r_T] ECSI inputs."
            )
        atom_encoder_channel_coords = (
            3 if separate_endpoint_atom_encoder else channel_coords
        )
        self.atom_embedder = AtomEmbedder(
            channel_z=channel_z,
            channel_atom=channel_atom,
            channel_atompair=channel_atompair,
            use_structure=True,
        )
        self.atom_attention_encoder = AtomAttentionEncoder(
            channel_atom=channel_atom,
            channel_atompair=channel_atompair,
            channel_token=channel_a,
            channel_coords=atom_encoder_channel_coords,
            num_blocks=atom_encoder_blocks,
            num_heads=atom_encoder_heads,
            use_structure=True,
            ckpt_atom_stack=ckpt_atom_stack,
            kernel_backend=self.kernel_backend,
        )
        if separate_endpoint_atom_encoder:
            self.endpoint_atom_attention_encoder = AtomAttentionEncoder(
                channel_atom=channel_atom,
                channel_atompair=channel_atompair,
                channel_token=channel_a,
                channel_coords=atom_encoder_channel_coords,
                num_blocks=atom_encoder_blocks,
                num_heads=atom_encoder_heads,
                use_structure=True,
                ckpt_atom_stack=ckpt_atom_stack,
                kernel_backend=self.kernel_backend,
            )
            self.layernorm_a_t = LayerNorm(channel_a, create_offset=False)
            self.layernorm_a_endpoint = LayerNorm(channel_a, create_offset=False)
            self.endpoint_fusion = LinearNoBias(
                channel_a * 2,
                channel_a,
                init="final",
                precision=32,
            )

        # === Full token-level attention === #
        self.layernorm_s = LayerNorm(channel_s, create_offset=False)
        self.linear_s_to_a = LinearNoBias(
            channel_s, channel_a, init="final", precision=32
        )
        self.layernorm_z = LayerNorm(channel_z, create_offset=False)
        self.linear_z_to_bias = LinearNoBias(
            channel_z, token_transformer_blocks * token_transformer_heads, precision=32
        )

        self.token_transformer = CachedGlobalTransformerStack(
            channel_a=channel_a,
            channel_s=channel_s,
            num_blocks=token_transformer_blocks,
            num_heads=token_transformer_heads,
            blocks_per_ckpt=blocks_per_ckpt,
            kernel_backend=self.kernel_backend,
        )

        self.layernorm_a = LayerNorm(channel_a, create_offset=False)

        # === Local token-level attention decoder === #
        self.atom_attention_decoder = AtomAttentionDecoder(
            channel_a=channel_a,
            channel_atom=channel_atom,
            channel_atompair=channel_atompair,
            num_blocks=atom_decoder_blocks,
            num_heads=atom_decoder_heads,
            ckpt_atom_stack=ckpt_atom_stack,
            kernel_backend=self.kernel_backend,
        )

    # === Main forward function for training === #
    def forward(
        self,
        f_input: FoldingInput,
        r_noisy: torch.Tensor,
        c_noise: torch.Tensor,
        s_inputs: torch.Tensor,
        z: torch.Tensor,
        atom_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Training forward pass of the AF3 diffusion module
        See Section 3.7 Algorithm 20: Diffusion Module in the AF3 paper.

        Parameters
        ----------
        f_input : FoldingInput
            The folding input.
        r_noisy: torch.Tensor
            The noisy atom positions, shape [B, N, La, 3],
            where N is number of diffusion samples and La is number of atoms.
        c_noise: torch.Tensor
            The diffusion noise level (or sigmas), shape [B, N].
        s_inputs: torch.Tensor
            The input single representation, shape [B, Lt, c_s].
        z: torch.Tensor
            The trunk pair representation, shape [B, Lt, Lt, c_z].
        atom_mask: torch.Tensor | None
            Atoms the coordinate stack may attend to and pool from, shape [B, La].
            Defaults to `f_input.atom.pad_mask`. A caller may narrow it to hide
            atoms whose coordinates are undefined; the reference-conformer features
            built by `get_atom_embeddings` still cover the full `pad_mask`.

        Returns
        -------
        r_update : torch.Tensor
            The scaled updated atom positions, shape [B, N, La, 3].
        """
        token_index = f_input.atom.token_index  # [B, Lt]
        if atom_mask is None:
            atom_mask = f_input.atom.pad_mask  # [B, La]
        token_mask = f_input.token.pad_mask  # [B, Lt]

        s = self.get_single_conditioning(s_inputs, c_noise)  # [B, N, Lt, c_s]
        z = self.get_pair_conditioning(f_input, z)  # [B, Lt, Lt, c_z]
        q, c, p = self.get_atom_embeddings(f_input, z)
        pair_bias = self.get_pair_bias(z)  # [B, Nblock, H, Lt, Lt]
        r_update = self.step(
            r_noisy,
            q,
            c,
            p,
            token_index,
            atom_mask,
            s,
            pair_bias,
            token_mask,
        )
        return r_update

    # === Multiple forward functions for different parts of the diffusion module === #
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
        return self.pair_conditioning(f_input, z)

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
        B, L, _, c_z = z.shape
        N, H = self.token_transformer_blocks, self.token_transformer_heads
        pair_bias = self.linear_z_to_bias(self.layernorm_z(z)).view(B, L, L, N, H)
        pair_bias = pair_bias.permute(0, 3, 4, 1, 2)  # [B, N, H, L, L]
        pair_bias = pair_bias.contiguous()
        return pair_bias.to(torch.float32)

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
        return self.single_conditioning(s_inputs, c_noise).to(torch.float32)

    def get_atom_embeddings(
        self,
        f_input: FoldingInput,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare the inputs which are static across diffusion steps.

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
        q, c, p = self.atom_embedder(f_input, z)
        return q, c, p

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
        # === Local attention on atom-level and aggregate to coarse-grained token === #
        # Add diffusion sample dimension
        q = q.unsqueeze(-3)  # [B, 1, La, c_atom]
        c = c.unsqueeze(-3)  # [B, 1, La, c_atom]
        p = p.unsqueeze(-5)  # [B, 1, W, Lq, Lk, c_atompair]
        atom_mask = atom_mask.unsqueeze(-2)  # [B, 1, La]
        token_mask = token_mask.unsqueeze(-2)  # [B, 1, Lt]
        pair_bias = pair_bias.unsqueeze(-5)  # [B, 1, Nblock, H, Lt, Lt]
        token_index = token_index.unsqueeze(-2)  # [B, 1, La]

        # NOTE: Add extra dimension for the number of diffusion samples, N.
        r_noisy = r_noisy * atom_mask[..., None]

        if self.separate_endpoint_atom_encoder:
            r_t, r_T = r_noisy[..., :3], r_noisy[..., 3:]
            a, q_skip, c_skip, p_skip = self.atom_attention_encoder(
                q,  # [B, 1, La, c_atom]
                c,  # [B, 1, La, c_atom]
                p,  # [B, 1, W, Lq, Lk, c_atompair]
                r_noisy=r_t,  # [B, N, La, 3]
                token_index=token_index,  # [B, 1, Lt]
                mask=atom_mask,  # [B, 1, La]
                num_tokens=token_mask.shape[-1],
            )
            a_endpoint, _, _, _ = self.endpoint_atom_attention_encoder(
                q,  # [B, 1, La, c_atom]
                c,  # [B, 1, La, c_atom]
                p,  # [B, 1, W, Lq, Lk, c_atompair]
                r_noisy=r_T,  # [B, N, La, 3]
                token_index=token_index,  # [B, 1, Lt]
                mask=atom_mask,  # [B, 1, La]
                num_tokens=token_mask.shape[-1],
            )
            a = self.endpoint_fusion(
                torch.cat(
                    (self.layernorm_a_t(a), self.layernorm_a_endpoint(a_endpoint)),
                    dim=-1,
                )
            )
        else:
            a, q_skip, c_skip, p_skip = self.atom_attention_encoder(
                q,  # [B, 1, La, c_atom]
                c,  # [B, 1, La, c_atom]
                p,  # [B, 1, W, Lq, Lk, c_atompair]
                r_noisy=r_noisy,  # [B, N, La, 3]
                token_index=token_index,  # [B, 1, Lt]
                mask=atom_mask,  # [B, 1, La]
                num_tokens=token_mask.shape[-1],
            )
        del q, c, p, r_noisy

        # Shape:
        # - a: [B, N, Lt, c_token]
        # - q_skip: [B, N, La, c_atom]
        # - c_skip: [B, 1, La, c_atom]
        # - p_skip: [B, 1, W, Lq, Lk, c_atompair]

        # === Full attention on token-level === #
        a = a.float()  # Convert to float32 for stability.
        a = a + self.linear_s_to_a(self.layernorm_s(s))  # [B, N, Lt, c_token]

        a = self.token_transformer(
            a,  # [B, N, Lt, c_token]
            s,  # [B, N, Lt, c_s]
            pair_bias,  # [B, 1, Nblock, H, Lt, Lt]
            mask=token_mask,  # [B, 1, Lt]
        )

        a = self.layernorm_a(a)  # [B, N, Lt, c_token]

        # === Broadcast token to atoms and run Local Atom Attention === #
        r_update = self.atom_attention_decoder(
            a,  # [B, N, Lt, c_token]
            q_skip,  # [B, N, La, c_atom]
            c_skip,  # [B, 1, La, c_atom]
            p_skip,  # [B, 1, W, Lq, Lk, c_atompair]
            token_index=token_index,  # [B, 1, La]
            mask=atom_mask,  # [B, 1, La]
        )  # -> [B, N, La, 3]

        r_update = r_update * atom_mask[..., None]  # Mask out padded atoms.

        return r_update
