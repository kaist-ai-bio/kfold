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

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from kfold.constants.sequence import PAD_TOKEN_INDEX
from kfold.data.types.model_input import FoldingInput
from kfold.model.layers.seq_enc.transformer_stack import TransformerStack
from kfold.utils.config import configurable

HF_REPO_ID = "kaist-ai-bio/kfold-assets"
HF_FILENAME = "weights/rna_seq_1b.pth"


@configurable
class RNASequenceEncoder(torch.nn.Module):
    @dataclass(kw_only=True)
    class Config:
        """Configuration for the RNA sequence encoder.

        Attributes
        ----------
        model_path: str | None
            Path to a local RNA encoder checkpoint. If unset, download from Hugging
            Face.
        cache_dir: str | None
            Directory used to cache weights downloaded from Hugging Face.
        vocab_size: int
            Size of the input token vocabulary.
        d_model: int
            Dimension of model hidden states and embeddings.
        n_heads: int
            Number of attention heads in the transformer.
        n_layers: int
            Number of transformer layers.
        use_moe: bool
            Whether to use Mixture of Experts (MoE) FFN layers instead of dense FFN.
        """

        model_path: str | None = None
        cache_dir: str | None = None
        vocab_size: int = 64
        d_model: int = 960
        n_heads: int = 15
        n_layers: int = 30
        use_moe: bool = True

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg: RNASequenceEncoder.Config = cfg

        # Construct the PLM without allocating or initializing its parameters.
        with torch.device("meta"):
            self.embed = torch.nn.Embedding(cfg.vocab_size, cfg.d_model)
            self.transformer = TransformerStack(
                cfg.d_model, cfg.n_heads, cfg.n_layers, use_moe=cfg.use_moe
            )

        if cfg.model_path is None:
            path = hf_hub_download(
                repo_id=HF_REPO_ID, filename=HF_FILENAME, cache_dir=cfg.cache_dir
            )
        else:
            path = Path(cfg.model_path)
        state_dict = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        state_dict = {
            k: v for k, v in state_dict.items() if not k.startswith("sequence_head")
        }
        self.load_state_dict(state_dict, strict=True, assign=True)
        del state_dict

        # RoPE caches are intentionally absent from the checkpoint. Recreate them
        # after assigning parameters so no buffer remains on the meta device.
        self.transformer.rotary.init_buffers(device="cpu")
        meta_tensors = [
            name
            for name, tensor in (*self.named_parameters(), *self.named_buffers())
            if tensor.is_meta
        ]
        if meta_tensors:
            raise RuntimeError(
                "Checkpoint loading left tensors on the meta device: "
                + ", ".join(meta_tensors[:5])
            )
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    @property
    def n_layers(self) -> int:
        return self.cfg.n_layers

    @property
    def d_model(self) -> int:
        return self.cfg.d_model

    @property
    def n_heads(self) -> int:
        return self.cfg.n_heads

    def forward(self, f_input: FoldingInput) -> torch.Tensor:
        """Forward pass of sequence representation module.

        Parameters
        ----------
        f_input: FoldingInput
            The input features

        Returns
        -------
        x_token: torch.Tensor
            Tensor of shape (B, Ntoken, N, D) containing sequence representations,
            where N is the number of layers and D is the model dimension.
        """
        with (
            torch.autocast(f_input.device.type, dtype=torch.bfloat16),
            torch.no_grad(),
        ):
            return self._forward(f_input)

    def _forward(
        self,
        f_input: FoldingInput,
    ) -> torch.Tensor:
        """Forward pass of sequence representation module.

        Parameters
        ----------
        f_input: FoldingInput
            The input features

        Returns
        -------
        x_token: torch.Tensor
            Tensor of shape (B, Ntoken, Nlayer+1, D) containing sequence representations,
            where D is the model dimension.
        """
        # NOTE: padding tokens have seq_id=-1, which will be masked out in
        # attention computation. (entity_id is 1-indexed for valid tokens)
        input_ids = f_input.sequence.seq_token_id
        seq_id = f_input.sequence.asym_id
        pos_id = f_input.sequence.pos_id

        # === Mask out invalid sequence tokens === #
        seq_mask = f_input.sequence.pad_mask & f_input.sequence.is_rna
        if not seq_mask.any():
            return self.embed.weight.new_zeros(
                (*f_input.token.pad_mask.shape, self.n_layers + 1, self.d_model)
            )
        seq_id = seq_id.masked_fill(~seq_mask, -1)
        input_ids = input_ids.masked_fill(~seq_mask, PAD_TOKEN_INDEX)

        # === Forward pass === #
        x = self.embed(input_ids)
        x_list = [x]
        rotary = self.transformer.rotary(pos_id)
        for block in self.transformer.blocks:
            x = block(x, seq_id, rotary)
            x_list.append(x)
        x = torch.stack(x_list, dim=-2)  # [B, Nseq, Nlayer+1, D]

        # sequence -> token index mapping
        seq_token_idx = f_input.token.seq_token_index.clamp(min=0)  # [B, Ntokens]
        seq_token_idx = seq_token_idx[..., None, None].expand(
            -1, -1, self.n_layers + 1, self.d_model
        )
        x_out = x.gather(1, seq_token_idx)

        # Mask out invalid tokens
        token_mask = f_input.token.pad_mask & f_input.token.is_rna
        x_out.masked_fill_(~token_mask[..., None, None], 0.0)
        return x_out
