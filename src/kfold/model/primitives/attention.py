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

import math

import torch
import torch.nn.functional as F

from .utils import permute_final_dims

try:
    from cuequivariance_torch import attention_pair_bias as _cueq_attention_pair_bias
except ImportError:
    _cueq_attention_pair_bias = None


def attention_pair_bias(
    module: torch.nn.Module,
    x_q: torch.Tensor,
    x_k: torch.Tensor,
    pair_bias: torch.Tensor,
    mask: torch.Tensor,
    *,
    kernel_backend: str,
    call_site: str,
) -> torch.Tensor:
    """Route normalized attention inputs to the selected implementation.

    Parameters
    ----------
    module : torch.nn.Module
        Attention module providing QKV, gate, and output projections.
    x_q : torch.Tensor
        Normalized query features of shape (*, Lq, C).
    x_k : torch.Tensor
        Normalized key/value features of shape (*, Lk, C).
    pair_bias : torch.Tensor
        Pair bias broadcastable to (*, H, Lq, Lk).
    mask : torch.Tensor
        Valid-key mask broadcastable to (*, Lk).
    kernel_backend : str
        Implementation: "torch", "sdpa", "cuequiv", or "triton".
    call_site : str
        Attention context used by the Triton implementation.

    Returns
    -------
    torch.Tensor
        Gated attention output with the same shape as x_q.
    """
    if kernel_backend == "triton":
        return triton_attention_pair_bias(
            module, x_q, x_k, pair_bias, mask, call_site=call_site
        )
    if kernel_backend == "sdpa":
        kernel = sdpa_attention_pair_bias
    elif kernel_backend == "cuequiv":
        kernel = cueq_attention_pair_bias
    elif kernel_backend == "torch":
        kernel = torch_attention_pair_bias
    else:
        raise ValueError(f"Unsupported attention backend: {kernel_backend!r}")

    q = module.linear_q(x_q)
    k = module.linear_k(x_k)
    v = module.linear_v(x_k)
    q, k, v = (
        tensor.unflatten(-1, (module.num_heads, module.head_dim)).transpose(-2, -3)
        for tensor in (q, k, v)
    )
    return kernel(
        s=x_q,
        q=q,
        k=k,
        v=v,
        pair_bias=pair_bias,
        mask=mask,
        w_proj_g=module.linear_g.weight,
        b_proj_g=module.linear_g.bias,
        w_proj_o=module.linear_out.weight,
        b_proj_o=module.linear_out.bias,
        inf=module.inf,
    )


@torch.compiler.disable
def triton_attention_pair_bias(
    module,
    x_q: torch.Tensor,
    x_k: torch.Tensor,
    pair_bias: torch.Tensor,
    mask: torch.Tensor,
    *,
    call_site: str,
) -> torch.Tensor:
    """Run the Triton APB implementation for normalized K-Fold inputs."""
    from kfold.kernels.triton.attention_pair_bias import (
        triton_attention_pair_bias,
    )

    return triton_attention_pair_bias(
        module,
        x_q,
        x_k,
        pair_bias,
        mask,
        call_site=call_site,
    )


def _attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    bias: torch.Tensor | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    """Compute the attention operation.
    Parameters
    ----------
    query : torch.Tensor
        The query tensor of shape (..., H, Lq, Dh)
    key : torch.Tensor
        The key tensor of shape (..., H, Lk, Dh)
    value : torch.Tensor
        The value tensor of shape (..., H, Lk, Dh)
    bias : Optional[torch.Tensor]
        The attention bias of shape (..., H, Lq, Lk), default None
    scale : Optional[float]
        The scaling factor for the query-key dot product, default None

    Returns
    -------
    out : torch.Tensor
        The output tensor of shape (..., H, Lq, Dh)
    """
    query = query.to(torch.float32)
    key = key.to(torch.float32)
    bias = bias.to(torch.float32) if bias is not None else None
    if scale is None:
        scale = 1 / math.sqrt(query.shape[-1])

    with torch.autocast(query.device.type, enabled=False):
        # Compute attention weights
        attn = torch.einsum("...qc,...kc->...qk", query * scale, key)  # [*, H, Lq, Lk]

        # Add attention bias
        if bias is not None:
            attn = attn + bias

        # Softmax
        attn = attn.softmax(dim=-1).to(value.dtype)

    # Compute output
    out = torch.einsum("...qk,...kc->...qc", attn, value)

    return out


@torch.compiler.disable
def cueq_attention_pair_bias(
    s: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pair_bias: torch.Tensor,
    mask: torch.Tensor,
    w_proj_g: torch.Tensor,
    w_proj_o: torch.Tensor,
    b_proj_g: torch.Tensor | None,
    b_proj_o: torch.Tensor | None,
    inf: float = 1e6,
    eps: float = 1e-5,
    attn_scale: float | None = None,
) -> torch.Tensor:
    """Compute gated attention with pair bias using cuEquivariance."""
    if _cueq_attention_pair_bias is None:
        raise ImportError(
            "cuequivariance_torch is not installed. Please install it to use the kernel."
        )
    # Check batch dimensions match
    if not (s.shape[:-3] == q.shape[:-4] == k.shape[:-4] == v.shape[:-4]):
        raise ValueError(
            f"cueq_attention_pair_bias supports inputs with the same batch dims: "
            f"s={s.shape}, q={q.shape}, k={k.shape}, v={v.shape}"
        )
    # Check sequence length dimensions match
    if not (s.shape[-2] == q.shape[-2] == k.shape[-2] == v.shape[-2]):
        raise ValueError(
            f"cueq_attention_pair_bias supports inputs with the same sequence "
            f"length dims: s={s.shape}, q={q.shape}, k={k.shape}, v={v.shape}"
        )
    # Merge batch dims for compatibility
    batch_dims = s.shape[:-2]
    s = s.flatten(0, -3)  # [*, L, D]
    q = q.flatten(0, -4)  # [*, H, Lq, Dh]
    k = k.flatten(0, -4)  # [*, H, Lk, Dh]
    v = v.flatten(0, -4)  # [*, H, Lk, Dh]
    pair_bias = pair_bias.flatten(0, -4)  # [*, H, Lq, Lk]
    mask = mask.flatten(0, -2)  # [*, Lk]

    out, _ = _cueq_attention_pair_bias(
        s=s,  # [B*M, L, D]
        q=q,  # [B*M, H, Lq, Dh]
        k=k,  # [B*M, H, Lk, Dh]
        v=v,  # [B*M, H, Lv, Dh]
        z=pair_bias,  # [B, H, Lq, Lk]
        mask=mask,  # [B, Lk] or [B*M, Lk]
        num_heads=q.shape[-3],  # int
        w_proj_z=None,  # is_cached_z_proj=True
        w_proj_g=w_proj_g,  # [D, D]
        w_proj_o=w_proj_o,  # [D, D]
        w_ln_z=None,  # is_cached_z_proj=True
        b_ln_z=None,  # is_cached_z_proj=True
        b_proj_z=None,  # is_cached_z_proj=True
        b_proj_g=b_proj_g,  # [D]
        b_proj_o=b_proj_o,  # [D]
        inf=inf,  # float
        eps=eps,  # float
        attn_scale=attn_scale,  # float or None
        return_z_proj=False,
        is_cached_z_proj=True,
    )
    out = out.unflatten(0, batch_dims)  # restore batch dims
    return out


def torch_attention_pair_bias(
    s: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pair_bias: torch.Tensor,
    mask: torch.Tensor,
    w_proj_g: torch.Tensor,
    w_proj_o: torch.Tensor,
    b_proj_g: torch.Tensor | None,
    b_proj_o: torch.Tensor | None,
    inf: float = 1e6,
    attn_scale: float | None = None,
) -> torch.Tensor:
    """Compute gated attention with pair bias using explicit PyTorch operations."""
    # Mask out invalid positions in pair bias
    pair_bias = (
        pair_bias - inf * ((~mask.bool()).float()[..., None, None, :])
    )  # [*, H, Lq, Lk]

    # === Attention === #
    Av = _attention(
        q,  # [*, H, Lq, Dh]
        k,  # [*, H, Lk, Dh]
        v,  # [*, H, Lk, Dh]
        bias=pair_bias,  # [*, H, Lq, Lk]
        scale=attn_scale,
    )  # [*, H, Lq, Dh]
    Av = permute_final_dims(Av, (1, 0, 2)).reshape(s.shape)  # [*, Lq, D]

    g = torch.sigmoid(F.linear(s, w_proj_g, b_proj_g))  # [*, L, D]
    out = F.linear(g * Av, w_proj_o, b_proj_o)  # [*, L, D]
    return out


def sdpa_attention_pair_bias(
    s: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pair_bias: torch.Tensor,
    mask: torch.Tensor,
    w_proj_g: torch.Tensor,
    w_proj_o: torch.Tensor,
    b_proj_g: torch.Tensor | None,
    b_proj_o: torch.Tensor | None,
    inf: float = 1e9,
    attn_scale: float | None = None,
) -> torch.Tensor:
    """Compute gated attention with pair bias using PyTorch SDPA.

    Parameters
    ----------
    s : torch.Tensor
        Query features of shape (*, Lq, C).
    q : torch.Tensor
        Queries of shape (*, H, Lq, C_h).
    k : torch.Tensor
        Keys of shape (*, H, Lk, C_h).
    v : torch.Tensor
        Values of shape (*, H, Lk, C_h).
    pair_bias : torch.Tensor
        Pair bias broadcastable to (*, H, Lq, Lk).
    mask : torch.Tensor
        Valid-key mask broadcastable to (*, Lk).
    w_proj_g : torch.Tensor
        Gate projection weight.
    w_proj_o : torch.Tensor
        Output projection weight.
    b_proj_g : torch.Tensor or None
        Gate projection bias.
    b_proj_o : torch.Tensor or None
        Output projection bias.
    inf : float, default=1e9
        Magnitude of the negative bias for invalid keys.
    attn_scale : float or None, default=None
        Attention scale; None uses the inverse square root of C_h.

    Returns
    -------
    torch.Tensor
        Gated attention output with the same shape as s.
    """
    bias = pair_bias.to(q.dtype).masked_fill(~mask.bool()[..., None, None, :], -inf)
    bias = bias.expand(*q.shape[:-2], q.shape[-2], k.shape[-2])
    # PyTorch efficient SDPA can omit its backward LSE workspace when only the
    # additive bias requires grad (frozen diffusion, trainable conditioning).
    # A local Q leaf requests that workspace without unfreezing model parameters.
    if (
        torch.is_grad_enabled()
        and bias.requires_grad
        and not any(tensor.requires_grad for tensor in (q, k, v))
    ):
        q = q.detach().requires_grad_(True)
    # Merge sample and local-window axes so fused SDPA receives rank-four inputs.
    attention = F.scaled_dot_product_attention(
        q.reshape(-1, *q.shape[-3:]),
        k.reshape(-1, *k.shape[-3:]),
        v.reshape(-1, *v.shape[-3:]),
        attn_mask=bias.reshape(-1, *bias.shape[-3:]),
        scale=attn_scale,
    )
    attention = attention.transpose(-3, -2).reshape_as(s)
    gate = torch.sigmoid(F.linear(s, w_proj_g, b_proj_g))
    return F.linear(gate * attention, w_proj_o, b_proj_o)
