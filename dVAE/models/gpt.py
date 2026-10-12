import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def norm(x: Tensor, eps: float = 1e-6) -> Tensor:
    """RMSNorm helper with backward-compatibility for PyTorch < 2.0."""
    if hasattr(F, "rms_norm"):
        return F.rms_norm(x, (x.size(-1),), eps=eps)
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


class Yarn(nn.Module):
    """Rotary position embeddings with frequency scaling (YARN-style)."""

    def __init__(self, head_dim: int, max_seq_len: int):
        super().__init__()
        self.head_dim = int(head_dim)
        self.max_seq_len = int(max_seq_len)
        self.attn_scale = 0.1
        self.reset()

    def reset(self):
        # Build base angular frequencies
        half_dim = self.head_dim // 4
        steps = torch.linspace(0, 1, steps=half_dim, dtype=torch.float32)
        base_freq = (1.0 / 1024.0) ** steps
        angular_freq = torch.cat([base_freq, torch.zeros(half_dim, dtype=torch.float32)])

        t = torch.arange(self.max_seq_len, dtype=torch.float32)
        theta = torch.outer(t, angular_freq)

        self.register_buffer("angular_freq", angular_freq, persistent=False)
        self.register_buffer("cos", theta.cos(), persistent=False)
        self.register_buffer("sin", theta.sin(), persistent=False)
        self.attn_scale = 0.1

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if self.cos.device != device:
            self.cos = self.cos.to(device)
            self.sin = self.sin.to(device)
        cos = self.cos[:seq_len].to(dtype=dtype)
        sin = self.sin[:seq_len].to(dtype=dtype)
        return cos, sin


def rotary(x_BTHD: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    T = x_BTHD.size(-3)
    cos = cos[:T][None, :, None, :]
    sin = sin[:T][None, :, None, :]
    x1, x2 = x_BTHD.chunk(2, dim=-1)
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat((y1, y2), dim=-1)


def causal_scaled_dot_product_attention(
    q: Tensor, k: Tensor, v: Tensor, is_causal: bool = True, scale: float = None
) -> Tensor:
    """Attention implementation that automatically falls back on PyTorch < 2.0."""
    if hasattr(F, "scaled_dot_product_attention"):
        return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=scale)

    # Manual fallback
    d_k = q.size(-1)
    if scale is None:
        scale = 1.0 / math.sqrt(d_k)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if is_causal:
        T_q, T_k = q.size(-2), k.size(-2)
        mask = torch.triu(
            torch.full((T_q, T_k), float("-inf"), device=q.device, dtype=q.dtype),
            diagonal=1,
        )
        scores = scores + mask
    attn_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, v)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim: int, num_heads: int, causal: bool = True):
        super().__init__()
        self.causal = causal
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dim = dim
        self.hdim = num_heads * head_dim
        assert self.hdim == self.dim, f"hdim ({self.hdim}) must match dim ({self.dim})"

        std = 0.5 * (self.dim ** -0.5)
        bound = (3.0 ** 0.5) * std
        self.qkvo_w = nn.Parameter(torch.empty(self.hdim, self.dim * 4))
        with torch.no_grad():
            self.qkvo_w.view(4, self.hdim, self.dim)[:3].uniform_(-bound, bound)
            self.qkvo_w.view(4, self.hdim, self.dim)[3].zero_()

    def _qkv(self, x: Tensor, cos_sin):
        B, T, D = x.shape
        w_qkv = self.qkvo_w.view(4, self.hdim, self.dim)[:3].flatten(end_dim=1).type_as(x)
        qkv = F.linear(x, w_qkv).view(B, T, 3 * self.num_heads, self.head_dim)
        q, k, v = qkv.chunk(3, dim=-2)
        q, k = norm(q), norm(k)

        cos, sin = cos_sin
        q = rotary(q, cos, sin)
        k = rotary(k, cos, sin)
        return q, k, v

    def forward(self, x: Tensor, cos_sin, scale: float):
        B, T, D = x.shape
        q, k, v = self._qkv(x, cos_sin)  # [B, T, nH, Hd]
        q = q.transpose(1, 2)  # [B, nH, T, Hd]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        y = causal_scaled_dot_product_attention(
            q, k, v, is_causal=self.causal, scale=scale
        )
        y = y.transpose(1, 2).contiguous().view(B, T, self.dim)
        w_out = self.qkvo_w.view(4, self.hdim, self.dim)[3].type_as(y)
        return F.linear(y, w_out)


class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.c_fc = nn.Parameter(torch.empty(dim, hdim))
        self.c_proj = nn.Parameter(torch.empty(dim, hdim))
        std = 0.5 * (dim ** -0.5)
        bound = (3.0 ** 0.5) * std
        with torch.no_grad():
            self.c_fc.uniform_(-bound, bound)
            self.c_proj.zero_()

    def forward(self, x: Tensor) -> Tensor:
        h = F.linear(x, self.c_fc.T.type_as(x))
        h = F.relu(h).square()
        return F.linear(h, self.c_proj.type_as(x))


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        head_dim: int,
        num_heads: int,
        causal: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attn = CausalSelfAttention(dim, head_dim, num_heads, causal=causal)
        self.mlp = MLP(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, cos_sin, scale: float) -> Tensor:
        x = x + self.dropout(self.attn(norm(x), cos_sin, scale))
        x = x + self.dropout(self.mlp(norm(x)))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        model_dim: int,
        max_seq_len: int,
        embeddings_cls=None,
        causal: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert model_dim % num_heads == 0, f"model_dim {model_dim} must be divisible by num_heads {num_heads}"
        head_dim = model_dim // num_heads

        self.embed = embeddings_cls() if embeddings_cls is not None else None
        self.yarn = Yarn(head_dim, max_seq_len)
        self.blocks = nn.ModuleList(
            [
                Block(model_dim, head_dim, num_heads, causal=causal, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.L = num_layers
        self.max_seq_len = max_seq_len

    def forward(self, x: Tensor) -> Tensor:
        if self.embed is not None:
            x = self.embed(x)
        B, T, D = x.shape
        cos, sin = self.yarn(T, device=x.device, dtype=x.dtype)
        x = norm(x)
        for i in range(self.L):
            x = self.blocks[i](x, (cos, sin), self.yarn.attn_scale)
        return norm(x)

