import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex rotation.
    From Moondream implementation.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using precomputed frequencies.
    From Moondream implementation.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # Adjust freqs_cis shape for broadcasting
    freqs_cis = freqs_cis[:xq_.shape[-2], :]
    shape = [1] * (xq_.ndim - 1) + [xq_.shape[-1]]
    freqs_cis = freqs_cis.view(*shape)

    # Apply rotation in complex space
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class CrossAttention(nn.Module):
    """
    Cross attention module with rotary embeddings and flash attention support.
    """
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.n_heads
        self.dropout = config.dropout

        # Rotary embedding parameters
        self.rotary_ndims = int(self.head_dim * config.partial_rotary_factor)
        self.rotary_ndims = self.rotary_ndims - (self.rotary_ndims % 2)  # Make even

        # Attention projections
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias)

        # Setup rotary embeddings
        self.freqs_cis = precompute_freqs_cis(
            self.rotary_ndims,
            config.max_seq_len,
            config.rope_theta
        )

        # Enable Flash Attention if available
        self.flash = hasattr(F, 'scaled_dot_product_attention')

    def forward(self, x, context=None, attention_mask=None):
        batch_size = x.size(0)

        # Project queries, keys, and values
        q = self.q_proj(x)
        k = self.k_proj(context if context is not None else x)
        v = self.v_proj(context if context is not None else x)

        # Reshape for attention
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.n_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.n_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.n_heads)

        # Split rotary and non-rotary parts
        q_rot = q[..., :self.rotary_ndims]
        q_pass = q[..., self.rotary_ndims:]
        k_rot = k[..., :self.rotary_ndims]
        k_pass = k[..., self.rotary_ndims:]

        # Apply rotary embeddings
        q_rot, k_rot = apply_rotary_emb(
            q_rot, k_rot,
            self.freqs_cis.to(q.device)
        )

        # Recombine rotary and non-rotary parts
        q = torch.cat([q_rot, q_pass], dim=-1)
        k = torch.cat([k_rot, k_pass], dim=-1)

        # Compute attention
        if self.flash:
            # Use Flash Attention when available
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0.0,
            )
        else:
            # Manual attention computation
            scale = 1 / math.sqrt(self.head_dim)
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale

            if attention_mask is not None:
                attn = attn.masked_fill(attention_mask == 0, float('-inf'))

            attn = F.softmax(attn, dim=-1)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            out = torch.matmul(attn, v)

        # Reshape and project output
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.o_proj(out)

        return out

class Block(nn.Module):
    """
    Transformer block with optional cross-attention.
    """
    def __init__(self, config, is_cross=False):
        super().__init__()
        # Pre-attention norm
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=1e-5)

        # Attention layers
        self.self_attn = CrossAttention(config)
        if is_cross:
            self.cross_attn = CrossAttention(config)
            self.ln_cross = nn.LayerNorm(config.hidden_size, eps=1e-5)

        # MLP
        self.mlp = MLP(config)
        self.is_cross = is_cross

    def forward(self, x, context=None, mask=None, pos_offset=0):
        # Self attention
        x = x + self.self_attn(self.ln1(x), attention_mask=mask)

        # Cross attention
        if self.is_cross and context is not None:
            x = x + self.cross_attn(
                self.ln_cross(x),
                context=context
            )

        # MLP
        x = x + self.mlp(self.ln2(x))
        return x

class MLP(nn.Module):
    """
    MLP block with GELU activation.
    """
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=config.bias
        )
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=config.bias
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
