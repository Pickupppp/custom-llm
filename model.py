import math
from typing import Union, Optional

import torch
from torch import nn


class CustomRMSNorm(nn.Module):
    """
    实现 RMSNorm 层。
    LayerNorm 是减去样本均值，除以样本方差，然后乘以缩放参数。
    RMSNorm 可以看作均值为0的特殊情况。
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


class CustomRotaryEmbedding(nn.Module):
    """
    实现旋转位置编码。
    """

    def __init__(
        self,
        dim,
        max_position_embeddings: int = 2048,
        base: int = 10000,
        device: Union[str, torch.device] = None,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        # 保存固定状态，但不成为模型参数
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(
            seq_len=self.max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        """
        设置 cos 和 sin 缓存。
        """
        self.max_seq_len_cached = seq_len

        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        # cos_cached / sin_cached 的 shape 为 (seq_len, dim)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, seq_len=None):
        if seq_len > self.max_position_embeddings:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(x.dtype),
            self.sin_cached[:seq_len].to(x.dtype),
        )


def rotate_half(x: torch.Tensor):
    """
    将隐藏层一半维度旋转
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    """
    对 q 和 k 进行旋转位置编码

    :param q: 查询向量
    :param k: 关键词向量
    :param cos: 旋转位置编码余弦部分
    :param sin: 旋转位置编码正弦部分
    :param position_ids: 位置索引
    :return 使用旋转位置编码后的 q 和 k
    """
    cos = cos[position_ids].unsqueeze(dim=1)
    sin = sin[position_ids].unsqueeze(dim=1)
    q_embed = (q * cos) + rotate_half(q) * sin
    k_embed = (k * cos) + rotate_half(k) * sin
    return q_embed, k_embed


class CustomMLP(nn.Module):
    """
    实现升维和降维
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.functional.gelu

    def forward(self, x: torch.Tensor):
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x) * gate
        return self.down_proj(up)


class CustomAttention(nn.Module):
    """
    多头注意力机制
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = self.config.max_position_embeddings
        self.rope_theta = self.config.rope_theta
        self.attention_dropout = self.config.attention_dropout

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=True
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=True
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=True
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        self.rotary_emb = CustomRotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, seq_len=seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        attn_weights = torch.matmul(
            query_states, key_states.transpose(-1, -2)
        ) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, seq_len, self.head_dim)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, seq_len, self.head_dim):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, seq_len, self.head_dim)}, but is {attention_mask.size()}"
                )

            attn_weights += attention_mask

        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, seq_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output


if __name__ == "__main__":
    a = torch.rand((2, 2, 10))
    print(a.shape)
    print(a.size())
