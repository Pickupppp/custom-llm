import math
from typing import Lsit, Tuple, Union, Optional

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


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
        self.num_heads = self.config.num_attention_heads
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

        if attn_weights.size() != (bsz, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, seq_len, seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, seq_len, seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, seq_len, seq_len)}, but is {attention_mask.size()}"
                )

            # 使用混合精度时 -1e9 会报错 RuntimeError: value cannot be converted to type at::Half without overflow
            # attn_weights.masked_fill_(attention_mask, -1e4)
            # 设置为 float(-inf) 损失可能变成 nan
            attn_weights.masked_fill_(attention_mask, -1e9)

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


class CustomDecoderLayer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = CustomAttention(config)
        self.mlp = CustomMLP(config)
        self.input_layernorm = CustomRMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = CustomRMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        residual = hidden_states

        # layernorm 归一化
        hidden_states = self.input_layernorm(hidden_states)

        # 自注意力
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        # 残差连接
        hidden_states += residual

        # 前馈网络部分
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states += residual
        return hidden_states


def _init_weights(config, modules):
    """
    初始化权重，对 embedding 层进行特殊处理
    """
    std = config.initializer_range
    for m in modules:
        if isinstance(m, nn.Linear):
            # nn.init.xavier_normal_(m.weight)
            m.weight.data.normal_(mean=0.0, std=std)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Embedding):
            m.weight.data.normal_(mean=0.0, std=std)
            if m.padding_idx is not None:
                m.weight.data[m.padding_idx].zero_()


def _update_causal_mask(
    attention_mask: torch.LongTensor, input_tensor: torch.FloatTensor
) -> torch.Tensor:
    """
    创建 causal_mask
    :param attention_mask: (bsz, seq_len)
    :param input_tensor: (bsz, seq_len, hidden_size)
    """
    device = input_tensor.device
    if input_tensor.dim() == 3:
        bsz, seq_len, _ = input_tensor.shape
    elif input_tensor.dim() == 2:
        bsz, seq_len = input_tensor.shape
    else:
        raise ValueError(
            f"Input tensor should have 2 or 3 dimensions, but has {input_tensor.dim()}"
        )

    assert (
        bsz == attention_mask.shape[0]
    ), f"batch size should be equal, but got {bsz} and {attention_mask.shape[0]}"

    # 处理 causal_mask
    causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(device)

    # 处理 padding mask
    if attention_mask.dim() == 2:
        padding_mask = attention_mask[:, None, None, :]  # (bsz, 1, 1, seq_len)
    elif attention_mask.dim() == 4:
        padding_mask = attention_mask
    else:
        raise ValueError(
            f"Attention mask dim should be `2` or `4`, but is {attention_mask.dim()}"
        )

    padding_mask = (padding_mask == 0).to(device)
    combined_mask = padding_mask | causal_mask
    return combined_mask


class CustomPreTrainedModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=self.padding_idx
        )
        self.layers = nn.ModuleList(
            [CustomDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = CustomRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.graident_checkpoint = False
        _init_weights(config, self.modules())

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        input_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:

        # 对于输入的处理
        if input_ids is not None and input_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            _, seq_len = input_ids.shape
        elif input_embeds is not None:
            _, seq_len, _ = input_embeds.shape
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        # 位置索引
        if position_ids is None:
            device = input_ids.device if input_ids is not None else input_embeds.device
            position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, seq_len)
        else:
            position_ids = position_ids.view(-1, seq_len).long()

        if input_embeds is None:
            input_embeds = self.embed_tokens(input_ids)

        attention_mask = _update_causal_mask(attention_mask, input_embeds)

        hidden_states = input_embeds

        for decoder_layer in self.layers:
            if self.training and self.graident_checkpoint:
                layer_outputs = checkpoint(
                    decoder_layer, hidden_states, attention_mask, position_ids
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )

            hidden_states = layer_outputs

        hidden_states = self.norm(hidden_states)
        return hidden_states


class CustomForCausalLM(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.model = CustomPreTrainedModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        _init_weights(config, self.modules())

    def enable_gradient_checkpoint(self):
        self.model.graident_checkpoint = True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        input_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            input_embeds=input_embeds,
        )

        logits: torch.Tensor = self.lm_head(outputs)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(
                -1, self.vocab_size
            )  # [bsz, seq_len, vocab] => [bsz * seq_len, vocab]
            shift_labels = shift_labels.view(-1)  # [bsz, seq_len] => [bsz * seq_len]

            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        return (logits, loss)

    def generate(
        self,
        input_ids: torch.IntTensor,
        stop_tokens: Lsit[int],
        attention_mask: torch.IntTensor,
        max_new_tokens: Optional[int] = 50,
        return_type: Optional[str] = None,
    ):
        # 去除最后的 <eos>
        input_ids = input_ids[:, :-1]
        attention_mask = attention_mask[:, :-1]

        stop_tokens_id = torch.tensor(stop_tokens, dtype=torch.int32).unsqueeze(0)

        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits, _ = self(input_ids=input_ids, attention_mask=attention_mask)
            logits = logits[:, -1, :]
            _, token_ids = torch.max(logits, dim=-1, keepdim=True)
            input_ids = torch.concat((input_ids, token_ids), dim=-1)
            attention_mask = torch.concat(
                (attention_mask, torch.ones(attention_mask.shape[0], 1)), dim=-1
            )

            mask = token_ids == stop_tokens_id
            stopped = mask.any(dim=1).all().item()
            if stopped:
                break
        if return_type == "pt":
            return input_ids
        else:
            return input_ids.tolist()

    def generate(self, text: str, tokenizer, max_new_tokens=50):
        device = next(self.parameters()).device
        outputs = tokenizer.encode(text)
        # 去除最后的 <eos>
        original_ids = outputs.ids[:-1]
        original_attention_mask = outputs.attention_mask[:-1]
        self.eval()
        for _ in range(max_new_tokens):
            input_ids = torch.tensor([original_ids]).to(device)
            attention_mask = torch.tensor([original_attention_mask]).to(device)
            with torch.no_grad():
                logtis, _ = self(input_ids=input_ids, attention_mask=attention_mask)
            logtis = logtis[:, -1, :]
            _, token_ids = torch.max(logtis, dim=-1)
            original_ids.append(token_ids.item())
            original_attention_mask.append(1)

            if token_ids == tokenizer.token_to_id("<|eos|>"):
                break
        return original_ids


if __name__ == "__main__":
    # a = torch.rand((2, 2, 10))
    # print(a[:, -1, :].shape)
    # b = torch.rand((1, 2, 10))
    # print(b[:, -1, :].shape)
    # _, c = torch.max(b[:, -1, :], dim=-1, keepdim=True)
    # print(c.shape)
    # print(a.tolist())
    # print(a.size())
    a = torch.rand(2, 5, 10)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    print(mask.shape)
    cal_mask = _update_causal_mask(mask, a)
    print(cal_mask)
