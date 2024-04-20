import torch
from torch import nn


class CustomRMSNorm(nn.Module):
    """
    实现 RMSNorm 层。
    LayerNorm 是减去样本均值，除以样本方差，然后乘以缩放参数。
    RMSNorm 可以看作均值为0的特殊情况。
    """

    def __init__(self, hidden_size, eps=1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)
