from torch.nn import Module, Parameter
from torch import Tensor
import torch


class LayerNorm(Module):
    """层归一化."""

    def __init__(self, d_model: int, epsilon: float = 1e-6) -> None:
        """初始化组件.

        Args:
            d_model: 输入和输出的维度.
            epsilon: 防止除零的小常数, 默认1e-6.
        """
        super().__init__()
        self.gamma: Parameter = Parameter(torch.ones(d_model))
        self.beta: Parameter = Parameter(torch.zeros(d_model))
        self.epsilon: float = epsilon

    def forward(self, x: Tensor) -> Tensor:
        """前馈计算.

        Args:
            x: 输入张量, shape: (batch_size, seq_len, d_model).

        Returns:
            返回层归一化后的输出.
        """
        mean: Tensor = x.mean(dim=-1, keepdim=True)  # (batch_size, seq_len, 1)
        std: Tensor = x.std(dim=-1, keepdim=True)  # (batch_size, seq_len, 1)
        return self.gamma * (x - mean) / (std + self.epsilon) + self.beta
