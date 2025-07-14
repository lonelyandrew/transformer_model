import math
import torch
from torch import Tensor
from torch.nn import Dropout, Module


class PositionalEncoding(Module):
    """位置编码."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        """初始化组件.

        Args:
            d_model: 输入和输出的维度.
            dropout: 丢弃率, 默认0.1.
            max_len: 最大长度, 默认5000.
        """
        super().__init__()
        self.dropout: Dropout = Dropout(dropout)

        pe: Tensor = torch.zeros(max_len, d_model)
        pos: Tensor = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 因为底数是10000, 所以数值膨胀的很快, 容易出现大指数或者小指数
        # 所以使用对数来避免这个数值上溢或下溢
        div_term: Tensor = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """前馈计算.

        Args:
            x: 输入张量, shape: (batch_size, seq_len, d_model).

        Returns:
            返回位置编码后的输出.
        """
        x = x + self.pe[:, : x.size(1), :]  # (batch_size, seq_len, d_model)
        return self.dropout(x)
