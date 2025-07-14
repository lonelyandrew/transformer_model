from torch.nn import Module, Linear
from torch import Tensor


class PositionwiseFeedForward(Module):
    """位置前馈网络(Position-wise Feed-Forward Network).

    Args:
        d_model: 输入和输出的维度.
        d_ff: 中间层的维度.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        """初始化组件.

        Args:
            d_model: 输入和输出的维度.
            d_ff: 中间层的维度.
            dropout: 丢弃率, 默认0.1.
        """
        super().__init__()
        self.w_1: Linear = Linear(d_model, d_ff)
        self.w_2: Linear = Linear(d_ff, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """前馈计算.

        Args:
            x: 输入张量, shape: (batch_size, seq_len, d_model).
        """
        return self.w_2(self.w_1(x).relu())
