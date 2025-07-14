from torch.nn import Module, Dropout
from torch import Tensor


class ResidualConnection(Module):
    """残差连接."""

    def __init__(self, dropout: float = 0.1) -> None:
        """初始化组件.

        Args:
            dropout: 丢弃率, 默认0.1.
        """
        super().__init__()
        self.dropout: Dropout = Dropout(dropout)

    def forward(self, x: Tensor, sublayer: Module) -> Tensor:
        """前馈计算.

        Args:
            x: 输入张量, shape: (batch_size, seq_len, d_model).
            sublayer: 子层, 通常是注意力层或前馈层.

        Returns:
            返回残差连接后的输出.
        """
        return x + self.dropout(sublayer(x))
