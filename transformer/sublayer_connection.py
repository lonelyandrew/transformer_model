from torch import Tensor
from torch.nn import Module
from transformer.layer_norm import LayerNorm
from transformer.residual_connection import ResidualConnection


class SublayerConnection(Module):
    """子层连接."""

    def __init__(self, d_model: int, epsilon: float = 1e-9, dropout: float = 0.1) -> None:
        """初始化组件.

        Args:
            d_model: 输入和输出的维度.
            epsilon: 防止除零的小常数, 默认1e-9.
            dropout: 丢弃率, 默认0.1.
        """
        super().__init__()
        self.residual: ResidualConnection = ResidualConnection(dropout)
        self.norm: LayerNorm = LayerNorm(d_model, epsilon)

    def forward(self, x: Tensor, sublayer: Module) -> Tensor:
        """前馈计算.

        Args:
            x: 输入张量, shape: (batch_size, seq_len, d_model).
            sublayer: 子层, 通常是注意力层或前馈层.

        Returns:
            返回子层连接后的输出.
        """
        return self.norm(self.residual(x, sublayer))
