from torch import Tensor
from torch.nn import Module, ModuleList
from transformer.attention.multi_head_attention import MultiHeadAttention
from transformer.positionwise_feed_forward import PositionwiseFeedForward
from transformer.sublayer_connection import SublayerConnection


class EncoderLayer(Module):
    """编码器层."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        """初始化组件.

        Args:
            d_model: 输入和输出的维度.
            num_heads: 注意力头的数量.
            d_ff: 前馈神经网络的维度.
            dropout: 丢弃率, 默认0.1.
        """
        super().__init__()
        # 自注意力层
        self.self_attention: MultiHeadAttention = MultiHeadAttention(d_model, num_heads)
        # 前馈神经网络
        self.feed_forward: PositionwiseFeedForward = PositionwiseFeedForward(d_model, d_ff)
        # 子层连接
        self.sublayer_connections: ModuleList = ModuleList([SublayerConnection(d_model) for _ in range(2)])
        self.d_model: int = d_model

    def forward(self, x: Tensor, src_mask: Tensor) -> Tensor:
        """前馈计算.

        Args:
            x: 输入张量, shape: (batch_size, seq_len, d_model).
            src_mask: 掩码, shape: (batch_size, seq_len, seq_len).

        Returns:
            返回编码器层的输出.
        """
        x = self.sublayer_connections[0](x, lambda x: self.self_attention(x, x, x, src_mask))
        x = self.sublayer_connections[1](x, self.feed_forward)
        return x
