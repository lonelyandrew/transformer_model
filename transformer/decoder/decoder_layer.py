from torch import Tensor
from torch.nn import Module, ModuleList
from transformer.attention.multi_head_attention import MultiHeadAttention
from transformer.positionwise_feed_forward import PositionwiseFeedForward
from transformer.sublayer_connection import SublayerConnection


class DecoderLayer(Module):
    """解码器层."""

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

        # 编码器-解码器注意力层
        self.cross_attention: MultiHeadAttention = MultiHeadAttention(d_model, num_heads)

        # 前馈神经网络
        self.feed_forward: PositionwiseFeedForward = PositionwiseFeedForward(d_model, d_ff, dropout)

        # 子层连接
        self.sublayer_connections: ModuleList = ModuleList([SublayerConnection(d_model) for _ in range(3)])
        self.d_model: int = d_model

    def forward(self, x: Tensor, memory: Tensor, src_mask: Tensor, tgt_mask: Tensor) -> Tensor:
        """前馈计算.

        Args:
            x: 输入张量, shape: (batch_size, seq_len, d_model).
            memory: 编码器输出, shape: (batch_size, seq_len, d_model).
            src_mask: 源掩码, shape: (batch_size, seq_len, seq_len).
            tgt_mask: 目标掩码, shape: (batch_size, seq_len, seq_len).

        Returns:
            返回解码器层的输出.
        """
        x = self.sublayer_connections[0](x, lambda x: self.self_attention(x, x, x, tgt_mask))
        x = self.sublayer_connections[1](x, lambda x: self.cross_attention(x, memory, memory, src_mask))
        return self.sublayer_connections[2](x, self.feed_forward)
