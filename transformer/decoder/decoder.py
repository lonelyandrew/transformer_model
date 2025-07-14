from torch import Tensor
from torch.nn import Module, ModuleList
from transformer.decoder.decoder_layer import DecoderLayer
from transformer.layer_norm import LayerNorm


class Decoder(Module):
    """解码器."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, num_layers: int, dropout: float = 0.1) -> None:
        """初始化组件.

        Args:
            d_model: 输入和输出的维度.
            num_heads: 注意力头的数量.
            d_ff: 前馈神经网络的维度.
            num_layers: 解码器层的数量.
            dropout: 丢弃率, 默认0.1.
        """
        super().__init__()
        self.layers: ModuleList = ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm: LayerNorm = LayerNorm(d_model)

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
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
