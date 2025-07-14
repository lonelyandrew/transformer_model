from torch import Tensor
from torch.nn import LayerNorm, Module, ModuleList
from transformer.encoder.encoder_layer import EncoderLayer


class Encoder(Module):
    """编码器."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, num_layers: int, dropout: float = 0.1) -> None:
        """初始化组件.

        Args:
            d_model: 输入和输出的维度.
            num_heads: 注意力头的数量.
            d_ff: 前馈神经网络的维度.
            num_layers: 编码器层的数量.
            dropout: 丢弃率, 默认0.1.
        """
        super().__init__()
        self.layers: ModuleList = ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm: LayerNorm = LayerNorm(d_model)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """前馈计算.

        Args:
            x: 输入张量, shape: (batch_size, seq_len, d_model).
            mask: 掩码, shape: (batch_size, seq_len, seq_len).

        Returns:
            返回编码器层的输出.
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
