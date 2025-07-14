from torch.nn import Module
from torch import Tensor

from transformer.attention import Attention


class SelfAttention(Module):
    """自注意力机制."""

    def __init__(self, d_model: int) -> None:
        """初始化模块.

        Args:
            d_model: 表征向量维度.
        """
        super().__init__()
        self.attention: Attention = Attention(d_model)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """前馈计算.

        Args:
            x: 输入张量, shape: (batch_size, seq_len, d_model)
            mask: 注意力掩码, shape: (batch_size, seq_len, seq_len)

        Returns:
            返回注意力输出.
        """
        return self.attention(x, x, x, mask)
