from torch.nn import Module
from torch import Tensor

from transformer.attention import Attention


class CrossAttention(Module):
    """交叉注意力机制."""

    def __init__(self, d_model: int) -> None:
        """初始化模块.

        Args:
            d_model: 表征向量维度.
        """
        super().__init__()
        self.attention: Attention = Attention(d_model)

    def forward(self, q: Tensor, kv: Tensor, mask: Tensor | None = None) -> Tensor:
        """前馈计算.

        Args:
            q: 查询张量, shape: (batch_size, seq_len_q, d_model)
            kv: 键值对张量, shape: (batch_size, seq_len_kv, d_model)
            mask: 注意力掩码, shape: (batch_size, seq_len_q, seq_len_kv)

        Returns:
            返回注意力输出.
        """
        return self.attention(q, kv, kv, mask)
