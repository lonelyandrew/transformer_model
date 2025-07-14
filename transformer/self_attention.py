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
