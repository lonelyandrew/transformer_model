from torch import Tensor
from torch.nn import Module, Linear

from transformer.attention.utils import scaled_dot_product_attention


class Attention(Module):
    """单头注意力机制."""

    def __init__(self, d_model: int) -> None:
        """初始化模块.

        Args:
            d_model: 表征向量维度.
        """
        super().__init__()
        self.d_model: int = d_model

        # 对Q, K, V分别定义一个线性变换层
        self.w_q: Linear = Linear(d_model, d_model)
        self.w_k: Linear = Linear(d_model, d_model)
        self.w_v: Linear = Linear(d_model, d_model)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """前馈计算.

        Args:
            q: 查询张量, shape: (batch_size, seq_len_q, d_model)
            k: 键张量, shape: (batch_size, seq_len_k, d_model)
            v: 值张量, shape: (batch_size, seq_len_v, d_model)
            mask: 注意力掩码, shape: (batch_size, seq_len_q, seq_len_k)

        Returns:
            返回注意力输出和注意力权重.
        """
        q = self.w_q(q)  # (batch_size, seq_len_q, d_model)
        k = self.w_k(k)  # (batch_size, seq_len_k, d_model)
        v = self.w_v(v)  # (batch_size, seq_len_v, d_model)

        return scaled_dot_product_attention(q, k, v, mask)
