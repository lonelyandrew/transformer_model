from torch import Tensor
from torch.nn import Module, Linear

from transformer.attention.utils import scaled_dot_product_attention


class MultiHeadAttention(Module):
    """多头注意力机制."""

    def __init__(self, d_model: int, num_heads: int) -> None:
        """初始化模块.

        Args:
            d_model: 表征向量维度.
            num_heads: 多头注意力机制的头数.
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"

        self.d_model: int = d_model
        self.num_heads: int = num_heads
        self.head_dim: int = d_model // num_heads

        self.w_q: Linear = Linear(d_model, self.head_dim)
        self.w_k: Linear = Linear(d_model, self.head_dim)
        self.w_v: Linear = Linear(d_model, self.head_dim)
        self.w_o: Linear = Linear(d_model, d_model)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> Tensor:
        """前馈计算.

        Args:
            q: 查询张量, shape: (batch_size, seq_len_q, d_model)
            k: 键张量, shape: (batch_size, seq_len_k, d_model)
            v: 值张量, shape: (batch_size, seq_len_v, d_model)
            mask: 注意力掩码, shape: (batch_size, seq_len_q, seq_len_k)

        Returns:
            返回注意力输出.
        """
        batch_size: int = q.shape[0]

        seq_len_q: int = q.shape[1]
        seq_len_k: int = k.shape[1]
        seq_len_v: int = v.shape[1]

        # 线性变换
        q = self.w_q(q)  # (batch_size, seq_len_q, d_model)
        k = self.w_k(k)  # (batch_size, seq_len_k, d_model)
        v = self.w_v(v)  # (batch_size, seq_len_v, d_model)

        # 切分
        # (batch_size, num_heads, seq_len_q/seq_len_k/seq_len_v, head_dim)
        q = q.reshape(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, seq_len_v, self.num_heads, self.head_dim).transpose(1, 2)

        # 多头注意力计算
        # (batch_size, seq_len_q, num_heads, head_dim)
        attention_output, _ = scaled_dot_product_attention(q, k, v, mask)

        # 拼接多头注意力输出, 恢复维度
        # (batch_size, seq_len_q, d_model)
        concat_output: Tensor = attention_output.transpose(1, 2).reshape(batch_size, -1, self.d_model)

        return self.w_o(concat_output)
