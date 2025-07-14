import math

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Module, Linear


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

    @classmethod
    def scaled_dot_product_attention(
        cls, q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """缩放点积注意力机制计算.

        Args:
            q: 查询张量, shape: (batch_size, seq_len_q, d_model)
            k: 键张量, shape: (batch_size, seq_len_k, d_model)
            v: 值张量, shape: (batch_size, seq_len_v, d_model)
            mask: 注意力掩码, shape: (batch_size, seq_len_q, seq_len_k)

        Returns:
            返回注意力输出和注意力权重.
        """
        d_model: int = q.shape[-1]

        # 缩放点积计算
        # (batch_size, seq_len_q, seq_len_k)
        scores: Tensor = torch.bmm(q, k.transpose(-2, -1)) / math.sqrt(d_model)

        # 掩码填充-inf
        # (batch_size, seq_len_q, seq_len_k)
        # 填充-inf是因为后面要进行Softmax计算, -inf对应的结果是0
        # 掩码应用于两种情况:
        #   1. 生成任务中的下文掩码
        #   2. batch处理中对变长序列的padding位掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Softmax激活函数
        # (batch_size, seq_len_q, seq_len_k)
        attention_weights: Tensor = F.softmax(scores, dim=-1)

        # 加权求和, 得到输出
        # (batch_size, seq_len_q, d_model)
        output: Tensor = torch.bmm(attention_weights, v)

        return output, attention_weights

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

        return self.scaled_dot_product_attention(q, k, v, mask)
