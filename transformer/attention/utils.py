import math

import torch
from torch import Tensor
import torch.nn.functional as F


def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """缩放点积注意力机制计算.

    Args:
        q: 查询张量, shape: (batch_size, num_heads, seq_len_q, d_model)
        k: 键张量, shape: (batch_size, num_heads, seq_len_k, d_model)
        v: 值张量, shape: (batch_size, num_heads, seq_len_v, d_model)
        mask: 注意力掩码, shape:
            - (1, 1, seq_len_q, seq_len_k) 用于decoder的masked self-attention
            - (batch_size, 1, seq_len_q, seq_len_k) 用于encoder-decoder的cross-attention
            - (batch_size, num_heads, seq_len_q, seq_len_k) 用于encoder的self-attention

    Returns:
        返回注意力输出和注意力权重.
    """
    d_model: int = q.shape[-1]

    # 缩放点积计算
    # (batch_size, num_heads, seq_len_q, seq_len_k)
    scores: Tensor = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_model)

    # 掩码填充-inf
    # (batch_size, num_heads, seq_len_q, seq_len_k)
    # 填充-inf是因为后面要进行Softmax计算, -inf对应的结果是0
    # 掩码应用于两种情况:
    #   1. 生成任务中的下文掩码
    #   2. batch处理中对变长序列的padding位掩码
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # Softmax激活函数
    # (batch_size, num_heads, seq_len_q, seq_len_k)
    attention_weights: Tensor = F.softmax(scores, dim=-1)

    # 加权求和, 得到输出
    # (batch_size, num_heads, seq_len_q, d_model)
    output: Tensor = torch.matmul(attention_weights, v)

    return output, attention_weights
