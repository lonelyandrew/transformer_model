from torch import Tensor
import torch


def create_look_ahead_mask(size: int) -> Tensor:
    """创建前瞻掩码.

    Args:
        size: 掩码大小.

    Returns:
        返回前瞻掩码.
    """
    mask: Tensor = torch.tril(torch.ones(size, size)).type(torch.bool)  # (size, size)
    return mask


def create_padding_mask(seq: Tensor, pad_token_id: int = 0) -> Tensor:
    """创建填充掩码.

    Args:
        seq: 输入张量, shape: (batch_size, seq_len).
        pad_token_id: 填充标记的ID, 默认0.

    Returns:
        返回填充掩码.
    """
    # 将填充标记的ID转换为True, 其他为False
    # (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)
    return (seq == pad_token_id).unsqueeze(1).unsqueeze(2)


def create_decoder_mask(target_seq: Tensor, pad_token_id: int = 0) -> Tensor:
    """创建解码器掩码.

    Args:
        target_seq: 目标序列, shape: (batch_size, seq_len).
        pad_token_id: 填充标记的ID, 默认0.

    Returns:
        返回解码器掩码.
    """
    padding_mask: Tensor = create_padding_mask(target_seq, pad_token_id)
    look_ahead_mask: Tensor = create_look_ahead_mask(target_seq.size(1))
    return look_ahead_mask.unsqueeze(0) & padding_mask
