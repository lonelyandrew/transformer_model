from torch.nn import Module
from torch import Tensor

from transformer.embedding.embeddings import Embeddings
from transformer.embedding.positional_encoding import PositionalEncoding


class TargetEmbedding(Module):
    """目标语言嵌入层."""

    def __init__(self, d_model: int, vocab_size: int, dropout: float = 0.1) -> None:
        """初始化组件.

        Args:
            d_model: 输入和输出的维度.
            vocab_size: 词汇表大小.
            dropout: 丢弃率, 默认0.1.
        """
        super().__init__()
        self.embeddings: Embeddings = Embeddings(d_model, vocab_size)
        self.positional_encoding: PositionalEncoding = PositionalEncoding(d_model, dropout)

    def forward(self, x: Tensor) -> Tensor:
        """前馈计算.

        Args:
            x: 输入张量, shape: (batch_size, seq_len).
        """
        # 词嵌入 + 位置编码
        # (batch_size, seq_len_tgt, d_model)
        return self.positional_encoding(self.embeddings(x))
