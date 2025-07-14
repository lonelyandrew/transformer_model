from torch.nn import Module, Embedding
from torch import Tensor


class Embeddings(Module):
    """词嵌入层."""

    def __init__(self, d_model: int, vocab_size: int) -> None:
        """初始化组件.

        Args:
            d_model: 输入和输出的维度.
            vocab_size: 词表大小.
        """
        super().__init__()
        self.embeddings: Embedding = Embedding(vocab_size, d_model)
        self.scale_factor: float = d_model**0.5

    def forward(self, x: Tensor) -> Tensor:
        """前馈计算.

        Args:
            x: 输入张量, shape: (batch_size, seq_len).

        Returns:
            返回词嵌入后的输出.
        """
        return self.embeddings(x) * self.scale_factor
