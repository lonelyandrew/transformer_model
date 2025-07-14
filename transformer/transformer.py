from torch import Tensor
from torch.nn import Linear, Module
from transformer.encoder import Encoder
from transformer.decoder import Decoder
from transformer.embedding import SourceEmbedding, TargetEmbedding
from transformer.mask import create_padding_mask, create_decoder_mask


class Transformer(Module):
    """Transformer模型."""

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        dropout: float = 0.1,
    ) -> None:
        """初始化组件.

        Args:
            src_vocab_size: 源语言词汇表大小.
            tgt_vocab_size: 目标语言词汇表大小.
            d_model: 输入和输出的维度.
            num_heads: 注意力头的数量.
            d_ff: 前馈神经网络的维度.
            num_layers: 编码器和解码器层的数量.
            dropout: 丢弃率, 默认0.1.
        """
        super().__init__()

        self.src_embeddings: SourceEmbedding = SourceEmbedding(d_model, src_vocab_size)
        self.tgt_embeddings: TargetEmbedding = TargetEmbedding(d_model, tgt_vocab_size)
        self.encoder: Encoder = Encoder(d_model, num_heads, d_ff, num_layers, dropout)
        self.decoder: Decoder = Decoder(d_model, num_heads, d_ff, num_layers, dropout)
        self.fc_out: Linear = Linear(d_model, tgt_vocab_size)

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        """前馈计算.

        Args:
            src: 源语言序列, shape: (batch_size, seq_len).
            tgt: 目标语言序列, shape: (batch_size, seq_len).
        """
        src_mask: Tensor = create_padding_mask(src)
        tgt_mask: Tensor = create_decoder_mask(tgt)

        encoder_output: Tensor = self.encoder(self.src_embeddings(src), src_mask)
        decoder_output: Tensor = self.decoder(self.tgt_embeddings(tgt), encoder_output, src_mask, tgt_mask)
        output: Tensor = self.fc_out(decoder_output)
        return output
