from torch import Tensor
import torch
from transformer.transformer import Transformer
from transformer.mask import create_padding_mask, create_decoder_mask


def main() -> None:
    # 假设
    batch_size: int = 32
    seq_len_src: int = 10
    seq_len_tgt: int = 15
    d_model: int = 512
    num_heads: int = 8
    d_ff: int = 2048
    num_layers: int = 6
    dropout: float = 0.1
    src_vocab_size: int = 10000
    tgt_vocab_size: int = 10000

    model: Transformer = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        dropout=dropout,
    )

    # 构造输入
    src: Tensor = torch.randint(0, 100, (batch_size, seq_len_src))  # (batch_size, seq_len_src)
    tgt: Tensor = torch.randint(0, 100, (batch_size, seq_len_tgt))  # (batch_size, seq_len_tgt)

    # 获取掩码用于打印编码器和解码器的输出
    src_mask: Tensor = create_padding_mask(src)
    tgt_mask: Tensor = create_decoder_mask(tgt)

    # 模型最终输出
    output: Tensor = model(src, tgt)

    # 打印各部分的输出形状
    print("Source embedding shape:", model.src_embeddings(src).shape)  # (batch_size, seq_len_src, d_model)
    print(
        "Encoder output shape:", model.encoder(model.src_embeddings(src), src_mask).shape
    )  # (batch_size, seq_len_src, d_model)
    print("Target embedding shape:", model.tgt_embeddings(tgt).shape)  # (batch_size, seq_len_tgt, d_model)
    print(
        "Decoder output shape:",
        model.decoder(
            model.tgt_embeddings(tgt), model.encoder(model.src_embeddings(src), src_mask), src_mask, tgt_mask
        ).shape,
    )  # (batch_size, seq_len_tgt, d_model)
    print("Final output shape:", output.shape)  # (batch_size, seq_len_tgt, tgt_vocab_size)


if __name__ == "__main__":
    main()
