import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from cmib.model.positional_encoding import PositionalEmbedding


class TransformerModel(nn.Module):
    def __init__(
        self,
        seq_len: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        dropout: float = 0.5,
        out_dim=91,
        num_labels=15
    ):
        super().__init__()
        self.model_type = "Transformer"
        self.seq_len = seq_len
        self.d_model = d_model
        self.nhead = nhead
        self.d_hid = d_hid
        self.nlayers = nlayers
        self.cond_emb = nn.Embedding(num_labels, d_model)
        self.pos_embedding = PositionalEmbedding(seq_len=seq_len, d_model=d_model)
        encoder_layers = TransformerEncoderLayer(
            d_model, nhead, d_hid, dropout, activation="gelu"
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.decoder = nn.Linear(d_model, out_dim)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor, cond_code: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size, embedding_dim]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, embedding_dim]
        """
        cond_embedding = self.cond_emb(cond_code).permute(1, 0, 2)
        output = self.pos_embedding(src)
        output = torch.cat([cond_embedding, output], dim=0)
        output = self.transformer_encoder(output, src_mask)
        output = self.decoder(output)
        return output, cond_embedding
