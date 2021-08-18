import torch
from torch import nn, Tensor
from torch.nn.modules.activation import ReLU
from rmi.model.plu import PLU
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from rmi.model.positional_encoding import PositionalEncoding
import math



class TransformerModel(nn.Module):

    def __init__(self, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, 91)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


class InputEncoder(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=512, out_dim=256):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = PLU(x)
        x = self.fc2(x)
        x = PLU(x)
        return x


class InfoganCodeEncoder(nn.Module):
    def __init__(self, input_dim, out_dim):
        super().__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim

        self.fc1 = nn.Linear(input_dim, out_dim//2)
        self.fc2 = nn.Linear(out_dim//2, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = PLU(x)
        x = self.fc2(x)
        return x


class LSTMNetwork(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256 * 3, num_layer=1, device="cpu"):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layer)

        self.device = device

    def init_hidden(self, batch_size):
        self.h = torch.zeros(
            (self.num_layer, batch_size, self.hidden_dim), device=self.device
        )
        self.c = torch.zeros(
            (self.num_layer, batch_size, self.hidden_dim), device=self.device
        )

    def forward(self, x):
        x, (self.h, self.c) = self.lstm(x, (self.h, self.c))
        return x


class Decoder(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=512, out_dim=256):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2, bias=True)
        self.fc3 = nn.Linear(hidden_dim // 2, out_dim - 4, bias=True)
        self.fc_contact = nn.Linear(hidden_dim // 2, 4, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = PLU(x)
        x = self.fc2(x)
        x = PLU(x)
        hidden_out = self.fc3(x)
        contact = self.fc_contact(x)
        contact_out = self.sigmoid(contact)
        return hidden_out, contact_out


class Discriminator(nn.Module):
    def __init__(self, input_dim, out_dim):
        super().__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim

        self.disc_block = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.out_dim),
        )

    def forward(self, x):
        x = self.disc_block(x)
        return x

class NDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim

        self.regular_gan = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 1)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.regular_gan(x)
        return x

class QDiscriminator(nn.Module):
    def __init__(self, input_dim, discrete_code_dim):
        super().__init__()
        self.input_dim = input_dim
        self.discrete_code_dim = discrete_code_dim

        self.infogan_q = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, self.discrete_code_dim)
        )

    def forward(self, x):
        q = self.infogan_q(x)
        return q



class InfoGANDiscriminator(nn.Module):
    # refer: 3.5 Motion discriminators, 3.7.2 sliding critics
    def __init__(self, input_dim=128, hidden_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.conv1d_1 = nn.Conv1d(self.input_dim, self.hidden_dim, kernel_size=1, padding=0, dilation=1)
        self.conv1d_31 = nn.Conv1d(self.input_dim, self.hidden_dim, kernel_size=3, padding=1, dilation=1)
        self.conv1d_32 = nn.Conv1d(self.input_dim, self.hidden_dim, kernel_size=3, padding=2, dilation=2)
        self.conv1d_33 = nn.Conv1d(self.input_dim, self.hidden_dim, kernel_size=3, padding=3, dilation=3)
        self.conv1d_35 = nn.Conv1d(self.input_dim, self.hidden_dim, kernel_size=3, padding=5, dilation=5)
        self.conv1d_1x1 = nn.Conv1d(5 * self.hidden_dim, 1, kernel_size=1)

    def forward(self, x):
        out_conv1d_1 = F.relu(self.conv1d_1(x))
        out_conv1d_31 = F.relu(self.conv1d_31(x))
        out_conv1d_32 = F.relu(self.conv1d_32(x))
        out_conv1d_33 = F.relu(self.conv1d_33(x))
        out_conv1d_35 = F.relu(self.conv1d_35(x))

        conv_out = torch.cat([out_conv1d_1, out_conv1d_31, out_conv1d_32, out_conv1d_33, out_conv1d_35], dim=1)
        out = self.conv1d_1x1(conv_out)
        return out


class DInfoGAN(nn.Module):
    def __init__(self, input_dim=30):
        super().__init__()
        self.input_dim = input_dim

        self.conv_to_gan = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, 1)
        )

    def forward(self, x):
        out = self.conv_to_gan(x)
        return out

class QInfoGAN(nn.Module):
    def __init__(self, input_dim=30, discrete_code_dim=4, continuous_code_dim=2):
        super().__init__()
        self.input_dim = input_dim
        self.discrete_code_dim = discrete_code_dim
        self.continuous_code_dim = continuous_code_dim

        self.conv_to_infogan_discrete = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, self.discrete_code_dim)
        )

        self.conv_to_infogan_continuous_mu = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, self.continuous_code_dim)
        )

        self.conv_to_infogan_continuous_var = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, self.continuous_code_dim)
        )


    def forward(self, x):
        q_discrete = self.conv_to_infogan_discrete(x)

        mu = self.conv_to_infogan_continuous_mu(x)
        var = torch.exp(self.conv_to_infogan_continuous_var(x))
        return q_discrete, mu, var