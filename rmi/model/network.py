import torch
from torch import nn, Tensor
from torch.nn.modules.activation import ReLU
from torch.nn.modules.transformer import TransformerDecoderLayer
from rmi.model.plu import PLU
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from rmi.model.positional_encoding import PositionalEmbedding
import math
from torch.nn import Transformer



class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 dim_feedforward: int = 512,
                 out_dim: int=91,
                 dropout: float = 0.1,
                 bottleneck_dim: int=256):
        super(Seq2SeqTransformer, self).__init__()

        self.src_pos_emb = PositionalEmbedding(d_model=emb_size)
        encoder_layers = TransformerEncoderLayer(emb_size, nhead, dim_feedforward, dropout, activation='relu')
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        self.encoder_bottleneck1 = nn.Linear(3072, 1024)
        self.encoder_bottleneck2 = nn.Linear(1024, bottleneck_dim)

        self.decoder_bottleneck1 = nn.Linear(bottleneck_dim, 1024)
        self.decoder_bottleneck2 = nn.Linear(1024, 3072)
        self.trg_pos_emb= PositionalEmbedding(d_model=emb_size)
        decoder_layers = TransformerDecoderLayer(emb_size, nhead, dim_feedforward, dropout, activation='relu')
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_decoder_layers)


        self.generator = nn.Linear(emb_size, out_dim)


    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor=None,
                tgt_mask: Tensor=None,
                src_padding_mask: Tensor=None,
                tgt_padding_mask: Tensor=None,
                memory_key_padding_mask: Tensor=None):
        src_emb = self.src_pos_emb(src)
        tf_enc_out = self.transformer_encoder(src_emb)
        output = torch.flatten(tf_enc_out.permute(1,0,2), start_dim=1)
        output = F.relu(self.encoder_bottleneck1(output))
        output = self.encoder_bottleneck2(output)

        output = F.relu(self.decoder_bottleneck1(output))
        output = self.decoder_bottleneck2(output)
        output = output.reshape(src_emb.shape)
        output = self.trg_pos_emb(output)
        output = self.transformer_decoder(output, tf_enc_out)
        output = self.generator(output)
        return output

    def encode(self, src: Tensor):
        src_emb = self.src_pos_emb(src)
        tf_enc_out = self.transformer_encoder(src_emb)
        output = torch.flatten(tf_enc_out.permute(1,0,2), start_dim=1)
        output = F.relu(self.encoder_bottleneck1(output))
        output = self.encoder_bottleneck2(output)
        return output

    def decode(self, context_vector: Tensor, memory: Tensor, tgt_mask: Tensor):
        output = F.relu(self.decoder_bottleneck1(context_vector))
        output = self.decoder_bottleneck2(output)
        output = output.reshape(memory.shape)
        output = self.trg_pos_emb(output)
        output = self.transformer_decoder(output, memory)
        output = self.generator(output)
        return output


def compute_gradient_penalty(D, real_samples, fake_samples, src_mask, phi=1.0):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand((1, real_samples.size(1), 1), device=real_samples.get_device())
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates, src_mask)
    fake = torch.ones([real_samples.shape[1], 1], requires_grad=False, device=real_samples.get_device())
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.contiguous()
    gradient_penalty = ((gradients.norm(2, dim=(0,2)) - phi) ** 2).mean()
    return gradient_penalty


class LinearProjection(nn.Module):

    def __init__(self, latent_dim:1024, d_model=95, seq_len=32):
        super().__init__()
        self.seq_len=32
        self.projection = nn.Linear(latent_dim, d_model * seq_len)

    def forward(self, noise):
        x = self.projection(noise)
        x = x.reshape(self.seq_len, x.shape[0], -1)
        return x


        
class TransformerGenerator(nn.Module):

    def __init__(self, latent_dim: int, seq_len: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5, out_dim=91, device='cpu'):
        super().__init__()
        self.model_type = 'TransformerGenerator'
        self.seq_len = seq_len
        self.projection = nn.Linear(latent_dim, d_model * seq_len)
        self.pos_embedding= PositionalEmbedding(d_model=d_model)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, activation='relu')
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, out_dim)

    def forward(self, noise: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size, embedding_dim]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, embedding_dim]
        """
        x = F.relu(self.projection(noise))
        x = x.reshape(self.seq_len, x.shape[0], -1)
        src = self.pos_embedding(x)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output

class TransformerDiscriminator(nn.Module):

    def __init__(self, seq_len: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5, out_dim=91, device='cpu'):
        super().__init__()
        self.model_type = 'TransformerDiscriminator'
        self.seq_len = seq_len
        self.pos_embedding= PositionalEmbedding(d_model=d_model)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, activation='relu')
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, out_dim)

        self.motion_discriminator = MotionDiscriminator()


    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size, embedding_dim]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, embedding_dim]
        """
        src = self.pos_embedding(src)
        output = self.transformer_encoder(src, src_mask)
        output = F.leaky_relu(self.decoder(output).permute(1,2,0))
        output = self.motion_discriminator(output)
        return output


class TransformerModel(nn.Module):

    def __init__(self, seq_len: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5, out_dim=91, device='cpu'):
        super().__init__()
        self.model_type = 'Transformer'
        self.seq_len = seq_len
        self.pos_embedding= PositionalEmbedding(seq_len=seq_len, d_model=d_model)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, activation='gelu')
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model
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
        output = self.pos_embedding(src, cond_code)
        output = self.transformer_encoder(output, src_mask)
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


class MotionDiscriminator(nn.Module):

    def __init__(self, in_channels=95, out_channels=32, out_dim=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1d_1 = nn.Conv1d(self.in_channels, 128, kernel_size=5)
        self.conv1d_2 = nn.Conv1d(128, out_channels, kernel_size=3)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(832, out_dim)

    def forward(self, x):
        x = F.leaky_relu(self.conv1d_1(x))
        x = F.leaky_relu(self.conv1d_2(x))
        x = self.flatten(x)
        x = self.linear1(x)
        return x
        

class DInfoGAN(nn.Module):
    def __init__(self, input_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.conv_to_gan = nn.Linear(input_dim, 1)
    def forward(self, x):
        out = self.conv_to_gan(x)
        return out

class QInfoGAN(nn.Module):
    def __init__(self, input_dim=128, discrete_code_dim=4, continuous_code_dim=2):
        super().__init__()
        self.input_dim = input_dim
        self.discrete_code_dim = discrete_code_dim
        self.continuous_code_dim = continuous_code_dim

        self.conv_to_infogan_discrete = nn.Sequential(
            nn.Linear(input_dim, self.discrete_code_dim)
        )

        if self.continuous_code_dim > 0:
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

        if self.continuous_code_dim > 0:
            mu = self.conv_to_infogan_continuous_mu(x)
            var = torch.exp(self.conv_to_infogan_continuous_var(x))
        else:
            mu, var = None, None
        return q_discrete, mu, var

class InfoGANCRH(nn.Module):

    def __init__(self, input_dim=95, hidden_dim=32, out_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.conv1d_1 = nn.Conv1d(self.input_dim, self.hidden_dim, kernel_size=3)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(896, 128)
        self.linear2 = nn.Linear(128, out_dim)

    def forward(self, x):
        x = F.leaky_relu(self.conv1d_1(x))
        x = self.flatten(x)
        x = F.leaky_relu(self.linear1(x))
        x = self.linear2(x)
        return x