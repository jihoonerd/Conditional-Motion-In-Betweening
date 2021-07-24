import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU
from rmi.model.plu import PLU
import torch.nn.functional as F


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
        x = self.sigmoid(self.regular_gan(x))
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
    def __init__(self, input_dim=128, hidden_dim=128, discrete_code_dim=4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.discrete_code_dim = discrete_code_dim

        self.conv1d_1 = nn.Conv1d(self.input_dim, self.hidden_dim, kernel_size=1)
        self.conv1d_32 = nn.Conv1d(self.input_dim, self.hidden_dim, kernel_size=3, padding=2, dilation=2)
        self.conv1d_33 = nn.Conv1d(self.input_dim, self.hidden_dim, kernel_size=3, padding=3, dilation=3)
        self.conv1d_35 = nn.Conv1d(self.input_dim, self.hidden_dim, kernel_size=3, padding=5, dilation=5)
        self.conv1d_1x1 = nn.Conv1d(4 * self.hidden_dim, 1, kernel_size=1)

        self.conv_to_gan = nn.Linear(30, 1)
        self.sigmoid = nn.Sigmoid()

        self.conv_to_infogan = nn.Sequential(
            nn.Linear(30, 16),
            nn.ReLU(),
            nn.Linear(16, self.discrete_code_dim)
        )

    def forward(self, x):
        out_conv1d_1 = F.relu(self.conv1d_1(x))
        out_conv1d_32 = F.relu(self.conv1d_32(x))
        out_conv1d_33 = F.relu(self.conv1d_33(x))
        out_conv1d_35 = F.relu(self.conv1d_35(x))

        out = torch.cat([out_conv1d_1, out_conv1d_32, out_conv1d_33, out_conv1d_35], dim=1)
        conv_out = self.conv1d_1x1(out)
        gan_out = self.conv_to_gan(conv_out[:,0,:])
        regular_gan_out = self.sigmoid(gan_out)

        q_discrete = self.conv_to_infogan(conv_out[:,0,:])
        return regular_gan_out, q_discrete