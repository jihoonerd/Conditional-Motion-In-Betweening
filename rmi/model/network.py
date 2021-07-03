import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU
from rmi.model.plu import PLU


class InputEncoder(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=512, out_dim=256):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.fc3 = nn.Linear(hidden_dim, out_dim, bias=True)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
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
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim // 2, bias=True)
        self.fc5 = nn.Linear(hidden_dim // 2, out_dim - 4, bias=True)
        self.fc_contact = nn.Linear(hidden_dim // 2, 4, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        hidden_out = self.fc5(x)
        contact = self.fc_contact(x)
        contact_out = self.sigmoid(contact)
        return hidden_out, contact_out


class Discriminator(nn.Module):
    # refer: 3.5 Motion discriminators, 3.7.2 sliding critics
    def __init__(self, input_dim=128, hidden_dim=128, out_dim=1, length=3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.length = length

        self.fc1 = nn.Conv1d(
            self.input_dim, self.hidden_dim, kernel_size=self.length, bias=True
        )
        self.fc2 = nn.Conv1d(
            self.hidden_dim, self.hidden_dim // 2, kernel_size=1, bias=True
        )
        self.fc3 = nn.Conv1d(self.hidden_dim // 2, out_dim, kernel_size=1, bias=True)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

class SinglePoseDiscriminator(nn.Module):
    def __init__(self, input_dim, discrete_code_dim):
        super().__init__()
        self.input_dim = input_dim
        self.discrete_code_dim =  discrete_code_dim

        self.single_pose_disc = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim),
            nn.ReLU(),
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )

        self.regular_gan = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 1)
        )

        self.infogan_q = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, self.discrete_code_dim)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        discriminator_out = self.single_pose_disc(x)
        regular_gan_out = self.sigmoid(self.regular_gan(discriminator_out))

        q_out = self.infogan_q(discriminator_out)
        return regular_gan_out, q_out


class InfoGANDiscriminator(nn.Module):
    # refer: 3.5 Motion discriminators, 3.7.2 sliding critics
    def __init__(self, input_dim=128, hidden_dim=128, discrete_code_dim=4, out_dim=1, length=3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.discrete_code_dim = discrete_code_dim
        self.length = length

        self.regular_gan = nn.Sequential(
            nn.Conv1d(self.input_dim, self.hidden_dim, kernel_size=self.length),
            nn.ReLU(),
            nn.Conv1d(self.hidden_dim, self.hidden_dim // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(self.hidden_dim // 2, out_dim, kernel_size=1),
        )

        self.infogan_q = nn.Sequential(
            nn.Conv1d(self.input_dim, self.hidden_dim, kernel_size=self.length),
            nn.LeakyReLU(0.1),
            nn.Conv1d(self.hidden_dim, self.hidden_dim // 2, kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(self.hidden_dim // 2, discrete_code_dim, kernel_size=1),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        regular_gan_out = self.sigmoid(self.regular_gan(x))
        q_out = self.infogan_q(x)
        q_discrete = torch.mean(q_out, dim=2)  # TODO: validate this. Is it okay to take mean for in-window data?
        return regular_gan_out, q_discrete
