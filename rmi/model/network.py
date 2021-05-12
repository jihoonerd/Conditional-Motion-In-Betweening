import torch
import torch.nn as nn
from rmi.model.plu import PLU


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
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        regular_gan_out = self.sigmoid(self.regular_gan(x))
        q_out = self.infogan_q(x) # TODO: Discriminator 설계 윈도우로 한번에 처리하는게 맞을까.
        # appendix보면 위치정보 기반으로 윈도우 한번에 평균내버림 코드에 의해 만들어진 모션의 집합이니깐 그냥 써도 괜찮지 않을까?
        # 그러면 discriminator loss BCE는 어떻게 처리하지. LSGAN은 평균내서 square해버린다.
        # 모션 전체에 대해서 평균 후 판단. 각각의 윈도우에서 gan out이 있는데 전체 평균
        # 대응해보면 각각의 infogan out에 대ㅐ서 전체 평균, 여기서는 BCE의 평균을 해야하지 않을까
        # [IMPORTANT] 야 discriminator 건드릴 필요가 있나? 그냥 gan loss쓰면되고 generator loss만 반영하면 될 것 같은데? 확인해보자.
        q_out_mean = torch.mean(q_out, dim=2)
        q_discrete = self.softmax(q_out_mean)
        return regular_gan_out, q_discrete
