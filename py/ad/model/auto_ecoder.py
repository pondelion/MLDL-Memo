import math
import torch
import torch.nn as nn
import numpy as np
from .. import DEVICE


class AutoEncoder(nn.Module):

    def __init__(self, input_dim, output_dim=2):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Linear(64, 12),
            nn.BatchNorm1d(12),
            nn.ReLU(True),
            nn.Linear(12, output_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(output_dim, 12),
            nn.BatchNorm1d(12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, input_dim),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def square_loss(self, x):
        if len(x.shape) == 1:
            x = np.expand_dims(x, axis=0)

        x = torch.stack([torch.from_numpy(d).float().to(DEVICE) for d in x])

        reconst = self.forward(x).detach().cpu().numpy()
        x = np.array(x)

        return ((x - reconst)**2).sum(axis=1)


class CNNAutoEncoder(nn.Module):

    def __init__(self, input_dim=31, output_dim=2, in_channels=1):
        super(CNNAutoEncoder, self).__init__()

        self._encoder1 = nn.Sequential(
            nn.Conv1d(in_channels, 16, 3, 2, 1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv1d(16, 16*2, 4, 2, 1),
            nn.BatchNorm1d(16*2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv1d(16*2, 16*4, 4, 2, 1),
            nn.BatchNorm1d(16*4),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv1d(16*4, 16*8, 4, 2, 1),
            nn.BatchNorm1d(16*8),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv1d(16*8, output_dim, 4, 2, 1),
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU(negative_slope=0.1),
        )

        self._decorder1 = nn.Sequential(
            nn.ConvTranspose1d(output_dim, 16*8, 4, 2, 1),
            nn.BatchNorm1d(16*8),
            nn.ReLU(),
            nn.ConvTranspose1d(16*8, 16*4, 4, 2, 1),
            nn.BatchNorm1d(16*4),
            nn.ReLU(),
            nn.ConvTranspose1d(16*4, 16*2, 4, 2, 1),
            nn.BatchNorm1d(16*2),
            nn.ReLU(),
            nn.ConvTranspose1d(16*2, 16, 4, 2, 1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, 3, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def encode(self, x):
        x = self._encoder1(x)
        return x

    def decode(self, x):
        x = self._decorder1(x)
        return x

    def reduce(self, x):
        return self.encode(x).squeeze(-1)

    def square_loss(self, x):
        if len(x.shape) == 1:
            # (N) → (B(1), C(1), N)
            x = np.expand_dims(x, axis=0)
            x = np.expand_dims(x, axis=0)
        elif len(x.shape) == 2:
            # (B, N) → (B, C(1), N)
            x = np.expand_dims(x, axis=1)

        x = torch.stack([torch.from_numpy(d).float() for d in x])

        reconst = self.forward(x).detach().numpy()
        x = np.array(x)

        return ((x - reconst)**2)[:, 0, :].sum(axis=1)


class CNNAutoEncoder2D(nn.Module):

    def __init__(self, input_dim_x=31, input_dim_y=31, output_dim=2, in_channels=1):
        super(CNNAutoEncoder2D, self).__init__()

        self._in_dim_x = input_dim_x
        self._in_dim_y = input_dim_y
        self._in_channels = in_channels

        self._encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, 2, 1), 
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(16, 16*2, 4, 2, 1), 
            nn.BatchNorm2d(16*2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(16*2, 16*4, 4, 2, 1),
            nn.BatchNorm2d(16*4),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(16*4, 16*8, 4, 2, 1),
            nn.BatchNorm2d(16*8),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(16*8, output_dim, 4, 2, 1),
            nn.BatchNorm2d(output_dim),
            nn.LeakyReLU(negative_slope=0.1),
        )

        self._decorder1 = nn.Sequential(
            nn.ConvTranspose2d(output_dim, 16*8, 4, 2, 1),
            nn.BatchNorm2d(16*8),
            nn.ReLU(),
            nn.ConvTranspose2d(16*8, 16*4, 4, 2, 1),
            nn.BatchNorm2d(16*4),
            nn.ReLU(),
            nn.ConvTranspose2d(16*4, 16*2, 4, 2, 1),
            nn.BatchNorm2d(16*2),
            nn.ReLU(),
            nn.ConvTranspose2d(16*2, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def encode(self, x):
        x = self._encoder1(x)
        return x

    def decode(self, x):
        x = self._decorder1(x)
        return x

    def reduce(self, x):
        return self.encode(x).squeeze(-1).squeeze(-1)

    def square_loss(self, x):
        if len(x.shape) == 2:
            # (N_Y, N_X) → (B(1), C(1), N_Y, N_X)
            x = np.expand_dims(x, axis=0)
            x = np.expand_dims(x, axis=0)
        elif len(x.shape) == 3:
            # (B, N_Y, N_X) → (B, C(1), N_Y, N_X)
            x = np.expand_dims(x, axis=1)

        x = torch.stack([torch.from_numpy(d).float() for d in x]).to(DEVICE)

        reconst = self.forward(x).cpu().detach().numpy()
        x = np.array(x)

        return ((x - reconst)**2)[:, 0, :].sum(axis=(1, 2))

    def _get_fc_layer_in_dim(self):
        dummy_data = torch.randn(3, self._in_channels, self._in_dim_x, self._in_dim_y)
        for layer in self._encoder1:
            dummy_data = layer(dummy_data)
        dummy_data = dummy_data.view(dummy_data.shape[0], -1)
        return dummy_data.shape[1]


class VariationalAutoEncoder(nn.Module):

    def __init__(
        self,
        input_dim: int=31,
        output_dim: int=2,
        in_channels: int=1,
        out_channels: int=64,
        k_size: int=4,
        stride: int=2,
        z_dim: int=100,
        hidden_dim: int=1024,
        device: str=DEVICE,
    ):
        super(VariationalAutoEncoder, self).__init__()

        self._in_dim = input_dim
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._k_size = k_size
        self._stride = stride
        self._z_dim = z_dim
        self._device = device

        self._encoder = nn.Sequential(
            nn.Conv1d(self._in_channels, self._out_channels, self._k_size-1, self._stride, 1, bias=False),
            nn.BatchNorm1d(self._out_channels),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv1d(self._out_channels, 2*self._out_channels, self._k_size, self._stride, 1, bias=False),
            nn.BatchNorm1d(2*self._out_channels),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv1d(2*self._out_channels, 4*self._out_channels, self._k_size, self._stride, 1, bias=False),
            nn.BatchNorm1d(4*self._out_channels),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv1d(4*self._out_channels, 8*self._out_channels, self._k_size, self._stride, 1, bias=False),
            nn.BatchNorm1d(8*self._out_channels),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv1d(8*self._out_channels, output_dim, self._k_size, self._stride, 1, bias=False),
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU(negative_slope=0.1),
        )
        fc_layer_in_dim = self._get_fc_layer_in_dim()
        self._fc_mu = nn.Linear(fc_layer_in_dim, z_dim)
        self._fc_log_sigma_sq = nn.Linear(fc_layer_in_dim, z_dim)
        self._fc = nn.Linear(z_dim, hidden_dim)

        self._decorder = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim, 8*self._out_channels, self._k_size, self._stride, 1, bias=False),
            nn.BatchNorm1d(8*self._out_channels),
            nn.LeakyReLU(negative_slope=0.1),
            nn.ConvTranspose1d(8*self._out_channels, 4*self._out_channels, self._k_size, self._stride, 1, bias=False),
            nn.BatchNorm1d(4*self._out_channels),
            nn.LeakyReLU(negative_slope=0.1),
            nn.ConvTranspose1d(4*self._out_channels, 2*self._out_channels, self._k_size, self._stride, 1, bias=False),
            nn.BatchNorm1d(2*self._out_channels),
            nn.LeakyReLU(negative_slope=0.1),
            nn.ConvTranspose1d(2*self._out_channels, self._out_channels, self._k_size, self._stride, 1, bias=False),
            nn.BatchNorm1d(self._out_channels),
            nn.LeakyReLU(negative_slope=0.1),
            nn.ConvTranspose1d(self._out_channels, self._in_channels, self._k_size-1, self._stride, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encode(x)
        x = self.decode(z)
        return x

    def encode(self, x):
        h = self._encoder(x)
        h = h.view(h.size(0), h.size(1)*h.size(2))
        mu = self._fc_mu(h)
        log_sigma_sq = self._fc_log_sigma_sq(h)
        z = self._sampling(mu, log_sigma_sq)
        z = self._fc(z)
        z = z.unsqueeze(-1)
        return z

    def decode(self, x):
        return self._decorder(x)

    def reduce(self, x):
        return self.encode(x).squeeze(-1)

    def _get_fc_layer_in_dim(self):
        dummy_data = torch.randn(3, self._in_channels, self._in_dim)
        for layer in self._encoder:
            dummy_data = layer(dummy_data)
        dummy_data = dummy_data.view(dummy_data.shape[0], -1)
        return dummy_data.shape[1]

    def _sampling(self, mu, log_sigma_sq):
        epsilon = torch.randn(mu.size()).float().to(self._device)
        sigma = torch.exp(log_sigma_sq / 2).to(self._device)
        return mu + sigma * epsilon

    def square_loss(self, x):
        if len(x.shape) == 1:
            # (N) → (B(1), C(1), N)
            x = np.expand_dims(x, axis=0)
            x = np.expand_dims(x, axis=0)
        elif len(x.shape) == 2:
            # (B, N) → (B, C(1), N)
            x = np.expand_dims(x, axis=1)

        x = torch.stack([torch.from_numpy(d).float() for d in x])

        reconst = self.forward(x).detach().numpy()
        x = np.array(x)

        return ((x - reconst)**2)[:, 0, :].sum(axis=1)

    def set_device(self, device):
        self._device = device


class LSTMAutoEncoder(nn.Module):

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_layers: int=4,
        hidden_size: int=2,
        device: str=DEVICE,
        batch_n: int=1000,
    ):
        super(LSTMAutoEncoder, self).__init__()

        self._in_dim = in_dim
        self._out_dim = out_dim
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._device = device
        self._batch_n = batch_n

        self._encoder = nn.LSTM(
            self._in_dim,
            self._hidden_size,
            self._num_layers,
            batch_first=True
        )

        self._decorder = nn.LSTM(
            self._hidden_size,
            self._out_dim,
            self._num_layers,
            batch_first=True
        )

    def forward(self, x):
        x, _ = self.encode(x)
        x, _ = self.decode(x)
        return x

    def encode(self, x):
        encoded, hidden = self._encoder(x)
        return encoded, hidden

    def decode(self, x):
        decoded, hidden = self._decorder(x)
        return decoded, hidden

    def set_device(self, device):
        self._device = device
