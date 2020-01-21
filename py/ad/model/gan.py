import torch.nn as nn


class Generator(nn.Module):

    def __init__(self, in_channel=1, base_channel=64, latent_dim=100):
        super(Generator, self).__init__()

        self._in_channel = in_channel
        self._base_channel = base_channel
        self._latent_dim = latent_dim

        self._layer = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, base_channel*8, kernel_size=2, stride=2),
            nn.BatchNorm1d(base_channel*8),
            nn.ReLU(),
            nn.ConvTranspose1d(base_channel*8, base_channel*4, kernel_size=2, stride=2),
            nn.BatchNorm1d(base_channel*4),
            nn.ReLU(),
            nn.ConvTranspose1d(base_channel*4, base_channel*2, kernel_size=2, stride=2),
            nn.BatchNorm1d(base_channel*2),
            nn.ReLU(),
            nn.ConvTranspose1d(base_channel*2, base_channel, kernel_size=2, stride=2),
            nn.BatchNorm1d(base_channel),
            nn.ConvTranspose1d(base_channel, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, z):
        #z = z.view(-1, self._latent_dim, 1)
        z = self._layer(z)
        return z


class Discriminator(nn.Module):

    def __init__(self, in_channel=1, base_channel=64, out_channel=256, latent_dim=100):
        super(Discriminator, self).__init__()

        self._in_channel = in_channel
        self._base_channel = base_channel
        self._latent_dim = latent_dim
        self._out_channel = out_channel

        self._layer = nn.Sequential(
            nn.Conv1d(in_channel, out_channel//4, kernel_size=2, stride=2),
            nn.BatchNorm1d(out_channel//4),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv1d(out_channel//4, out_channel//3, kernel_size=2, stride=2),
            nn.BatchNorm1d(out_channel//3),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv1d(out_channel//3, out_channel//2, kernel_size=2, stride=2),
            nn.BatchNorm1d(out_channel//2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv1d(out_channel//2, out_channel, kernel_size=2, stride=2),
            nn.BatchNorm1d(out_channel),
            nn.LeakyReLU(),
            nn.Conv1d(out_channel, 1, kernel_size=2, stride=2),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self._layer(x).squeeze()
        return x
