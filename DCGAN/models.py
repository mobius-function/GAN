import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim, channels, output_size=64, output_channels=3):
        super(Generator, self).__init__()

        self.init_size = output_size // 16
        self.l1 = nn.Sequential(nn.Linear(latent_dim, channels[0] * self.init_size**2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(channels[0]),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(channels[0], channels[1], 3, stride=1, padding=1),
            nn.BatchNorm2d(channels[1], 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(channels[1], channels[2], 3, stride=1, padding=1),
            nn.BatchNorm2d(channels[2], 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(channels[2], channels[3], 3, stride=1, padding=1),
            nn.BatchNorm2d(channels[3], 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(channels[3], output_channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self, channels, input_size=64, input_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_channels, out_channels, bn=True):
            block = [nn.Conv2d(in_channels, out_channels, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_channels, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(input_channels, channels[0], bn=False),
            *discriminator_block(channels[0], channels[1]),
            *discriminator_block(channels[1], channels[2]),
            *discriminator_block(channels[2], channels[3]),
        )

        ds_size = input_size // 2**4
        self.adv_layer = nn.Sequential(nn.Linear(channels[3] * ds_size**2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity
