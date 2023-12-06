import torch.nn as nn
import torch.nn.functional as F
import torch
import pdb

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # self.model = nn.Sequential(
        #     *discriminator_block(in_channels, 64, normalization=False),
        #     *discriminator_block(64, 128),
        #     *discriminator_block(128, 256),
        #     *discriminator_block(256, 512),
        #     nn.ZeroPad2d((1, 0, 1, 0)),
        #     nn.Conv2d(512, 1, 4, padding=1, bias=False)
        # )
        self.model = nn.Sequential(
            *discriminator_block(in_channels, 32, normalization=False),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(256, 1, 4, padding=1, bias=False)
        )

    def forward(self, img):
        # Concatenate image and condition image by channels to produce input
        return self.model(img)