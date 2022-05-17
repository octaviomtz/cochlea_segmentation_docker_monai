from typing import List

import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision


class ConvBlock(nn.Module):
    """two conv-> relu layers stacked"""

    def __init__(self, input_ch: int, output_ch: int, kernal_size: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(input_ch, output_ch, kernal_size)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(output_ch, output_ch, kernal_size)

    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))


class Encoder(nn.Module):
    def __init__(
        self,
        layers: List[int] = [3, 64, 128, 256, 512, 1024],
        kernal_size: int = 3,
        pool: int = 2,
    ):
        super().__init__()
        self.encoder = nn.ModuleList(
            [
                ConvBlock(layers[i], layers[i + 1], kernal_size)
                for i in range(len(layers) - 1)
            ]
        )
        self.pool_layer = nn.MaxPool2d(pool)

    def forward(self, x):
        encoder = []
        for block in self.encoder:
            x = block(x)
            encoder.append(x)
            x = self.pool_layer(x)
        return encoder


class Decoder(nn.Module):
    def __init__(self, layers=[1024, 512, 256, 128, 64], kernal_size=3):
        super().__init__()
        self.layers = layers
        self.decoder = nn.ModuleList(
            [
                ConvBlock(layers[i], layers[i + 1], kernal_size)
                for i in range(len(layers) - 1)
            ]
        )
        self.upconv = nn.ModuleList(
            [
                nn.ConvTranspose2d(layers[i], layers[i + 1], 2, 2)
                for i in range(len(layers) - 1)
            ]
        )

    def forward(self, x, encoder):
        for i in range(len(self.layers) - 1):
            x = self.upconv[i](x)
            connect = self.crop(encoder[i], x)
            x = torch.cat([x, connect], dim=1)
            x = self.decoder[i](x)
        return x

    def crop(self, input_tensor, target_tensor):
        _, _, H, W = target_tensor.shape
        reshaped = torchvision.transforms.CenterCrop([H, W])(input_tensor)
        return reshaped


class Unet(nn.Module):
    def __init__(
        self,
        encoder_layers=[3, 64, 128, 256],
        decoder_layers=[256, 128, 64],
        retain_dimension=True,
        num_class=1,
        outputsize=(128, 128),
    ):
        super().__init__()
        self.encoder = Encoder(encoder_layers)
        self.decoder = Decoder(decoder_layers)
        self.output = nn.Conv2d(decoder_layers[-1], num_class, 1)
        self.retain_dimension = retain_dimension
        self.outputsize = outputsize

    def forward(self, x):
        enco = self.encoder(x)
        output = self.decoder(enco[::-1][0], enco[::-1][1:])
        output = self.output(output)
        if self.retain_dimension:
            output = F.interpolate(output, self.outputsize)
        return output


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice
