import torch

from modules.base import BaseModule
from modules.layers import Conv1dWithInitialization


LINEAR_SCALE=5000


class FeatureWiseLinearModulation(BaseModule):
    def __init__(self, in_channels, out_channels, input_dscaled_by):
        super(FeatureWiseLinearModulation, self).__init__()
        self.signal_conv = torch.nn.Sequential(*[
            Conv1dWithInitialization(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            torch.nn.LeakyReLU(0.2)
        ])
        # self.positional_encoding = PositionalEncoding(in_channels)
        self.scale_conv = Conv1dWithInitialization(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.shift_conv = Conv1dWithInitialization(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, x):
        outputs = self.signal_conv(x)
        # outputs = outputs + self.positional_encoding(noise_level).unsqueeze(-1)
        scale, shift = self.scale_conv(outputs), self.shift_conv(outputs)
        return scale, shift


class FeatureWiseAffine(BaseModule):
    def __init__(self):
        super(FeatureWiseAffine, self).__init__()

    def forward(self, x, scale, shift):
        outputs = scale * x + shift
        return outputs
