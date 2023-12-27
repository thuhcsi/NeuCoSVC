#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright (c) 2021 Kazuhiro KOBAYASHI <root.4mac@gmail.com>
#
# Distributed under terms of the MIT license.

"""

"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayers(nn.Module):
    in_channels = 80
    conv_channels = 256
    out_channels = 222
    kernel_size = 3
    dilation_size = 1
    group_size = 8
    n_conv_layers = 3
    use_causal = False
    conv_type = "original"

    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            if k not in self.__class__.__dict__.keys():
                raise ValueError(f"{k} not in arguments {self.__class__}.")
            setattr(self, k, v)
        if self.conv_type == "ddsconv":
            self.conv_layers = self.ddsconv()
        elif self.conv_type == "original":
            self.conv_layers = self.original_conv()
        else:
            raise ValueError(f"Unsupported conv_type: {self.conv_type}")

    def forward(self, x):
        """
        x: (B, T, in_channels)
        y: (B, T, out_channels)
        """
        return self.conv_layers(x)

    def original_conv(self):
        modules = []
        modules += [
            Conv1d(
                self.in_channels,
                self.conv_channels,
                self.kernel_size,
                self.dilation_size,
                1,
                self.use_causal,
            ),
            nn.ReLU(),
        ]
        for i in range(self.n_conv_layers):
            modules += [
                Conv1d(
                    self.conv_channels,
                    self.conv_channels,
                    self.kernel_size,
                    self.dilation_size,
                    self.group_size,
                    self.use_causal,
                ),
                nn.ReLU(),
            ]
        modules += [
            Conv1d(
                self.conv_channels,
                self.out_channels,
                self.kernel_size,
                self.dilation_size,
                1,
                self.use_causal,
            ),
        ]
        return nn.Sequential(*modules)

    def ddsconv(self):
        modules = []
        modules += [
            Conv1d(
                in_channels=self.in_channels,
                out_channels=self.conv_channels,
                kernel_size=1,
                dilation_size=1,
                group_size=1,
                use_causal=self.use_causal,
            )
        ]
        for i in range(self.n_conv_layers):
            if self.dilation_size == 1:
                dilation_size = self.kernel_size ** i
            else:
                dilation_size = self.dilation_size ** i
            modules += [
                DepthSeparableConv1d(
                    channels=self.conv_channels,
                    kernel_size=self.kernel_size,
                    dilation_size=dilation_size,
                    use_causal=self.use_causal,
                )
            ]
        modules += [
            Conv1d(
                in_channels=self.conv_channels,
                out_channels=self.out_channels,
                kernel_size=1,
                dilation_size=1,
                group_size=1,
                use_causal=self.use_causal,
            )
        ]
        return nn.Sequential(*modules)

    def remove_weight_norm(self):
        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:
                return

        self.apply(_remove_weight_norm)


class Conv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        dilation_size=1,
        group_size=1,
        use_causal=False,
    ):
        super().__init__()
        self.use_causal = use_causal
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) * dilation_size

        self.conv1d = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation_size,
            groups=group_size,
        )
        nn.init.kaiming_normal_(self.conv1d.weight)

    def forward(self, x):
        """
        x: (B, T, D)
        y: (B, T, D)
        """
        x = x.transpose(1, 2)
        y = self.conv1d(x)
        # NOTE(k2kobayashi): kernel_size=1 does not discard padding
        if self.kernel_size > 1 and self.conv1d.padding != (0,):
            if self.use_causal:
                y = y[..., : -self.padding]
            else:
                y = y[..., self.padding // 2 : -self.padding // 2]
        return y.transpose(1, 2)


class DepthSeparableConv1d(nn.Module):
    def __init__(self, channels, kernel_size, dilation_size, use_causal=False):
        super().__init__()
        sep_conv = Conv1d(
            channels,
            channels,
            kernel_size,
            dilation_size,
            group_size=channels,
            use_causal=use_causal,
        )
        conv1d = Conv1d(
            channels,
            channels,
            kernel_size=1,
            dilation_size=1,
            group_size=1,
            use_causal=use_causal,
        )
        ln1 = nn.LayerNorm(channels)
        ln2 = nn.LayerNorm(channels)
        gelu1 = nn.GELU()
        gelu2 = nn.GELU()
        modules = [sep_conv, ln1, gelu1, conv1d, ln2, gelu2]
        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        y = self.layers(x)
        return x + y


class DFTLayer(nn.Module):
    def __init__(self, n_fft=1024):
        super().__init__()
        self.n_fft = n_fft
        wsin, wcos = self._generate_fourier_kernels(n_fft=n_fft)
        self.register_buffer(
            "wsin", torch.tensor(wsin, dtype=torch.float), persistent=False
        )
        self.register_buffer(
            "wcos", torch.tensor(wcos, dtype=torch.float), persistent=False
        )

    @staticmethod
    def _generate_fourier_kernels(n_fft, window="hann"):
        freq_bins = n_fft
        s = np.arange(0, n_fft, 1.0)
        wsin = np.empty((freq_bins, 1, n_fft))
        wcos = np.empty((freq_bins, 1, n_fft))
        for k in range(freq_bins):
            wsin[k, 0, :] = np.sin(2 * np.pi * k * s / n_fft)
            wcos[k, 0, :] = np.cos(2 * np.pi * k * s / n_fft)
        return wsin.astype(np.float32), wcos.astype(np.float32)

    def forward(self, x, imag=None, inverse=False):
        if not inverse:
            return self.dft(x)
        else:
            return self.idft(x, imag)

    def dft(self, real):
        real = real.transpose(0, 1)
        imag = F.conv1d(real, self.wsin, stride=self.n_fft + 1).permute(2, 0, 1)
        real = F.conv1d(real, self.wcos, stride=self.n_fft + 1).permute(2, 0, 1)
        return real, -imag

    def idft(self, real, imag):
        real = real.transpose(0, 1)
        imag = imag.transpose(0, 1)
        a1 = F.conv1d(real, self.wcos, stride=self.n_fft + 1)
        a2 = F.conv1d(real, self.wsin, stride=self.n_fft + 1)
        b1 = F.conv1d(imag, self.wcos, stride=self.n_fft + 1)
        b2 = F.conv1d(imag, self.wsin, stride=self.n_fft + 1)
        imag = a2 + b1
        real = a1 - b2
        return (
            (real / self.n_fft).permute(2, 0, 1),
            (imag / self.n_fft).permute(2, 0, 1),
        )
