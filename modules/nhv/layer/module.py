#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright (c) 2021 Kazuhiro KOBAYASHI <root.4mac@gmail.com>
#
# Distributed under terms of the MIT license.

"""

"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

from .layer import Conv1d, ConvLayers


class CCepLTVFilter(nn.Module):
    def __init__(
        self,
        in_channels,
        conv_channels=256,
        ccep_size=222,
        kernel_size=3,
        dilation_size=1,
        group_size=8,
        fft_size=1024,
        hop_size=256,
        n_ltv_layers=3,
        n_ltv_postfilter_layers=1,
        use_causal=False,
        conv_type="original",
        feat2linear_fn=None,
        ltv_postfilter_type="conv",
        ltv_postfilter_kernel_size=128,
    ):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.window_size = hop_size * 2
        self.ccep_size = ccep_size
        self.use_causal = use_causal
        self.feat2linear_fn = feat2linear_fn
        self.ltv_postfilter_type = ltv_postfilter_type
        self.ltv_postfilter_kernel_size = ltv_postfilter_kernel_size
        self.n_ltv_postfilter_layers = n_ltv_postfilter_layers

        win_norm = self.window_size // (hop_size * 2)  # only for hanning window
        # periodic must be True to become OLA 1
        win = torch.hann_window(self.window_size, periodic=True) / win_norm
        self.conv = ConvLayers(
            in_channels=in_channels,
            conv_channels=conv_channels,
            out_channels=ccep_size,
            kernel_size=kernel_size,
            dilation_size=dilation_size,
            group_size=group_size,
            n_conv_layers=n_ltv_layers,
            use_causal=use_causal,
            conv_type=conv_type,
        )
        self.ltv_postfilter_fn = self._get_ltv_postfilter_fn()

        idx = torch.arange(1, ccep_size // 2 + 1).float()
        quef_norm = torch.cat([torch.flip(idx, dims=[-1]), idx], dim=-1)
        self.padding = (self.fft_size - self.ccep_size) // 2
        self.register_buffer("quef_norm", quef_norm)
        self.register_buffer("win", win)

    def forward(self, x, z):
        """
        x: B, T, D
        z: B, 1, T * hop_size
        """
        # inference complex cepstrum
        ccep = self.conv(x) / self.quef_norm

        # apply LTV filter and overlap
        log_mag = None if self.feat2linear_fn is None else self.feat2linear_fn(x)
        y = self._ccep2impulse(ccep, ref=log_mag)
        z = self._conv_impulse(z, y)
        z = self._ola(z)
        if self.ltv_postfilter_fn is not None:
            z = self.ltv_postfilter_fn(z.transpose(1, 2)).transpose(1, 2)
        return z

    def _apply_ref_mag(self, real, ref):
        # TODO(k2kobayashi): it requires to consider following line.
        # this mask eliminates very small amplitude values (-100).
        # ref = ref * (ref > -100)
        real[..., : self.fft_size // 2 + 1] += ref
        real[..., self.fft_size // 2 :] += torch.flip(ref[..., 1:], dims=[-1])
        return real

    def _ccep2impulse(self, ccep, ref=None):
        ccep = F.pad(ccep, (self.padding, self.padding))
        y = torch.fft.fft(ccep, n=self.fft_size, dim=-1)
        # NOTE(k2kobayashi): we assume intermediate log amplitude as 10log10|mag|
        if ref is not None:
            y.real = self._apply_ref_mag(y.real, ref)
        # logarithmic to linear
        mag, phase = torch.pow(10, y.real / 10), y.imag
        real, imag = mag * torch.cos(phase), mag * torch.sin(phase)
        y = torch.fft.ifft(torch.complex(real, imag), n=self.fft_size + 1, dim=-1)
        return y.real

    def _conv_impulse(self, z, y):
        # (B, T * hop_size + hop_size)
        # z = F.pad(z, (self.hop_size // 2, self.hop_size // 2)).squeeze(1)
        z = F.pad(z, (self.hop_size, 0)).squeeze(1)
        z = z.unfold(-1, self.window_size, step=self.hop_size)  # (B, T, window_size)
        z = F.pad(z, (self.fft_size // 2, self.fft_size // 2))
        z = z.unfold(-1, self.fft_size + 1, step=1)  # (B, T, window_size, fft_size + 1)
        # y: (B, T, fft_size + 1) -> (B, T, fft_size + 1, 1)
        # z: (B, T, window_size, fft_size + 1)
        # output: (B, T, window_size)
        output = torch.matmul(z, y.unsqueeze(-1)).squeeze(-1)
        return output

    def _conv_impulse_old(self, z, y):
        z = F.pad(z, (self.window_size // 2 - 1, self.window_size // 2)).squeeze(1)
        z = z.unfold(-1, self.window_size, step=self.hop_size)  # (B, 1, T, window_size)

        z = F.pad(z, (self.fft_size // 2 - 1, self.fft_size // 2))
        z = z.unfold(-1, self.fft_size, step=1)  # (B, 1, T, window_size, fft_size)

        # z = matmul(z, y) -> (B, 1, T, window_size) where
        # z: (B, 1, T, window_size, fft_size)
        # y: (B, T, fft_size) -> (B, 1, T, fft_size, 1)
        z = torch.matmul(z, y.unsqueeze(-1)).squeeze(-1)
        return z

    def _ola(self, z):
        z = z * self.win
        l, r = torch.chunk(z, 2, dim=-1)  # (B, 1, T, window_size // 2)
        z = l + torch.roll(r, 1, dims=-2)  # roll a frame of right side
        z = z.reshape(z.size(0), 1, -1)
        return z

    def _get_ltv_postfilter_fn(self):
        if self.ltv_postfilter_type == "ddsconv":
            fn = ConvLayers(
                in_channels=1,
                conv_channels=64,
                out_channels=1,
                kernel_size=5,
                dilation_size=2,
                n_conv_layers=self.n_ltv_postfilter_layers,
                use_causal=self.use_causal,
                conv_type="ddsconv",
            )
        elif self.ltv_postfilter_type == "conv":
            fn = Conv1d(
                in_channels=1,
                out_channels=1,
                kernel_size=self.ltv_postfilter_kernel_size,
                use_causal=self.use_causal,
            )
        elif self.ltv_postfilter_type is None:
            fn = None
        else:
            raise ValueError(f"Invalid ltv_postfilter_type: {self.ltv_postfilter_type}")
        return fn


class SinusoidsGenerator(nn.Module):
    def __init__(
        self,
        hop_size,
        fs=24000,
        harmonic_amp=0.1,
        n_harmonics=200,
        use_uvmask=False,
    ):
        super().__init__()
        self.fs = fs
        self.harmonic_amp = harmonic_amp
        self.upsample = nn.Upsample(scale_factor=hop_size, mode="linear")
        self.use_uvmask = use_uvmask
        self.n_harmonics = n_harmonics
        harmonics = torch.arange(1, self.n_harmonics + 1).unsqueeze(-1)
        self.register_buffer("harmonics", harmonics)

    def forward(self, cf0):
        f0 = self.upsample(cf0.transpose(1, 2))
        uv = torch.zeros(f0.size()).to(f0.device)
        nonzero_indices = torch.nonzero(f0, as_tuple=True)
        uv[nonzero_indices] = 1.0
        harmonic = self.generate_sinusoids(f0, uv).reshape(cf0.size(0), 1, -1)
        return self.harmonic_amp * harmonic

    def generate_sinusoids(self, f0, uv):
        mask = self.anti_aliacing_mask(f0 * self.harmonics)
        rads = f0.cumsum(dim=-1) * 2.0 * math.pi / self.fs * self.harmonics
        harmonic = torch.sum(torch.cos(rads) * mask, dim=1, keepdim=True)
        if self.use_uvmask:
            harmonic = uv * harmonic
        return harmonic

    def anti_aliacing_mask(self, f0_with_harmonics, use_soft_mask=False):
        if use_soft_mask:
            return torch.sigmoid(-(f0_with_harmonics - self.fs / 2.0))
        else:
            return (f0_with_harmonics < self.fs / 2.0).float()
