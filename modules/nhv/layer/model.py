#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright (c) 2021 Kazuhiro KOBAYASHI <root.4mac@gmail.com>
#
# Distributed under terms of the MIT license.

"""

"""

import joblib
import torch
import torch.nn as nn

from .layer import Conv1d, ConvLayers
from .module import CCepLTVFilter, SinusoidsGenerator
from .preprocess import LogMelSpectrogram2LogMagnitude, LogMelSpectrogramScaler


class NeuralHomomorphicVocoder(nn.Module):
    fs = 24000
    fft_size = 1024
    hop_size = 256
    in_channels = 80
    conv_channels = 256
    ccep_size = 222
    out_channels = 1
    kernel_size = 3
    dilation_size = 1
    group_size = 8
    fmin = 80
    fmax = 7600
    roll_size = 24
    n_ltv_layers = 3
    n_postfilter_layers = 4
    n_ltv_postfilter_layers = 1
    harmonic_amp = 0.1
    noise_std = 0.03
    use_causal = False
    use_reference_mag = False
    use_tanh = False
    use_uvmask = True
    use_weight_norm = True
    conv_type = "original"
    postfilter_type = None
    ltv_postfilter_type = None
    ltv_postfilter_kernel_size = 128
    scaler_file = None

    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            if k not in self.__class__.__dict__.keys():
                raise ValueError(f"{k} not in arguments {self.__class__}.")
            setattr(self, k, v)
        # load scaler
        self.feat_scaler_fn = self._load_feat_scaler(self.scaler_file, ext="mlfb")

        # feat to linear spectrogram if use_reference_mag
        self.feat2linear_fn = self._get_feat2linear_fn(ext="mlfb")

        # impulse generator
        self.impulse_generator = SinusoidsGenerator(
            hop_size=self.hop_size,
            fs=self.fs,
            harmonic_amp=self.harmonic_amp,
            use_uvmask=self.use_uvmask,
        )

        # LTV modules
        self.ltv_params = self._get_ltv_params()
        self.ltv_harmonic = CCepLTVFilter(
            **self.ltv_params, feat2linear_fn=self.feat2linear_fn
        )
        self.ltv_noise = CCepLTVFilter(**self.ltv_params)

        # post filter
        self.postfilter_fn = self._get_postfilter_fn()

        if self.use_weight_norm:
            self._apply_weight_norm()

    def forward(self, z, x, cf0):
        """
        z: (B, 1, T * hop_size)
        x: (B, T, D) ## D is the total dimension, including loudness, PPGs and spk_embs.
        cf0: (B, T, 1) ## frame-level pitch
        ### uv: (B, T, 1)  ## frame-level uv symbols, u is 0 and v is 1.
        """
        if self.feat_scaler_fn is not None:
            x = self.feat_scaler_fn(x)

        harmonic = self.impulse_generator(cf0)
        sig_harm = self.ltv_harmonic(x, harmonic)
        sig_noise = self.ltv_noise(x, z)
        y = sig_harm + sig_noise

        if self.postfilter_fn is not None:
            y = self.postfilter_fn(y.transpose(1, 2)).transpose(1, 2)

        y = torch.tanh(y) if self.use_tanh else torch.clamp(y, -1, 1)
        y = y.reshape(x.size(0), self.out_channels, -1)

        return harmonic, y


    @torch.no_grad()
    def inference(self, c):
        """Interface for PWG decoder
        c: (T, D)
        """
        c = c.unsqueeze(0)
        z = torch.normal(0, self.noise_std, (1, c.size(1) * self.hop_size)).to(c.device)
        x, cf0, uv = torch.split(c, [self.in_channels, 1, 1], dim=-1)
        y = self._forward(z, x, cf0, uv)
        return y.squeeze(0)

    def remove_weight_norm(self):
        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:
                return

        self.apply(_remove_weight_norm)

    def _apply_weight_norm(self):
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)

    def _get_ltv_params(self):
        return {
            "in_channels": self.in_channels,
            "conv_channels": self.conv_channels,
            "ccep_size": self.ccep_size,
            "kernel_size": self.kernel_size,
            "dilation_size": self.dilation_size,
            "group_size": self.group_size,
            "fft_size": self.fft_size,
            "hop_size": self.hop_size,
            "n_ltv_layers": self.n_ltv_layers,
            "n_ltv_postfilter_layers": self.n_ltv_postfilter_layers,
            "use_causal": self.use_causal,
            "conv_type": self.conv_type,
            "ltv_postfilter_type": self.ltv_postfilter_type,
            "ltv_postfilter_kernel_size": self.ltv_postfilter_kernel_size,
        }

    @staticmethod
    def _load_feat_scaler(scaler_file, ext="mlfb"):
        if scaler_file is not None:
            if ext == "mlfb":
                fn = LogMelSpectrogramScaler(joblib.load(scaler_file)[ext])
            elif ext == "lsp":
                fn = None
                raise NotImplementedError("lsp scaler is not implemented.")
        else:
            fn = None
        return fn

    def _get_feat2linear_fn(self, ext="mlfb"):
        if self.use_reference_mag:
            if ext == "mlfb":
                fn = LogMelSpectrogram2LogMagnitude(
                    fs=self.fs,
                    fft_size=self.fft_size,
                    n_mels=self.in_channels,
                    fmin=self.fmin,
                    fmax=self.fmax,
                    roll_size=self.roll_size,
                    melspc_scaler_fn=self.feat_scaler_fn,
                )
            elif ext == "lsp":
                fn = None
                raise NotImplementedError("lsp to linear is not implemented.")
        else:
            fn = None
        return fn

    def _get_postfilter_fn(self):
        if self.postfilter_type == "ddsconv":
            fn = ConvLayers(
                in_channels=1,
                conv_channels=64,
                out_channels=1,
                kernel_size=5,
                dilation_size=2,
                n_conv_layers=self.n_postfilter_layers,
                use_causal=self.use_causal,
                conv_type="ddsconv",
            )
        elif self.postfilter_type == "conv":
            fn = Conv1d(
                in_channels=1,
                out_channels=1,
                kernel_size=self.fft_size,
                use_causal=self.use_causal,
            )
        elif self.postfilter_type is None:
            fn = None
        else:
            raise ValueError(f"Invalid postfilter_type: {self.postfilter_type}")
        return fn
