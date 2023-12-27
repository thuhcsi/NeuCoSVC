#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright (c) 2021 Kazuhiro KOBAYASHI <root.4mac@gmail.com>
#
# Distributed under terms of the MIT license.

"""

"""

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LogMelSpectrogram(nn.Module):
    def __init__(
        self,
        fs=24000,
        hop_size=128,
        fft_size=1024,
        win_length=None,
        window="hann",
        center=True,
        pad_mode="reflect",
        n_mels=80,
        fmin=None,
        fmax=None,
        scaler_file=None,
    ):
        super().__init__()
        self.mag_layer = Magnitude(fs, hop_size, fft_size, win_length, window)
        self.mel_layer = Magnitude2LogMelSpectrogram(fs, fft_size, n_mels, fmin, fmax)
        if scaler_file is not None:
            self.melspc_scaler = LogMelSpectrogramScaler(scaler_file)
        else:
            self.melspc_scaler = None

    def forward(self, x):
        mag = self.mag_layer(x)
        log_melspc = self.mel_layer(mag)
        if self.melspc_scaler is not None:
            log_melspc = self.scaler_layer(log_melspc)
        return log_melspc


class Magnitude(torch.nn.Module):
    def __init__(
        self,
        fs=24000,
        hop_size=128,
        fft_size=1024,
        win_length=None,
        window="hann",
        center=True,
        pad_mode="reflect",
        return_complex=True,
    ):
        super().__init__()
        self.hop_size = hop_size
        self.fft_size = fft_size
        self.win_length = fft_size if win_length is None else win_length
        self.window = window
        self.center = center
        self.pad_mode = pad_mode
        self.return_complex = return_complex

    def forward(self, x):
        """
        x: (B, 1, T)
        ret: (B, T, fft_size // 2 + 1)
        """
        f = getattr(torch, f"{self.window}_window")
        window = f(self.win_length, dtype=x.dtype, device=x.device)
        y = torch.stft(
            x,
            n_fft=self.fft_size,
            win_length=self.win_length,
            hop_length=self.hop_size,
            window=window,
            center=self.center,
            pad_mode=self.pad_mode,
            return_complex=self.return_complex,
        )
        return y.abs().transpose(1, 2)


class Magnitude2LogMelSpectrogram(torch.nn.Module):
    def __init__(
        self, fs=24000, fft_size=1024, n_mels=80, fmin=None, fmax=None, eps=1.0e-10
    ):
        super().__init__()
        self.eps = eps
        fmin = 0 if fmin is None else fmin
        fmax = fs / 2 if fmax is None else fmax
        mel_basis = librosa.filters.mel(
            fs, fft_size, n_mels=n_mels, fmin=fmin, fmax=fmax
        )
        self.register_buffer("mel_basis", torch.from_numpy(mel_basis.T).float())

    def forward(self, x):
        melspc = torch.matmul(x, self.mel_basis)
        log_melspc = torch.clamp(melspc, min=self.eps).log10()
        return log_melspc


class LogMelSpectrogram2LogMagnitude(nn.Module):
    def __init__(
        self,
        fs,
        fft_size,
        n_mels=80,
        fmin=None,
        fmax=None,
        eps=1.0e-10,
        roll_size=24,
        melspc_scaler_fn=None,
    ):
        super().__init__()
        self.eps = eps
        self.roll_size = roll_size
        self.melspc_scaler_fn = melspc_scaler_fn
        fmin = 0 if fmin is None else fmin
        fmax = fs / 2 if fmax is None else fmax

        mel_basis = librosa.filters.mel(
            fs, fft_size, n_mels=n_mels, fmin=fmin, fmax=fmax
        )
        inv_mel_basis = np.linalg.pinv(mel_basis)
        self.register_buffer("inv_mel_basis", torch.from_numpy(inv_mel_basis.T).float())

    def forward(self, x):
        if self.melspc_scaler_fn is not None:
            # denorm mlfb
            x = self.melspc_scaler_fn.inverse_transform(x)

        x = torch.pow(10.0, x)
        spc = torch.matmul(x, self.inv_mel_basis)
        log_spc = 10 * torch.clamp(spc, min=self.eps).log10()
        z = F.pad(log_spc, (self.roll_size // 2 - 1, self.roll_size // 2))
        z = z.unfold(-1, self.roll_size, step=1)
        log_spc = torch.median(z, dim=-1)[0]
        return log_spc


class CepstrumLiftering(nn.Module):
    def __init__(self, lifter_size=None):
        super().__init__()
        if lifter_size <= 0:
            raise ValueError("lifter_size must be > 0.")
        else:
            self.lifter_size = lifter_size

    def forward(self, x):
        cep = torch.fft.ifft(x, dim=-1)
        cep[..., self.lifter_size : -self.lifter_size] = 0
        x = torch.fft.fft(cep, dim=-1)
        return x


class LogMelSpectrogramScaler(nn.Module):
    def __init__(self, scaler):
        super().__init__()
        self.register_parameter(
            "mean",
            nn.Parameter(torch.from_numpy(scaler.mean_).float(), requires_grad=False),
        )
        self.register_parameter(
            "scale",
            nn.Parameter(
                torch.from_numpy(scaler.var_).float().sqrt(), requires_grad=False
            ),
        )

    def forward(self, x):
        return (x - self.mean) / self.scale

    def inverse_transform(self, x):
        return x * self.scale + self.mean
