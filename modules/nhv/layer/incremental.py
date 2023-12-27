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

from .layer import ConvLayers, DFTLayer
from .model import NeuralHomomorphicVocoder
from .module import CCepLTVFilter, SinusoidsGenerator


class IncrementalCacheConvClass(nn.Module):
    def __init__(self):
        super().__init__()
        # remain handles to remove old hooks
        self.handles = []

    def _forward_without_cache(self, x):
        raise NotImplementedError("Please implement _forward_without_cache")

    def forward(self, caches, *inputs):
        self.caches = caches
        self.new_caches = []
        self.cache_num = 0
        x = self._forward(*inputs)
        return x, self.new_caches

    def reset_caches(self, *args, hop_size=128, batch_size=1):
        self.caches = []
        self.receptive_sizes = []
        self._initialize_caches(batch_size=batch_size, hop_size=hop_size)
        # set ordering hook
        self._set_pre_hooks(cache_ordering=True)
        # caclulate order of inference
        _ = self._forward_without_cache(*args)
        # remove hook handles for ordering
        [h.remove() for h in self.handles]
        # set concatenate hook
        self._set_pre_hooks(cache_ordering=False)
        # make cache zeros
        self.caches = [torch.zeros_like(c) for c in self.caches]
        # remove conv padding
        self._remove_padding()
        return self.caches

    def _initialize_caches(self, batch_size=1, hop_size=128):
        self.caches_dict = {}
        self.receptive_sizes_dict = {}
        for k, m in self.named_modules():
            if isinstance(m, nn.Conv1d):
                if m.kernel_size[0] > 1:
                    receptive_size = self._get_receptive_size_1d(m)
                    # NOTE(k2kobayashi): postfilter_fn requires to accept
                    # hop_size length input
                    if "postfilter_fn" in k:
                        receptive_size += hop_size - 1
                    self.caches_dict[id(m)] = torch.randn(
                        (batch_size, m.in_channels, receptive_size)
                    )
                    self.receptive_sizes_dict[id(m)] = receptive_size

    def _set_pre_hooks(self, cache_ordering=True):
        if cache_ordering:
            func = self._cache_ordering
        else:
            func = self._concat_cache
        for k, m in self.named_modules():
            if isinstance(m, nn.Conv1d):
                if m.kernel_size[0] > 1:
                    self.handles.append(m.register_forward_pre_hook(func))

    def _concat_cache(self, module, inputs):
        def __concat_cache(inputs, cache, receptive_size):
            inputs = torch.cat([cache, inputs[0]], axis=-1)
            inputs = inputs[..., -receptive_size:]
            return inputs

        cache = self.caches[self.cache_num]
        receptive_size = self.receptive_sizes[self.cache_num]
        inputs = __concat_cache(inputs, cache, receptive_size)
        self.new_caches += [inputs]
        self.cache_num += 1
        return inputs

    def _cache_ordering(self, module, inputs):
        self.caches.append(self.caches_dict[id(module)])
        self.receptive_sizes.append(self.receptive_sizes_dict[id(module)])

    def _remove_padding(self):
        def __remove_padding(m):
            if isinstance(m, torch.nn.Conv1d):
                m.padding = (0,)
            if isinstance(m, torch.nn.Conv2d):
                m.padding = (0, 0)

        self.apply(__remove_padding)

    @staticmethod
    def _get_receptive_size_1d(m):
        return (m.kernel_size[0] - 1) * m.dilation[0] + 1


class IncrementalNeuralHomomorphicVocoder(
    NeuralHomomorphicVocoder, IncrementalCacheConvClass
):
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
    use_causal = False
    use_reference_mag = False
    use_tanh = False
    use_uvmask = False
    use_weight_norm = True
    conv_type = "original"
    postfilter_type = "ddsconv"
    ltv_postfilter_type = "conv"
    ltv_postfilter_kernel_size = 128
    scaler_file = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert kwargs["use_causal"], "Require use_causal"
        self.impulse_generator = IncrementalSinusoidsGenerator(
            hop_size=self.hop_size, fs=self.fs, use_uvmask=self.use_uvmask
        )
        self.ltv_harmonic = IncrementalCCepLTVFilter(
            **self.ltv_params, feat2linear_fn=self.feat2linear_fn
        )
        self.ltv_noise = IncrementalCCepLTVFilter(**self.ltv_params)
        self.window_size = self.ltv_harmonic.window_size

    def _forward_without_cache(self, *inputs):
        super()._forward(*inputs)

    def forward(self, z, x, f0, uv, ltv_caches, conv_caches):
        self.caches = conv_caches
        self.new_caches = []
        self.cache_num = 0
        y, new_ltv_caches = self._incremental_forward(z, x, f0, uv, ltv_caches)
        return y, new_ltv_caches, self.new_caches

    def _incremental_forward(self, z, x, cf0, uv, ltv_caches):
        if self.feat_scaler_fn is not None:
            x = self.feat_scaler_fn(x)

        # impulse
        impulse, impulse_cache = self.impulse_generator.incremental_forward(
            cf0, uv, ltv_caches[0]
        )

        # ltv for harmonic
        harmonic = self._concat_ltv_input_cache(ltv_caches[1], impulse)
        ltv_harm = self.ltv_harmonic.incremental_forward(x, harmonic)
        sig_harm = ltv_caches[2][..., -self.hop_size :] + ltv_harm[..., : self.hop_size]
        if self.ltv_harmonic.ltv_postfilter_fn is not None:
            sig_harm = self.ltv_harmonic.ltv_postfilter_fn(
                sig_harm.transpose(1, 2)
            ).transpose(1, 2)

        # ltv for noise
        noise = self._concat_ltv_input_cache(ltv_caches[3], z)
        ltv_noise = self.ltv_noise.incremental_forward(x, noise)
        sig_noise = (
            ltv_caches[4][..., -self.hop_size :] + ltv_noise[..., : self.hop_size]
        )
        if self.ltv_noise.ltv_postfilter_fn is not None:
            sig_noise = self.ltv_noise.ltv_postfilter_fn(
                sig_noise.transpose(1, 2)
            ).transpose(1, 2)

        # superimpose
        y = sig_harm + sig_noise

        if self.postfilter_fn is not None:
            y = self.postfilter_fn(y.transpose(1, 2)).transpose(1, 2)

        y = torch.tanh(y) if self.use_tanh else torch.clamp(y, -1, 1)

        new_ltv_caches = [impulse_cache, harmonic, ltv_harm, noise, ltv_noise]
        return y.reshape(1, self.out_channels, -1), new_ltv_caches

    def reset_ltv_caches(self):
        ltv_caches = []
        # impulse generator
        ltv_caches += [torch.zeros(1, 1, 1)]
        # ltv harm
        ltv_caches += [torch.zeros(1, 1, self.window_size)]
        ltv_caches += [torch.zeros(1, 1, self.window_size)]
        # ltv noise
        ltv_caches += [torch.zeros(1, 1, self.window_size)]
        ltv_caches += [torch.zeros(1, 1, self.window_size)]
        return ltv_caches

    def _concat_ltv_input_cache(self, cache, z):
        z = torch.cat([cache, z], axis=-1)
        z = z[..., self.hop_size :]
        return z


class IncrementalSinusoidsGenerator(SinusoidsGenerator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def incremental_forward(self, cf0, uv, cache):
        f0, uv = self.upsample(cf0.transpose(1, 2)), self.upsample(uv.transpose(1, 2))
        harmonic, new_cache = self.incremental_generate_sinusoids(f0, uv, cache)
        harmonic = self.harmonic_amp * harmonic.reshape(cf0.size(0), 1, -1)
        return harmonic, new_cache

    def incremental_generate_sinusoids(self, f0, uv, cache):
        mask = self.anti_aliacing_mask(f0 * self.harmonics)
        # f0[..., 0] = f0[..., 0] + cache
        f0 = torch.cat([cache, f0], axis=-1)
        cumsum = torch.cumsum(f0, dim=-1)[..., 1:]
        rads = cumsum * 2.0 * math.pi / self.fs * self.harmonics
        harmonic = torch.sum(torch.cos(rads) * mask, dim=1, keepdim=True)
        if self.use_uvmask:
            harmonic = uv * harmonic
        new_cache = cumsum[..., -1:] % self.fs
        return harmonic, new_cache


class IncrementalConvLayers(ConvLayers, IncrementalCacheConvClass):
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
        for k, v in kwargs.items():
            if k not in self.__class__.__dict__.keys():
                raise ValueError(f"{k} not in arguments {self.__class__}.")
            setattr(self, k, v)
        assert kwargs["use_causal"], "Require use_causal"
        super().__init__(**kwargs)

    def _forward_without_cache(self, *inputs):
        super().forward(*inputs)

    def forward(self, x, conv_caches):
        self.caches = conv_caches
        self.new_caches = []
        self.cache_num = 0
        x = self.conv_layers(x)
        return x, self.new_caches


class IncrementalCCepLTVFilter(CCepLTVFilter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv_dft = DFTLayer(n_fft=self.fft_size)
        self.conv_idft = DFTLayer(n_fft=self.fft_size + 1)
        self.padding = (self.fft_size - self.ccep_size) // 2

    def incremental_forward(self, x, z):
        """Input tensor size
        x: (1, 1, input_size)
        z: (1, 1, fft_size + hop_size)
        """
        # inference complex cepstrum
        ccep = self.conv(x) / self.quef_norm
        log_mag = None if self.feat2linear_fn is None else self.feat2linear_fn(x)
        y = self._dft_ccep2impulse(ccep, ref=log_mag)

        # convolve to a frame
        z = F.pad(z, (self.fft_size // 2, self.fft_size // 2))
        z = F.conv1d(z, y)
        return z * self.win

    def _dft_ccep2impulse(self, ccep, ref=None):
        ccep = F.pad(ccep, (self.padding, self.padding))
        real, imag = self.conv_dft(ccep)
        if ref is not None:
            real = self._apply_ref_mag(real, ref)
        mag, phase = torch.pow(10, real / 10), imag
        real, imag = mag * torch.cos(phase), mag * torch.sin(phase)
        real, _ = self.conv_idft(F.pad(real, (0, 1)), F.pad(imag, (0, 1)), inverse=True)
        return real
