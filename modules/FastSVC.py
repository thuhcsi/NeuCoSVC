import numpy as np
import torch
import torch.nn.functional as F

from modules.base import BaseModule
from modules.layers import Conv1dWithInitialization
from modules.upsampling import UpsamplingBlock as UBlock
from modules.downsampling import DownsamplingBlock as DBlock
from modules.linear_modulation import FeatureWiseLinearModulation as FiLM

from modules.nhv import NeuralHomomorphicVocoder

device_str = 'cuda' if torch.cuda.is_available() else 'cpu'

class SVCNN(BaseModule):
    """
    WaveGrad is a fully-convolutional mel-spectrogram conditional
    vocoder model for waveform generation introduced in
    "WaveGrad: Estimating Gradients for Waveform Generation" paper (link: https://arxiv.org/pdf/2009.00713.pdf).
    The concept is built on the prior work on score matching and diffusion probabilistic models.
    Current implementation follows described architecture in the paper.
    """
    def __init__(self, config):
        super(SVCNN, self).__init__()

        # Construct NHV module.
        self.hop_size = config.data_config.hop_size
        self.noise_std = config.model_config.nhv_noise_std
        self.nhv_cat_type = config.model_config.nhv_cat_type
        self.harmonic_type = config.model_config.harmonic_type

        self.nhv = NeuralHomomorphicVocoder(fs=config.data_config.sampling_rate, hop_size=self.hop_size, in_channels=config.model_config.nhv_inchannels, fmin=80, fmax=7600)

        # Building upsampling branch (mels -> signal)
        self.ublock_preconv = Conv1dWithInitialization(
            in_channels=config.model_config.nhv_inchannels-1,
            out_channels=config.model_config.upsampling_preconv_out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        upsampling_in_sizes = [config.model_config.upsampling_preconv_out_channels] \
            + config.model_config.upsampling_out_channels[:-1]
        self.ublocks = torch.nn.ModuleList([
            UBlock(
                in_channels=in_size,
                out_channels=out_size,
                factor=factor,
                dilations=dilations
            ) for in_size, out_size, factor, dilations in zip(
                upsampling_in_sizes,
                config.model_config.upsampling_out_channels,
                config.model_config.factors,
                config.model_config.upsampling_dilations
            )
        ])
        self.ublock_postconv = Conv1dWithInitialization(
            in_channels=config.model_config.upsampling_out_channels[-1],
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # Building downsampling branch (starting from signal)
        self.ld_dblock_preconv = Conv1dWithInitialization(
            in_channels=1,
            out_channels=config.model_config.downsampling_preconv_out_channels,
            kernel_size=5,
            stride=1,
            padding=2
        )
        self.pitch_dblock_preconv = Conv1dWithInitialization(
            in_channels=config.model_config.num_harmonic,
            out_channels=config.model_config.downsampling_preconv_out_channels,
            kernel_size=5,
            stride=1,
            padding=2
        )

        downsampling_in_sizes = [config.model_config.downsampling_preconv_out_channels] \
            + config.model_config.downsampling_out_channels[:-1]
        self.ld_dblocks = torch.nn.ModuleList([
            DBlock(
                in_channels=in_size,
                out_channels=out_size,
                factor=factor,
                dilations=dilations
            ) for in_size, out_size, factor, dilations in zip(
                downsampling_in_sizes,
                config.model_config.downsampling_out_channels,
                config.model_config.factors[1:][::-1],
                config.model_config.downsampling_dilations
            )
        ])
        self.pitch_dblocks = torch.nn.ModuleList([
            DBlock(
                in_channels=in_size,
                out_channels=out_size,
                factor=factor,
                dilations=dilations
            ) for in_size, out_size, factor, dilations in zip(
                downsampling_in_sizes,
                config.model_config.downsampling_out_channels,
                config.model_config.factors[1:][::-1],
                config.model_config.downsampling_dilations
            )
        ])

        # Building FiLM connections (in order of downscaling stream)
        film_in_sizes = [24] + config.model_config.downsampling_out_channels
        film_out_sizes = config.model_config.upsampling_out_channels[::-1]
        film_factors = [1] + config.model_config.factors[1:][::-1]
        self.ld_films = torch.nn.ModuleList([
            FiLM(
                in_channels=in_size,
                out_channels=out_size,
                input_dscaled_by=np.product(film_factors[:i+1])  # for proper positional encodings initialization
            ) for i, (in_size, out_size) in enumerate(
                zip(film_in_sizes, film_out_sizes)
            )
        ])
        self.pitch_films = torch.nn.ModuleList([
            FiLM(
                in_channels=in_size,
                out_channels=out_size,
                input_dscaled_by=np.product(film_factors[:i+1])  # for proper positional encodings initialization
            ) for i, (in_size, out_size) in enumerate(
                zip(film_in_sizes, film_out_sizes)
            )
        ])

    def forward(self, wavlm, pitch, ld):
        """
        Computes forward pass of neural network.
        :param mels (torch.Tensor): mel-spectrogram acoustic features of shape [B, n_mels, T//hop_length]
        :param yn (torch.Tensor): noised signal `y_n` of shape [B, T]
        :return (torch.Tensor): epsilon noise
        """
        ## Prepare inputs
        # wavlm: B, 1024, T
        # pitch: B, T
        # ld: B, T
        assert len(wavlm.shape) == 3 # B, n_mels, T

        pitch = pitch.unsqueeze(1)        
        ld = ld.unsqueeze(1)
        assert len(pitch.shape) == 3  # B, 1, T
        assert len(ld.shape) == 3  # B, 1, T

        # Generate NHV conditions
        if self.nhv_cat_type == 'PLS':
            nhv_ld = ld
            nhv_wavlm = F.interpolate(wavlm, size=nhv_ld.shape[2], mode='nearest')
            nhv_conditions = torch.cat((nhv_ld, nhv_wavlm), dim=1) # B, (1+1024), T
        else:
            raise NameError('Unknown nhv cat type: {self.nhv_cat_type}')
        
        nhv_conditions = nhv_conditions.transpose(1, 2) # B, T, n_emb

        # Generate NHV harmonic signals
        nhv_noise = torch.normal(0, self.noise_std, (nhv_conditions.size(0), 1, nhv_conditions.size(1)*self.hop_size)).to(nhv_conditions.device)
        nhv_pitch = pitch.transpose(1, 2) # B, T, 1

        raw_harmonic, filtered_harmonic = self.nhv(nhv_noise, nhv_conditions, nhv_pitch)

        # Linear interpolate loudness to audio_rate
        upsampled_ld = F.interpolate(ld, scale_factor=self.hop_size, mode='linear')
        if self.harmonic_type == 0:
            upsampled_pitch = raw_harmonic
        elif self.harmonic_type == 1:
            upsampled_pitch = filtered_harmonic
        elif self.harmonic_type == 2:
            upsampled_pitch = torch.cat((raw_harmonic, filtered_harmonic), dim=1)
        else:
            raise NameError(f'unknown harmonic type: {self.harmonic_type}')

        # Downsampling stream + Linear Modulation statistics calculation
        ld_statistics = []
        dblock_outputs = self.ld_dblock_preconv(upsampled_ld)
        scale, shift = self.ld_films[0](x=dblock_outputs)
        ld_statistics.append([scale, shift])
        for dblock, film in zip(self.ld_dblocks, self.ld_films[1:]):
            dblock_outputs = dblock(dblock_outputs)
            scale, shift = film(x=dblock_outputs)
            ld_statistics.append([scale, shift])
        ld_statistics = ld_statistics[::-1]

        pitch_statistics = []
        dblock_outputs = self.pitch_dblock_preconv(upsampled_pitch)
        scale, shift = self.pitch_films[0](x=dblock_outputs)
        pitch_statistics.append([scale, shift])
        for dblock, film in zip(self.pitch_dblocks, self.pitch_films[1:]):
            dblock_outputs = dblock(dblock_outputs)
            scale, shift = film(x=dblock_outputs)
            pitch_statistics.append([scale, shift])
        pitch_statistics = pitch_statistics[::-1]
        
        # Upsampling stream
        condition = wavlm
        ublock_outputs = self.ublock_preconv(condition)
        for i, ublock in enumerate(self.ublocks):
            ld_scale, ld_shift = ld_statistics[i]
            pitch_scale, pitch_shift = pitch_statistics[i]
            ublock_outputs = ublock(x=ublock_outputs, scale=ld_scale+pitch_scale, shift=ld_shift+pitch_shift)
        outputs = self.ublock_postconv(ublock_outputs)
        return outputs.squeeze(1)

