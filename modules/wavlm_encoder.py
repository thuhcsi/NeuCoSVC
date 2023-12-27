import resampy
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from torch import Tensor
from torchaudio.sox_effects import apply_effects_tensor

from modules.wavlm.WavLM import WavLM, WavLMConfig


class WavLMEncoder(nn.Module):

    def __init__(self,
        ckpt_path,
        device='cpu'
    ):
        """ 
        Load the WavLM large checkpoint from the original paper. See https://github.com/microsoft/unilm/tree/master/wavlm for details. 
        
        Args:
            ckpt_path : checkpoint path of WavLM.

        """
        super().__init__()
        wavlm_check_point = torch.load(ckpt_path)
        cfg = WavLMConfig(wavlm_check_point['cfg'])
        wavlm = WavLM(cfg)
        wavlm.load_state_dict(wavlm_check_point['model'])
        wavlm = wavlm.to(device)
        
        # store wavlm
        self.wavlm = wavlm.eval()
        self.device = torch.device(device)
        self.sr = 16000
    
    @torch.inference_mode()
    def get_features(self, path, output_layer=None, weights=None, vad_trigger_level=0):
        """
        Returns the features of the waveform at `path` as a tensor of shape (seq_len, dim).
        Optionally, performs Voice Activity Detection (VAD) trimming on the start and end of the waveform
        using the `vad_trigger_level`.

        If the `output_layer` is specified, the result of the corresponding layer is returned.
        If the `weights` are specified, the weighted result of the corresponding layers is returned.
        If neither `output_layer` nor `weights` are specified, the result of all layers is returned.

        Args:
            path (str or torch.Tensor): Path to the audio waveform file or a tensor representing the waveform.
            output_layer (int, optional): Index of the layer to extract the features from. Defaults to None.
            weights (torch.Tensor, optional): Weights to apply to the features of each layer. Defaults to None.
            vad_trigger_level (float, optional): VAD trigger level for trimming silence. Defaults to 0.

        Returns:
            torch.Tensor: Extracted WavLM features of the waveform.

        """
        # load audio
        if type(path) in [str, Path]:
            x, sr = torchaudio.load(path, normalize=True)
            if sr != self.sr:
                print(f'Original audio sr is {sr}, change it to {self.sr}.')
                x = resampy.resample(x.numpy(), sr, self.sr, axis=1)
                x = torch.from_numpy(x).to(dtype=torch.float)
                sr = self.sr
        else:
            x: Tensor = path
            sr = self.sr
            if x.dim() == 1: x = x[None]
        assert sr == self.sr, f"input audio sample rate must be 16kHz. Got {sr}"
        
        # trim silence from front and back
        if vad_trigger_level > 1e-3:
            transform = T.Vad(sample_rate=sr, trigger_level=vad_trigger_level)
            x_front_trim = transform(x)
            waveform_reversed, sr = apply_effects_tensor(x_front_trim, sr, [["reverse"]])
            waveform_reversed_front_trim = transform(waveform_reversed)
            waveform_end_trim, sr = apply_effects_tensor(
                waveform_reversed_front_trim, sr, [["reverse"]]
            )
            x = waveform_end_trim

        # extract the representation of each layer
        wav_input_16khz = x.to(self.device)
        if output_layer is not None:
            # use fastpath
            features = self.wavlm.extract_features(wav_input_16khz, output_layer=output_layer, ret_layer_results=False)[0]
            features = torch.squeeze(features)
        else:
            # use slower weighted
            rep, layer_results = self.wavlm.extract_features(wav_input_16khz, output_layer=self.wavlm.cfg.encoder_layers, ret_layer_results=True)[0]
            features = torch.cat([x.transpose(0, 1) for x, _ in layer_results], dim=0) # (n_layers, seq_len, dim)
            # save full sequence
            if weights is not None:
                features = (features*weights[:, None] ).sum(dim=0) # (1, seq_len, dim)
        
        return features