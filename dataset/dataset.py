import numpy as np
import torch
from torch.utils.data import Dataset
import json
import random
from pathlib import Path
import soundfile as sf

class SVCDataset(Dataset):
    def __init__(self, root, n_samples, sampling_rate, hop_size, mode):
        self.root = Path(root)
        self.n_samples = n_samples
        self.sampling_rate = sampling_rate
        self.hop_size = hop_size
        self.n_frames = int(n_samples/hop_size)
        
        with open(self.root / f"{mode}.json") as file:
            metadata = json.load(file)

        self.metadata = []
        for audio_path, wavlm_path, pitch_path, ld_path in metadata:
            self.metadata.append([audio_path, wavlm_path, pitch_path, ld_path])
            
        print(mode, 'n_samples:', n_samples, 'metadata:', len(self.metadata))
        random.shuffle(self.metadata)
    
    def load_wav(self, audio_path):
        wav, fs = sf.read(audio_path)
        assert fs == self.sampling_rate, f'Audio {audio_path} sampling rate is not {self.sampling_rate} Hz.'
        peak = np.abs(wav).max()
        if peak > 1.0:
            wav /= peak
        return wav

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        audio_path, wavlm_path, pitch_path, ld_path = self.metadata[index]

        audio = self.load_wav(audio_path)
        wavlm = torch.load(wavlm_path)
        if isinstance(wavlm, torch.Tensor):
            wavlm = wavlm.numpy().T # (1024, T)
        else:
            wavlm = np.squeeze(wavlm)
        pitch = np.load(pitch_path)
        ld = np.load(ld_path)

        wavlm_frames = int(self.n_frames/2)

        assert pitch.shape[0] == ld.shape[0], f'{audio_path}: Length Mismatch: pitch length ({pitch.shape[0]}), ld length ({ld.shape[0]})'

        # Align features, the hop size for wavlm is 20 ms, while the hop size for pitch/ld is 10 ms.
        seq_len = wavlm.shape[-1] * 2
        if seq_len > pitch.shape[0]:
            p = seq_len - pitch.shape[0]
            pitch = np.pad(pitch, (0, p), mode='edge')
            ld = np.pad(ld, (0, p), mode='edge')
        else:
            pitch = pitch[:seq_len]
            ld = ld[:seq_len]

        # To ensure upsampling/downsampling will be processed in a right way for full signals
        p = seq_len * self.hop_size - audio.shape[-1]
        if p > 0:
            audio = np.pad(audio, (0, p), mode='reflect')
        else:
            audio = audio[:seq_len * self.hop_size]

        if audio.shape[0] >= self.n_samples:
            pos = random.randint(0, wavlm.shape[-1] - wavlm_frames)
            wavlm = wavlm[:, pos:pos+wavlm_frames]
            pitch = pitch[pos*2:pos*2+self.n_frames]
            ld = ld[pos*2:pos*2+self.n_frames]
            audio = audio[pos*2*self.hop_size:(pos*2+self.n_frames)*self.hop_size]
        else:
            wavlm = np.pad(wavlm, ((0, 0), (0, wavlm_frames - wavlm.shape[-1])), mode='edge')
            pitch = np.pad(pitch, (0, self.n_frames-pitch.shape[0]), mode='edge')
            ld = np.pad(ld, (0, self.n_frames-ld.shape[0]), mode='edge')
            audio = np.pad(audio, (0, self.n_samples-audio.shape[0]), mode='edge')

        assert audio.shape[0] == self.n_samples, f'{audio_path}: audio length is not enough, {wavlm.shape}, {audio.shape}, {p}'
        assert pitch.shape[0] == self.n_frames, f'{audio_path}: pitch length is not enough, {wavlm.shape}, {pitch.shape}, {self.n_frames}'
        assert ld.shape[0] == self.n_frames, f'{audio_path}: ld length is not enough, {wavlm.shape}, {ld.shape}, {self.n_frames}'
        assert wavlm.shape[-1] == wavlm_frames, f'{audio_path}: wavlm length is not enough, {wavlm.shape}, {self.n_frames}'

        return (torch.from_numpy(wavlm).to(dtype=torch.float), torch.from_numpy(pitch).to(dtype=torch.float), torch.from_numpy(ld).to(dtype=torch.float)), torch.from_numpy(audio).to(dtype=torch.float)