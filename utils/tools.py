import os
import torch
import numpy as np
import soundfile as sf
import librosa


def fast_cosine_dist(source_feats: torch.Tensor, matching_pool: torch.Tensor, device='cpu'):
    """
    Computes the cosine distance between source features and a matching pool of features.
    Like torch.cdist, but fixed dim=-1 and for cosine distance.

    Args:
        source_feats (torch.Tensor): Tensor of source features with shape (n_source_feats, feat_dim).
        matching_pool (torch.Tensor): Tensor of matching pool features with shape (n_matching_feats, feat_dim).
        device (str, optional): Device to perform the computation on. Defaults to 'cpu'.

    Returns:
        torch.Tensor: Tensor of cosine distances between the source features and the matching pool features.

    """
    source_feats = source_feats.to(device)
    matching_pool = matching_pool.to(device)
    source_norms = torch.norm(source_feats, p=2, dim=-1)
    matching_norms = torch.norm(matching_pool, p=2, dim=-1)
    dotprod = -torch.cdist(source_feats[None].to(device), matching_pool[None], p=2)[0]**2 + source_norms[:, None]**2 + matching_norms[None]**2
    dotprod /= 2

    dists = 1 - ( dotprod / (source_norms[:, None] * matching_norms[None]) )
    return dists


def load_wav(wav_path, sr=None):
    """
    Loads a waveform from a wav file.

    Args:
        wav_path (str): Path to the wav file.
        sr (int, optional): Target sample rate. 
            If `sr` is specified and the loaded audio has a different sample rate, an AssertionError is raised. 
            Defaults to None.

    Returns:
        Tuple[np.ndarray, int]: Tuple containing the loaded waveform as a NumPy array and the sample rate.

    """
    wav, fs = librosa.load(wav_path, sr=sr)
    assert wav.ndim == 1, 'Single-channel audio is required.'
    assert sr is None or fs == sr, f'{sr} kHz audio is required. Got {fs}'
    peak = np.abs(wav).max()
    if peak > 1.0:
        wav /= peak
    return wav, fs


class ConfigWrapper(object):
    """
    Wrapper dict class to avoid annoying key dict indexing like:
    `config.sample_rate` instead of `config["sample_rate"]`.
    """
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = ConfigWrapper(**v)
            self[k] = v
      
    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def to_dict_type(self):
        return {
            key: (value if not isinstance(value, ConfigWrapper) else value.to_dict_type())
            for key, value in dict(**self).items()
        }

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


def save_checkpoint(steps, epochs, model, optimizer, scheduler, checkpoint_path, dst_train=False):
    """Save checkpoint.

    Args:
        checkpoint_path (str): Checkpoint path to be saved.

    """
    state_dict = {
        "optimizer": {
            "generator": optimizer["generator"].state_dict(),
            "discriminator": optimizer["discriminator"].state_dict(),
        },
        "scheduler": {
            "generator": scheduler["generator"].state_dict(),
            "discriminator": scheduler["discriminator"].state_dict(),
        },
        "steps": steps,
        "epochs": epochs,
    }
    if dst_train:
        state_dict["model"] = {
            "generator": model["generator"].module.state_dict(),
            "discriminator": model["discriminator"].module.state_dict(),
        }
    else:
        state_dict["model"] = {
            "generator": model["generator"].state_dict(),
            "discriminator": model["discriminator"].state_dict(),
        }

    if not os.path.exists(os.path.dirname(checkpoint_path)):
        os.makedirs(os.path.dirname(checkpoint_path))
    torch.save(state_dict, checkpoint_path)


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, load_only_params=False, dst_train=False):
    """Load checkpoint.

    Args:
        checkpoint_path (str): Checkpoint path to be loaded.
        load_only_params (bool): Whether to load only model parameters.

    """
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if dst_train:
        model["generator"].module.load_state_dict(
            state_dict["model"]["generator"]
        )
        model["discriminator"].module.load_state_dict(
            state_dict["model"]["discriminator"]
        )
    else:
        model["generator"].load_state_dict(state_dict["model"]["generator"])
        model["discriminator"].load_state_dict(
            state_dict["model"]["discriminator"]
        )
    optimizer["generator"].load_state_dict(
        state_dict["optimizer"]["generator"]
    )
    optimizer["discriminator"].load_state_dict(
        state_dict["optimizer"]["discriminator"]
    )
    scheduler["generator"].load_state_dict(
        state_dict["scheduler"]["generator"]
    )
    scheduler["discriminator"].load_state_dict(
        state_dict["scheduler"]["discriminator"]
    )
    
    steps = state_dict["steps"]
    epochs = state_dict["epochs"]

    return steps, epochs
