import os
import json
import time
import argparse
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from modules.FastSVC import SVCNN
from modules.wavlm_encoder import WavLMEncoder
from utils.pitch_ld_extraction import extract_loudness, extract_pitch_ref as extract_pitch
from utils.tools import ConfigWrapper, fast_cosine_dist

# the 6th layer features of wavlm are used to audio synthesis
SPEAKER_INFORMATION_LAYER = 6
# the mean of last 5 layers features of wavlm are used for matching in kNN
CONTENT_INFORMATION_LAYER = [20, 21, 22, 23, 24]


def VoiceConverter(test_utt: str, ref_utt: str, out_path: str, svc_mdl: SVCNN, wavlm_encoder: WavLMEncoder, f0_factor: float, speech_enroll=False, device=torch.device('cpu')):
    """
    Perform singing voice conversion and save the resulting waveform to `out_path`.

    Args:
        test_utt (str): Path to the source singing waveform (24kHz, single-channel).
        ref_utt (str): Path to the reference waveform from the target speaker (single-channel, not less than 16kHz).
        out_path (str): Path to save the converted singing audio.
        svc_mdl (SVCNN): Loaded FastSVC model with neural harmonic filters.
        wavlm_encoder (WavLMEncoder): Loaded WavLM Encoder.
        f0_factor (float): F0 shift factor.
        speech_enroll (bool, optional): Whether the reference audio is a speech clip or a singing clip. Defaults to False.
        device (torch.device, optional): Device to perform the conversion on. Defaults to cpu.

    """
    # Preprocess audio and extract features.
    print('Processing feats.')
    applied_weights = F.one_hot(torch.tensor(CONTENT_INFORMATION_LAYER), num_classes=25).float().mean(axis=0).to(device)[:, None]

    ld = extract_loudness(test_utt)
    pitch, f0_factor = extract_pitch(test_utt, ref_utt, predefined_factor=f0_factor, speech_enroll=speech_enroll)
    assert pitch.shape[0] == ld.shape[0], f'{test_utt} Length Mismatch: pitch length ({pitch.shape[0]}), ld length ({ld.shape[0]}).'

    query_feats = wavlm_encoder.get_features(test_utt, weights=applied_weights)
    matching_set = wavlm_encoder.get_features(ref_utt, weights=applied_weights)
    synth_set = wavlm_encoder.get_features(ref_utt, output_layer=SPEAKER_INFORMATION_LAYER)

    # Calculate the distance between the query feats and the matching feats
    dists = fast_cosine_dist(query_feats, matching_set, device=device)
    best = dists.topk(k=4, largest=False, dim=-1)
    # Replace query features with corresponding nearest synth feats
    prematched_wavlm = synth_set[best.indices].mean(dim=1).transpose(0, 1)  # (T, 1024)

    # Align the features: the hop_size of the wavlm feature is twice that of pitch and loudness.
    seq_len = prematched_wavlm.shape[1] * 2
    if seq_len > pitch.shape[0]:
        p = seq_len - pitch.shape[0]
        pitch = np.pad(pitch, (0, p), mode='edge')
        ld = np.pad(ld, (0, p), mode='edge')
    else:
        pitch = pitch[:seq_len]
        ld = ld[:seq_len]

    in_feats = [prematched_wavlm.unsqueeze(0), torch.from_numpy(pitch).to(dtype=torch.float).unsqueeze(0),
                torch.from_numpy(ld).to(dtype=torch.float).unsqueeze(0)]
    in_feats = tuple([x_.to(device) for x_ in in_feats])

    # Inference
    print('Inferencing.')
    with torch.no_grad():
        y_ = svc_mdl(*in_feats)

    # Save converted audio.
    print('Saving audio.')
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    y_ = y_.unsqueeze(0)
    y_ = np.clip(y_.view(-1).cpu().numpy(), -1, 1)
    sf.write(out_path, y_, 24000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--src_wav_path',
        required=True, type=str, help='The audio path for the source singing utterance.'
    )
    parser.add_argument(
        '--ref_wav_path',
        required=True, type=str, help='The audio path for the reference utterance.'
    )
    parser.add_argument(
        '--out_path',
        required=True, type=str, help='The audio path for the reference utterance.'
    )
    parser.add_argument(
        '-cfg', '--config_file',
        type=str, default='configs/config.json',
        help='The model configuration file.'
    )
    parser.add_argument(
        '-ckpt', '--ckpt_path',
        type=str,
        default='pretrained/model.pkl',
        help='The model checkpoint path for loading.'
    )
    parser.add_argument(
        '-f0factor', '--f0_factor', type=float, default=0.0,
        help='Adjust the pitch of the source singing to match the vocal range of the target singer. \
            The default value is 0.0, which means no pitch adjustment is applied (equivalent to f0_factor = 1.0)'
    )
    parser.add_argument(
        '--speech_enroll', action='store_true',
        help='When using speech as the reference audio, the pitch of the reference audio will be increased by 1.2 times \
            when performing pitch shift to cover the pitch gap between singing and speech. \
            Note: This option is invalid when f0_factor is specified.'
    )

    args = parser.parse_args()

    t0 = time.time()

    f0factor = args.f0_factor
    speech_enroll_flag = args.speech_enroll

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'using {device} for inference.')

    # Loading model and parameters.
    # load svc model
    cfg = args.config_file
    model_path = args.ckpt_path

    print('Loading svc model configurations.')
    with open(cfg) as f:
        config = ConfigWrapper(**json.load(f))

    svc_mdl = SVCNN(config)

    state_dict = torch.load(model_path, map_location='cpu')
    svc_mdl.load_state_dict(state_dict['model']['generator'], strict=False)
    svc_mdl.to(device)
    svc_mdl.eval()
    # load wavlm model
    wavlm_encoder = WavLMEncoder(ckpt_path='pretrained/WavLM-Large.pt', device=device)
    print('wavlm loaded.')
    # End loading model and parameters.
    t1 = time.time()
    print(f'loading models cost {t1-t0:.2f}s.')

    VoiceConverter(test_utt=args.src_wav_path, ref_utt=args.ref_wav_path, out_path=args.out_path,
                   svc_mdl=svc_mdl, wavlm_encoder=wavlm_encoder,
                   f0_factor=f0factor, speech_enroll=speech_enroll_flag, device=device)

    t2 = time.time()
    print(f'converting costs {t2-t1:.2f}s.')
