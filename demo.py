import os
import torch
import torchaudio as ta
import warnings
import numpy as np
import librosa.display
import requests
import re
from speechbrain.inference import EncoderClassifier

warnings.filterwarnings('ignore')
from torch.cuda.amp import autocast
from torchaudio.prototype.pipelines import HIFIGAN_VOCODER_V3_LJSPEECH
import matplotlib.pyplot as plt
from f_params2 import *
from f_tiny_diffusion import TinyDiffusion


LOG_DIR = "log/n3"
SAMPLE_DIR = os.path.join(LOG_DIR, "samples")
os.makedirs(SAMPLE_DIR, exist_ok=True)

# HiFi-GAN Vocoder
bundle = HIFIGAN_VOCODER_V3_LJSPEECH
VOCODER = bundle.get_vocoder().to("cuda").eval()
SAMPLE_RATE = 22050


vocab = {
    "<PAD>": 0,
    "<UNK>": 1,
    "<SILENCE>": 1,
    "AA0": 2,
    "AA1": 3,
    "AA2": 4,
    "AE0": 5,
    "AE1": 6,
    "AE2": 7,
    "AH0": 8,
    "AH1": 9,
    "AH2": 10,
    "AO0": 11,
    "AO1": 12,
    "AO2": 13,
    "AW0": 14,
    "AW1": 15,
    "AW2": 16,
    "AY0": 17,
    "AY1": 18,
    "AY2": 19,
    "B": 20,
    "CH": 21,
    "D": 22,
    "DH": 23,
    "EH0": 24,
    "EH1": 25,
    "EH2": 26,
    "ER0": 27,
    "ER1": 28,
    "ER2": 29,
    "EY0": 30,
    "EY1": 31,
    "EY2": 32,
    "F": 33,
    "G": 34,
    "HH": 35,
    "IH0": 36,
    "IH1": 37,
    "IH2": 38,
    "IY0": 39,
    "IY1": 40,
    "IY2": 41,
    "JH": 42,
    "K": 43,
    "L": 44,
    "M": 45,
    "N": 46,
    "NG": 47,
    "OW0": 48,
    "OW1": 49,
    "OW2": 50,
    "OY0": 51,
    "OY1": 52,
    "OY2": 53,
    "P": 54,
    "R": 55,
    "S": 56,
    "SH": 57,
    "T": 58,
    "TH": 59,
    "UH0": 60,
    "UH1": 61,
    "UH2": 62,
    "UW0": 63,
    "UW1": 64,
    "UW2": 65,
    "V": 66,
    "W": 67,
    "Y": 68,
    "Z": 69,
    "ZH": 70,
    "spn": 71
}


def download_cmudict(path="cmudict.dict"):
    if not os.path.exists(path):
        url = "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            with open(path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            print(f"下载CMUdict到 {path}")
        except Exception as e:
            raise ValueError(f"cant download CMUdict：{e}")
    return path


def load_cmudict(path):
    cmudict = {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith(';;;'): continue
                parts = line.strip().split()
                if len(parts) < 2: continue
                word = parts[0].lower()
                if '(' in word:
                    word = word.split('(')[0]
                phonemes = parts[1:]
                if word not in cmudict:
                    cmudict[word] = phonemes
    except Exception as e:
        raise ValueError(f"cant load CMUdict：{e}")
    return cmudict


def text_to_phonemes(text, cmudict, vocab):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    phonemes = []
    for word in words:
        if word in cmudict:
            phonemes.extend(cmudict[word])
            phonemes.append('<SILENCE>')
        else:
            phonemes.append('<UNK>')
    if phonemes and phonemes[-1] == '<SILENCE>':
        phonemes.pop()
    phoneme_ids = [vocab.get(p, vocab['<UNK>']) for p in phonemes]
    return phonemes, phoneme_ids


def save_plot(mel_tensor, filepath, mel_mean=-8.122798642, mel_std=2.1809869538):
    mel_np = mel_tensor.numpy()
    print(
        f"[Save Plot Raw] min={mel_np.min():.2f}, max={mel_np.max():.2f}, mean={mel_np.mean():.2f}, std={mel_np.std():.2f}")
    print(f"[Save Plot Raw] Percentage near 0 dB: {(mel_np > -1.0).mean() * 100:.2f}%")
    hist, bins = np.histogram(mel_np.flatten(), bins=50, range=(-30, 0))
    print(f"[Save Plot Raw] Histogram near 0 dB: {hist[-5:] / hist.sum() * 100} (bins: {bins[-6:]})")
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_np, sr=SAMPLE_RATE, hop_length=256, fmin=0, fmax=8000, cmap='viridis')
    plt.colorbar(format='%+2.0f')
    plt.title('Mel Spectrogram (Raw Log-Mel)')
    plt.xlabel('Time')
    plt.ylabel('Mel Frequency')
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


def sample_and_save(wav_path, text):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





    model = TinyDiffusion(
        n_feats=n_feats, dec_dim=dec_dim, spk_emb_dim=decoder_spk_emb_dim,
        beta_min=beta_min, beta_max=beta_max, pe_scale=pe_scale, n_timesteps=n_timesteps,
        d_phoneme=d_phoneme, d_speaker=d_speaker
    ).to(device)
    RESUME_CHECKPOINT = os.path.join("logs/test30/checkpoints", "tiny_diffusion_epoch_112.pt")
    if os.path.exists(RESUME_CHECKPOINT):
        print(f"Loading model from {RESUME_CHECKPOINT}")
        ckpt = torch.load(RESUME_CHECKPOINT, map_location=device)
        state_dict = ckpt.get('model', ckpt)
        model.load_state_dict(state_dict, strict=False)
        print("Model loaded successfully.")
    else:
        raise FileNotFoundError(f"Checkpoint {RESUME_CHECKPOINT} not found.")
    model.eval()


    ecapa = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/ecapa",
        run_opts={"device": device}
    ).to(device).eval()
    waveform, sr = ta.load(wav_path)
    if sr != SAMPLE_RATE:
        waveform = ta.functional.resample(waveform, sr, SAMPLE_RATE)
    waveform = waveform.to(device)
    with torch.no_grad():
        spk_emb = ecapa.encode_batch(waveform).squeeze().cpu()


    cmudict_path = download_cmudict()
    cmudict = load_cmudict(cmudict_path)
    phonemes, phoneme_ids = text_to_phonemes(text, cmudict, vocab)


    print("转换后的phonemes:", ' '.join(phonemes))
    print("对应的phoneme_ids:", phoneme_ids)


    if not phoneme_ids:
        raise ValueError("文本转换后无有效phonemes。")
    x = torch.tensor(phoneme_ids, dtype=torch.long).unsqueeze(0).to(device)
    x_lengths = torch.tensor([len(phoneme_ids)], dtype=torch.long).to(device)
    spk = spk_emb.unsqueeze(0).to(device)


    with torch.no_grad():
        with autocast():
            gen_mel = model.sample(x, spk, x_lengths, n_timesteps=200, temperature=1.5, cfg=False, cfg_scale=1.0)


    gen_mel_path = os.path.join(SAMPLE_DIR, 'sample_gen_mel.png')
    save_plot(gen_mel.squeeze(0).cpu(), gen_mel_path)


    wav = VOCODER(gen_mel.to(device))
    wav = wav.squeeze(1)
    wav_rms = wav.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-6)
    wav = wav * (0.2 / wav_rms)
    wav = torch.tanh(wav)
    pred_wav_path = os.path.join(SAMPLE_DIR, 'sample_gen.wav')
    ta.save(pred_wav_path, wav.cpu(), SAMPLE_RATE)

    pred_rms = wav.pow(2).mean(dim=-1).sqrt().item()
    pred_duration = wav.shape[-1] / SAMPLE_RATE
    print(f"Generated WAV Path: {pred_wav_path}")
    print(f"Generated RMS: {pred_rms:.6f}")
    print(f"Generated Duration: {pred_duration:.2f} seconds")
    print(f"Text: {text.strip()}")


if __name__ == "__main__":
    
    wav_path = "data/LibriTTS-R/train-clean-100/103/1241/103_1241_000000_000001.wav"#path for extracting speaker embedding
    text = "This is a test."
    sample_and_save(wav_path, text)