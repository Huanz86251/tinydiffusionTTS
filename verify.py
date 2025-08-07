# import os
# import torch
# import torchaudio as ta
# from torch.utils.data import DataLoader
# from torch.cuda.amp import autocast
# from torchaudio.prototype.pipelines import HIFIGAN_VOCODER_V3_LJSPEECH
# import matplotlib.pyplot as plt
# import scipy.ndimage
# import scipy.signal
# from f_params2 import *
# from f_tiny_diffusion import TinyDiffusion
# from f_data_stage2 import PhonemeMelSpeakerDataset, PhonemeMelSpeakerBatchCollate
#
# # Paths
# TRAIN_FILELIST = "data/LibriTTS-R/full_libritts_aligned_filelist.txt"
# SPEAKER_EMBED_PATH = "data/LibriTTS-R/speaker_embeddings_192_r_aligned.pt"
# LOG_DIR = "log/n2"
# SAMPLE_DIR = os.path.join(LOG_DIR, "samples")
# os.makedirs(SAMPLE_DIR, exist_ok=True)
#
# # HiFi-GAN Vocoder
# bundle = HIFIGAN_VOCODER_V3_LJSPEECH
# VOCODER = bundle.get_vocoder().to("cuda").eval()
# SAMPLE_RATE = 22050
#
#
# def save_plot(mel_tensor, filepath, mel_mean=-8.122798642, mel_std=2.1809869538):
#     mel_np = mel_tensor.numpy()
#     print(
#         f"[Save Plot Raw] min={mel_np.min():.2f}, max={mel_np.max():.2f}, mean={mel_np.mean():.2f}, std={mel_np.std():.2f}")
#     print(f"[Save Plot Raw] Percentage near 0 dB: {(mel_np > -1.0).mean() * 100:.2f}%")
#     mel_np = (mel_np - mel_mean) / (mel_std + 1e-8)
#     print(
#         f"[Save Plot Norm] min={mel_np.min():.2f}, max={mel_np.max():.2f}, mean={mel_np.mean():.2f}, std={mel_np.std():.2f}")
#     plt.figure(figsize=(10, 4))
#     vmin = mel_np.min()
#     vmax = mel_np.max()
#     plt.imshow(mel_np, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
#     plt.colorbar(format='%+2.0f')
#     plt.title('Mel Spectrogram (Re-standardized to N(0,1))')
#     plt.xlabel('Time')
#     plt.ylabel('Mel Frequency')
#     plt.tight_layout()
#     plt.savefig(filepath)
#     plt.close()
#
#
# def sample_and_save():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     # Dataset
#     full_dataset = PhonemeMelSpeakerDataset(TRAIN_FILELIST, SPEAKER_EMBED_PATH)
#
#     # Model
#     model = TinyDiffusion(
#         n_feats=n_feats, dec_dim=dec_dim, spk_emb_dim=decoder_spk_emb_dim,
#         beta_min=beta_min, beta_max=beta_max, pe_scale=pe_scale, n_timesteps=n_timesteps,
#         d_phoneme=d_phoneme, d_speaker=d_speaker
#     ).to(device)
#     # Load pre-trained model
#     RESUME_CHECKPOINT = os.path.join("logs/test30/checkpoints", "tiny_diffusion_epoch_112.pt")
#     if os.path.exists(RESUME_CHECKPOINT):
#         print(f"Loading model from {RESUME_CHECKPOINT}")
#         ckpt = torch.load(RESUME_CHECKPOINT, map_location=device)
#         state_dict = ckpt.get('model', ckpt)
#         model.load_state_dict(state_dict, strict=False)
#         print("Model loaded successfully.")
#     else:
#         raise FileNotFoundError(f"Checkpoint {RESUME_CHECKPOINT} not found.")
#
#     model.eval()
#
#     # Sample one example
#     with torch.no_grad():
#         sample_batch = full_dataset.sample_test_batch(1)
#         x = sample_batch[0]['x'].unsqueeze(0).to(device)
#         spk = sample_batch[0]['spk'].unsqueeze(0).to(device)
#         x_lengths = torch.tensor([sample_batch[0]['x_lengths']]).to(device)
#         gt_wav = sample_batch[0]['gt_wave']
#         gt_text = sample_batch[0]['text']
#         gt_wav_path = sample_batch[0]['wav_path']
#
#         # Generate predicted Mel
#         with autocast():
#             gen_mel = model.sample(x, spk, x_lengths, n_timesteps=200, temperature=0.8, cfg=True, cfg_scale=1.5)
#
#         # MEL 后处理
#         gen_mel_np = gen_mel.cpu().numpy()
#         gen_mel_np = scipy.ndimage.gaussian_filter(gen_mel_np, sigma=0.5)
#         gen_mel = torch.from_numpy(gen_mel_np).to(device)
#         gen_mel = torch.clamp(gen_mel, min=-30.0, max=0.0)
#         gen_mel = gen_mel * 0.4 - 6.0  # 加强压缩高值
#         gen_mel = torch.clamp(gen_mel, min=-30.0, max=-1.0)
#         low_freq_boost = torch.linspace(1.4, 1.0, steps=gen_mel.shape[1]).view(-1, 1).to(device)
#         gen_mel = gen_mel * low_freq_boost
#         gen_mel = torch.clamp(gen_mel, min=-30.0, max=-1.0)
#
#         # Save MEL spectrograms
#         gen_mel_path = os.path.join(SAMPLE_DIR, 'sample_gen_mel.png')
#         save_plot(gen_mel.squeeze(0).cpu(), gen_mel_path, mel_mean=full_dataset.mel_mean, mel_std=full_dataset.mel_std)
#         gt_mel_path = os.path.join(SAMPLE_DIR, 'sample_gt_mel.png')
#         save_plot(sample_batch[0]['y'].cpu(), gt_mel_path, mel_mean=full_dataset.mel_mean, mel_std=full_dataset.mel_std)
#
#         # Generate and save WAV
#         wav = VOCODER(gen_mel.to(device))
#         wav_rms = wav.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-6)
#         wav = wav * (0.2 / wav_rms)  # 提高 RMS
#         wav_np = wav.squeeze().cpu().numpy()  # Squeeze to 1D for filtfilt
#         b, a = scipy.signal.butter(4, 8000 / (SAMPLE_RATE / 2), btype='low')
#         wav_np = scipy.signal.filtfilt(b, a, wav_np)
#         wav_np = wav_np.copy()
#         wav = torch.from_numpy(wav_np).float()
#         wav = torch.tanh(wav)
#         pred_wav_path = os.path.join(SAMPLE_DIR, 'sample_gen.wav')
#         ta.save(pred_wav_path, wav.unsqueeze(0), SAMPLE_RATE)  # unsqueeze 确保 2D
#
#         # Save ground truth WAV
#         gt_wav_save_path = os.path.join(SAMPLE_DIR, 'sample_gt.wav')
#         ta.save(gt_wav_save_path, gt_wav.unsqueeze(0), SAMPLE_RATE)
#
#         # Print metrics
#         gt_rms = gt_wav.pow(2).mean().sqrt().item()
#         pred_rms = wav.pow(2).mean().sqrt().item()
#         gt_duration = gt_wav.shape[0] / SAMPLE_RATE
#         pred_duration = wav.shape[0] / SAMPLE_RATE
#         print(f"Ground Truth WAV Path: {gt_wav_save_path}")
#         print(f"Predicted WAV Path: {pred_wav_path}")
#         print(f"Ground Truth RMS: {gt_rms:.6f}")
#         print(f"Predicted RMS: {pred_rms:.6f}")
#         print(f"Ground Truth Duration: {gt_duration:.2f} seconds")
#         print(f"Predicted Duration: {pred_duration:.2f} seconds")
#         print(f"Text: {gt_text.strip()}")
#
#
# if __name__ == "__main__":
#     sample_and_save()
import os
import torch
import torchaudio as ta
import warnings
import numpy as np
import librosa.display
warnings.filterwarnings('ignore')
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torchaudio.prototype.pipelines import HIFIGAN_VOCODER_V3_LJSPEECH
import matplotlib.pyplot as plt
from f_params2 import *
from f_tiny_diffusion import TinyDiffusion
from f_data_stage2 import PhonemeMelSpeakerDataset, PhonemeMelSpeakerBatchCollate

# Paths
TRAIN_FILELIST = "data/LibriTTS-R/full_libritts_aligned_filelist.txt"
SPEAKER_EMBED_PATH = "data/LibriTTS-R/speaker_embeddings_192_r_aligned.pt"
LOG_DIR = "log/n2"
SAMPLE_DIR = os.path.join(LOG_DIR, "samples")
os.makedirs(SAMPLE_DIR, exist_ok=True)

# HiFi-GAN Vocoder
bundle = HIFIGAN_VOCODER_V3_LJSPEECH
VOCODER = bundle.get_vocoder().to("cuda").eval()
SAMPLE_RATE = 22050

def save_plot(mel_tensor, filepath, mel_mean=-8.122798642, mel_std=2.1809869538):
    mel_np = mel_tensor.numpy()
    print(f"[Save Plot Raw] min={mel_np.min():.2f}, max={mel_np.max():.2f}, mean={mel_np.mean():.2f}, std={mel_np.std():.2f}")
    print(f"[Save Plot Raw] Percentage near 0 dB: {(mel_np > -1.0).mean() * 100:.2f}%")
    hist, bins = np.histogram(mel_np.flatten(), bins=50, range=(-30, 0))
    print(f"[Save Plot Raw] Histogram near 0 dB: {hist[-5:]/hist.sum()*100} (bins: {bins[-6:]})")
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_np, sr=SAMPLE_RATE, hop_length=256, fmin=0, fmax=8000, cmap='viridis')
    plt.colorbar(format='%+2.0f')
    plt.title('Mel Spectrogram (Raw Log-Mel)')
    plt.xlabel('Time')
    plt.ylabel('Mel Frequency')
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

def sample_and_save():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset
    full_dataset = PhonemeMelSpeakerDataset(TRAIN_FILELIST, SPEAKER_EMBED_PATH)

    # Model
    model = TinyDiffusion(
        n_feats=n_feats, dec_dim=dec_dim, spk_emb_dim=decoder_spk_emb_dim,
        beta_min=beta_min, beta_max=beta_max, pe_scale=pe_scale, n_timesteps=n_timesteps,
        d_phoneme=d_phoneme, d_speaker=d_speaker
    ).to(device)
    # Load pre-trained model
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

    # Sample one example
    with torch.no_grad():
        sample_batch = full_dataset.sample_test_batch(1)
        x = sample_batch[0]['x'].unsqueeze(0).to(device)
        spk = sample_batch[0]['spk'].unsqueeze(0).to(device)
        x_lengths = torch.tensor([sample_batch[0]['x_lengths']]).to(device)
        gt_wav = sample_batch[0]['gt_wave']
        gt_text = sample_batch[0]['text']
        gt_wav_path = sample_batch[0]['wav_path']

        # Generate predicted Mel
        with autocast():
            gen_mel = model.sample(x, spk, x_lengths, n_timesteps=200, temperature=1.5, cfg=False, cfg_scale=1.0)

        # Save MEL spectrograms (no post-processing)
        gen_mel_path = os.path.join(SAMPLE_DIR, 'sample_gen_mel.png')
        save_plot(gen_mel.squeeze(0).cpu(), gen_mel_path, mel_mean=full_dataset.mel_mean, mel_std=full_dataset.mel_std)
        gt_mel_path = os.path.join(SAMPLE_DIR, 'sample_gt_mel.png')
        save_plot(sample_batch[0]['y'].cpu(), gt_mel_path, mel_mean=full_dataset.mel_mean, mel_std=full_dataset.mel_std)

        # Generate and save WAV (no post-processing except RMS)
        wav = VOCODER(gen_mel.to(device))
        wav = wav.squeeze(1)  # 明确调整为 [1, T]
        wav_rms = wav.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-6)
        wav = wav * (0.2 / wav_rms)  # Keep RMS normalization
        wav = torch.tanh(wav)
        pred_wav_path = os.path.join(SAMPLE_DIR, 'sample_gen.wav')
        ta.save(pred_wav_path, wav.cpu(), SAMPLE_RATE)

        # Save ground truth WAV
        gt_wav_save_path = os.path.join(SAMPLE_DIR, 'sample_gt.wav')
        ta.save(gt_wav_save_path, gt_wav.unsqueeze(0), SAMPLE_RATE)

        # Print metrics
        gt_rms = gt_wav.pow(2).mean().sqrt().item()
        pred_rms = wav.pow(2).mean(dim=-1).sqrt().item()
        gt_duration = gt_wav.shape[0] / SAMPLE_RATE
        pred_duration = wav.shape[-1] / SAMPLE_RATE
        print(f"Ground Truth WAV Path: {gt_wav_save_path}")
        print(f"Predicted WAV Path: {pred_wav_path}")
        print(f"Ground Truth RMS: {gt_rms:.6f}")
        print(f"Predicted RMS: {pred_rms:.6f}")
        print(f"Ground Truth Duration: {gt_duration:.2f} seconds")
        print(f"Predicted Duration: {pred_duration:.2f} seconds")
        print(f"Text: {gt_text.strip()}")

if __name__ == "__main__":
    sample_and_save()