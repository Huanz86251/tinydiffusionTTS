import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import warnings
from torch.cuda.amp import autocast, GradScaler
import os
from torchaudio.prototype.pipelines import HIFIGAN_VOCODER_V3_LJSPEECH
warnings.filterwarnings('ignore')
import torchaudio as ta
from f_params2 import *
from f_tiny_diffusion import TinyDiffusion
from f_data_stage2 import PhonemeMelSpeakerDataset, PhonemeMelSpeakerBatchCollate
from utils import plot_tensor
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import os


def save_plot(mel_tensor, filepath):
    mel_np = mel_tensor.numpy()  # 转numpy

    # 打印stats（normalized范围）
    print(f"[Save Plot] Stats: min={mel_np.min():.2f}, max={mel_np.max():.2f}, mean={mel_np.mean():.2f}")

    # 用imshow（直接用你的代码思路）
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_np, aspect='auto', origin='lower', cmap='plasma', vmin=-3, vmax=3)  # 针对normalized Mel
    plt.colorbar(format='%+2.0f')
    plt.title('Generated Mel Spectrogram (Normalized)')
    plt.xlabel('Time')
    plt.ylabel('Mel Frequency')
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
bundle = HIFIGAN_VOCODER_V3_LJSPEECH
VOCODER = bundle.get_vocoder().to("cuda").eval()
def estimate_mel_mean_std(dataset):
    total_frames = 0
    mean_sum = 0.0
    sq_sum = 0.0

    for i in range(len(dataset)):
        mel = dataset[i]['y']  # shape: [80, T]
        mel = mel.flatten()    # 展平成 1D
        total_frames += mel.numel()
        mean_sum += mel.sum().item()
        sq_sum += (mel ** 2).sum().item()

    mean = mean_sum / total_frames
    var = sq_sum / total_frames - mean ** 2
    std = var ** 0.5

    return mean, std

# Paths
TRAIN_FILELIST = "data/LibriTTS-R/full_libritts_aligned_filelist.txt"
SPEAKER_EMBED_PATH = "data/LibriTTS-R/speaker_embeddings_192_r_aligned.pt"
LOG_DIR = "logs/test30"  # New dir to avoid messing up original TensorBoard
CKPT_DIR = os.path.join(LOG_DIR, "checkpoints")
SAMPLE_DIR = os.path.join(LOG_DIR, "samples")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)

# Config for subset and loading
SUBSET_RATIO = 1.0 # ~32% of training data
LOAD_MODEL = True  # Set to False to start from scratch
RESUME_CHECKPOINT = os.path.join("logs/test29/checkpoints",
                                 "tiny_diffusion_epoch_120.pt")  # Path to your existing checkpoint (adjust if needed)
START_EPOCH = 96 if LOAD_MODEL else 0  # Continue from after epoch 10 if loading
LOG_EVERY = 50
# HiFi-GAN Vocoder

SAMPLE_RATE = 22050



def train():
    # Dataset and Dataloader
    full_dataset = PhonemeMelSpeakerDataset(TRAIN_FILELIST, SPEAKER_EMBED_PATH)
    collate_fn = PhonemeMelSpeakerBatchCollate(pad_idx=full_dataset .pad_idx)  # Pass pad_idx from dataset
    # Subsample train_dataset to ~32%
    subset_size = int(SUBSET_RATIO * len(full_dataset ))
    dataset = torch.utils.data.random_split(full_dataset , [subset_size, len(full_dataset ) - subset_size])[0]
    # mean,std=estimate_mel_mean_std(dataset)
    # print(f"Mean,std {mean,std} *************************************")

    # Split into train/val after subsample
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=8,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=8,
                            pin_memory=True)

    # Model
    model = TinyDiffusion(
        n_feats=n_feats, dec_dim=dec_dim, spk_emb_dim=decoder_spk_emb_dim,
        beta_min=beta_min, beta_max=beta_max, pe_scale=pe_scale, n_timesteps=n_timesteps,
        d_phoneme=d_phoneme, d_speaker=d_speaker
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5, eps=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5,min_lr=5e-5)

    writer = SummaryWriter(LOG_DIR)  # New fresh TensorBoard log

    scaler = GradScaler()
    # Optional load model (only state_dict, no optimizer)
    if LOAD_MODEL and os.path.exists(RESUME_CHECKPOINT):
        print(f"Loading model from {RESUME_CHECKPOINT}")
        ckpt = torch.load(RESUME_CHECKPOINT, map_location=device)
        state_dict = ckpt.get('model', ckpt)
        # buffer_keys = [k for k in state_dict if
        #                k in ['betas', 'alphas_cumprod', 'alphas_cumprod_prev', 'sqrt_alphas_cumprod',
        #                      'sqrt_one_minus_alphas_cumprod', 'posterior_variance', 'alphas']]
        # for k in buffer_keys:
        #     del state_dict[k]
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)
        optimizer.load_state_dict(ckpt['optimizer'])
        print("✅ Optimizer state restored.")
        scheduler.load_state_dict(ckpt['scheduler'])
        print("✅ Scheduler state restored.")
        scaler.load_state_dict(ckpt['scaler'])
        print("✅ GradScaler state restored.")

    else:
        print("Starting from scratch (no load or checkpoint not found).")

    # Always create fresh optimizer and scheduler (reset momentum etc.)


    # Calculate initial global_step based on START_EPOCH
    steps_per_epoch = len(train_loader)
    global_step = START_EPOCH * steps_per_epoch

    # Training loop
    for epoch in range(START_EPOCH,
                       START_EPOCH + n_epochs):  # Run exactly 10 more epochs (adjust n_epochs in params if needed)

        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{START_EPOCH + n_epochs}"):
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            spk = batch['spk'].to(device)
            x_lengths = batch['x_lengths'].to(device)
            y_lengths = batch['y_lengths'].to(device)
            y_max_length = batch['y'].shape[-1]
            gt_dur = batch['dur']
            if gt_dur is not None:
                gt_dur = gt_dur.to(device)
            getwav=batch['gt_wave'].to(device)
            optimizer.zero_grad()
            with autocast():
                diff_loss, prior_loss, durloss,wavloss,overshoot_loss,varloss = model.compute_loss(y, x, spk, x_lengths, y_lengths, y_max_length, gt_dur,getwav)
                # target = 0.2
                # scale = (target / (durloss.detach() + 1e-8)).clamp(0.1, 300.)
                # durloss = durloss * scale
                loss = 2.2 * diff_loss + 0.4 * prior_loss + 1.0*durloss+0.4*wavloss+0.1*varloss

            # Debug NaN
            if torch.isnan(loss).any():
                print(f"NaN detected in loss at epoch {epoch + 1}, batch {global_step + 1}")
                cond_dbg = model.get_cond(x, spk, x_lengths, y_max_length=None)
                print(
                    f"y_nan: {torch.isnan(y).any()} | "
                    f"mu_phon_nan: {torch.isnan(cond_dbg['mu_phon']).any()} | "
                    f"logw_nan: {torch.isnan(cond_dbg['logw']).any() if cond_dbg['logw'] is not None else 'None'}"
                )

            scaler.scale(loss).backward()
            nan_detected = False
            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"NaN gradient in {name}")
                    nan_detected = True
            if nan_detected:
                optimizer.zero_grad()  # Skip update
                continue
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            global_step += 1
            if global_step % LOG_EVERY == 0:
                writer.add_scalar('Loss/train', loss.item(), global_step)
                writer.add_scalar('Loss/train_diff', diff_loss.item(), global_step)
                writer.add_scalar('Loss/train_prior', prior_loss.item(), global_step)
                writer.add_scalar('Loss/train_dur_raw', durloss.item(), global_step)
                writer.add_scalar('Loss/train_wav', wavloss.item(), global_step)
                writer.add_scalar('Loss/train_var', varloss.item(), global_step)
                writer.add_scalar('Loss/train_overshoot', overshoot_loss.item(), global_step)
        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch + 1}: Avg Train Loss = {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch['x'].to(device)
                y = batch['y'].to(device)
                spk = batch['spk'].to(device)
                x_lengths = batch['x_lengths'].to(device)
                y_lengths = batch['y_lengths'].to(device)
                y_max_length = batch['y'].shape[-1]
                gt_dur = batch['dur']
                getwav = batch['gt_wave'].to(device)
                if gt_dur is not None:
                    gt_dur = gt_dur.to(device)

                with autocast():
                    diff_loss, prior_loss, durloss,wavloss,overshoot_loss,varloss = model.compute_loss(y, x, spk, x_lengths, y_lengths, y_max_length, gt_dur,getwav)

                    # target = 0.2
                    # scale = (target / (durloss.detach() + 1e-8)).clamp(0.1, 300.)
                    # durloss = durloss * scale
                    loss = 2.2 * diff_loss + 0.4 * prior_loss + 1.0*durloss+0.4*wavloss+0.1*varloss
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        print(f"Epoch {epoch + 1}: Avg Val Loss = {avg_val_loss:.4f}")

        scheduler.step(avg_val_loss)

        # Sample and generate
        with torch.no_grad():
            sample_batch = full_dataset.sample_test_batch(1)
            x = sample_batch[0]['x'].unsqueeze(0).to(device)
            spk = sample_batch[0]['spk'].unsqueeze(0).to(device)
            x_lengths = torch.tensor([sample_batch[0]['x_lengths']]).to(device)
            with autocast():
                gen_mel = model.sample(x, spk, x_lengths, n_timesteps=200, temperature=0.8,cfg=False,cfg_scale=0.5)
            save_plot(gen_mel.squeeze(0).cpu(), os.path.join(SAMPLE_DIR, f'epoch_{epoch + 1}_gen_mel.png'))
            wav = VOCODER(gen_mel.to(device)) # Update to direct call for torchaudio HiFi-GAN
            wav_rms = wav.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-6)
            wav = wav * (0.06 / wav_rms)
            wav = torch.tanh(wav)
            wav = wav.squeeze(0).cpu()
            ta.save(os.path.join(SAMPLE_DIR, f'epoch_{epoch + 1}_gen.wav'), wav, SAMPLE_RATE)
            gt_text = sample_batch[0]['text']  # 假设你的 sample_test_batch 返回了 'text' 字段
            with open(os.path.join(SAMPLE_DIR, f'epoch_{epoch + 1}_gen.txt'), 'w', encoding='utf-8') as f:
                f.write(gt_text.strip() + '\n')
        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None,
                'scaler': scaler.state_dict() if scaler else None,
                'epoch': epoch,
                'global_step': global_step,
            }, os.path.join(CKPT_DIR, f'tiny_diffusion_epoch_{epoch + 1}.pt'))
    writer.close()


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train()