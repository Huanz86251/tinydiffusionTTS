
import os
import torch
import torchaudio as ta
import json
from torch.utils.data import Dataset
import random
import torch.nn.functional as F
DEBUG=False
def dprint(*args, **kwargs):

    if DEBUG:
        print(*args, **kwargs)

class PhonemeMelSpeakerDataset(Dataset):
    def __init__(self, filelist_path="/root/autodl-tmp/goodluck/data/LibriTTS-R/full_libritts_aligned_filelist.txt",
                 speaker_embed_path="/root/autodl-tmp/goodluck/data/LibriTTS-R/speaker_embeddings_192_r_aligned.pt",
                 vocab_path="/root/autodl-tmp/goodluck/data/LibriTTS-R/phone2idx.json",
                 sample_rate=22050, hop_length=256, n_fft=1024, n_mels=80, f_min=0, f_max=8000,
                 mel_mean=-8.122798642, mel_std=2.1809869538):
        super().__init__()
        self.escape_counter = 0
        self.filelist_path = filelist_path
        self.speaker_embed_path = speaker_embed_path
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.mel_params = {
            'n_fft': n_fft,
            'n_mels': n_mels,
            'hop_length': hop_length,
            'win_length': hop_length * 4,
            'f_min': f_min,
            'f_max': f_max,
            'power': 1.0,
            'normalized': True,
            'norm': 'slaney',
            'mel_scale': 'slaney'
        }
        self.mel_mean = mel_mean
        self.mel_std = mel_std
        self.melgenerator = ta.transforms.MelSpectrogram(sample_rate=self.sample_rate, **self.mel_params)
        # Load filelist: speaker_id | wav_path | text
        with open(filelist_path, 'r', encoding='utf-8') as f:
            self.filelist = [line.strip().split('|', 2) for line in f.readlines()]

        # Load speaker embeddings dict
        self.speaker_embs = torch.load(speaker_embed_path)

        # Load phoneme vocab
        with open(vocab_path, 'r') as f:
            self.vocab = json.load(f)
        self.vocab_size = len(self.vocab)
        self.pad_idx = self.vocab['<PAD>']  # 0 for <PAD>

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):

        while True:

            speaker_id, pt_path, text = self.filelist[idx]
            wav_path = pt_path.replace('.pt', '.wav')
            mel_path = pt_path.replace('.pt', '.mel.pt')
            pt_data = torch.load(pt_path)
            phoneme_ids = pt_data['phoneme_ids']
            durations = pt_data['durations']
            if len(phoneme_ids) != len(durations):
                idx = (idx + 1) % len(self.filelist)  # 尝试下一个样本
                self.escape_counter+=1
                continue
            speaker_emb = self.speaker_embs[speaker_id]
            if os.path.exists(mel_path):
                # 加载缓存的 Mel 频谱图
                mel = torch.load(mel_path)
                wav, sr = ta.load(wav_path)
                if sr != self.sample_rate:
                    wav = ta.functional.resample(wav, sr, self.sample_rate)
            else:
                wav, sr = ta.load(wav_path)
                if sr != self.sample_rate:
                    wav = ta.functional.resample(wav, sr, self.sample_rate)
                mel = self.melgenerator(wav)
                mel = torch.log(torch.clamp(mel, min=1e-5))
                mel = (mel - self.mel_mean) / (self.mel_std + 1e-8)
                os.makedirs(os.path.dirname(mel_path), exist_ok=True)
                torch.save(mel, mel_path)
            x_lengths = phoneme_ids.shape[0]
            y_lengths = mel.shape[-1]
            dur_sum = durations.sum().item()
            threshold = 25
            diff = dur_sum - y_lengths

            if abs(diff) > 2 * threshold:
                idx = (idx + 1) % len(self.filelist)  # 尝试下一个样本
                self.escape_counter += 1
                continue
            elif abs(diff) > 0.5*threshold:
                nonzero_idx = (durations > 0).nonzero(as_tuple=True)[0]
                if len(nonzero_idx) == 0:
                    idx = (idx + 1) % len(self.filelist)  # 尝试下一个样本
                    self.escape_counter += 1
                    continue

                # 按 duration 值从大到小排序
                sorted_idx = nonzero_idx[torch.argsort(durations[nonzero_idx], descending=True)]

                n_fix = min(abs(int(diff)), len(sorted_idx))  # 修复次数不超过有效 token 数
                for i in sorted_idx[:n_fix]:
                    if diff > 0:
                        durations[i] -= 1
                    elif durations[i] > 1:
                        durations[i] += 1
            if DEBUG:
                dprint(
                    f"[Dataset getitem] wav shape: {wav.shape}, sr: {self.sample_rate}, mel shape: {mel.shape}, y_lengths: {y_lengths}")
            return {
                'x': phoneme_ids,
                'dur': durations,
                'y': mel.squeeze(0),
                'spk': speaker_emb,
                'x_lengths': x_lengths,
                'y_lengths': y_lengths,
                'text': text,
                'wav_path': wav_path,
                'gt_wave': wav.squeeze(0)
            }





    def sample_test_batch(self, n_samples=1):
        indices = random.sample(range(len(self)), n_samples)
        return [self[i] for i in indices]

class PhonemeMelSpeakerBatchCollate:
    def __init__(self, pad_idx=0):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        max_T = max(item['x_lengths'] for item in batch)
        max_S = max(item['y_lengths'] for item in batch)

        x = torch.stack([torch.nn.functional.pad(item['x'], (0, max_T - item['x_lengths']), value=self.pad_idx) for item in batch])
        dur = torch.stack([torch.nn.functional.pad(item['dur'], (0, max_T - item['x_lengths']), value=0.0) for item in batch])
        y = torch.stack([torch.nn.functional.pad(item['y'], (0, max_S - item['y_lengths'])) for item in batch])
        spk = torch.stack([item['spk'] for item in batch])
        x_lengths = torch.tensor([item['x_lengths'] for item in batch])
        y_lengths = torch.tensor([item['y_lengths'] for item in batch])
        hop_length = 256  # 从 Dataset 拿
        max_audio = max(item['y_lengths'] for item in batch) * hop_length + hop_length
        gt_wave = torch.stack([F.pad(item['gt_wave'], (0, max_audio - item['gt_wave'].shape[0]), value=0.0) for item in
                               batch])  # [B, max_audio]

        if DEBUG:
            dprint(f"[Collate] gt_wave shape: {gt_wave.shape}, max_audio: {max_audio}")
        return {
            'x': x.long(),
            'dur': dur,
            'y': y,
            'spk': spk,
            'x_lengths': x_lengths,
            'y_lengths': y_lengths,
            'gt_wave': gt_wave
        }
