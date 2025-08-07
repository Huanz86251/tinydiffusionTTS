import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import Wav2Vec2Model
from speechbrain.inference import HIFIGAN
from torch.cuda.amp import autocast
import os
from torchaudio.prototype.pipelines import HIFIGAN_VOCODER_V3_LJSPEECH
import torch
import torchaudio
import torchaudio as ta
import numpy as np
os.makedirs('./stft_test_output', exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEBUG = False
local_model_path = os.path.abspath("./wav2vec2_local/facebook/wav2vec2-base-960h")


def dprint(*args, **kwargs):

    if DEBUG:
        print(*args, **kwargs)



def _resample_to_16k(wav, orig_sr=22050):
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)  # 强制转 mono
    if DEBUG:
        dprint(f"[_resample_to_16k] Before resample: shape={wav.shape}, sr={orig_sr}")
    resampled_wav = ta.functional.resample(wav, orig_sr, 16000)
    if DEBUG:
        dprint(f"[_resample_to_16k] After resample: shape={resampled_wav.shape}, target sr=16000")
    return resampled_wav
# Helper functions
def sequence_mask(lengths, max_len=None):
    if max_len is None:
        max_len = lengths.max()
    ids = torch.arange(0, max_len, device=lengths.device)
    mask = (ids < lengths.unsqueeze(1)).float()
    if DEBUG:
        dprint(f"[sequence_mask] lengths: {lengths}, max_len: {max_len}, mask.shape: {mask.shape}, sum(dim=1): {mask.sum(dim=1)}, isnan: {torch.isnan(mask).any()}")
    return mask

# FiLM
class FiLM(nn.Module):
    def __init__(self, cond_dim, out_channels):
        super().__init__()
        self.scale = nn.Conv1d(cond_dim, out_channels, 1)
        self.shift = nn.Conv1d(cond_dim, out_channels, 1)

    def forward(self, x, cond):
        if DEBUG:
            dprint(f"[FiLM input] x.shape: {x.shape}, cond.shape: {cond.shape}, min/max x: {x.min()}/{x.max()}, cond: {cond.min()}/{cond.max()}")
        gamma = self.scale(cond)[:, :, :x.shape[2]].clamp(-2.0, 2.0)
        beta = self.shift(cond)[:, :, :x.shape[2]]
        if DEBUG:
            dprint(f"[FiLM gamma] shape: {gamma.shape}, min/max: {gamma.min()}/{gamma.max()}")
            dprint(f"[FiLM beta] shape: {beta.shape}, min/max: {beta.min()}/{beta.max()}")
        output = x * (1 + gamma) + beta
        if DEBUG:
            dprint(f"[FiLM output] shape: {output.shape}, min/max: {output.min()}/{output.max()}, isnan: {torch.isnan(output).any()}")
        return output

# CrossAttentionBlock
class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, cond_dim, num_heads=4):
        super().__init__()
        self.query_proj = nn.Conv1d(dim, dim, 1)
        self.key_proj = nn.Conv1d(cond_dim, dim, 1)
        self.value_proj = nn.Conv1d(cond_dim, dim, 1)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, cond, key_padding_mask=None):
        if DEBUG:
            dprint(f"[CrossAttn input] x.shape: {x.shape}, cond.shape: {cond.shape}, key_padding_mask sum: {key_padding_mask.sum() if key_padding_mask is not None else 'None'}")
        q = self.query_proj(x).permute(0, 2, 1)
        k = self.key_proj(cond).permute(0, 2, 1)
        v = self.value_proj(cond).permute(0, 2, 1)
        if DEBUG:
            dprint(f"[CrossAttn q] shape: {q.shape}, min/max: {q.min()}/{q.max()}")
            dprint(f"[CrossAttn k] shape: {k.shape}, min/max: {k.min()}/{k.max()}")
            dprint(f"[CrossAttn v] shape: {v.shape}, min/max: {v.min()}/{v.max()}")
        out, _ = self.attn(q, k, v, key_padding_mask=key_padding_mask)
        out = self.norm(out + q)
        out = out.permute(0, 2, 1)
        if DEBUG:
            dprint(f"[CrossAttn output] shape: {out.shape}, min/max: {out.min()}/{out.max()}, isnan: {torch.isnan(out).any()}")
        return out

# Modern UNet block
class ModernUNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, is_down=True, cond_dim=0, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.self_attn = nn.MultiheadAttention(out_channels, num_heads=4, batch_first=True) if out_channels >= 64 else None
        self.film = FiLM(cond_dim, out_channels) if cond_dim > 0 else None
        self.is_down = is_down
        if is_down:
            self.pool = nn.MaxPool1d(2, ceil_mode=True)  # Use ceil_mode to handle odd lengths
        else:
            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond=None, key_padding_mask=None):
        if DEBUG:
            dprint(f"[UNetBlock input] x.shape: {x.shape}, cond: {cond.shape if cond is not None else 'None'}, key_padding_mask sum: {key_padding_mask.sum() if key_padding_mask is not None else 'None'}")
        residual = self.residual(x)
        if DEBUG:
            dprint(f"[UNet residual] shape: {residual.shape}, min/max: {residual.min()}/{residual.max()}")
        x = self.conv1(x)
        if DEBUG:
            dprint(f"[UNet after conv1] shape: {x.shape}, min/max: {x.min()}/{x.max()}")
        x = self.norm1(x)
        if DEBUG:
            dprint(f"[UNet after norm1] shape: {x.shape}, min/max: {x.min()}/{x.max()}")
        x = self.relu(x)
        x = torch.clamp(self.conv2(x), -15.0, 15.0)

        if DEBUG:
            dprint(f"[UNet after conv2] shape: {x.shape}, min/max: {x.min()}/{x.max()}")
        x = self.norm2(x)
        if DEBUG:
            dprint(f"[UNet after norm2] shape: {x.shape}, min/max: {x.min()}/{x.max()}")
        x = self.relu(x)
        x = self.dropout(x)
        if self.self_attn is not None:
            current_T = x.shape[2]
            attn_mask = key_padding_mask[:, :current_T] if key_padding_mask is not None else None
            x_perm = x.permute(0, 2, 1)
            x_perm, _ = self.self_attn(x_perm, x_perm, x_perm, key_padding_mask=attn_mask)
            x = x_perm.permute(0, 2, 1) + x
            if DEBUG:
                dprint(f"[UNet after self_attn] shape: {x.shape}, min/max: {x.min()}/{x.max()}")
        if self.film is not None and cond is not None:
            x = self.film(x, cond)
            x = torch.clamp(x, -15.0, 15.0)
            if DEBUG:
                dprint(f"[UNet after film] shape: {x.shape}, min/max: {x.min()}/{x.max()}")
        x = x + residual
        if DEBUG:
            dprint(f"[UNet after residual add] shape: {x.shape}, min/max: {x.min()}/{x.max()}")
        if self.is_down:
            x = self.pool(x)
            if DEBUG:
                dprint(f"[UNet after pool] shape: {x.shape}, min/max: {x.min()}/{x.max()}")
        else:
            x = self.upsample(x)
            if DEBUG:
                dprint(f"[UNet after upsample] shape: {x.shape}, min/max: {x.min()}/{x.max()}")
        return x

# EnhancedSinusoidalPosEmb
class EnhancedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(20000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if DEBUG:
            dprint(f"[PosEmb] input x: {x.shape}, output emb.shape: {emb.shape}, min/max: {emb.min()}/{emb.max()}")
        return emb

# MLPConditioner
class MLPConditioner(nn.Module):
    def __init__(self, vocab_size=72, d_phoneme=768, d_speaker=192, d_hidden=512, d_mu=80, d_speakerout=64, num_heads=4, num_layers=2):
        super().__init__()
        self.phoneme_emb = nn.Embedding(vocab_size, d_phoneme)
        self.pos_emb = EnhancedSinusoidalPosEmb(d_phoneme)
        self.speaker_mlp = nn.Sequential(
            nn.Linear(d_speaker, d_phoneme),  # Project to d_phoneme for attn compatibility
            nn.LayerNorm(d_phoneme),
            nn.ReLU(),
        )
        self.speaker_mlp_decoder = nn.Sequential(
            nn.Linear(d_speaker, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_speakerout),
            nn.Tanh()  # 替换 LeakyReLU，确保输出 [-1,1]
        )
        # Multi-layer Transformer blocks for phoneme processing
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'self_attn': nn.MultiheadAttention(d_phoneme, num_heads=num_heads, batch_first=True),
                'cross_attn': nn.MultiheadAttention(d_phoneme, num_heads=num_heads, batch_first=True),
                'ffn': nn.Sequential(
                    nn.Linear(d_phoneme, d_hidden),
                    nn.ReLU(),
                    nn.Linear(d_hidden, d_phoneme),
                    nn.LayerNorm(d_phoneme),
                    nn.Dropout(0.1),
                ),
                'norm1': nn.LayerNorm(d_phoneme),
                'norm2': nn.LayerNorm(d_phoneme),
                'norm3': nn.LayerNorm(d_phoneme),
            }))
        self.norm_after_layers = nn.LayerNorm(d_phoneme)
        self.mu_proj = nn.Linear(d_phoneme, d_mu)
        self.duration_predictor = nn.Sequential(
            nn.Linear(d_phoneme, d_hidden // 2),
            nn.ReLU(),
            nn.Linear(d_hidden // 2, 1)
        )

    def forward(self, phoneme_ids, speaker_emb, x_lengths, predict_dur=True, y_max_length=None):
        with torch.autocast(device_type='cuda',enabled=False):
            B, T = phoneme_ids.shape
            if DEBUG:
                dprint(f"[Conditioner input] phoneme_ids.shape: {phoneme_ids.shape}, speaker_emb.shape: {speaker_emb.shape}, x_lengths: {x_lengths}, y_max_length: {y_max_length}")
            phoneme_emb = self.phoneme_emb(phoneme_ids)  # [B, T, d_phoneme]
            if DEBUG:
                dprint(f"[Conditioner phoneme_emb] shape: {phoneme_emb.shape}, min/max: {phoneme_emb.min()}/{phoneme_emb.max()}, isnan: {torch.isnan(phoneme_emb).any()}")
            pos = torch.arange(0, T, device=phoneme_ids.device)
            phoneme_emb = phoneme_emb + self.pos_emb(pos).expand(B, -1, -1)
            if DEBUG:
                dprint(f"[Conditioner after pos_emb] shape: {phoneme_emb.shape}, min/max: {phoneme_emb.min()}/{phoneme_emb.max()}")
            phoneme_emb = torch.clamp(phoneme_emb, min=-10.0, max=10.0)
            if DEBUG:
                dprint(f"[Conditioner after clamp] shape: {phoneme_emb.shape}, min/max: {phoneme_emb.min()}/{phoneme_emb.max()}, isnan: {torch.isnan(phoneme_emb).any()}")
            s = self.speaker_mlp(speaker_emb).unsqueeze(1)  # [B, 1, d_phoneme] for cross-attn
            if DEBUG:
                dprint(f"[Conditioner s] shape: {s.shape}, min/max: {s.min()}/{s.max()}")
            key_padding_mask = ~sequence_mask(x_lengths, T).bool()  # [B, T] for padding mask (True for pad)
            if DEBUG:
                dprint(f"[Conditioner key_padding_mask] shape: {key_padding_mask.shape}, sum: {key_padding_mask.sum()}, isnan: {torch.isnan(key_padding_mask.float()).any()}")
            # Multi-layer Transformer
            x = phoneme_emb
            for layer_idx, layer in enumerate(self.layers):
                # Self-attn on phoneme tokens
                x_res = x
                if DEBUG:
                    dprint(f"[Conditioner layer{layer_idx} self_attn input] x.shape: {x.shape}, min/max: {x.min()}/{x.max()}")
                x_out, _ = layer['self_attn'](x, x, x, key_padding_mask=key_padding_mask)
                x = layer['norm1'](x_out + x_res)
                if DEBUG:
                    dprint(f"[Conditioner layer{layer_idx} after self_attn] shape: {x.shape}, min/max: {x.min()}/{x.max()}, isnan: {torch.isnan(x).any()}")

                # Cross-attn with speaker (speaker as KV, phoneme as Q)
                x_res = x
                x_out, _ = layer['cross_attn'](x, s, s)
                x = layer['norm2'](x_out + x_res)
                if DEBUG:
                    dprint(f"[Conditioner layer{layer_idx} after cross_attn] shape: {x.shape}, min/max: {x.min()}/{x.max()}, isnan: {torch.isnan(x).any()}")

                # FFN
                x_res = x
                x = layer['ffn'](x)
                x = layer['norm3'](x + x_res)
                if DEBUG:
                    dprint(f"[Conditioner layer{layer_idx} after ffn] shape: {x.shape}, min/max: {x.min()}/{x.max()}, isnan: {torch.isnan(x).any()}")

            if torch.isnan(x).any() or torch.isinf(x).any():
                if DEBUG:
                    dprint(
                    f"Extreme/NaN at x after layers: norm={x.norm().item():.4f}, min={x.min().item():.4f}, max={x.max().item():.4f}, isnan={torch.isnan(x).any().item()}, shape={x.shape}, key_padding_mask sum={key_padding_mask.sum().item()}")
            x = self.norm_after_layers(x)
            if DEBUG:
                dprint(f"[Conditioner after norm_after_layers] shape: {x.shape}, min/max: {x.min()}/{x.max()}")
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)  # Zero pad in x before proj/predictor
            if DEBUG:
                dprint(f"[Conditioner after masked_fill pad] shape: {x.shape}, min/max: {x.min()}/{x.max()}")
            mu_phon = self.mu_proj(x).transpose(1, 2)  # [B, d_mu, T]
            if DEBUG:
                dprint(f"[Conditioner mu_phon] shape: {mu_phon.shape}, min/max: {mu_phon.min()}/{mu_phon.max()}")
            spk_feat = self.speaker_mlp_decoder(F.normalize(speaker_emb, dim=-1))
            if DEBUG:
                dprint(f"[Conditioner spk_feat] shape: {spk_feat.shape}, min/max: {spk_feat.min()}/{spk_feat.max()}")
            x_mask = sequence_mask(x_lengths, mu_phon.shape[2]).unsqueeze(1)
            if DEBUG:
                dprint(f"[Conditioner x_mask] shape: {x_mask.shape}, sum(dim=2): {x_mask.sum(dim=2)}")
            logw = None
            dur = None
            if predict_dur:
                logw = self.duration_predictor(x).squeeze(-1)  # [B, T]
                if DEBUG:
                    dprint(f"[Conditioner logw raw] shape: {logw.shape}, min/max: {logw.min()}/{logw.max()}")
                logw = torch.clamp(logw, min=-10.0, max=10.0)
                if DEBUG:
                    dprint(f"[Conditioner logw clamp] min/max: {logw.min()}/{logw.max()}")
                logw = logw.masked_fill(key_padding_mask, -10.0)  # Fill pad with low value
                if DEBUG:
                    dprint(f"[Conditioner logw after mask] min/max: {logw.min()}/{logw.max()}")
                dur = torch.exp(logw)  # [B, T]
                dur = dur.masked_fill(key_padding_mask, 0.0)  # Zero dur for pad
                dur = torch.clamp(dur, min=1.0, max=50.0)
                if DEBUG:
                    dprint(f"[Conditioner dur] shape: {dur.shape}, min/max: {dur.min()}/{dur.max()}, sum(dim=1): {dur.sum(dim=1)}")
            return {'mu_phon': mu_phon, 'spk_feat': spk_feat, 'mask': x_mask, 'logw': logw, 'dur': dur}

# TinyDiffusion
class TinyDiffusion(nn.Module):
    def __init__(self, n_feats=80, dec_dim=384, spk_emb_dim=64, beta_min=1e-4, beta_max=0.05, pe_scale=1000, n_timesteps=200, d_phoneme=768, d_speaker=192):
        super().__init__()
        self.n_feats = n_feats
        self.dec_dim = dec_dim
        self.spk_emb_dim = spk_emb_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.pe_scale = pe_scale
        self.n_timesteps = n_timesteps
        self.cfg_scale = 1.5

        self.conditioner = MLPConditioner(vocab_size=72, d_phoneme=d_phoneme, d_speaker=d_speaker, d_hidden=512, d_mu=n_feats, d_speakerout=spk_emb_dim)
        s = 0.008  # 常见偏移值，避免t=0问题
        t = torch.arange(0, n_timesteps + 1, dtype=torch.float32)  # [0, 1, ..., T]
        alpha_bar = torch.cos((t / n_timesteps + s) / (1 + s) * math.pi / 2) ** 2  # alpha_bar_t = cos^2(...)
        alphas = alpha_bar[1:] / alpha_bar[:-1]  # alpha_t = alpha_bar_t / alpha_bar_{t-1}
        betas = torch.clamp(1. - alphas, min=0.0001, max=0.999)  # betas = 1 - alpha_t，clip避免极端值
        print(f"[Init] 使用cosine调度, betas.min()={betas.min().item():.6f}, betas.max()={betas.max().item():.6f}")

        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('posterior_variance', posterior_variance)
        self.time_emb = EnhancedSinusoidalPosEmb(dec_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(dec_dim, dec_dim * 4),
            nn.ReLU(),
            nn.Linear(dec_dim * 4, dec_dim)
        )
        self.time_proj = nn.Conv1d(dec_dim, n_feats, 1)  # Project to n_feats only, add separately
        self.spk_proj = nn.Conv1d(spk_emb_dim, dec_dim, 1)
        self.mu_proj = nn.Conv1d(n_feats, dec_dim, 1)
        # Reduce depth to 4 layers for "tiny" efficiency
        self.down1 = ModernUNetBlock(n_feats, dec_dim, cond_dim=dec_dim)
        self.cross_attn1 = CrossAttentionBlock(dec_dim, dec_dim)
        self.down2 = ModernUNetBlock(dec_dim, dec_dim * 2, cond_dim=dec_dim)
        self.cross_attn2 = CrossAttentionBlock(dec_dim * 2, dec_dim)
        self.down3 = ModernUNetBlock(dec_dim * 2, dec_dim * 4, cond_dim=dec_dim)
        self.cross_attn3 = CrossAttentionBlock(dec_dim * 4, dec_dim)
        self.down4 = ModernUNetBlock(dec_dim * 4, dec_dim * 6, cond_dim=dec_dim)
        self.cross_attn4 = CrossAttentionBlock(dec_dim * 6, dec_dim)
        self.bottleneck_conv = nn.Conv1d(dec_dim * 6, dec_dim * 6, 1)
        self.bottleneck_relu = nn.ReLU()
        self.bottleneck_attn = nn.MultiheadAttention(dec_dim * 6, num_heads=24, batch_first=True)
        self.up4 = ModernUNetBlock(dec_dim * 12, dec_dim * 4, is_down=False, cond_dim=dec_dim)
        self.up3 = ModernUNetBlock(dec_dim * 8, dec_dim * 2, is_down=False, cond_dim=dec_dim)
        self.up2 = ModernUNetBlock(dec_dim * 4, dec_dim, is_down=False, cond_dim=dec_dim)
        self.up1 = ModernUNetBlock(dec_dim * 2, dec_dim, is_down=False, cond_dim=dec_dim)
        self.out = nn.Sequential(
            nn.Conv1d(dec_dim, n_feats, 1),
 # Bound output for stability
        )
        bundle = HIFIGAN_VOCODER_V3_LJSPEECH
        self.vocoder= bundle.get_vocoder().to("cuda").eval()

        #self.wav2vec = Wav2Vec2Model.from_pretrained(
#     "facebook/wav2vec2-base-960h",
#     cache_dir="./wav2vec2_local",  # 就是你现在这个目录
#     local_files_only=True
# ).eval().cuda()   # eval 模式，避免梯度


        self.mel_mean, self.mel_std = -8.122798642, 2.1809869538

    def forward(self, z, t, cond, mask):
        B, _, T = z.shape
        if DEBUG:
            dprint(f"[Diffusion forward input] z.shape: {z.shape}, t.shape: {t.shape}, cond['mu_x'].shape: {cond['mu_x'].shape}, mask.shape: {mask.shape}")
        t_emb = self.time_emb(t * self.pe_scale).to(z.dtype)
        if DEBUG:
            dprint(f"[Diffusion t_emb] shape: {t_emb.shape}, min/max: {t_emb.min()}/{t_emb.max()}")
        t_emb = self.time_mlp(t_emb).unsqueeze(-1).expand(-1, -1, T)
        if DEBUG:
            dprint(f"[Diffusion t_emb after mlp expand] shape: {t_emb.shape}")
        t_emb = self.time_proj(t_emb)  # [B, n_feats, T]
        if DEBUG:
            dprint(f"[Diffusion t_emb after proj] shape: {t_emb.shape}, min/max: {t_emb.min()}/{t_emb.max()}")
        spk = self.spk_proj(cond['spk_feat'].unsqueeze(-1).expand(-1, -1, T))  # [B, dec_dim, T]
        if DEBUG:
            dprint(f"[Diffusion spk] shape: {spk.shape}, min/max: {spk.min()}/{spk.max()}")
        mu = self.mu_proj(cond['mu_x'])  # [B, dec_dim, T]
        if DEBUG:
            dprint(f"[Diffusion mu] shape: {mu.shape}, min/max: {mu.min()}/{mu.max()}")
        input_cond = spk + mu  # [B, dec_dim, T]
        if DEBUG:
            dprint(f"[Diffusion input_cond] shape: {input_cond.shape}, min/max: {input_cond.min()}/{input_cond.max()}")
        z = z + t_emb  # Add time to z separately
        if DEBUG:
            dprint(f"[Diffusion z after + t_emb] shape: {z.shape}, min/max: {z.min()}/{z.max()}")
        key_padding_mask = ~mask.squeeze(1).bool()  # [B, T] for attn
        if DEBUG:
            dprint(f"[Diffusion key_padding_mask] shape: {key_padding_mask.shape}, sum: {key_padding_mask.sum()}")
        d1 = self.down1(z, input_cond, key_padding_mask=key_padding_mask)
        if DEBUG:
            dprint(f"[Diffusion d1] shape: {d1.shape}, min/max: {d1.min()}/{d1.max()}")
        d1 = self.cross_attn1(d1, input_cond, key_padding_mask=key_padding_mask)
        if DEBUG:
            dprint(f"[Diffusion d1 after attn] shape: {d1.shape}, min/max: {d1.min()}/{d1.max()}")
        d2 = self.down2(d1, input_cond, key_padding_mask=key_padding_mask)
        if DEBUG:
            dprint(f"[Diffusion d2] shape: {d2.shape}, min/max: {d2.min()}/{d2.max()}")
        d2 = self.cross_attn2(d2, input_cond, key_padding_mask=key_padding_mask)
        if DEBUG:
            dprint(f"[Diffusion d2 after attn] shape: {d2.shape}, min/max: {d2.min()}/{d2.max()}")
        d3 = self.down3(d2, input_cond, key_padding_mask=key_padding_mask)
        if DEBUG:
            dprint(f"[Diffusion d3] shape: {d3.shape}, min/max: {d3.min()}/{d3.max()}")
        d3 = self.cross_attn3(d3, input_cond, key_padding_mask=key_padding_mask)
        if DEBUG:
            dprint(f"[Diffusion d3 after attn] shape: {d3.shape}, min/max: {d3.min()}/{d3.max()}")
        d4 = self.down4(d3, input_cond, key_padding_mask=key_padding_mask)
        if DEBUG:
            dprint(f"[Diffusion d4] shape: {d4.shape}, min/max: {d4.min()}/{d4.max()}")
        d4 = self.cross_attn4(d4, input_cond, key_padding_mask=key_padding_mask)
        if DEBUG:
            dprint(f"[Diffusion d4 after attn] shape: {d4.shape}, min/max: {d4.min()}/{d4.max()}")
        b = self.bottleneck_conv(d4)
        if DEBUG:
            dprint(f"[Diffusion bottleneck_conv] shape: {b.shape}, min/max: {b.min()}/{b.max()}")
        b = self.bottleneck_relu(b)
        if DEBUG:
            dprint(f"[Diffusion bottleneck_relu] shape: {b.shape}, min/max: {b.min()}/{b.max()}")
        b_perm = b.permute(0, 2, 1)
        current_T = b.shape[2]
        attn_mask = key_padding_mask[:, :current_T] if key_padding_mask is not None else None
        b_perm, _ = self.bottleneck_attn(b_perm, b_perm, b_perm, key_padding_mask=attn_mask)
        b = b_perm.permute(0, 2, 1)
        if DEBUG:
            dprint(f"[Diffusion bottleneck_attn] shape: {b.shape}, min/max: {b.min()}/{b.max()}")
        # Align lengths for skip connections by cropping if necessary
        u4 = self.up4(torch.cat([b[:, :, :d4.shape[2]], d4], dim=1), input_cond, key_padding_mask=key_padding_mask)
        if DEBUG:
            dprint(f"[Diffusion u4] shape: {u4.shape}, min/max: {u4.min()}/{u4.max()}")
        u3 = self.up3(torch.cat([u4[:, :, :d3.shape[2]], d3], dim=1), input_cond, key_padding_mask=key_padding_mask)
        if DEBUG:
            dprint(f"[Diffusion u3] shape: {u3.shape}, min/max: {u3.min()}/{u3.max()}")
        u2 = self.up2(torch.cat([u3[:, :, :d2.shape[2]], d2], dim=1), input_cond, key_padding_mask=key_padding_mask)
        if DEBUG:
            dprint(f"[Diffusion u2] shape: {u2.shape}, min/max: {u2.min()}/{u2.max()}")
        u1 = self.up1(torch.cat([u2[:, :, :d1.shape[2]], d1], dim=1), input_cond, key_padding_mask=key_padding_mask)
        if DEBUG:
            dprint(f"[Diffusion u1] shape: {u1.shape}, min/max: {u1.min()}/{u1.max()}")

        epsilon = torch.clamp(self.out(u1), -5.0, 5.0)
        if DEBUG:
            dprint(f"[Diffusion epsilon raw] shape: {epsilon.shape}, min/max: {epsilon.min()}/{epsilon.max()}")
        target_T = z.shape[2]
        if epsilon.shape[2] != target_T:
            epsilon = F.interpolate(epsilon, size=target_T, mode='linear', align_corners=False)
            if DEBUG:
                dprint(f"[Diffusion epsilon after interpolate] shape: {epsilon.shape}, min/max: {epsilon.min()}/{epsilon.max()}")
        return epsilon



    # def compute_perceptual_loss(self, y_hat, y, gt_wave, mask):
    #     device = y_hat.device
    #     if DEBUG:
    #         dprint(f"[Perceptual] y_hat device: {device}, dtype: {y_hat.dtype}, shape: {y_hat.shape}")
    #
    #     # denorm Mel
    #     y_hat_denorm = y_hat * self.mel_std + self.mel_mean
    #     if DEBUG:
    #         dprint(
    #             f"[Perceptual] y_hat_denorm device: {y_hat_denorm.device}, dtype: {y_hat_denorm.dtype}, min/max: {y_hat_denorm.min():.2f}/{y_hat_denorm.max():.2f}")
    #
    #     with autocast():
    #         pred_wave = self.vocoder(y_hat_denorm.to(device)) # torchaudio HiFi-GAN 接口，直接输入 Mel [B, 80, T]
    #         if DEBUG:
    #             dprint(
    #                 f"[Perceptual] pred_wave device: {pred_wave.device}, dtype: {pred_wave.dtype}, shape: {pred_wave.shape}")
    #
    #     # 处理长度不匹配
    #     min_len = min(pred_wave.shape[-1], gt_wave.shape[-1])
    #     pred_wave = pred_wave[..., :min_len]
    #     gt_wave = gt_wave[..., :min_len]
    #
    #     # resample 到 Wav2Vec2 输入（16000Hz）
    #     pred_wave_res = _resample_to_16k(pred_wave.squeeze(1))  # [B, time]
    #     gt_wave_res = _resample_to_16k(gt_wave.squeeze(1))
    #
    #     # 提取 Wav2Vec2 特征
    #     with torch.no_grad():
    #         pred_feats = self.wav2vec(pred_wave_res, output_hidden_states=True).hidden_states[9]
    #         gt_feats = self.wav2vec(gt_wave_res, output_hidden_states=True).hidden_states[9]
    #
    #     # 应用 mask
    #     feat_len = pred_feats.shape[1]
    #     mask_time = mask.squeeze(1)[:, :feat_len].unsqueeze(-1).expand(-1, -1, pred_feats.shape[-1]).float()
    #
    #     # 计算 L2 损失
    #     perceptual_loss = F.mse_loss(pred_feats * mask_time, gt_feats * mask_time)
    #
    #     return perceptual_loss

    def compute_stft_loss(self, y_hat, y, gt_wave, mask):
        """
        Compute multi-resolution STFT loss on waveform level.

        Args:
            y_hat (torch.Tensor): Predicted mel spectrogram [B, n_feats, Ty]
            y (torch.Tensor): Ground truth mel spectrogram [B, n_feats, Ty]
            gt_wave (torch.Tensor): Ground truth waveform [B, (1,) T_wave]  # 支持 2D 或 3D
            mask (torch.Tensor): Mask for valid frames [B, 1, Ty]

        Returns:
            torch.Tensor: Scalar STFT loss
        """
        device = y_hat.device
        B = y_hat.shape[0]  # 批次大小
        if DEBUG:
            dprint(
                f"[STFT Loss input] y_hat.shape: {y_hat.shape}, min/max: {y_hat.min():.2f}/{y_hat.max():.2f}, "
                f"y.shape: {y.shape}, gt_wave.shape: {gt_wave.shape}, dim: {gt_wave.dim()}, "
                f"mask.shape: {mask.shape}, mask sum(dim=2): {mask.sum(dim=2)}, "
                f"isnan: {torch.isnan(y_hat).any() or torch.isnan(gt_wave).any()}, batch_size B: {B}")

        # Denormalize mel spectrogram (no pre-offset)
        y_hat_denorm = y_hat * self.mel_std + self.mel_mean
        y_hat_denorm = torch.clamp(y_hat_denorm, min=-25.0, max=2.0)  # Clamp to reasonable ln scale
        if DEBUG:
            dprint(
                f"[STFT Loss denorm] y_hat_denorm.shape: {y_hat_denorm.shape}, "
                f"min/max: {y_hat_denorm.min():.2f}/{y_hat_denorm.max():.2f}, "
                f"mean/std: {y_hat_denorm.mean():.2f}/{y_hat_denorm.std():.2f}, "
                f"isnan: {torch.isnan(y_hat_denorm).any()}")
            if y_hat_denorm.min() < -20.0 or y_hat_denorm.max() > 0.0:
                dprint(f"[STFT Loss denorm warning] y_hat_denorm out of expected range [-20, 0]: "
                       f"min={y_hat_denorm.min():.2f}, max={y_hat_denorm.max():.2f}")

        # Generate waveform using HiFi-GAN
        with torch.cuda.amp.autocast():
            pred_wave = self.vocoder(y_hat_denorm.to(device))  # [B, 1, T_wave]
        if DEBUG:
            dprint(
                f"[STFT Loss pred_wave raw] shape: {pred_wave.shape}, dim: {pred_wave.dim()}, "
                f"min/max: {pred_wave.min():.2f}/{pred_wave.max():.2f}, "
                f"mean/std: {pred_wave.mean():.2f}/{pred_wave.std():.2f}, isnan: {torch.isnan(pred_wave).any()}")
            if pred_wave.shape[0] != B:
                dprint(f"[STFT Loss pred_wave warning] batch size {pred_wave.shape[0]} != B {B}")
            if pred_wave.dim() != 3:
                dprint(f"[STFT Loss pred_wave warning] dim {pred_wave.dim()} != 3, expected [B, 1, T_wave]")
            if pred_wave.dim() == 3 and pred_wave.shape[1] != 1:
                dprint(f"[STFT Loss pred_wave warning] channel dim {pred_wave.shape[1]} != 1")

        # Align waveform lengths
        # 统一 gt_wave 到 [B, 1, T_wave]
        if gt_wave.dim() == 2:
            gt_wave = gt_wave.unsqueeze(1)  # [B, T] -> [B, 1, T]
        if DEBUG:
            dprint(f"[STFT Loss pre-align] gt_wave after unsqueeze: {gt_wave.shape}, dim: {gt_wave.dim()}")
            if gt_wave.dim() != 3 or gt_wave.shape[0] != B or gt_wave.shape[1] != 1:
                dprint(f"[STFT Loss pre-align warning] gt_wave shape {gt_wave.shape} != expected [B, 1, T_wave]")

        min_len = min(pred_wave.shape[-1], gt_wave.shape[-1])
        pred_wave = pred_wave[..., :min_len]
        gt_wave = gt_wave[..., :min_len]
        if DEBUG:
            dprint(
                f"[STFT Loss aligned] min_len: {min_len}, pred_wave.shape: {pred_wave.shape}, dim: {pred_wave.dim()}, "
                f"gt_wave.shape: {gt_wave.shape}, dim: {gt_wave.dim()}")
            expected_len = mask.sum(dim=2).max().item() * 256  # 假设 hop_length=256
            if min_len < expected_len * 0.8:
                dprint(f"[STFT Loss aligned warning] min_len {min_len} < 80% of expected {expected_len}")
            if pred_wave.shape[0] != B:
                dprint(f"[STFT Loss aligned warning] pred_wave batch size {pred_wave.shape[0]} != B {B}")
            if pred_wave.dim() != 3:
                dprint(f"[STFT Loss aligned warning] pred_wave dim {pred_wave.dim()} != 3")
            if pred_wave.dim() == 3 and pred_wave.shape[1] != 1:
                dprint(f"[STFT Loss aligned warning] pred_wave channel dim {pred_wave.shape[1]} != 1")
            if gt_wave.shape != pred_wave.shape:
                dprint(
                    f"[STFT Loss aligned warning] gt_wave shape {gt_wave.shape} != pred_wave shape {pred_wave.shape}")

        # Post-offset on waveform: scale to match GT RMS (no dynamic RMS during STFT)
        pred_rms = pred_wave.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-6)  # [B, 1, 1]
        gt_rms = gt_wave.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-6)  # [B, 1, 1]
        pred_peak = pred_wave.abs().max(dim=-1, keepdim=True)[0]
        gt_peak = gt_wave.abs().max(dim=-1, keepdim=True)[0]
        if DEBUG:
            dprint(
                f"[STFT Loss pre-offset] pred_rms.shape: {pred_rms.shape}, gt_rms.shape: {gt_rms.shape}, "
                f"pred_rms: {pred_rms.mean().item():.4f}, gt_rms: {gt_rms.mean().item():.4f}, "
                f"pred_peak: {pred_peak.mean().item():.4f}, gt_peak: {gt_peak.mean().item():.4f}")
            if pred_rms.shape != torch.Size([B, 1, 1]):
                dprint(f"[STFT Loss pre-offset warning] pred_rms shape {pred_rms.shape} != [B, 1, 1]")
            if gt_rms.shape != torch.Size([B, 1, 1]):
                dprint(f"[STFT Loss pre-offset warning] gt_rms shape {gt_rms.shape} != [B, 1, 1]")
            if pred_rms.min() < 1e-4:
                dprint(
                    f"[STFT Loss pre-offset warning] pred_rms {pred_rms.mean().item():.4f} too low, check vocoder input")
        scale_factor = gt_rms / pred_rms  # [B, 1, 1]
        if DEBUG:
            dprint(
                f"[STFT Loss scale factor] shape: {scale_factor.shape}, min/max: {scale_factor.min():.2f}/{scale_factor.max():.2f}")
            if scale_factor.shape != torch.Size([B, 1, 1]):
                dprint(f"[STFT Loss scale factor warning] scale_factor shape {scale_factor.shape} != [B, 1, 1]")
        pred_wave = pred_wave * scale_factor
        pred_wave = torch.tanh(pred_wave)  # Soft clip to avoid distortion
        if DEBUG:
            post_pred_rms = pred_wave.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-6)
            post_pred_peak = pred_wave.abs().max(dim=-1, keepdim=True)[0]
            dprint(
                f"[STFT Loss post-offset] pred_wave.shape: {pred_wave.shape}, dim: {pred_wave.dim()}, "
                f"min/max: {pred_wave.min():.2f}/{pred_wave.max():.2f}, "
                f"mean/std: {pred_wave.mean():.2f}/{pred_wave.std():.2f}, isnan: {torch.isnan(pred_wave).any()}, "
                f"post_pred_rms: {post_pred_rms.mean().item():.4f}, post_pred_peak: {post_pred_peak.mean().item():.4f}")
            if pred_wave.shape != torch.Size([B, 1, min_len]):
                dprint(f"[STFT Loss post-offset warning] pred_wave shape {pred_wave.shape} != expected [B, 1, min_len]")

        # 修改部分：保存 pred_wave 和 gt_wave，只保存一次
        if not hasattr(self, '_saved_wav'):
            self._saved_wav = True
            os.makedirs('./stft_test_output', exist_ok=True)
        
            # 获取第一个样本 [B, 1, T] → [1, T]
            pred = pred_wave[0].detach().cpu()  # shape: [1, T]
            gt = gt_wave[0].detach().cpu()  # shape: [1, T]

            # 若张量维度为 2（[1, T]），就可以直接保存；若为 [T]，需 unsqueeze
            if pred.dim() == 1:
                pred = pred.unsqueeze(0)  # [T] → [1, T]
            if gt.dim() == 1:
                gt = gt.unsqueeze(0)

            # 保证是 [channels, time]
            assert pred.dim() == 2 and gt.dim() == 2, f"pred.dim={pred.dim()}, gt.dim={gt.dim()}"

            # 保存音频
            ta.save('./stft_test_output/pred.wav', pred, 22050)
            ta.save('./stft_test_output/gt.wav', gt, 22050)
        # Multi-resolution STFT parameters
        n_ffts = [2048, 1024, 512]
        hop_lengths = [512, 256, 128]
        win_lengths = [2048, 1024, 512]
        stft_loss = 0.0
        spec_loss = 0.0
        total_frames = 0

        for i, (n_fft, hop_length, win_length) in enumerate(zip(n_ffts, hop_lengths, win_lengths)):
            # Compute STFT
            stft = torchaudio.transforms.Spectrogram(
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window_fn=torch.hann_window,
                power=2,
                normalized=True
            ).to(device)

            # 统一处理 waveform 到 [B, T_wave] 格式
            if pred_wave.dim() == 3:
                pred_wave_flat = pred_wave.squeeze(1)  # [B, 1, T] -> [B, T]
            else:
                raise ValueError(
                    f"[STFT Loss scale {i}] pred_wave dim {pred_wave.dim()} invalid, expected 3D [B, 1, T_wave], got {pred_wave.shape}")

            if gt_wave.dim() == 3:
                gt_wave_flat = gt_wave.squeeze(1)  # [B, 1, T] -> [B, T]
            elif gt_wave.dim() == 2:
                gt_wave_flat = gt_wave  # [B, T]
            else:
                raise ValueError(
                    f"[STFT Loss scale {i}] gt_wave dim {gt_wave.dim()} invalid, expected 2D or 3D, got {gt_wave.shape}")

            if DEBUG:
                dprint(f"[STFT Loss scale {i}] pred_wave before squeeze: {pred_wave.shape}, dim: {pred_wave.dim()}")
                dprint(
                    f"[STFT Loss scale {i}] pred_wave_flat.shape: {pred_wave_flat.shape}, dim: {pred_wave_flat.dim()}, "
                    f"gt_wave_flat.shape: {gt_wave_flat.shape}, dim: {gt_wave_flat.dim()}")
                if pred_wave_flat.shape[0] != B:
                    dprint(
                        f"[STFT Loss scale {i} warning] pred_wave_flat batch size {pred_wave_flat.shape[0]} != B {B}")
                if gt_wave_flat.shape[0] != B:
                    dprint(
                        f"[STFT Loss scale {i} warning] gt_wave_flat batch size {gt_wave_flat.shape[0]} != B {B}")
                if pred_wave_flat.dim() != 2:
                    dprint(
                        f"[STFT Loss scale {i} warning] pred_wave_flat dim {pred_wave_flat.dim()} != 2, shape: {pred_wave_flat.shape}")
                if gt_wave_flat.dim() != 2:
                    dprint(
                        f"[STFT Loss scale {i} warning] gt_wave_flat dim {gt_wave_flat.dim()} != 2, shape: {gt_wave_flat.shape}")

            # 断言 waveform 是 2D [B, T]
            assert pred_wave_flat.dim() == 2, f"[STFT Loss scale {i}] pred_wave_flat dim {pred_wave_flat.dim()} != 2, shape: {pred_wave_flat.shape}"
            assert gt_wave_flat.dim() == 2, f"[STFT Loss scale {i}] gt_wave_flat dim {gt_wave_flat.dim()} != 2, shape: {gt_wave_flat.shape}"
            assert pred_wave_flat.shape[
                       0] == B, f"[STFT Loss scale {i}] pred_wave_flat batch size {pred_wave_flat.shape[0]} != B {B}"
            assert gt_wave_flat.shape[
                       0] == B, f"[STFT Loss scale {i}] gt_wave_flat batch size {gt_wave_flat.shape[0]} != B {B}"

            # STFT on waveforms [B, T_wave] -> [B, freq, T_spec]
            pred_spec = stft(pred_wave_flat)  # [B, freq, T_spec]
            gt_spec = stft(gt_wave_flat)  # [B, freq, T_spec]

            if DEBUG:
                dprint(
                    f"[STFT Loss scale {i}] n_fft={n_fft}, hop={hop_length}, "
                    f"pred_spec.shape: {pred_spec.shape}, min/max: {pred_spec.min():.2f}/{pred_spec.max():.2f}, "
                    f"mean/std: {pred_spec.mean():.2f}/{pred_spec.std():.2f}, isnan: {torch.isnan(pred_spec).any()}")
                dprint(
                    f"[STFT Loss scale {i}] gt_spec.shape: {gt_spec.shape}, min/max: {gt_spec.min():.2f}/{gt_spec.max():.2f}, "
                    f"mean/std: {gt_spec.mean():.2f}/{gt_spec.std():.2f}, isnan: {torch.isnan(gt_spec).any()}")
                if pred_spec.shape[0] != B:
                    dprint(f"[STFT Loss scale {i} warning] pred_spec batch size {pred_spec.shape[0]} != B {B}")
                if pred_spec.dim() != 3:
                    dprint(
                        f"[STFT Loss scale {i} warning] pred_spec dim {pred_spec.dim()} != 3, shape: {pred_spec.shape}")
                if pred_spec.shape[1] != n_fft // 2 + 1:
                    dprint(
                        f"[STFT Loss scale {i} warning] pred_spec freq {pred_spec.shape[1]} != n_fft/2+1 {n_fft // 2 + 1}")
                if gt_spec.shape != pred_spec.shape:
                    dprint(
                        f"[STFT Loss scale {i} warning] gt_spec shape {gt_spec.shape} != pred_spec shape {pred_spec.shape}")

            # Align mask to spectrogram length
            T_spec = pred_spec.shape[-1]
            mask_resized = F.interpolate(mask.float(), size=T_spec, mode='nearest')  # [B, 1, T_spec]
            mask_spec = mask_resized.expand(-1, n_fft // 2 + 1, -1)  # [B, freq, T_spec]

            if DEBUG:
                dprint(
                    f"[STFT Loss scale {i}] mask_resized.shape: {mask_resized.shape}, sum: {mask_resized.sum().item()}")
                dprint(
                    f"[STFT Loss scale {i}] mask_spec.shape: {mask_spec.shape}, sum: {mask_spec.sum().item()}, "
                    f"sum(dim=2): {mask_spec.sum(dim=2)[:, 0]}, isnan: {torch.isnan(mask_spec).any()}")
                expected_frames = mask.sum(dim=2).sum().item() * (n_fft // 2 + 1) * (256 / hop_length)
                if mask_spec.sum().item() < expected_frames * 0.5:
                    dprint(
                        f"[STFT Loss scale {i} warning] mask_spec sum {mask_spec.sum().item()} < 50% of expected {expected_frames}")
                if mask_spec.shape[0] != B:
                    dprint(f"[STFT Loss scale {i} warning] mask_spec batch size {mask_spec.shape[0]} != B {B}")
                if mask_spec.shape[1] != n_fft // 2 + 1:
                    dprint(
                        f"[STFT Loss scale {i} warning] mask_spec freq {mask_spec.shape[1]} != n_fft/2+1 {n_fft // 2 + 1}")

            # 断言 mask_spec 形状匹配 pred_spec
            assert mask_spec.shape == pred_spec.shape, f"[STFT Loss scale {i}] mask_spec {mask_spec.shape} does not match pred_spec {pred_spec.shape}"

            total_frames += mask_spec.sum()

            # Magnitude and log-magnitude L1 loss
            mag_loss = F.l1_loss(pred_spec, gt_spec, reduction='none')
            log_mag_loss = F.l1_loss(torch.log(pred_spec + 1e-6), torch.log(gt_spec + 1e-6), reduction='none')
            if DEBUG:
                dprint(
                    f"[STFT Loss scale {i}] mag_loss raw min/max: {mag_loss.min():.2f}/{mag_loss.max():.2f}, "
                    f"after mask sum: {(mag_loss * mask_spec).sum().item():.2f}")
                dprint(
                    f"[STFT Loss scale {i}] log_mag_loss raw min/max: {log_mag_loss.min():.2f}/{log_mag_loss.max():.2f}, "
                    f"after mask sum: {(log_mag_loss * mask_spec).sum().item():.2f}")
                if mag_loss.shape[0] != B:
                    dprint(f"[STFT Loss scale {i} warning] mag_loss batch size {mag_loss.shape[0]} != B {B}")
                if mag_loss.shape != pred_spec.shape:
                    dprint(
                        f"[STFT Loss scale {i} warning] mag_loss shape {mag_loss.shape} != pred_spec shape {pred_spec.shape}")

            # Apply mask and sum
            mag_loss = (mag_loss * mask_spec).sum()
            log_mag_loss = (log_mag_loss * mask_spec).sum()

            stft_loss += mag_loss
            spec_loss += log_mag_loss

        # Average over resolutions and valid frames
        stft_loss = (stft_loss + spec_loss) / len(n_ffts) / total_frames.clamp(min=1)
        if DEBUG:
            dprint(
                f"[STFT Loss final] loss: {stft_loss.item():.4f}, total_frames: {total_frames.item()}, "
                f"isnan: {torch.isnan(stft_loss).any()}")
            expected_total_frames = mask.sum().item() * sum(n_fft // 2 + 1 for n_fft in n_ffts) / len(n_ffts)
            if total_frames.item() < expected_total_frames * 0.5:
                dprint(
                    f"[STFT Loss final warning] total_frames {total_frames.item()} < 50% of expected {expected_total_frames}")

        return stft_loss
    def get_cond(self, phoneme_ids, speaker_emb, x_lengths, y_max_length=None):
        if DEBUG:
            dprint(f"[get_cond] phoneme_ids.shape: {phoneme_ids.shape}, speaker_emb.shape: {speaker_emb.shape}, x_lengths: {x_lengths}, y_max_length: {y_max_length}")
        cond = self.conditioner(phoneme_ids, speaker_emb, x_lengths, y_max_length=y_max_length)
        if DEBUG:
            dprint(f"[get_cond output] keys: {cond.keys()}, mu_phon: {cond['mu_phon'].shape}, dur: {cond['dur'].shape if cond['dur'] is not None else 'None'}")
        return cond


    def compute_loss(self, y, phoneme_ids, spk, x_lengths, y_lengths, y_max_length, gt_dur=None,gt_wave=None):
        B = y.size(0)
        if DEBUG:
            dprint(f"[compute_loss input] y.shape: {y.shape}, phoneme_ids.shape: {phoneme_ids.shape}, spk.shape: {spk.shape}, x_lengths: {x_lengths}, y_lengths: {y_lengths}, y_max_length: {y_max_length}, gt_dur: {gt_dur.shape if gt_dur is not None else 'None'}")
        cond = self.get_cond(phoneme_ids, spk, x_lengths, y_max_length=None)  # 不用提前给 y_max_length
        mu_phon = cond['mu_phon']          # [B, 80, T_phon]
        logw = cond['logw']                # [B, T_phon]
        dur = cond['dur']                  # [B, T_phon]

        # --- (1) 得到 Ty 和 y_max_length ---
        Ty = y_lengths                     # tensor [B]
        if y_max_length is None:
            y_max_length = int(Ty.max().item())

        # --- (2) 展开 mu_phon 到 mu_x ---

        expanded_mu = []
        for b in range(B):
            Tb = x_lengths[b].item()

            if gt_dur is not None:
                d = gt_dur[b, :Tb].clamp(min=1.0, max=50.0)
                #d = dur[b, :Tb].clamp(min=1.0, max=50.0)
            elif dur is not None:
                d = dur[b, :Tb].clamp(min=1.0, max=50.0)
            else:
                mu_b = F.interpolate(mu_phon[b:b + 1], size=Ty[b].item(), mode='linear', align_corners=False).squeeze(0)
                if Ty[b].item() < y_max_length:
                    mu_b = F.pad(mu_b, (0, y_max_length - Ty[b].item()), value=0.)
                expanded_mu.append(mu_b)
                continue

            dur_int = d.round().long()  # [Tb]
            current_sum = dur_int.sum().item()
            target_ty = Ty[b].item()
            if current_sum != target_ty:
                diff = target_ty - current_sum
                if diff > 0:
                    dur_int += (torch.arange(Tb, device=d.device) < diff % Tb).long()
                    dur_int += diff // Tb
                elif diff < 0:
                    dur_int = torch.clamp(dur_int - (torch.arange(Tb, device=d.device) < (-diff) % Tb).long(), min=1)
                    dur_int += diff // Tb
                    dur_int = torch.clamp(dur_int, min=1)

            cum = torch.cumsum(dur_int, dim=0)
            idx = torch.searchsorted(
                cum,
                torch.arange(1, target_ty + 1, device=d.device)
            )
            idx = idx.clamp(0, Tb - 1)
            mu_b = mu_phon[b, :, idx]
            if target_ty < y_max_length:
                mu_b = F.pad(mu_b, (0, y_max_length - target_ty), value=0.)
            expanded_mu.append(mu_b)
        mu_x = torch.stack(expanded_mu)  # [B, 80, y_max_length]

        # --- (3) 归一化 mu_x ---
        # --- (3) 无归一化（y已标准化到N(0,1)） ---
        mask = sequence_mask(Ty, y_max_length).unsqueeze(1).float()  # [B,1,T]
        mu_x = mu_x.masked_fill(~mask.bool(), 0.0)  # 只填充 pad=0，防泄露
        if DEBUG:
            dprint(
                f"[compute_loss mu_x no norm] shape: {mu_x.shape}, min/max: {mu_x.min()}/{mu_x.max()}, mean/std: {mu_x.mean()}/{mu_x.std()}")
        mu_x = torch.clamp(mu_x, min=-10.0, max=10.0)  # 可选：clip防溢出（N(0,1)安全范围），若稳定可删

        # --- (4) 更新 cond & 扩散 ---
        cond['mu_x'] = mu_x
        cond['mask'] = mask
        if DEBUG:
            dprint(f"[compute_loss mask] shape: {mask.shape}, sum(dim=2): {mask.sum(dim=2)}")

        t = torch.randint(0, self.n_timesteps, (y.shape[0],), device=y.device)
        if not hasattr(self, '_firsttest'):
            self._firsttest = True
            t = torch.randint(self.n_timesteps//3, self.n_timesteps, (y.shape[0],), device=y.device)
        noise = torch.randn_like(y)
        if DEBUG:
            dprint(
                f"[Noise check] t: {t.tolist()}, sigma_t: {[self.sqrt_one_minus_alphas_cumprod[t[i]].item() for i in range(B)]}")
            dprint(
                f"[Noise shape] y.shape: {y.shape}, noise.shape: {noise.shape}, isnan noise: {torch.isnan(noise).any()}")
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        # print(
        #     f"[Debug t={t[0].item()}] sqrt_alphas_cumprod_t={sqrt_alpha_cumprod_t[0, 0, 0].item():.4f}, sqrt_one_minus_alphas_cumprod_t={sqrt_one_minus_alpha_cumprod_t[0, 0, 0].item():.4f}")
        z = sqrt_alpha_cumprod_t * y + sqrt_one_minus_alpha_cumprod_t * noise
        if DEBUG:
            dprint(f"[z shape] z.shape: {z.shape}, isnan z: {torch.isnan(z).any()}, min/max: {z.min()}/{z.max()}")
        epsilon_pred = self(z, t, cond, mask)
        if DEBUG:
            dprint(
                f"[epsilon_pred] shape: {epsilon_pred.shape}, isnan: {torch.isnan(epsilon_pred).any()}, min/max: {epsilon_pred.min()}/{epsilon_pred.max()}")
        y_hat = (z - sqrt_one_minus_alpha_cumprod_t * epsilon_pred) / sqrt_alpha_cumprod_t
        y_hat = torch.clamp(y_hat, -8.0, 8.0)
        if DEBUG:
            dprint(
                f"[epsilon_pred] shape: {epsilon_pred.shape}, isnan: {torch.isnan(epsilon_pred).any()}, min/max: {epsilon_pred.min()}/{epsilon_pred.max()}")
        diff_raw = F.mse_loss(epsilon_pred, noise, reduction='none')
        if not hasattr(self, '_saved_mel'):
            self._saved_mel = True
            z_sample = z[0].detach().cpu().numpy()  # 加噪mel
            y_sample = y[0].detach().cpu().numpy()  # 原始mel
            y_hat_sample = y_hat[0].detach().cpu().numpy()  # denoise后的mel

            # 保存为 .npy 文件
            np.save('./stft_test_output/z_noise.npy', z_sample)
            np.save('./stft_test_output/y_original.npy', y_sample)
            np.save('./stft_test_output/y_hat_denoise.npy', y_hat_sample)

            # 打印统计信息
            print(
                f"[MEL Stats] z_noise: min={z_sample.min():.4f}, max={z_sample.max():.4f}, mean={z_sample.mean():.4f}, std={z_sample.std():.4f}")
            print(
                f"[MEL Stats] y_original: min={y_sample.min():.4f}, max={y_sample.max():.4f}, mean={y_sample.mean():.4f}, std={y_sample.std():.4f}")
            print(
                f"[MEL Stats] y_hat_denoise: min={y_hat_sample.min():.4f}, max={y_hat_sample.max():.4f}, mean={y_hat_sample.mean():.4f}, std={y_hat_sample.std():.4f}")
        valid = mask.expand_as(diff_raw).float()
        diff_loss = (diff_raw * valid).sum() / valid.sum().clamp(min=1)
        prior_loss_raw = F.l1_loss(mu_x, y, reduction='none') * mask.expand_as(y)
        validp = mask.expand_as(prior_loss_raw).float()
        prior_loss = (prior_loss_raw * validp).sum() / validp.sum().clamp(min=1)

        # --- dur_loss ---
        dur_loss = torch.tensor(0.0, device=y.device)
        if logw is not None:
            x_mask = sequence_mask(x_lengths, logw.shape[1]).float()
            if DEBUG:
                dprint(f"[dur_loss x_mask] shape: {x_mask.shape}, sum(dim=1): {x_mask.sum(dim=1)}")
            pred_dur = torch.exp(logw)
            pred_dur = pred_dur.masked_fill(~x_mask.bool(), 0.0)
            pred_dur = torch.clamp(pred_dur, min=1.0, max=50.0)
            if DEBUG:
                dprint(
                    f"[dur_loss pred_dur] shape: {pred_dur.shape}, min/max: {pred_dur.min()}/{pred_dur.max()}, sum: {pred_dur.sum(1)}")
            # L_len: 绝对帧差，归一化
            pred_len = pred_dur.sum(1)
            L_len = ((pred_len - y_lengths.float()).abs() / y_lengths.float().mean().clamp(min=1.)).mean()
            if DEBUG:
                dprint(f"[dur_loss L_len] pred_len: {pred_len}, y_lengths: {y_lengths}, value: {L_len}")
            # L_tv
            dur_diff = (torch.log(pred_dur + 1e-6)[:, 1:] - torch.log(pred_dur + 1e-6)[:, :-1]).abs()
            tv_mask = x_mask[:, 1:]
            L_tv = (dur_diff * tv_mask).sum() / tv_mask.sum().clamp_min(1.)
            L_tv = L_tv.clamp(min=0)
            if DEBUG:
                dprint(f"[dur_loss L_tv] value: {L_tv}")
            # L_dur_gt
            L_dur_gt = torch.tensor(0.0, device=y.device)
            if gt_dur is not None:
                gt_dur = gt_dur.masked_fill(~x_mask.bool(), 0.0)
                gt_dur = torch.clamp(gt_dur, min=1.0, max=50.0)
                L_dur_gt_raw = F.l1_loss(torch.log(pred_dur + 1e-6), torch.log(gt_dur + 1e-6), reduction='none')
                L_dur_gt = (L_dur_gt_raw * x_mask).sum() / x_mask.sum().clamp_min(1.)
                if DEBUG:
                    dprint(f"[dur_loss L_dur_gt] value: {L_dur_gt}")
            # 条件化权重
            if gt_dur is not None:
                dur_loss = 0.0 * L_len + 0.0 * L_tv + 1.0 * L_dur_gt
            else:
                dur_loss = 1.0 * L_len + 0.3 * L_tv
            if DEBUG:
                dprint(f"[dur_loss final] L_len: {L_len}, L_tv: {L_tv}, L_dur_gt: {L_dur_gt}, total: {dur_loss}")

        if gt_wave is not None:
            perceptual_l = self.compute_stft_loss(y_hat, y, gt_wave, mask)

            if DEBUG:
                dprint(
                    f"[compute_loss y_hat vs y] y_hat min/max: {y_hat.min():.2f}/{y_hat.max():.2f}, y min/max: {y.min():.2f}/{y.max():.2f}")
                diff = F.mse_loss(y_hat, y)
                dprint(f"[compute_loss y_hat vs y MSE] {diff.item():.4f}")
        else:
            perceptual_l = torch.tensor(0.0, device=y.device)  # 如果无 gt_wave，跳过
        if DEBUG:
            print(
                f"Dur sum: {dur.sum(1)}, y_lengths: {y_lengths}, diff%: {((dur.sum(1) - y_lengths).abs() / y_lengths).mean().item():.4f}")

        # if DEBUG:
        #     dprint(
        #         f"[recon_loss] shape: {recon_loss.shape}, min/max: {recon_loss.min()}/{recon_loss.max()}, mean: {recon_loss.mean()}, isnan: {torch.isnan(recon_loss).any()}")
        def masked_var(x, mask, dim=[1, 2]):
            mask_exp = mask.expand_as(x).float()
            num_valid = mask_exp.sum(dim=dim).clamp(min=1.0)

            # 用 clipped mean 代替 median，提升鲁棒性但保持稠密梯度
            clipped_x = torch.clamp(x, min=-5.0, max=5.0)  # clip outliers
            mean = (clipped_x * mask_exp).sum(dim=dim) / num_valid

            sq_diff = ((x - mean.view(-1, 1, 1)) ** 2 * mask_exp)
            sq_diff = torch.clamp(sq_diff, 0.0, 5.0)  # 如原版，防爆炸
            var = sq_diff.sum(dim=dim) / num_valid
            return var.mean() # scalar，平均过batch
        var_y = masked_var(y, mask, dim=[1, 2])
        if DEBUG:
            dprint(
                f"[compute_loss y_hat vs y] y_hat min/max: {y_hat.min():.2f}/{y_hat.max():.2f}, y min/max: {y.min():.2f}/{y.max():.2f}")
            diff = F.mse_loss(y_hat, y)
            dprint(f"[compute_loss y_hat vs y MSE] {diff.item():.4f}")
        if DEBUG:
            pad_lengths = y_max_length - y_lengths
            dprint(f"[Var Log] y_lengths: {y_lengths}, pad_lengths: {pad_lengths}, y_max_length: {y_max_length}")
            var_y_no_mask = y.var(dim=[1, 2]).mean()
            var_y_hat_no_mask = y_hat.var(dim=[1, 2]).mean()
            dprint(f"[Var Log No Mask] var_y: {var_y_no_mask.item():.4f}, var_y_hat: {var_y_hat_no_mask.item():.4f}")
        var_y_hat = masked_var(y_hat, mask, dim=[1, 2])
        var_loss = (var_y_hat - var_y).abs().mean()  # L1防敏感
        # if DEBUG:
        #     dprint(
        #         f"[Var Log Masked] var_y: {var_y.item():.4f}, var_y_hat: {var_y_hat.item():.4f}, var_loss: {var_loss.item():.4f}")
        # if torch.isnan(var_loss).any():
        #     print(
        #         f"[NaN in var_loss] var_y: {var_y}, var_y_hat: {var_y_hat}, sq_diff min/max: {sq_diff.min()}/{sq_diff.max()}")
        valid = mask.expand_as(y_hat).float()  # [B, 1, T] -> [B, 80, T]
        excess = F.relu(y_hat - 4.0)
        overshoot_loss = ((excess ** 2) * valid).sum() / valid.sum().clamp(min=1)
        #var_loss = torch.tensor(0.0, device=device)
        var_loss = torch.clamp(var_loss, 0.0, 10.0)

        return diff_loss, prior_loss, dur_loss,perceptual_l,overshoot_loss,var_loss

    def sample(self, phoneme_ids, spk, x_lengths, n_timesteps=None, temperature=0.5, y_max_length=None, cfg=True,cfg_scale=0.5):
        if n_timesteps is None:
            n_timesteps = self.n_timesteps

        # 1) 先拿到 logw / mu_phon
        cond = self.get_cond(phoneme_ids, spk, x_lengths, y_max_length=None)
        logw = cond['logw']  # [B, T_phon]
        mu_phon = cond['mu_phon']  # [B, 80, T_phon]
        B, _, T_phon = mu_phon.shape
        device = phoneme_ids.device

        # 2) raw_dur & mask
        x_mask = sequence_mask(x_lengths, T_phon).float().to(device)
        raw_dur = torch.exp(logw).masked_fill(~x_mask.bool(), 0.0)
        raw_dur = torch.clamp(raw_dur, min=1.0, max=50.0)
        if DEBUG:
            dprint(f"[sample raw_dur] shape: {raw_dur.shape}, sum: {raw_dur.sum(1)}")

        # 3) 估计 mel 总长度
        if y_max_length is None:
            y_lengths_pred = raw_dur.sum(1).round().int().clamp(min=1)
            y_max_length = int(y_lengths_pred.max().item())
        else:
            y_lengths_pred = torch.clamp(raw_dur.sum(1).round().int(), min=1, max=y_max_length)
        if DEBUG:
            dprint(f"[sample y_lengths_pred] values: {y_lengths_pred}, y_max_length: {y_max_length}")

        # 4) 直接round，无需scale
        dur_int = raw_dur.round().long().clamp(min=1, max=50)
        for b in range(B):
            Tb = x_lengths[b].item()
            current_sum = dur_int[b, :Tb].sum().item()
            target_ty = y_lengths_pred[b].item()
            if current_sum != target_ty:
                diff = target_ty - current_sum
                if diff > 0:
                    dur_int[b, :Tb] += (torch.arange(Tb, device=device) < diff % Tb).long()
                    dur_int[b, :Tb] += diff // Tb
                elif diff < 0:
                    dur_int[b, :Tb] = torch.clamp(
                        dur_int[b, :Tb] - (torch.arange(Tb, device=device) < (-diff) % Tb).long(),
                        min=1
                    )
                    dur_int[b, :Tb] += diff // Tb
                    dur_int[b, :Tb] = torch.clamp(dur_int[b, :Tb], min=1)
            if DEBUG:
                dprint(f"[sample dur_int b={b}] sum: {dur_int[b].sum()} vs target: {target_ty}")

        # 5) 展开 mu_phon -> mu_x
        expanded_mu = []
        for b in range(B):
            Ty = y_lengths_pred[b].item()
            cum = torch.cumsum(dur_int[b, :x_lengths[b]], dim=0)
            idx = torch.searchsorted(cum, torch.arange(1, Ty + 1, device=device))
            idx = idx.clamp(0, T_phon - 1)
            mu_b = mu_phon[b, :, idx]
            if Ty < y_max_length:
                mu_b = F.pad(mu_b, (0, y_max_length - Ty), value=0.)
            expanded_mu.append(mu_b)
        mu_x = torch.stack(expanded_mu, dim=0)

        # 6) 归一化
        # 7) 无归一化（y已标准化到N(0,1)）
        mask = sequence_mask(y_lengths_pred, y_max_length).unsqueeze(1).float().to(device)
        mu_x = mu_x.masked_fill(~mask.bool(), 0.0)  # 只填充 pad=0
        if DEBUG:
            dprint(
                f"[sample mu_x no norm] shape: {mu_x.shape}, min/max: {mu_x.min()}/{mu_x.max()}, mean/std: {mu_x.mean()}/{mu_x.std()}")
        #mu_x = torch.clamp(mu_x, min=-10.0, max=10.0)  # 可选 clip

        # 7) Diffusion 采样
        cond['mu_x'] = mu_x
        cond['mask'] = mask
        z = mu_x + torch.randn(B, self.n_feats, y_max_length, device=device)
        null_cond = {'mu_x': torch.zeros_like(mu_x), 'spk_feat': torch.zeros_like(cond['spk_feat']), 'mask': mask}

        for i in reversed(range(n_timesteps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            eps = self(z, t, cond, mask)
            if cfg:
                eps_null = self(z, t, null_cond, mask)
                eps = (1 + cfg_scale) * eps - cfg_scale * eps_null
            alpha_t = self.alphas[t].view(-1, 1, 1)
            sigma_t = torch.sqrt(self.posterior_variance[t]).view(-1, 1, 1)
            z = (z - (1 - alpha_t) / torch.sqrt(1 - self.alphas_cumprod[t].view(-1, 1, 1)) * eps) / torch.sqrt(alpha_t)
            z = torch.clamp(z, -5.0, 5.0)
            z += temperature * sigma_t * torch.randn_like(z)
        mel_mean, mel_std = -8.122798642, 2.1809869538
        z = z * mel_std + mel_mean  # 必须保留，匹配HiFi-GAN输入
        z = torch.clamp(z, min=-40.0, max=2.0)
        return z