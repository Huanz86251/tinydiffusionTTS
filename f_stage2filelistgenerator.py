
import os
import json
import torch
import torchaudio as ta
import textgrid
import glob
from tqdm import tqdm
from speechbrain.inference import EncoderClassifier
from collections import Counter
import random
import string
import time
import re

def normalize(text):
    return re.sub(r'[^a-z0-9]', '', text.lower())

# 配置
LIBRITTS_ROOT = "/root/autodl-tmp/goodluck/data/LibriTTS-R"
SPLITS = ["train-clean-100", "train-clean-360"]  # 成功后再加 train-other-500
SAMPLE_RATE = 22050
SAVE_DIR = "/root/autodl-tmp/goodluck/data/LibriTTS-R"
os.makedirs(SAVE_DIR, exist_ok=True)
SPK_EMB_PATH = os.path.join(SAVE_DIR, "speaker_embeddings_192_r_aligned.pt")
VOCAB_PATH = os.path.join(SAVE_DIR, "phone2idx.json")
FILELIST_PATH = os.path.join(SAVE_DIR, "full_libritts_aligned_filelist.txt")
MAX_PER_SPEAKER = 5

# 初始化 ECAPA
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ecapa = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/ecapa",
    run_opts={"device": DEVICE}
).eval()

# 收集 wav 和 alignments
speaker_wavs = {}
phone_counter = Counter()
failed_samples = []
valid_samples = []

for split in SPLITS:
    print(f"\n🔹 Processing split: {split}")
    wav_dir = os.path.join(LIBRITTS_ROOT, split)
    align_dir = os.path.join(LIBRITTS_ROOT, f"{split}-alignments")
    if not os.path.isdir(wav_dir) or not os.path.isdir(align_dir):
        print(f"❌ Missing wav_dir {wav_dir} or align_dir {align_dir}")

        continue

    wav_files = glob.glob(wav_dir + "/**/*.wav", recursive=True)
    for wav_path in tqdm(wav_files, desc=f"Processing {split}"):
        wav_id = os.path.basename(wav_path).replace('.wav', '')  # e.g., 100_121669_000000_000000
        speaker_id = wav_id.split('_')[0]
        txt_path = wav_path.replace('.wav', '.normalized.txt')
        tg_path = os.path.join(align_dir, f"{wav_id}.TextGrid")

        # 验证 wav、txt 和 TextGrid 存在
        if not (os.path.exists(wav_path) and os.path.exists(txt_path) and os.path.exists(tg_path)):
            failed_samples.append(wav_id)
            print(f"❌ Missing wav/txt/TextGrid for {wav_id}")
            continue

        # 验证文本匹配
        try:
            with open(txt_path, 'r', encoding='utf-8') as txt_f:
                local_text = txt_f.read().strip().lower()
                local_text = local_text.translate(str.maketrans('', '', string.punctuation))
            tg = textgrid.TextGrid.fromFile(tg_path)
            tg_text = ''
            for tier in tg.tiers:
                if tier.name == "words":
                    tg_text = ' '.join(interval.mark.strip().lower() for interval in tier if interval.mark.strip())
                    tg_text = tg_text.translate(str.maketrans('', '', string.punctuation))
                    break
            if normalize(tg_text) != normalize(local_text):
                failed_samples.append(wav_id)

                print(f"\n❌ Text mismatch example for {wav_id}:")
                print("------ TextGrid text ------")
                print(tg_text)
                print("------ .normalized.txt text ------")
                print(local_text)
                print("---------------------------\n")
                mismatch_debug_printed = True


                continue
        except Exception as e:
            failed_samples.append(wav_id)
            print(f"❌ Failed to parse TextGrid or read txt for {wav_id}: {e}")
            continue

        # 提取 phones 和 durations
        try:
            phones = []
            durations = []
            for tier in tg.tiers:
                if tier.name == "phones":
                    for interval in tier:
                        phone = interval.mark if interval.mark else "<SILENCE>"
                        phones.append(phone)
                        dur_frames = (interval.maxTime - interval.minTime) * (SAMPLE_RATE / 256)  # hop=256
                        durations.append((round(dur_frames)))
                    break
            if not phones or not durations or len(phones) != len(durations) or sum(durations) == 0:
                failed_samples.append(wav_id)
                print(f"❌ Invalid phones/durations for {wav_id}")
                continue
            phone_counter.update(phones)
        except Exception as e:
            failed_samples.append(wav_id)
            print(f"❌ Failed to parse phones/durations for {wav_id}: {e}")
            continue

        # 收集 wav 用于 speaker embedding
        try:
            waveform, sr = ta.load(wav_path)
            if sr != SAMPLE_RATE:
                waveform = ta.functional.resample(waveform, sr, SAMPLE_RATE)
            if speaker_id not in speaker_wavs:
                speaker_wavs[speaker_id] = []
            if len(speaker_wavs[speaker_id]) < MAX_PER_SPEAKER:
                speaker_wavs[speaker_id].append(waveform)
            valid_samples.append((speaker_id, wav_path, local_text, phones, durations))
        except Exception as e:
            failed_samples.append(wav_id)
            print(f"❌ Failed to load audio for {wav_id}: {e}")
            continue

# 生成 speaker embeddings
speaker_embeddings = {}
failed_speakers = []
for speaker_id, wavs in tqdm(speaker_wavs.items(), desc="Extracting speaker embeddings"):
    try:
        selected_wavs = random.sample(wavs, min(len(wavs), MAX_PER_SPEAKER))
        merged_waveform = torch.cat(selected_wavs, dim=1).to(DEVICE)
        with torch.no_grad():
            emb = ecapa.encode_batch(merged_waveform).squeeze(0).squeeze(0).cpu()
        speaker_embeddings[speaker_id] = emb
    except Exception as e:
        failed_speakers.append(speaker_id)
        print(f"❌ Failed to compute embedding for speaker {speaker_id}: {e}")

# 保存 speaker embeddings
torch.save(speaker_embeddings, SPK_EMB_PATH)
print(f"✅ Saved {len(speaker_embeddings)} speaker embeddings to {SPK_EMB_PATH}")
if failed_speakers:
    print(f"⚠️ Failed speakers: {len(failed_speakers)} {failed_speakers[:10]}")
if failed_samples:
    print(f"⚠️ Failed samples: {len(failed_samples)} {failed_samples[:10]}")

# 构建 phone2idx
sorted_phones = sorted(phone_counter.keys())
phone2idx = {"<PAD>": 0, "<UNK>": 1}
phone2idx.update({ph: idx + 1 for idx, ph in enumerate(sorted_phones)})
with open(VOCAB_PATH, "w") as f:
    json.dump(phone2idx, f, indent=2)
print(f"✅ Saved phone2idx vocab ({len(phone2idx)} entries) to {VOCAB_PATH}")

# 生成 FileList 和 .pt 文件
with open(FILELIST_PATH, 'w', encoding='utf-8') as f:
    discarded = 0
    for speaker_id, wav_path, local_text, phones, durations in valid_samples:
        wav_id = os.path.basename(wav_path).replace('.wav', '')
        pt_path = wav_path.replace('.wav', '.pt')

        # 转换 phoneme_ids
        phoneme_ids = [phone2idx.get(ph, phone2idx.get('<UNK>', 0)) for ph in phones]

        # 保存 .pt
        try:
            os.makedirs(os.path.dirname(pt_path), exist_ok=True)
            torch.save({
                'phoneme_ids': torch.tensor(phoneme_ids, dtype=torch.long),
                'durations': torch.tensor(durations, dtype=torch.float)
            }, pt_path)
        except Exception as e:
            print(f"❌ Failed to save .pt for {wav_id}: {e}")
            discarded += 1
            continue

        # 写入 FileList
        f.write(f"{speaker_id}|{pt_path}|{local_text}\n")
vocab_size = len(phone2idx)
print(f"Generated FileList at {FILELIST_PATH}, discarded {discarded} incomplete/mismatched samples.")
print(f"Phoneme+Dur .pt files saved alongside each .wav in their directories.")
print(f"Vocab size: {vocab_size}. Update MLPConditioner vocab_size accordingly.")
