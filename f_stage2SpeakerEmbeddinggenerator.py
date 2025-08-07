import os
import random
import torch
import torchaudio as ta
from tqdm import tqdm
from speechbrain.inference import EncoderClassifier

FILELIST_PATH = "data/LibriTTS/full_libritts_filelist.txt"
OUTPUT_PT_FILE = "data/LibriTTS/speaker_embeddings_192_full.pt"
MAX_PER_SPEAKER = 10
SAMPLE_RATE = 16000

run_opts = {"device": "cuda" if torch.cuda.is_available() else "cpu"}

ecapa = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/ecapa",
    run_opts=run_opts
).eval()

# Group WAVs by speaker from filelist
speaker_wavs = {}
with open(FILELIST_PATH, "r") as f:
    lines = f.readlines()
for line in lines:
    speaker_id, wav_path, _ = line.strip().split('|', 2)
    if speaker_id not in speaker_wavs:
        speaker_wavs[speaker_id] = []
    speaker_wavs[speaker_id].append(wav_path)

speaker_embeddings = {}
for sid in tqdm(speaker_wavs, desc="Extracting ECAPA embeddings"):
    wav_paths = speaker_wavs[sid]
    selected_wavs = random.sample(wav_paths, k=min(MAX_PER_SPEAKER, len(wav_paths)))
    all_waveforms = []
    for wav_path in selected_wavs:
        wav, sr = ta.load(wav_path)
        if sr != SAMPLE_RATE:
            wav = ta.functional.resample(wav, sr, SAMPLE_RATE)
        all_waveforms.append(wav)
    merged_waveform = torch.cat(all_waveforms, dim=1).to(run_opts["device"])
    with torch.no_grad():
        emb = ecapa.encode_batch(merged_waveform).squeeze(0).squeeze(0).cpu()
        speaker_embeddings[sid] = emb

torch.save(speaker_embeddings, OUTPUT_PT_FILE)
print(f"Saved {len(speaker_embeddings)} speaker embeddings to {OUTPUT_PT_FILE}")