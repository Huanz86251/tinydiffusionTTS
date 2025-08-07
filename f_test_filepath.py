import os
from tqdm import tqdm

FILELIST_PATH = "data/LibriTTS/full_libritts_filelist.txt"

bad_format_lines = []
missing_wav_lines = []
missing_txt_lines = []
empty_text_lines = []
mismatch_lines = []

with open(FILELIST_PATH, "r", encoding="utf-8") as f:
    lines = f.readlines()

for idx, line in enumerate(tqdm(lines, desc="Checking filelist")):
    parts = line.strip().split("|", 2)

    if len(parts) != 3:
        bad_format_lines.append(idx + 1)
        continue

    speaker_id, wav_path, text = parts
    txt_path = wav_path.replace(".wav", ".normalized.txt")

 
    if not os.path.exists(wav_path):
        missing_wav_lines.append(idx + 1)

 
    if not os.path.exists(txt_path):
        missing_txt_lines.append(idx + 1)
        continue


    with open(txt_path, "r", encoding="utf-8") as f_txt:
        txt_content = f_txt.read().strip()
        if txt_content == "":
            empty_text_lines.append(idx + 1)
        elif txt_content != text.strip():
            mismatch_lines.append(idx + 1)


print("\n=== Filelist Validation Report ===")
print(f"Total lines checked: {len(lines)}")
print(f" Bad format lines: {len(bad_format_lines)}")
print(f" Missing .wav files: {len(missing_wav_lines)}")
print(f" Missing .normalized.txt files: {len(missing_txt_lines)}")
print(f" Empty text in .normalized.txt: {len(empty_text_lines)}")
print(f" Mismatch between filelist text and .normalized.txt: {len(mismatch_lines)}")
print("===================================")
