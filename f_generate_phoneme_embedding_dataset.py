import os
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from text2phonemesequence import Text2PhonemeSequence

MODEL_NAME = "vinai/xphonebert-base"
TEXT_ROOT = "data/LibriTTS/train-clean-100"  # 假设每个 speaker 下的 txt 都是我们要提取的
SELECTED_IDS_FILE = "data/selected_speakers.txt"
OUTPUT_DIR = "phoneme_embeddings_xphonebert"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
text2phone_model = Text2PhonemeSequence(language='eng-us', is_cuda=True)
model.eval()


with open(SELECTED_IDS_FILE, "r") as f:
    speaker_ids = [line.strip() for line in f if line.strip()]


for speaker_id in tqdm(speaker_ids, desc="Processing speakers"):
    speaker_path = os.path.join(TEXT_ROOT, speaker_id)
    if not os.path.isdir(speaker_path):
        print(f"❌ Skip: {speaker_id} path not found.")
        continue

    for chapter in os.listdir(speaker_path):
        chapter_path = os.path.join(speaker_path, chapter)
        if not os.path.isdir(chapter_path):
            continue

        for fname in os.listdir(chapter_path):
            if not fname.endswith(".normalized.txt"):
                continue
            txt_path = os.path.join(chapter_path, fname)


            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
                input_phonemes= text2phone_model.infer_sentence(text)
                if not input_phonemes.strip():
                    continue


            with torch.no_grad():
                inputs = tokenizer(input_phonemes, return_tensors="pt",add_special_tokens=False).to(DEVICE)
                outputs = model(**inputs)
                embeddings = outputs.last_hidden_state.squeeze(0).cpu()





            out_name = fname.replace('.normalized.txt', '.pt')
            out_path = os.path.join(OUTPUT_DIR, out_name)
            torch.save(embeddings, out_path)
