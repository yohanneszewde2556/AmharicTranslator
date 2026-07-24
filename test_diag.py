"""
Diagnostic script to test exact translations for ሰላም እንዴት ነህ and variations.
"""
import torch
import sentencepiece as spm
import re
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import DATA_VERSION
from src.inference import load_model, beam_search_decode, greedy_decode
from src.normalizer import normalize_amharic_text

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
_PROC_DIR      = os.path.join(BASE_DIR, "data", "processed", DATA_VERSION) \
                 if DATA_VERSION else os.path.join(BASE_DIR, "data", "processed")
SPM_MODEL      = os.path.join(_PROC_DIR, "am_en_bpe.model")
CHECKPOINT     = os.path.join(BASE_DIR, "checkpoints", "best_model.pt")

def normalize_attached_punctuation(text: str) -> str:
    # 1. Standardize homophones
    text = re.sub(r'[ሐኀ]', 'ሀ', text)
    text = re.sub(r'[ሑኁ]', 'ሁ', text)
    text = re.sub(r'[ሒኺ]', 'ሂ', text)
    text = re.sub(r'[ሓኻ]', 'ሃ', text)
    text = re.sub(r'[ሔኼ]', 'ሄ', text)
    text = re.sub(r'[ሕኅ]', 'ህ', text)
    text = re.sub(r'[ሖኆ]', 'ሆ', text)

    text = re.sub(r'ሠ', 'ሰ', text)
    text = re.sub(r'ሡ', 'ሱ', text)
    text = re.sub(r'ሢ', 'ሲ', text)
    text = re.sub(r'ሣ', 'ሳ', text)
    text = re.sub(r'ሤ', 'ሴ', text)
    text = re.sub(r'ሥ', 'ስ', text)
    text = re.sub(r'ሦ', 'ሶ', text)

    text = re.sub(r'ዐ', 'አ', text)
    text = re.sub(r'ዑ', 'ኡ', text)
    text = re.sub(r'ዒ', 'ኢ', text)
    text = re.sub(r'ዓ', 'ኣ', text)
    text = re.sub(r'ዔ', 'ኤ', text)
    text = re.sub(r'ዕ', 'እ', text)
    text = re.sub(r'ዖ', 'ኦ', text)

    text = re.sub(r'ፀ', 'ጸ', text)
    text = re.sub(r'ፁ', 'ጹ', text)
    text = re.sub(r'ፂ', 'ጺ', text)
    text = re.sub(r'ፃ', 'ጻ', text)
    text = re.sub(r'ፄ', 'ጼ', text)
    text = re.sub(r'ፅ', 'ጽ', text)
    text = re.sub(r'ፆ', 'ጾ', text)

    # 2. Collapse spaces around text
    text = re.sub(r'\s+', ' ', text).strip()

    # 3. Interrogative question mark handling: ensure '?' is attached without a leading space!
    question_words = ['እንዴት', 'ማን', 'ምን', 'ምንድን', 'ለምን', 'መቼ', 'የት', 'ስንት', 'እንደምን']
    has_qw = any(qw in text for qw in question_words)

    if (has_qw or '?' in text):
        text = text.rstrip('.!።? ') + '?'

    return text

def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(CHECKPOINT, device)
    sp = spm.SentencePieceProcessor()
    sp.load(SPM_MODEL)

    phrases = [
        "ሰላም እንዴት ነህ",
        "ሰላም እንዴት ነህ?",
        "እንዴት ነህ",
        "እንዴት ነህ?"
    ]

    print("=" * 65)
    for p in phrases:
        norm = normalize_attached_punctuation(p)
        ids = [1] + sp.encode(norm, out_type=int) + [2]
        out = sp.decode(beam_search_decode(model, torch.tensor([ids]).to(device), 50, device, 5).tolist())

        print(f"Raw Input  : '{p}'")
        print(f"Norm Input : '{norm}'")
        print(f"BPE Tokens : {sp.encode(norm, out_type=str)}")
        print(f"Translation: '{out}'\n")

if __name__ == "__main__":
    run()
