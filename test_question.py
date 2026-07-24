"""
Debug script for testing punctuation isolation vs question mark auto-inference.
"""
import torch
import sentencepiece as spm
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

def normalize_interrogative_amharic(text: str) -> str:
    norm = normalize_amharic_text(text)
    # List of common Amharic interrogative question words
    question_words = ['እንዴት', 'ማን', 'ምን', 'ምንድን', 'ለምን', 'መቼ', 'የት', 'ስንት', 'እንደምን']
    
    # If input contains a question word or already has ?, ensure it ends with isolated ' ?'
    has_question_word = any(qw in norm for qw in question_words)
    if (has_question_word or '?' in norm) and not norm.endswith('?'):
        norm = norm.rstrip('.!። ') + ' ?'
    
    return norm

def test_inputs():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(CHECKPOINT, device)
    sp = spm.SentencePieceProcessor()
    sp.load(SPM_MODEL)

    test_cases = [
        "ሰላም እንዴት ነህ?",
        "ሰላም እንዴት ነህ",
        "እንዴት ነህ?",
        "እንዴት ነህ"
    ]

    print("=" * 60)
    for text in test_cases:
        norm = normalize_interrogative_amharic(text)
        src_ids = [1] + sp.encode(norm, out_type=int) + [2]
        src = torch.tensor([src_ids], dtype=torch.long).to(device)
        out_ids = beam_search_decode(model, src, 50, device, beam_width=5)
        translation = sp.decode(out_ids.tolist())
        print(f"Raw Input: '{text}'  --> Interrogative Norm: '{norm}'")
        print(f"Tokens   : {sp.encode(norm, out_type=str)}")
        print(f"Output   : '{translation}'\n")

if __name__ == "__main__":
    test_inputs()
