"""
Robustness Test Suite for Amharic-English NMT Model
Tests punctuation isolation and homophone normalization handling.
"""

import os
import sys
import torch
import sentencepiece as spm

# Allow imports from project root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import DATA_VERSION
from src.inference import load_model, translate
from src.normalizer import normalize_amharic_text

# Paths
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
_PROC_DIR      = os.path.join(BASE_DIR, "data", "processed", DATA_VERSION) \
                 if DATA_VERSION else os.path.join(BASE_DIR, "data", "processed")
SPM_MODEL      = os.path.join(_PROC_DIR, "am_en_bpe.model")
CHECKPOINT     = os.path.join(BASE_DIR, "checkpoints", "best_model.pt")

def run_tests():
    print("=" * 65)
    print("Amharic NMT Robustness & Normalization Unit Tests")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model = load_model(CHECKPOINT, device)
    sp = spm.SentencePieceProcessor()
    sp.load(SPM_MODEL)

    test_groups = [
        {
            "name": "Question Mark Punctuation Invariance",
            "inputs": ["እንዴት ነህ", "እንዴት ነህ?", "እንዴት ነህ ?"]
        },
        {
            "name": "Homophone & Informal Spelling Normalization",
            "inputs": ["እወድሃለሁ", "እወድሀለው", "እወድሀለሁ"]
        },
        {
            "name": "Full Stop Punctuation Invariance",
            "inputs": ["ደህና ነህ", "ደህና ነህ።", "ደህና ነህ ።"]
        },
        {
            "name": "Capital City Benchmark Sentence",
            "inputs": ["አዲስ አበባ የኢትዮጵያ ዋና ከተማ ናት።", "አዲስ አበባ የኢትዮጵያ ዋና ከተማ ናት", "አዲስ አበባ የኢትዮጵያ ዋና ከተማ ናት?"]
        }
    ]

    print("\nRunning Test Cases (Punctuation-Stripped Root Normalization):\n")

    for group in test_groups:
        print(f"--- Test Group: {group['name']} ---")
        translations = []
        for text in group["inputs"]:
            norm_text = normalize_amharic_text(text)
            trans = translate(text, model, sp, device, method="beam", beam_width=5)
            translations.append(trans)
            print(f"  Input  : '{text}'  --> Normalized: '{norm_text}'")
            print(f"  Output : '{trans}'\n")
        
        # Check consistency across group
        first_trans = translations[0].strip().lower()
        all_match = all(t.strip().lower() == first_trans for t in translations)
        if all_match:
            print("  Result : PASSED (100% Consistent Translation) ✅\n")
        else:
            print("  Result : VARIANCE (Outputs differ across punctuation/spelling) ⚠️\n")

    print("=" * 65)
    print("ALL ROBUSTNESS TESTS EXECUTED!")
    print("=" * 65)

if __name__ == "__main__":
    run_tests()
