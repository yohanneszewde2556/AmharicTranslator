"""
Test greeting boundary normalization for ሰላም -> hello!
"""
import torch
import sentencepiece as spm
import re
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import DATA_VERSION
from src.inference import load_model, beam_search_decode

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
_PROC_DIR      = os.path.join(BASE_DIR, "data", "processed", DATA_VERSION) \
                 if DATA_VERSION else os.path.join(BASE_DIR, "data", "processed")
SPM_MODEL      = os.path.join(_PROC_DIR, "am_en_bpe.model")
CHECKPOINT     = os.path.join(BASE_DIR, "checkpoints", "best_model.pt")

def test_greeting():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(CHECKPOINT, device)
    sp = spm.SentencePieceProcessor()
    sp.load(SPM_MODEL)

    inputs = [
        "ሰላም እንዴት ነህ",
        "ሰላም! እንዴት ነህ",
        "ሰላም እንዴት ነሽ",
        "ጤና ይስጥልኝ! እንዴት ነህ",
        "ጤና ይስጥልኝ እንዴት ነህ"
    ]

    print("=" * 65)
    for p in inputs:
        # Add ! after initial greeting ሰላም if missing
        norm = re.sub(r'^(ሰላም)\s+', r'\1! ', p)
        if any(qw in norm for qw in ['እንዴት', 'ማን', 'ምን', 'ምንድን', 'ለምን', 'መቼ', 'የት', 'ስንት']) or '?' in norm:
            norm = norm.rstrip('.!።? ') + '?'

        ids = [1] + sp.encode(norm, out_type=int) + [2]
        out = sp.decode(beam_search_decode(model, torch.tensor([ids]).to(device), 50, device, 5).tolist())

        print(f"Raw Input  : '{p}'")
        print(f"Norm Input : '{norm}'")
        print(f"Translation: '{out}'\n")

if __name__ == "__main__":
    test_greeting()
