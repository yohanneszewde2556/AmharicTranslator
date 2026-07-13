"""
Phase 4 — Evaluation
Runs the trained model against test.csv and computes:
  - BLEU  : standard MT metric via SacreBLEU
  - chrF  : character n-gram metric — more reliable for Amharic's rich morphology
  - chrF++ : chrF with word n-grams added (SacreBLEU default chrf++)

Outputs a detailed report with per-sentence samples and aggregate scores.
"""
import torch
import sentencepiece as spm
import pandas as pd
import sacrebleu
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import MAX_LENGTH
from src.inference import load_model, translate


# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_CSV       = os.path.join(BASE_DIR, "data", "processed", "test.csv")
SPM_MODEL      = os.path.join(BASE_DIR, "data", "processed", "am_en_bpe.model")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")


def evaluate(
    checkpoint_path: str = None,
    test_csv: str = TEST_CSV,
    beam_width: int = 5,
    max_len: int = MAX_LENGTH,
    sample_count: int = 20
):
    """
    Full evaluation pipeline.

    Args:
        checkpoint_path : path to best_model.pt. Defaults to checkpoints/best_model.pt
        test_csv        : path to test split CSV
        beam_width      : beam width for beam search decoding
        max_len         : max tokens to generate per sentence
        sample_count    : number of example translations to print in report
    """

    # ── Setup ─────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("Amharic-English NMT — Evaluation")
    print("=" * 60)
    print(f"Device : {device}")

    if checkpoint_path is None:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")

    if not os.path.exists(checkpoint_path):
        print(f"ERROR: No checkpoint found at {checkpoint_path}")
        print("Run training first: python src/train.py")
        sys.exit(1)

    if not os.path.exists(test_csv):
        print(f"ERROR: Test CSV not found at {test_csv}")
        sys.exit(1)

    # ── Load model and tokenizer ───────────────────────────────────────────────
    model = load_model(checkpoint_path, device)

    sp = spm.SentencePieceProcessor()
    sp.load(SPM_MODEL)
    print(f"Tokenizer loaded from {SPM_MODEL}")

    # ── Load test data ─────────────────────────────────────────────────────────
    df = pd.read_csv(test_csv)
    df = df.dropna(subset=["amharic", "english"])
    df["amharic"] = df["amharic"].astype(str).str.strip()
    df["english"] = df["english"].astype(str).str.strip()
    df = df[(df["amharic"] != "") & (df["english"] != "")].reset_index(drop=True)

    print(f"Test pairs     : {len(df):,}")
    print(f"Beam width     : {beam_width}")
    print()
    print("Translating test set (this may take several minutes)...")
    print("-" * 60)

    # ── Run inference on entire test set ──────────────────────────────────────
    hypotheses = []   # model's translations
    references  = []  # ground truth English sentences

    start_time = time.time()

    for idx, row in df.iterrows():
        hypothesis = translate(
            row["amharic"],
            model,
            sp,
            device,
            method="beam",
            beam_width=beam_width,
            max_len=max_len
        )
        hypotheses.append(hypothesis)
        references.append(row["english"])

        # Progress print every 500 sentences
        if (idx + 1) % 500 == 0:
            elapsed = time.time() - start_time
            rate    = (idx + 1) / elapsed
            remaining = (len(df) - idx - 1) / rate
            print(f"  [{idx+1:>5}/{len(df)}]  "
                  f"Elapsed: {elapsed/60:.1f}m  "
                  f"ETA: {remaining/60:.1f}m")

    total_time = time.time() - start_time
    print(f"\nTranslation complete — {total_time/60:.1f} minutes total")
    print(f"Average speed   : {len(df)/total_time:.1f} sentences/sec")

    # ── Compute BLEU ──────────────────────────────────────────────────────────
    # SacreBLEU expects references as a list of lists (supports multiple refs)
    bleu_result = sacrebleu.corpus_bleu(hypotheses, [references])

    # ── Compute chrF ──────────────────────────────────────────────────────────
    # chrF: character n-gram F-score, better for morphologically rich languages
    chrf_result = sacrebleu.corpus_chrf(hypotheses, [references])

    # ── Compute chrF++ ────────────────────────────────────────────────────────
    # chrF++ adds word unigrams and bigrams on top of character n-grams
    chrfpp_result = sacrebleu.corpus_chrf(
        hypotheses, [references], word_order=2
    )

    # ── Print results ─────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  BLEU       : {bleu_result.score:.2f}")
    print(f"  chrF       : {chrf_result.score:.2f}")
    print(f"  chrF++     : {chrfpp_result.score:.2f}")
    print()
    print(f"  BLEU details : {bleu_result}")
    print("=" * 60)

    # ── Interpretation guide ──────────────────────────────────────────────────
    print()
    print("Score Interpretation (from-scratch, no pretrained weights):")
    bleu = bleu_result.score
    if bleu < 5:
        verdict = "Very low — model may not have converged. Check training loss."
    elif bleu < 12:
        verdict = "Low — model is learning but needs more data or epochs."
    elif bleu < 20:
        verdict = "Acceptable — within expected range for from-scratch Amharic NMT."
    elif bleu < 30:
        verdict = "Good — strong result for a from-scratch system on this dataset."
    else:
        verdict = "Excellent — outstanding result."
    print(f"  BLEU {bleu:.2f} → {verdict}")
    print()
    print("  Note: Use chrF as the primary metric for Amharic — character-level")
    print("  scoring is more reliable than word-level BLEU for Ge'ez script.")

    # ── Sample translations ───────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"SAMPLE TRANSLATIONS (first {sample_count})")
    print("=" * 60)
    for i in range(min(sample_count, len(df))):
        print(f"\n[{i+1}]")
        print(f"  Source (AM) : {df.iloc[i]['amharic']}")
        print(f"  Reference   : {references[i]}")
        print(f"  Hypothesis  : {hypotheses[i]}")

    # ── Save results to file ──────────────────────────────────────────────────
    results_path = os.path.join(BASE_DIR, "evaluation_results.txt")
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("AMHARIC-ENGLISH NMT EVALUATION RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Checkpoint : {checkpoint_path}\n")
        f.write(f"Test set   : {test_csv}\n")
        f.write(f"Pairs      : {len(df)}\n")
        f.write(f"Beam width : {beam_width}\n\n")
        f.write(f"BLEU   : {bleu_result.score:.4f}\n")
        f.write(f"chrF   : {chrf_result.score:.4f}\n")
        f.write(f"chrF++ : {chrfpp_result.score:.4f}\n\n")
        f.write("SAMPLE TRANSLATIONS\n")
        f.write("-" * 60 + "\n")
        for i in range(len(df)):
            f.write(f"\n[{i+1}]\n")
            f.write(f"  AM  : {df.iloc[i]['amharic']}\n")
            f.write(f"  REF : {references[i]}\n")
            f.write(f"  HYP : {hypotheses[i]}\n")

    print(f"\nFull results saved to: {results_path}")

    return {
        "bleu"  : bleu_result.score,
        "chrf"  : chrf_result.score,
        "chrfpp": chrfpp_result.score
    }


if __name__ == "__main__":
    evaluate()
