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
from src.inference import load_model, translate, translate_batch


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

    # Batch size for evaluation — larger = faster, uses more VRAM
    EVAL_BATCH_SIZE = 64

    print(f"Test pairs     : {len(df):,}")
    print(f"Eval batch size: {EVAL_BATCH_SIZE} (batched greedy for speed)")
    print(f"Beam width     : {beam_width} (used for sample translations only)")
    print()
    print("Translating test set with batched greedy decoding...")
    print("-" * 60)

    # ── Run batched greedy inference on entire test set ───────────────────────
    hypotheses = []
    references = df["english"].tolist()
    amharic_sentences = df["amharic"].tolist()

    start_time = time.time()
    total = len(amharic_sentences)

    for i in range(0, total, EVAL_BATCH_SIZE):
        batch_texts = amharic_sentences[i: i + EVAL_BATCH_SIZE]
        batch_hyps  = translate_batch(batch_texts, model, sp, device, max_len)
        hypotheses.extend(batch_hyps)

        if (i + EVAL_BATCH_SIZE) % 500 < EVAL_BATCH_SIZE:
            elapsed   = time.time() - start_time
            done      = min(i + EVAL_BATCH_SIZE, total)
            rate      = done / elapsed
            remaining = (total - done) / rate if rate > 0 else 0
            print(f"  [{done:>5}/{total}]  "
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

    # ── Sample translations — use beam search for quality ────────────────────
    print()
    print("=" * 60)
    print(f"SAMPLE TRANSLATIONS (first {sample_count}) — Beam Search")
    print("=" * 60)
    for i in range(min(sample_count, len(df))):
        beam_translation = translate(
            df.iloc[i]["amharic"], model, sp, device,
            method="beam", beam_width=beam_width, max_len=max_len
        )
        print(f"\n[{i+1}]")
        print(f"  Source (AM) : {df.iloc[i]['amharic']}")
        print(f"  Reference   : {references[i]}")
        print(f"  Hypothesis  : {beam_translation}")

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
