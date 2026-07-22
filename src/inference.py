"""
Phase 4 — Inference
Provides two decoding strategies:
  1. Greedy decoding  — fast, picks the single highest probability token at each step
  2. Beam search      — production standard, tracks top-k sequences by cumulative log prob
"""
import torch
import sentencepiece as spm
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    VOCAB_SIZE, PAD_IDX, BOS_IDX, EOS_IDX,
    D_MODEL, N_HEADS, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS,
    DIM_FEEDFORWARD, DROPOUT, MAX_LENGTH, DATA_VERSION
)
from src.model import Seq2SeqTransformer, generate_square_subsequent_mask


# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROC_DIR      = os.path.join(BASE_DIR, "data", "processed", DATA_VERSION) \
                 if DATA_VERSION else os.path.join(BASE_DIR, "data", "processed")
SPM_MODEL      = os.path.join(_PROC_DIR, "am_en_bpe.model")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")


# ── Model loader ──────────────────────────────────────────────────────────────
def load_model(checkpoint_path: str, device: torch.device) -> Seq2SeqTransformer:
    """
    Instantiates the Transformer and loads weights from a checkpoint file.
    """
    model = Seq2SeqTransformer(
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        d_model=D_MODEL,
        nhead=N_HEADS,
        src_vocab_size=VOCAB_SIZE,
        tgt_vocab_size=VOCAB_SIZE,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Model loaded from {checkpoint_path}  (epoch {checkpoint.get('epoch', '?')})")
    return model


# ── Greedy decoding ───────────────────────────────────────────────────────────
def greedy_decode(model: Seq2SeqTransformer,
                  src: torch.Tensor,
                  max_len: int,
                  device: torch.device) -> torch.Tensor:
    """
    Greedy decoding — at each step picks the single highest probability token.
    Fast but lower quality than beam search. Good for early testing.

    Args:
        model   : trained Seq2SeqTransformer
        src     : (1, src_len) token ID tensor — single source sentence
        max_len : maximum number of tokens to generate
        device  : cpu or cuda

    Returns:
        Tensor of generated token IDs (without BOS)
    """
    src = src.to(device)

    # Encode the source sentence once
    src_mask = torch.zeros((src.shape[1], src.shape[1]), device=device).bool()
    memory   = model.encode(src, src_mask)  # (1, src_len, d_model)

    # Start the decoder with BOS token
    ys = torch.tensor([[BOS_IDX]], dtype=torch.long, device=device)

    for _ in range(max_len):
        tgt_mask = generate_square_subsequent_mask(ys.shape[1]).to(device)
        out      = model.decode(ys, memory, tgt_mask)  # (1, tgt_len, d_model)

        # Project to vocabulary and pick the highest probability token
        prob     = model.generator(out[:, -1, :])      # (1, vocab_size)
        next_tok = prob.argmax(dim=-1).item()

        ys = torch.cat([ys, torch.tensor([[next_tok]], device=device)], dim=1)

        # Stop when EOS is generated
        if next_tok == EOS_IDX:
            break

    # Return generated tokens, excluding the leading BOS
    return ys[0, 1:]


# ── Beam search decoding ──────────────────────────────────────────────────────
def beam_search_decode(model: Seq2SeqTransformer,
                       src: torch.Tensor,
                       max_len: int,
                       device: torch.device,
                       beam_width: int = 5) -> torch.Tensor:
    """
    Beam search decoding — maintains the top `beam_width` candidate sequences
    at each step, selected by cumulative log probability sum.
    Production standard decoding strategy.

    Args:
        model      : trained Seq2SeqTransformer
        src        : (1, src_len) token ID tensor — single source sentence
        max_len    : maximum tokens to generate
        device     : cpu or cuda
        beam_width : number of beams to track (5 is standard)

    Returns:
        Tensor of the best generated token IDs (without BOS)
    """
    src = src.to(device)

    # Encode source once — shared across all beams
    src_mask = torch.zeros((src.shape[1], src.shape[1]), device=device).bool()
    memory   = model.encode(src, src_mask)  # (1, src_len, d_model)

    # Each beam is: (cumulative_log_prob, [token_ids], is_finished)
    # Start with a single beam containing just BOS
    beams = [(0.0, [BOS_IDX], False)]

    completed = []

    for step in range(max_len):
        if not beams:
            break

        all_candidates = []

        for score, tokens, finished in beams:
            # Don't expand finished beams
            if finished:
                completed.append((score, tokens))
                continue

            # Build current decoder input from this beam's tokens
            ys       = torch.tensor([tokens], dtype=torch.long, device=device)
            tgt_mask = generate_square_subsequent_mask(ys.shape[1]).to(device)

            # Expand memory to match batch dimension if needed
            mem = memory.expand(1, -1, -1)

            out      = model.decode(ys, mem, tgt_mask)       # (1, len, d_model)
            logits   = model.generator(out[:, -1, :])        # (1, vocab_size)
            log_prob = torch.log_softmax(logits, dim=-1)     # (1, vocab_size)

            # Get top beam_width candidate next tokens
            topk_log_probs, topk_ids = log_prob[0].topk(beam_width)

            for log_p, tok_id in zip(topk_log_probs.tolist(), topk_ids.tolist()):
                new_score  = score + log_p
                new_tokens = tokens + [tok_id]
                new_finished = (tok_id == EOS_IDX)

                if new_finished:
                    # Length normalisation: divide by sequence length applying penalty.
                    # alpha=0.65 penalizes runaway hallucinated sequences in beam search!
                    alpha = 0.65
                    normalised_score = new_score / (len(new_tokens) ** alpha)
                    completed.append((normalised_score, new_tokens))
                else:
                    all_candidates.append((new_score, new_tokens, False))

        # Keep only the top beam_width candidates for next step
        all_candidates.sort(key=lambda x: x[0], reverse=True)
        beams = all_candidates[:beam_width]

    # If nothing completed (hit max_len), force-complete the best active beam
    if not completed and beams:
        score, tokens, _ = beams[0]
        alpha = 0.65
        normalised_score = score / (max(len(tokens), 1) ** alpha)
        completed.append((normalised_score, tokens))

    # Pick the completed sequence with the highest normalised score
    completed.sort(key=lambda x: x[0], reverse=True)
    best_tokens = completed[0][1]

    # Remove BOS and EOS from output
    result = [t for t in best_tokens if t not in (BOS_IDX, EOS_IDX)]
    return torch.tensor(result, dtype=torch.long)


# ── High-level translate function ─────────────────────────────────────────────
def translate(text: str,
              model: Seq2SeqTransformer,
              sp: spm.SentencePieceProcessor,
              device: torch.device,
              method: str = "beam",
              beam_width: int = 5,
              max_len: int = MAX_LENGTH) -> str:
    """
    End-to-end translation: Amharic string → English string.

    Args:
        text       : Amharic input sentence
        model      : loaded Seq2SeqTransformer
        sp         : loaded SentencePieceProcessor
        device     : cpu or cuda
        method     : "beam" (default) or "greedy"
        beam_width : beam width when method="beam"
        max_len    : maximum output tokens

    Returns:
        Translated English string
    """
    model.eval()
    with torch.no_grad():
        # Tokenize: add BOS and EOS, move directly to device
        src_ids = [BOS_IDX] + sp.encode(text, out_type=int) + [EOS_IDX]
        src     = torch.tensor([src_ids], dtype=torch.long).to(device)  # (1, src_len)

        if method == "beam":
            out_ids = beam_search_decode(model, src, max_len, device, beam_width)
        else:
            out_ids = greedy_decode(model, src, max_len, device)

        # Decode token IDs back to string
        translation = sp.decode(out_ids.tolist())

    return translation


# ── Batched greedy decode for fast evaluation ─────────────────────────────────
def translate_batch(texts: list,
                    model: Seq2SeqTransformer,
                    sp: spm.SentencePieceProcessor,
                    device: torch.device,
                    max_len: int = MAX_LENGTH) -> list:
    """
    Translates a list of Amharic sentences using batched greedy decoding.
    Much faster than one-by-one beam search for bulk evaluation.

    Args:
        texts   : list of Amharic strings
        model   : loaded Seq2SeqTransformer
        sp      : loaded SentencePieceProcessor
        device  : cuda or cpu
        max_len : maximum output tokens

    Returns:
        List of translated English strings
    """
    from torch.nn.utils.rnn import pad_sequence

    model.eval()
    with torch.no_grad():
        # Tokenize all sentences
        encoded = [
            torch.tensor([BOS_IDX] + sp.encode(t, out_type=int) + [EOS_IDX],
                         dtype=torch.long)
            for t in texts
        ]

        # Dynamic pad to longest in this batch
        src = pad_sequence(encoded, batch_first=True,
                           padding_value=PAD_IDX).to(device)  # (B, src_len)

        batch_size = src.shape[0]

        # Build source padding mask
        src_key_padding_mask = (src == PAD_IDX)                    # (B, src_len)
        src_mask = torch.zeros(
            (src.shape[1], src.shape[1]), device=device
        ).bool()

        # Encode entire batch at once
        memory = model.encode(src, src_mask)                       # (B, src_len, d_model)

        # Decoder starts with BOS for every sentence in batch
        ys = torch.full((batch_size, 1), BOS_IDX,
                        dtype=torch.long, device=device)           # (B, 1)

        # Track which sentences have hit EOS
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Identify common punctuation tokens to force-stop hallucinations
        punct_tokens = set()
        for p in [".", "?", "!", "።"]:
            enc = sp.encode(p, out_type=int)
            if enc: punct_tokens.add(enc[-1]) # use last token if it encodes to multiple

        for _ in range(max_len):
            tgt_mask = generate_square_subsequent_mask(
                ys.shape[1]
            ).to(device)

            # Pass the src_key_padding_mask so decoder cross-attention ignores source padding
            out    = model.decode(ys, memory, tgt_mask, memory_key_padding_mask=src_key_padding_mask)
            logits = model.generator(out[:, -1, :])                # (B, vocab_size)

            next_tokens = logits.argmax(dim=-1, keepdim=True)      # (B, 1)

            # Detect which sentences just hit EOS BEFORE overwriting
            next_t = next_tokens.squeeze(1)
            
            # Force-stop hack to prevent runaway hallucinations after a period
            if ys.shape[1] > 5:
                for b_idx in range(batch_size):
                    if not finished[b_idx] and next_t[b_idx].item() in punct_tokens:
                        next_t[b_idx] = EOS_IDX
                        next_tokens[b_idx, 0] = EOS_IDX

            just_finished = (next_t == EOS_IDX)

            # For already-finished sentences, append PAD instead
            next_tokens[finished] = PAD_IDX

            ys = torch.cat([ys, next_tokens], dim=1)               # (B, tgt_len+1)

            # Update finished mask
            finished |= just_finished

            if finished.all():
                break

        # Decode each row back to string
        results = []
        for i in range(batch_size):
            token_ids = ys[i, 1:].tolist()   # skip BOS
            # Cut at EOS if present
            if EOS_IDX in token_ids:
                token_ids = token_ids[:token_ids.index(EOS_IDX)]
            # Remove PAD tokens
            token_ids = [t for t in token_ids if t != PAD_IDX]
            results.append(sp.decode(token_ids))

        return results


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        print("Train the model first by running: python src/train.py")
        sys.exit(1)

    # Load model and tokenizer
    model = load_model(checkpoint_path, device)
    sp    = spm.SentencePieceProcessor()
    sp.load(SPM_MODEL)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true", help="Run in interactive terminal mode")
    args, _ = parser.parse_known_args()

    if args.interactive:
        print("\n" + "=" * 55)
        print("Interactive Translation Mode (Beam Search)")
        print("Type 'exit' or 'quit' to stop.")
        print("=" * 55)
        while True:
            try:
                user_input = input("\nEnter Amharic: ")
                if user_input.strip().lower() in ['exit', 'quit']:
                    break
                if not user_input.strip():
                    continue
                
                translation = translate(user_input, model, sp, device, method="beam")
                print(f"English    : {translation}")
            except KeyboardInterrupt:
                break
    else:
        # Test sentences
        test_sentences = [
            "ውሻው ሰውን ነከሰ።",
            "እናቴ ቡና ትጠጣለች።",
            "አዲስ አበባ የኢትዮጵያ ዋና ከተማ ናት።",
        ]

        print("\n" + "=" * 55)
        print("Inference Test — Beam Search (width=5)")
        print("=" * 55)
        for sent in test_sentences:
            translation = translate(sent, model, sp, device, method="beam")
            print(f"\n  AM : {sent}")
            print(f"  EN : {translation}")

        print("\nRun with --interactive to type your own sentences!")
