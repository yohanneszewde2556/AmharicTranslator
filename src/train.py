"""
Phase 4 — Training Pipeline
Full training loop with:
  - FP16 Automatic Mixed Precision (AMP)
  - AdamW optimizer
  - Linear warmup + cosine decay LR scheduler
  - Gradient clipping
  - Per-epoch validation
  - Checkpoint saving on best validation loss
  - Early stopping after 5 epochs of no improvement
"""
import torch
import torch.nn as nn
import math
import time
import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    VOCAB_SIZE, PAD_IDX, BOS_IDX, EOS_IDX,
    D_MODEL, N_HEADS, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS,
    DIM_FEEDFORWARD, DROPOUT,
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    WARMUP_STEPS, NUM_EPOCHS, MAX_LENGTH,
    DATA_VERSION
)
from src.model import Seq2SeqTransformer, create_mask
from src.dataset import get_dataloader


# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# DATA_VERSION controls which processed split to use:
#   "v2" → data/processed/v2/  (expanded corpus, 232,914 pairs)
#   ""   → data/processed/     (original corpus,  229,472 pairs)
_PROC_DIR      = os.path.join(BASE_DIR, "data", "processed", DATA_VERSION) \
                 if DATA_VERSION else os.path.join(BASE_DIR, "data", "processed")
TRAIN_CSV      = os.path.join(_PROC_DIR, "train.csv")
VAL_CSV        = os.path.join(_PROC_DIR, "val.csv")
SPM_MODEL      = os.path.join(_PROC_DIR, "am_en_bpe.model")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
LOG_FILE       = os.path.join(BASE_DIR, "training_log.txt")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ── Logger — writes to stdout AND training_log.txt simultaneously ─────────────
class Logger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log      = open(filepath, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# ── LR Scheduler ──────────────────────────────────────────────────────────────
def get_lr(step: int, d_model: int = D_MODEL, warmup_steps: int = WARMUP_STEPS) -> float:
    """
    Transformer learning rate schedule from 'Attention Is All You Need'.
    Linear warmup for warmup_steps, then inverse square root decay.
    """
    if step == 0:
        step = 1
    return (d_model ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)


# ── Training for one epoch ─────────────────────────────────────────────────────
def train_epoch(model, dataloader, optimizer, criterion, scaler, scheduler, device):
    """
    Runs one full pass over the training set.
    Returns average loss over all batches.
    """
    model.train()
    total_loss = 0.0
    total_tokens = 0
    start_time = time.time()

    for batch_idx, (src, tgt) in enumerate(dataloader):
        src = src.to(device)
        tgt = tgt.to(device)

        # Teacher forcing: decoder input is tgt without the last token
        # Target labels are tgt without the first token (BOS)
        tgt_input  = tgt[:, :-1]   # (batch, tgt_len - 1)
        tgt_labels = tgt[:, 1:]    # (batch, tgt_len - 1)

        # Build all attention masks
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, tgt_input, PAD_IDX
        )

        optimizer.zero_grad()

        # FP16 AMP forward pass — doubles throughput on RTX 4000 Ada
        with torch.amp.autocast('cuda'):
            logits = model(
                src, tgt_input,
                src_mask, tgt_mask,
                src_padding_mask, tgt_padding_mask,
                src_padding_mask   # memory_key_padding_mask = src_padding_mask
            )
            # logits shape: (batch, tgt_len-1, vocab_size)
            # Reshape for CrossEntropyLoss: (batch * tgt_len-1, vocab_size)
            loss = criterion(
                logits.reshape(-1, VOCAB_SIZE),
                tgt_labels.reshape(-1)
            )

        # AMP backward pass
        scaler.scale(loss).backward()

        # Gradient clipping — prevents exploding gradients
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        # Step the LR scheduler every batch
        scheduler.step()

        # Count non-padding tokens for accurate loss averaging
        num_tokens = (tgt_labels != PAD_IDX).sum().item()
        total_loss   += loss.item() * num_tokens
        total_tokens += num_tokens

        # Print progress every 200 batches
        if (batch_idx + 1) % 200 == 0:
            elapsed = time.time() - start_time
            print(f"  Batch {batch_idx+1:>5}/{len(dataloader)} | "
                  f"Loss: {loss.item():.4f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                  f"Elapsed: {elapsed:.1f}s")

    return total_loss / total_tokens


# ── Validation for one epoch ───────────────────────────────────────────────────
def evaluate_epoch(model, dataloader, criterion, device):
    """
    Runs one full pass over the validation set.
    No gradients — purely for loss measurement.
    Returns average loss.
    """
    model.eval()
    total_loss   = 0.0
    total_tokens = 0

    with torch.no_grad():
        for src, tgt in dataloader:
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_input  = tgt[:, :-1]
            tgt_labels = tgt[:, 1:]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
                src, tgt_input, PAD_IDX
            )

            with torch.amp.autocast('cuda'):
                logits = model(
                    src, tgt_input,
                    src_mask, tgt_mask,
                    src_padding_mask, tgt_padding_mask,
                    src_padding_mask
                )
                loss = criterion(
                    logits.reshape(-1, VOCAB_SIZE),
                    tgt_labels.reshape(-1)
                )

            num_tokens    = (tgt_labels != PAD_IDX).sum().item()
            total_loss   += loss.item() * num_tokens
            total_tokens += num_tokens

    return total_loss / total_tokens


# ── Main training entry point ─────────────────────────────────────────────────
def train():
    # ── Argument parsing ──────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="Amharic-English NMT Training")
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from checkpoints/best_model.pt"
    )
    parser.add_argument(
        "--epochs", type=int, default=NUM_EPOCHS,
        help=f"Total epochs to train (default: {NUM_EPOCHS})"
    )
    args = parser.parse_args()

    # ── Tee stdout → training_log.txt ─────────────────────────────────────────
    sys.stdout = Logger(LOG_FILE)

    # ── Device setup ──────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("Amharic-English NMT — Training Pipeline")
    print("=" * 60)
    print(f"Device  : {device}")
    if device.type == "cuda":
        print(f"GPU     : {torch.cuda.get_device_name(0)}")
        print(f"VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()

    # ── DataLoaders ───────────────────────────────────────────────────────────
    print("Loading datasets...")
    # num_workers=4 uses 4 parallel CPU threads to feed the GPU
    # set to 0 if you get multiprocessing errors on Windows
    train_loader = get_dataloader(TRAIN_CSV, SPM_MODEL, BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = get_dataloader(VAL_CSV,   SPM_MODEL, BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Train batches : {len(train_loader):,}")
    print(f"Val batches   : {len(val_loader):,}")
    print()

    # ── Model ─────────────────────────────────────────────────────────────────
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

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters : {total_params:,}")
    print()

    # ── Loss, Optimizer, Scheduler, Scaler ────────────────────────────────────
    # ignore_index=PAD_IDX so padding tokens don't contribute to loss
    # label_smoothing=0.1 prevents overconfidence and improves generalisation
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)

    # lr=1.0 because the Noam schedule lambda already computes the full LR
    # Setting base lr to anything other than 1.0 would double-scale it
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1.0,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=WEIGHT_DECAY
    )

    # Noam schedule: lrate = d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)
    # This is the exact schedule from "Attention Is All You Need"
    # Peak LR ≈ 0.002 at step 4000 with d_model=512
    def noam_lambda(step):
        step = max(step, 1)
        return (D_MODEL ** -0.5) * min(step ** -0.5, step * (WARMUP_STEPS ** -1.5))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=noam_lambda)

    # GradScaler manages loss scaling for FP16 AMP
    scaler = torch.amp.GradScaler('cuda')

    # ── Resume from checkpoint ────────────────────────────────────────────────
    start_epoch       = 1
    best_val_loss     = float("inf")
    epochs_no_improve = 0
    early_stop_patience = 8

    checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")

    if args.resume and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        scheduler.load_state_dict(ck["scheduler_state_dict"])
        best_val_loss = ck["val_loss"]
        start_epoch   = ck["epoch"] + 1
        print(f"  Resumed at epoch {ck['epoch']}  |  best val loss so far: {best_val_loss:.4f}")
        print(f"  Continuing from epoch {start_epoch} → {args.epochs}")
        print()
    elif args.resume:
        print(f"WARNING: --resume passed but no checkpoint found at {checkpoint_path}")
        print("Starting from scratch instead.")
        print()

    # ── Training loop ─────────────────────────────────────────────────────────
    total_epochs = args.epochs
    print(f"Starting training  (epochs {start_epoch} → {total_epochs})")
    print("-" * 60)

    for epoch in range(start_epoch, total_epochs + 1):
        epoch_start = time.time()

        print(f"\nEpoch {epoch}/{total_epochs}")

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, scaler, scheduler, device
        )

        # Validate
        val_loss = evaluate_epoch(model, val_loader, criterion, device)

        epoch_time = time.time() - epoch_start

        # Perplexity = e^loss — a common readable metric for language models
        train_ppl = math.exp(train_loss)
        val_ppl   = math.exp(val_loss)

        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss : {train_loss:.4f}  |  Train PPL : {train_ppl:.2f}")
        print(f"  Val Loss   : {val_loss:.4f}  |  Val PPL   : {val_ppl:.2f}")
        print(f"  Time       : {epoch_time/60:.1f} min")

        # ── Checkpoint: save if validation loss improved ───────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
            torch.save({
                "epoch"           : epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss"        : val_loss,
                "train_loss"      : train_loss,
            }, checkpoint_path)
            print(f"  Checkpoint saved → {checkpoint_path}  (best val loss: {best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  No improvement. Patience: {epochs_no_improve}/{early_stop_patience}")

        # ── Early stopping ────────────────────────────────────────────────
        if epochs_no_improve >= early_stop_patience:
            print(f"\nEarly stopping triggered after {epoch} epochs.")
            print(f"Best validation loss: {best_val_loss:.4f}")
            break

    print("\n" + "=" * 60)
    print("Training complete.")
    print(f"Best model saved at: {os.path.join(CHECKPOINT_DIR, 'best_model.pt')}")
    print("=" * 60)


if __name__ == "__main__":
    train()
