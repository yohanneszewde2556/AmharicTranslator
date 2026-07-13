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
from torch.cuda.amp import GradScaler, autocast
import math
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    VOCAB_SIZE, PAD_IDX, BOS_IDX, EOS_IDX,
    D_MODEL, N_HEADS, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS,
    DIM_FEEDFORWARD, DROPOUT,
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    WARMUP_STEPS, NUM_EPOCHS, MAX_LENGTH
)
from src.model import Seq2SeqTransformer, create_mask
from src.dataset import get_dataloader


# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_CSV      = os.path.join(BASE_DIR, "data", "processed", "train.csv")
VAL_CSV        = os.path.join(BASE_DIR, "data", "processed", "val.csv")
SPM_MODEL      = os.path.join(BASE_DIR, "data", "processed", "am_en_bpe.model")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


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
        with autocast():
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

            with autocast():
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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.98),
        eps=1e-9
    )

    # LambdaLR applies our custom warmup+decay schedule per batch step
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=get_lr
    )

    # GradScaler manages loss scaling for FP16 AMP
    scaler = GradScaler()

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss    = float("inf")
    epochs_no_improve = 0
    early_stop_patience = 5

    print("Starting training...")
    print("-" * 60)

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()

        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")

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
