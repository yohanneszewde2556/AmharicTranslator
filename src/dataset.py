"""
Phase 4 — Dataset & DataLoader
Reads train/val/test CSVs, tokenizes with the trained BPE model,
and produces padded batches for the Transformer training loop.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import sentencepiece as spm
import os
import sys

# Allow imports from parent directory when running standalone
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    PAD_IDX, BOS_IDX, EOS_IDX, MAX_LENGTH, BATCH_SIZE
)


class AmharicEnglishDataset(Dataset):
    """
    Reads a CSV with 'amharic' and 'english' columns.
    Tokenizes both sides with the shared BPE model.
    Returns (src_ids, tgt_ids) as integer tensors.
    """

    def __init__(self, csv_path: str, sp_model_path: str, max_length: int = MAX_LENGTH):
        """
        Args:
            csv_path      : path to train.csv / val.csv / test.csv
            sp_model_path : path to am_en_bpe.model
            max_length    : sequences longer than this are dropped (prevents OOM)
        """
        print(f"Loading dataset from {csv_path} ...")
        df = pd.read_csv(csv_path)

        # Drop rows with missing values
        df = df.dropna(subset=["amharic", "english"])
        df["amharic"] = df["amharic"].astype(str).str.strip()
        df["english"] = df["english"].astype(str).str.strip()
        df = df[(df["amharic"] != "") & (df["english"] != "")].reset_index(drop=True)

        # Load the shared SentencePiece BPE model
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(sp_model_path)

        self.max_length = max_length
        self.pairs = []

        skipped = 0
        for _, row in df.iterrows():
            # Encode source (Amharic) — add BOS and EOS
            src_ids = [BOS_IDX] + self.sp.encode(row["amharic"], out_type=int) + [EOS_IDX]
            # Encode target (English) — add BOS and EOS
            tgt_ids = [BOS_IDX] + self.sp.encode(row["english"], out_type=int) + [EOS_IDX]

            # Skip sequences that exceed max_length to prevent OOM on GPU
            if len(src_ids) > max_length or len(tgt_ids) > max_length:
                skipped += 1
                continue

            self.pairs.append((
                torch.tensor(src_ids, dtype=torch.long),
                torch.tensor(tgt_ids, dtype=torch.long)
            ))

        print(f"  Loaded  : {len(self.pairs):,} pairs")
        if skipped:
            print(f"  Skipped : {skipped:,} pairs exceeding max_length={max_length}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]  # (src_tensor, tgt_tensor)


def collate_fn(batch):
    """
    Dynamic padding collate function.
    Pads each batch only to the longest sequence IN THAT BATCH,
    not to a fixed global length — saves significant VRAM.

    Args:
        batch : list of (src_tensor, tgt_tensor) tuples

    Returns:
        src_batch : (batch_size, max_src_len) padded with PAD_IDX
        tgt_batch : (batch_size, max_tgt_len) padded with PAD_IDX
    """
    src_batch, tgt_batch = zip(*batch)

    # pad_sequence pads to the longest tensor in the list
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)

    return src_batch, tgt_batch


def get_dataloader(csv_path: str, sp_model_path: str, batch_size: int = BATCH_SIZE,
                   shuffle: bool = True, num_workers: int = 0) -> DataLoader:
    """
    Convenience function that returns a ready-to-use DataLoader.

    Args:
        csv_path      : path to the CSV split file
        sp_model_path : path to am_en_bpe.model
        batch_size    : number of sentence pairs per batch
        shuffle       : True for training, False for val/test
        num_workers   : parallel data loading workers (0 = main process only)

    Returns:
        DataLoader with dynamic padding collate_fn applied
    """
    dataset = AmharicEnglishDataset(csv_path, sp_model_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True   # speeds up CPU→GPU transfer
    )


# ── Standalone plumbing test ──────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    from src.config import DATA_VERSION
    _PROC_DIR = os.path.join(BASE_DIR, "data", "processed", DATA_VERSION) \
                if DATA_VERSION else os.path.join(BASE_DIR, "data", "processed")
    TRAIN_CSV = os.path.join(_PROC_DIR, "train.csv")
    SPM_MODEL = os.path.join(_PROC_DIR, "am_en_bpe.model")

    if not os.path.exists(TRAIN_CSV):
        print(f"ERROR: {TRAIN_CSV} not found.")
        sys.exit(1)
    if not os.path.exists(SPM_MODEL):
        print(f"ERROR: {SPM_MODEL} not found.")
        sys.exit(1)

    print("=" * 55)
    print("Dataset & DataLoader Plumbing Test")
    print("=" * 55)

    loader = get_dataloader(TRAIN_CSV, SPM_MODEL, batch_size=4, shuffle=False)

    # Grab the first batch and inspect it
    src_batch, tgt_batch = next(iter(loader))

    print(f"\nFirst batch shapes:")
    print(f"  src_batch : {src_batch.shape}  (batch_size, src_seq_len)")
    print(f"  tgt_batch : {tgt_batch.shape}  (batch_size, tgt_seq_len)")

    print(f"\nFirst src sequence (token IDs):")
    print(f"  {src_batch[0].tolist()}")
    print(f"First tgt sequence (token IDs):")
    print(f"  {tgt_batch[0].tolist()}")

    # Verify BOS and EOS are in place
    assert src_batch[0][0].item() == BOS_IDX, "BOS token missing from source"
    assert tgt_batch[0][0].item() == BOS_IDX, "BOS token missing from target"

    print(f"\nPAD_IDX={PAD_IDX}  BOS_IDX={BOS_IDX}  EOS_IDX={EOS_IDX}")
    print(f"PAD tokens in src batch: {(src_batch == PAD_IDX).sum().item()}")
    print(f"PAD tokens in tgt batch: {(tgt_batch == PAD_IDX).sum().item()}")

    print("\nDataset plumbing test PASSED.")
