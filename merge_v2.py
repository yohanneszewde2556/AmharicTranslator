"""
Step 2 — merge_v2.py
Loads the original corpus and all five filtered v2 sources,
concatenates them, runs a final global deduplication pass,
and saves the result to data/raw/final_dataset_v2.csv.

The original data/raw/final_dataset.csv is never touched.
"""

import pandas as pd
import os

# ── paths ─────────────────────────────────────────────────────────────────────
LEGACY_PATH  = "data/raw/final_dataset.csv"
FILTERED_DIR = "data/v2/filtered"
OUTPUT_PATH  = "data/raw/final_dataset_v2.csv"

SOURCES = {
    "ted2020"       : f"{FILTERED_DIR}/ted2020.csv",
    "opensubtitles" : f"{FILTERED_DIR}/opensubtitles.csv",
    "globalvoices"  : f"{FILTERED_DIR}/globalvoices.csv",
    "gnome"         : f"{FILTERED_DIR}/gnome.csv",
    "rush_hour"     : f"{FILTERED_DIR}/rush_hour.csv",
}

# ── load legacy corpus ────────────────────────────────────────────────────────
print("="*60)
print("Loading legacy corpus...")
legacy = pd.read_csv(LEGACY_PATH)
print(f"  Legacy corpus : {len(legacy):>7,} pairs")

# ── load and tag each new source ──────────────────────────────────────────────
print("\nLoading filtered v2 sources...")
new_frames = []
for name, path in SOURCES.items():
    df = pd.read_csv(path)
    new_frames.append(df)
    print(f"  {name:<16}: {len(df):>5,} pairs")

total_new = sum(len(f) for f in new_frames)
print(f"  {'─'*28}")
print(f"  Total new     : {total_new:>5,} pairs")

# ── concatenate ───────────────────────────────────────────────────────────────
print("\nConcatenating...")
combined = pd.concat([legacy] + new_frames, ignore_index=True)
print(f"  Combined (pre-dedup) : {len(combined):>7,} pairs")

# ── global deduplication ──────────────────────────────────────────────────────
print("\nRunning global deduplication...")

before = len(combined)
combined = combined.dropna(subset=["amharic", "english"])
combined = combined[combined["amharic"].str.strip().str.len() > 0]
combined = combined[combined["english"].str.strip().str.len()  > 0]
after_nulls = len(combined)
print(f"  After null/empty drop  : {after_nulls:>7,}  (removed {before - after_nulls:,})")

before = after_nulls
combined = combined.drop_duplicates(subset=["amharic"])
after_am_dedup = len(combined)
print(f"  After Amharic dedup    : {after_am_dedup:>7,}  (removed {before - after_am_dedup:,})")

before = after_am_dedup
combined = combined.drop_duplicates(subset=["english"])
after_en_dedup = len(combined)
print(f"  After English dedup    : {after_en_dedup:>7,}  (removed {before - after_en_dedup:,})")

combined = combined.reset_index(drop=True)

# ── save ──────────────────────────────────────────────────────────────────────
os.makedirs("data/raw", exist_ok=True)
combined.to_csv(OUTPUT_PATH, index=False)

# ── final summary ─────────────────────────────────────────────────────────────
net_gain = len(combined) - len(legacy)

print("\n" + "="*60)
print("MERGE COMPLETE")
print("="*60)
print(f"  Legacy corpus size : {len(legacy):>7,} pairs")
print(f"  New corpus size    : {len(combined):>7,} pairs")
print(f"  Net gain           : {net_gain:>+7,} pairs")
print(f"  Growth             :  +{net_gain/len(legacy)*100:.2f}%")
print(f"\n  Saved → {OUTPUT_PATH}")
print("\nReady for Step 3 — src/preprocess.py")
