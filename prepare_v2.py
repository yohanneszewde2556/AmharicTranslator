"""
Step 1 — prepare_v2.py
Per-source filtering of data/v2/verified_additions.csv and rush_hour_clean.csv.
Produces five clean DataFrames saved to data/v2/filtered/ ready for merge_v2.py.

Sources and their specific filter logic:
  - ted2020       : min 5 words both sides (data is already very clean)
  - opensubtitles : min 4 words, remove majority-Latin Amharic rows
  - globalvoices  : min 6 words both sides (news text, should be full sentences)
  - gnome         : min 5 words both sides (strips short UI fragments),
                    remove underscore-pattern keys
  - rush_hour     : already cleaned by clean_rushhour.py, load directly
"""

import pandas as pd
import re
import os

# ── output directory ──────────────────────────────────────────────────────────
OUT_DIR = "data/v2/filtered"
os.makedirs(OUT_DIR, exist_ok=True)

# ── load source files ─────────────────────────────────────────────────────────
print("Loading source files...")
v2 = pd.read_csv("data/v2/verified_additions.csv")
rush = pd.read_csv("rush_hour_clean.csv")

print(f"  v2 verified_additions : {len(v2):,} rows")
print(f"  rush_hour_clean       : {len(rush):,} rows")

# ── helper functions ──────────────────────────────────────────────────────────

def word_count(text):
    return len(str(text).split()) if pd.notna(text) else 0

def is_majority_latin(text, threshold=0.30):
    """True if more than threshold fraction of alphabetic chars are Latin."""
    if not isinstance(text, str):
        return True
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return True
    latin = sum(1 for c in alpha if c.isascii())
    return latin / len(alpha) > threshold

def has_geez(text):
    """True if text contains at least one Ge'ez character."""
    if not isinstance(text, str):
        return False
    return bool(re.search(r"[\u1200-\u137F]", text))

def has_underscore_key(text):
    """True if text contains an underscore between word chars (GNOME key noise)."""
    return bool(re.search(r"\w_\w", str(text)))

def filter_source(df, name, min_am, min_en, extra_filters=None):
    """
    Apply standard length + quality filters to a DataFrame.
    extra_filters: list of boolean Series (True = DROP the row)
    Returns filtered DataFrame with only [amharic, english] columns.
    """
    start = len(df)

    # compute word counts
    df = df.copy()
    df["am_wc"] = df["amharic"].apply(word_count)
    df["en_wc"] = df["english"].apply(word_count)

    # length filter
    df = df[(df["am_wc"] >= min_am) & (df["en_wc"] >= min_en)].copy()
    after_len = len(df)

    # drop nulls / empty strings
    df = df.dropna(subset=["amharic", "english"])
    df = df[df["amharic"].str.strip().str.len() > 0]
    df = df[df["english"].str.strip().str.len() > 0]

    # extra source-specific filters
    if extra_filters:
        for mask in extra_filters:
            before = len(df)
            df = df[~mask.reindex(df.index, fill_value=False)].copy()

    # dedup within this source
    df = df.drop_duplicates(subset=["amharic"])
    df = df.drop_duplicates(subset=["english"])

    # keep only the two columns we need
    df = df[["amharic", "english"]].reset_index(drop=True)

    print(f"\n  [{name}]")
    print(f"    Input            : {start:,}")
    print(f"    After length     : {after_len:,}  (min_am={min_am}, min_en={min_en})")
    print(f"    After all filters: {len(df):,}")
    return df


# ════════════════════════════════════════════════════════════════════════════
# TED2020
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Filtering TED2020...")
ted_raw = v2[v2["source"] == "ted2020"].copy()
ted_clean = filter_source(ted_raw, "ted2020", min_am=5, min_en=5)
ted_clean.to_csv(f"{OUT_DIR}/ted2020.csv", index=False)
print(f"    Saved → {OUT_DIR}/ted2020.csv")


# ════════════════════════════════════════════════════════════════════════════
# OpenSubtitles
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Filtering OpenSubtitles...")
sub_raw = v2[v2["source"] == "opensubtitles"].copy()

# extra filter: rows where Amharic column is majority Latin (garbled subtitle OCR)
majority_latin_mask = sub_raw["amharic"].apply(is_majority_latin)
print(f"    Majority-Latin rows to drop: {majority_latin_mask.sum()}")

sub_clean = filter_source(
    sub_raw, "opensubtitles",
    min_am=4, min_en=4,
    extra_filters=[majority_latin_mask]
)
sub_clean.to_csv(f"{OUT_DIR}/opensubtitles.csv", index=False)
print(f"    Saved → {OUT_DIR}/opensubtitles.csv")


# ════════════════════════════════════════════════════════════════════════════
# GlobalVoices
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Filtering GlobalVoices...")
gv_raw = v2[v2["source"] == "globalvoices"].copy()

# extra filter: rows where English contains Ge'ez (column swap)
geez_in_en_mask = gv_raw["english"].apply(has_geez)
print(f"    Ge'ez-in-English rows to drop: {geez_in_en_mask.sum()}")

gv_clean = filter_source(
    gv_raw, "globalvoices",
    min_am=6, min_en=6,
    extra_filters=[geez_in_en_mask]
)
gv_clean.to_csv(f"{OUT_DIR}/globalvoices.csv", index=False)
print(f"    Saved → {OUT_DIR}/globalvoices.csv")


# ════════════════════════════════════════════════════════════════════════════
# GNOME
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Filtering GNOME...")
gnome_raw = v2[v2["source"] == "gnome"].copy()

# extra filter 1: underscore keys (GNOME localization key artifacts)
underscore_am_mask  = gnome_raw["amharic"].apply(has_underscore_key)
underscore_en_mask  = gnome_raw["english"].apply(has_underscore_key)
combined_underscore = underscore_am_mask | underscore_en_mask
print(f"    Underscore-key rows to drop: {combined_underscore.sum()}")

# extra filter 2: rows without any Ge'ez in Amharic column
no_geez_mask = ~gnome_raw["amharic"].apply(has_geez)
print(f"    No-Ge'ez rows to drop: {no_geez_mask.sum()}")

gnome_clean = filter_source(
    gnome_raw, "gnome",
    min_am=5, min_en=5,
    extra_filters=[combined_underscore, no_geez_mask]
)
gnome_clean.to_csv(f"{OUT_DIR}/gnome.csv", index=False)
print(f"    Saved → {OUT_DIR}/gnome.csv")


# ════════════════════════════════════════════════════════════════════════════
# Rush Hour (already clean — minimal safety checks only)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Filtering Rush Hour...")
rush_clean = filter_source(
    rush, "rush_hour",
    min_am=3, min_en=3
)
rush_clean.to_csv(f"{OUT_DIR}/rush_hour.csv", index=False)
print(f"    Saved → {OUT_DIR}/rush_hour.csv")


# ════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════════════════
total_new = sum([
    len(ted_clean),
    len(sub_clean),
    len(gv_clean),
    len(gnome_clean),
    len(rush_clean),
])

current_corpus = 229472

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"  TED2020        : {len(ted_clean):>5,} pairs")
print(f"  OpenSubtitles  : {len(sub_clean):>5,} pairs")
print(f"  GlobalVoices   : {len(gv_clean):>5,} pairs")
print(f"  GNOME          : {len(gnome_clean):>5,} pairs")
print(f"  Rush Hour      : {len(rush_clean):>5,} pairs")
print(f"  {'─'*30}")
print(f"  Total new      : {total_new:>5,} pairs")
print(f"  Current corpus : {current_corpus:>5,} pairs")
print(f"  Expected total : ~{current_corpus + total_new:>5,} pairs (before global dedup)")
print(f"\nAll filtered files saved to {OUT_DIR}/")
print("Ready for Step 2 — merge_v2.py")
