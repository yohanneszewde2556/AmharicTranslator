# Data Improvement Execution Plan

**Goal:** Merge the unused v2 dataset into the corpus, clean each source
correctly, retrain the tokenizer, and rebuild the splits — giving the model
better domain coverage and more training signal before the next training run.

**Current state:** 229,472 pairs in `data/raw/final_dataset.csv`.  
**Expected state after:** ~235,000–237,000 pairs with improved domain diversity.

---

## What We Are Working With

| Source | Pairs available | Status | Priority |
|--------|----------------|--------|----------|
| TED2020 | 991 | Accepted, not merged | ⭐ High |
| OpenSubtitles | 2,326 | Accepted, not merged | ⭐ High |
| GlobalVoices | 380 | Accepted, not merged | Medium |
| GNOME UI strings | 3,903 | Accepted, not merged | Low |
| Rush Hour subtitles | 1,009 | Cleaned, not merged | ⭐ High |

**Why these priorities?**

- **TED2020** (mean 9.9 AM / 14.2 EN words): General knowledge talks, natural
  spoken language, well-aligned. Directly addresses the formal-language bias.
- **OpenSubtitles** (mean 5.1 / 7.3 words): Short conversational sentences from
  film subtitles. Same style as Rush Hour — fills the dialogue gap.
- **Rush Hour** (1,009 pairs): Already cleaned and sitting as a loose CSV in the
  project root. Was never added to `merge_datasets.py`. Easy win.
- **GlobalVoices** (mean 15.4 / 21.1 words): News/blog text. Good domain diversity,
  reasonable quality (21% acceptance rate from original crawl).
- **GNOME** (mean 2.97 / 3.29 words): UI strings like "theme file name", "new game".
  Extremely short. Stylistically unlike all other data. Will be included but
  needs a stricter minimum-length filter (≥5 words) to avoid polluting the corpus
  with single-phrase fragments.

---

## Step-by-Step Execution Plan

---

### Step 1 — Filter and Prepare Each V2 Source

**Script to create:** `prepare_v2.py`

Load `data/v2/verified_additions.csv` and apply source-specific filters before
merging. Each source has different quality characteristics.

**TED2020 filter:**
- Minimum 5 words in both columns (the data is generally good; this just removes
  any edge cases that slipped through)
- No additional filters needed — 95.3% acceptance rate means this data is clean

**OpenSubtitles filter:**
- Minimum 4 words in both columns (subtitle data is naturally short)
- Remove rows where Amharic column is more than 30% Latin characters
  (subtitle OCR noise can produce garbled Amharic)
- Remove rows containing only proper nouns / names (single-word subtitle turns)

**GlobalVoices filter:**
- Minimum 6 words in both columns (news text should be sentence-level)
- Already good quality — light touch needed

**GNOME filter:**
- **Minimum 5 words in both columns** — this is the critical filter for GNOME.
  The raw accepted pairs include 2-word menu labels like "theme file name" /
  "new game". These are legitimate translations but they are so short and
  stylistically unusual (no verbs, no context) that they hurt more than they help.
  The ≥5 word filter will reduce GNOME's contribution from 3,903 to roughly
  500–800 sentence-length pairs, which is the right amount.
- Remove rows containing underscore patterns (GNOME keys like `file_manager`)

**Rush Hour:**
- Already cleaned by `clean_rushhour.py` and saved to `rush_hour_clean.csv`
- Load directly, no additional filtering needed

**Output of Step 1:** Five separate DataFrames, each with only `amharic` and
`english` columns, ready for merge.

---

### Step 2 — Merge Everything Into a New Raw Corpus

**Script to create:** `merge_v2.py`

This replaces (or extends) the existing `merge_datasets.py`. It loads the
original corpus and appends all v2 sources.

```
Load data/raw/final_dataset.csv          ← 229,472 pairs (base)
  + ted2020 (filtered)                   ← ~991 pairs
  + opensubtitles (filtered)             ← ~2,000–2,300 pairs
  + globalvoices (filtered)              ← ~370–380 pairs
  + gnome (filtered, ≥5 words)           ← ~500–800 pairs
  + rush_hour_clean.csv                  ← ~1,009 pairs
                                         ─────────────────
  = ~234,000–237,000 pairs (before global dedup)
```

**Global deduplication after concat:**
1. Drop exact Amharic duplicates
2. Drop exact English duplicates
3. Drop rows where either column is empty after strip

**Save to:** `data/raw/final_dataset_v2.csv`  
(Keep the original `final_dataset.csv` untouched as a backup)

---

### Step 3 — Re-run Normalization and Stratified Split

**Script:** `src/preprocess.py` (existing, no changes needed)

Run it pointing at the new corpus:

```
input:  data/raw/final_dataset_v2.csv
output: data/processed/train_v2.csv
        data/processed/val_v2.csv
        data/processed/test_v2.csv
```

The existing preprocessing logic handles everything correctly:
- Amharic colon normalization (`::` → `።`, `:` → `፡`)
- English lowercasing and non-ASCII stripping
- 100-word length filter
- Stratified 95 / 2.5 / 2.5 split with 5 quantile length buckets

No changes to `src/preprocess.py` are needed.

---

### Step 4 — Retrain the BPE Tokenizer

**Script:** `src/tokenizer.py` (existing, no changes needed)

The tokenizer must be retrained on the new training split. Running the old
tokenizer on new data is technically valid but suboptimal — the new sources
(especially TED2020 and OpenSubtitles) introduce new vocabulary patterns.
Retraining ensures the BPE merges are optimized for the full new corpus.

```
input:  data/processed/train_v2.csv
output: data/processed/am_en_bpe_v2.model
        data/processed/am_en_bpe_v2.vocab
```

Keep the old tokenizer files. Do not overwrite them until the new model is
confirmed to work correctly.

---

### Step 5 — Verify the New Corpus With EDA

**Script:** `eda.py` (existing, point at new files)

Before training, run a quick verification pass to confirm:

1. Total pair count is in the expected range (~234K–237K)
2. No missing values in either column
3. Domain distribution has improved — conversational share should rise from
   8% to roughly 10–12% due to TED2020 + OpenSubtitles + Rush Hour additions
4. Ge'ez coverage is still ≥99.9%
5. Length distribution is similar to the original (no outlier inflation)

This is a sanity check, not a full EDA. It should take less than 2 minutes to run.

---

### Step 6 — Update the Processed Data Pointers

Update `src/train.py` and `src/evaluate.py` to point at the new split files:

```
data/processed/train_v2.csv   (was train.csv)
data/processed/val_v2.csv     (was val.csv)
data/processed/test_v2.csv    (was test.csv)
data/processed/am_en_bpe_v2.model  (was am_en_bpe.model)
```

This is a small config change — either update the hardcoded paths or add a
config variable `DATA_VERSION = "v2"` to `src/config.py` and reference it
everywhere. The config variable approach is cleaner.

---

### Step 7 — Retrain the Model

Once Steps 1–6 are complete and verified, the model can be retrained from scratch
on the new corpus. No architecture changes are needed. The same hyperparameters
in `src/config.py` apply.

The expected improvement is in **chrF and BLEU on conversational and general-domain
test sentences**, not on Bible/religious text where the model is already well-covered.

---

## Summary Table

| Step | Script | Input | Output | Effort |
|------|--------|-------|--------|--------|
| 1 | `prepare_v2.py` (new) | `data/v2/verified_additions.csv` + `rush_hour_clean.csv` | 5 filtered DataFrames | Medium |
| 2 | `merge_v2.py` (new) | Above + `data/raw/final_dataset.csv` | `data/raw/final_dataset_v2.csv` | Small |
| 3 | `src/preprocess.py` (existing) | `final_dataset_v2.csv` | `train_v2.csv`, `val_v2.csv`, `test_v2.csv` | None (just run it) |
| 4 | `src/tokenizer.py` (existing) | `train_v2.csv` | `am_en_bpe_v2.model` | None (just run it) |
| 5 | `eda.py` (existing) | New processed files | Verification printout | None (just run it) |
| 6 | `src/config.py` + train/eval scripts | — | Updated path pointers | Small |
| 7 | `src/train.py` (existing) | New splits + tokenizer | New `best_model.pt` | Run on workstation |

Steps 1 and 2 are the only new code that needs to be written.
Steps 3–7 are existing scripts pointed at new files.

---

## What to Expect After Retraining

| Metric | Before (estimate) | After (estimate) |
|--------|------------------|-----------------|
| Corpus size | 229,472 pairs | ~235,000–237,000 pairs |
| Conversational domain share | ~8% | ~10–12% |
| Religious domain share | ~30% | ~28% (diluted, not removed) |
| BLEU on formal/religious text | baseline | similar (no regression) |
| BLEU on conversational text | weak | meaningfully improved |
| chrF overall | baseline | +1–3 points expected |

The gain will be most visible on short conversational sentences. Religious and
formal translation quality will stay the same or improve slightly due to the
larger training set.

---

## Files That Will Be Created

```
prepare_v2.py                    ← new: per-source filtering
merge_v2.py                      ← new: corpus assembly
data/raw/final_dataset_v2.csv    ← new: merged v2 corpus
data/processed/train_v2.csv      ← new: training split
data/processed/val_v2.csv        ← new: validation split
data/processed/test_v2.csv       ← new: test split
data/processed/am_en_bpe_v2.model ← new: retrained tokenizer
data/processed/am_en_bpe_v2.vocab ← new: retrained vocab
```

All existing files are kept as-is. Nothing is deleted or overwritten.

---

*Ready to start. Confirm and we will begin with Step 1.*
