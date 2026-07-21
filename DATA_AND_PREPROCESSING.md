# Data & Preprocessing — Detailed Technical Explanation

**Project:** Amharic → English Neural Machine Translation  
**Author:** Yohannes Zewde  
**Scope:** Everything from raw source files to the tokenized training splits ready for the Transformer

---

## Table of Contents

1. [Overview](#1-overview)
2. [Data Sources](#2-data-sources)
3. [Per-Source Cleaning](#3-per-source-cleaning)
4. [Merging Into the Master Corpus](#4-merging-into-the-master-corpus)
5. [Text Normalization](#5-text-normalization)
6. [Stratified Train / Val / Test Split](#6-stratified-train--val--test-split)
7. [SentencePiece BPE Tokenizer](#7-sentencepiece-bpe-tokenizer)
8. [V2 Data Expansion Pipeline](#8-v2-data-expansion-pipeline)
9. [EDA Findings Summary](#9-eda-findings-summary)
10. [Strengths of This Pipeline](#10-strengths-of-this-pipeline)
11. [Known Limitations and Risks](#11-known-limitations-and-risks)
12. [Recommendations](#12-recommendations)

---

## 1. Overview

The entire data pipeline converts five heterogeneous raw sources into a single clean,
normalized, tokenized parallel corpus of **229,472 Amharic-English sentence pairs**.
The pipeline is structured in five conceptual stages:

```
Raw Sources (5 files)
       │
       ▼
Per-Source Cleaning (clean_*.py)
       │
       ▼
Master Corpus Merge (merge_datasets.py)
       │
       ▼
Normalization + Stratified Split (src/preprocess.py)
       │
       ▼
BPE Tokenizer Training (src/tokenizer.py)
       │
       ▼
train.csv / val.csv / test.csv  +  am_en_bpe.model
```

Each stage is described in detail below.

---

## 2. Data Sources

Five sources were assembled, each contributing a different text style and domain:

### 2.1 AmharicDataset.xlsx — General Parallel Corpus (~5,000 pairs)

The original seed dataset. A general-purpose Amharic-English parallel corpus covering
everyday topics — family, society, law, proverbs, and common sentences. Originally a
three-column file (`amharic`, `oromo`, `english`); the Oromo column was dropped since
the model only targets the Amharic → English direction.

**Characteristics:** High quality, human-curated, diverse topics. Short to medium
sentence lengths. No obvious domain skew.

### 2.2 OPUS-100 (opus100_clean.xlsx) — Web-Crawled Parallel Text (~200,000 pairs)

The largest source by far, downloaded from the HuggingFace `Helsinki-NLP/opus-100`
dataset, `am-en` language pair. OPUS-100 is a massively multilingual corpus assembled
from a wide range of sources including news, religious texts (mainly JW300 — Jehovah's
Witnesses publications), and web pages.

**Characteristics:** This is the dominant source — roughly 87% of the final corpus by
volume. Because JW300 is a major OPUS component for Amharic, the corpus is heavily
skewed toward formal religious and instructional language. Requires careful cleaning to
remove noise (non-Amharic characters in the Amharic column, extreme lengths, duplicates).

### 2.3 FLORES Benchmark (flores_am_en.xlsx) — Evaluation-Grade Pairs (~1,000 pairs)

FLORES is a professionally translated benchmark dataset created by Meta AI specifically
to evaluate low-resource NMT systems. Every sentence pair was translated by a human
professional. It covers news and general informational text.

**Characteristics:** Highest quality source in the corpus. Including it in training is
a deliberate choice — it brings in verified, high-quality signal and slightly improves
coverage of formal news-style Amharic. The downside is a small risk of evaluation
contamination if the same FLORES split is used for testing, but since custom splits are
used here that risk is minimal.

### 2.4 Bible Parallel Text (bible_clean.xlsx) — Religious Narrative (~30,000 pairs)

A Bible parallel corpus aligned at the verse level, meaning each Amharic Bible verse is
paired with its English translation. Both Old and New Testament are included.

**Characteristics:** Long-form religious prose, verse-level alignment. Contains
bracket artifacts like `[to be]`, `[from]` inserted by the source to fill grammatical
gaps — these are removed during cleaning. Sentences are relatively long and syntactically
complex. Adds significant religious domain coverage on top of what OPUS already brings.

### 2.5 Rush Hour Subtitles (rush_hour_clean.csv) — Conversational (~1,009 pairs)

The only conversational source in the entire dataset. Extracted from subtitle files
(`.srt`) of the Rush Hour film series, manually aligned into Amharic-English pairs.
This source directly addresses the well-known weakness of NMT systems trained on
religious/formal text: poor performance on everyday spoken language.

**Characteristics:** Short, fragmented, informal sentences. Speaker labels in both
languages (e.g. `CARTER:`, `ሰው:`). Subtitle turn-taking dashes. Numbers and digit
sequences. Stray Latin characters in the Amharic column (noise from subtitles).
Requires the most aggressive multi-pass cleaning of any source.

---

## 3. Per-Source Cleaning

Each source had its own cleaning script because each source has its own specific noise
patterns. A generic cleaner would either miss source-specific problems or over-aggressively
remove clean data.

### 3.1 clean_opus.py — OPUS-100 Cleaning

**Steps applied:**
1. **Drop missing values** — remove any row where either column is empty or NaN
2. **Word count filter** — keep only sentences with 3–50 words in both languages.
   Below 3: too short to carry translatable meaning. Above 50: likely paragraph-level
   concatenations from the web crawl, not sentence pairs
3. **Ge'ez script check** — detect rows where the Amharic column contains zero Ge'ez
   characters (`\u1200–\u137F`). These are rows where the web crawl put Latin text in
   the wrong column, or where the alignment failed entirely
4. **Underscore noise removal** — remove rows with `word_word` patterns (underscores
   mid-word), a common artifact of CamelCase tokenization artifacts from web crawling
5. **Deduplication** — drop exact duplicates on the Amharic column, then on the English
   column separately. This removes cases where the same Amharic sentence was aligned to
   two different English translations (a known OPUS alignment problem)

**Result:** OPUS contributes the most pairs but also had the most noise removed.

### 3.2 clean_bible.py — Bible Cleaning

**Steps applied:**
1. **Word count filter** — 3–100 words in both languages (longer limit than OPUS because
   Bible verses are legitimately longer)
2. **Bracket artifact removal** — Bible translation convention uses `[word]` notation to
   mark words added for English grammatical completeness that have no direct Amharic
   equivalent. These are purely typographic and meaningless to the model:
   `re.sub(r'\[.*?\]', '', text)` removes them
3. **Space repair** — after bracket removal, words that were separated only by the
   bracket become concatenated (`"word1[x]word2"` → `"word1word2"`). A post-processing
   step re-inserts spaces at CamelCase boundaries and after commas
4. **Deduplication** — by Amharic column, then English column

### 3.3 clean_rushhour.py — Subtitle Cleaning (3-Pass Pipeline)

This is the most complex cleaner in the project, justified because subtitle data is
the most structurally noisy source.

**Pass 1 — Structural cleanup:**
- Strip speaker labels using regex. English labels match `ALL-CAPS word(s) + colon`
  (`CARTER:`, `MAN 1:`). Amharic labels match `any-word + optional-number + colon`
  (`ሰው:`, `ካርተር:`)
- Remove leading/trailing ellipsis artifacts (`...`, `…`) from both columns
- Drop rows flagged `CHECK` — rows where the length ratio (Amharic words / English words)
  was below 0.5 or above 2.0, indicating a likely misalignment
- Drop rows with ≤2 words in either language after label stripping
- Drop rows where Amharic is >30% Latin characters — this catches cases where the
  subtitle aligner put English text in the Amharic column

**Pass 2 — Deep text cleanup:**
- Remove all numbers and digit sequences from both columns (subtitle timecodes and
  scene numbers that leaked into the text)
- Strip leading dashes used for turn-taking (`-word` → `word`)
- For English: keep only letters, spaces, apostrophes, and basic punctuation
- For Amharic: keep only Ge'ez Unicode block (`\u1200–\u137F`), Ethiopic supplement
  (`\u1380–\u139F`), Ethiopic punctuation characters, and spaces. Everything else —
  including any remaining Latin characters — is removed

**Pass 3 — Final filter after cleanup:**
- Re-measure word counts after cleanup (some rows that looked valid before cleanup
  become too short after noise is removed)
- Drop rows with ≤2 words in either column
- Final deduplication

**Result:** 1,009 clean conversational pairs — the only dialogue-style data in the
corpus.

### 3.4 clean_csv_source.py — General CSV Source Cleaning

Applied to a secondary CSV source (`new_source2.csv`):
1. Drop nulls, apply word count filter (3–100 words)
2. Remove rows starting with numbers (table headers and list items that leaked in)
3. Remove rows where the English column contains Ge'ez characters — a sign that the
   columns were swapped during alignment
4. Deduplication on both columns
5. Space normalization — fix missing spaces at CamelCase boundaries and after commas
6. Remove date-only rows (rows containing only a date string contribute no translatable
   linguistic content)

---

## 4. Merging Into the Master Corpus

`merge_datasets.py` concatenates all five cleaned sources into a single DataFrame using
`pd.concat`, then applies a final global deduplication pass:

```python
combined = pd.concat([original, opus, flores, bible], ignore_index=True)
combined = combined.dropna()
combined = combined[combined["amharic"].str.strip() != ""]
combined = combined[combined["english"].str.strip() != ""]
combined = combined.drop_duplicates(subset=["amharic"])
combined = combined.drop_duplicates(subset=["english"])
```

The global deduplication is important. Individual sources were deduplicated within
themselves, but a sentence from the general corpus could still appear verbatim in the
OPUS crawl. This final pass catches those cross-source duplicates.

**Output:** `data/raw/final_dataset.csv` — 229,472 pairs, two columns (`amharic`,
`english`), no nulls, no duplicates.

---

## 5. Text Normalization

`src/preprocess.py` applies language-specific normalization before splitting. This is
distinct from cleaning — cleaning removes noise, normalization standardizes valid text.

### 5.1 Amharic Normalization

```python
def normalize_amharic(text):
    text = re.sub(r':{2,}', '።', text)   # '::' → Ethiopic full stop
    text = text.replace(':', '፡')         # ':' → Ethiopic word separator
    text = re.sub(r'\s+', ' ', text).strip()
```

**Why this matters:** Many Amharic digital texts use ASCII colon (`:`) as a substitute
for the Ethiopic word separator (`፡`) and double-colon (`::`) as a substitute for the
Ethiopic full stop (`።`). This is a keyboard-driven convention — most Amharic keyboards
map these characters to `:` for easier typing. Leaving them as colons would cause the
tokenizer to treat them as English punctuation, splitting them into Latin subword units
instead of treating them as native Amharic sentence delimiters. The normalization ensures
the Ge'ez punctuation system is consistently represented.

### 5.2 English Normalization

```python
def normalize_english(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s.,!?\'""-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
```

**Why lowercase:** The shared BPE vocabulary is trained on both Amharic and English text.
Amharic has no case distinction. Keeping English case would force the vocabulary to
allocate tokens for both `The` and `the`, `God` and `god`, etc., effectively halving
the vocabulary budget available for English. Lowercasing collapses all case variants into
single tokens and frees up vocabulary space for more meaningful subword units.

**Why strip non-ASCII:** Keeps the English vocabulary clean and predictable. The model
does not need to learn anything from stray Unicode punctuation, smart quotes, or
typographic em-dashes — these carry no translatable meaning and would create rare tokens
that the model sees too infrequently to learn meaningful representations for.

### 5.3 Length Filter

After normalization, sentences exceeding 100 words are removed:

```python
df = df[df['word_count'] <= 100].copy()
```

100 words corresponds to roughly 120–135 BPE tokens for English and slightly fewer for
Amharic (Amharic is morphologically denser). The model's `MAX_LENGTH` is set to 128
tokens. Keeping the 100-word filter ensures that almost no sentence will be truncated
during training, which would silently corrupt the target sequence and introduce noise
into the loss computation. The EDA confirmed that only 0.01% of sentences in the
processed train split exceed 128 tokens — the filter worked as intended.

### 5.4 Empty-Row Removal

After normalization, any row where either column became an empty string is dropped.
This handles edge cases where a sentence was entirely composed of the characters that
were stripped (e.g., a row containing only colons and spaces).

---

## 6. Stratified Train / Val / Test Split

The split uses a two-stage stratified sampling strategy rather than a simple random
split. This is one of the most important decisions in the preprocessing pipeline.

### Why Stratification Matters

A naive random split on 229K sentences will produce statistically valid splits, but it
risks an uneven distribution of sentence lengths across splits. Short sentences
(2–5 words) are much easier to translate than long sentences (40–100 words). If the
test set happens to contain more short sentences than the training set, evaluation
scores will be inflated. Stratification prevents this.

### Implementation

```python
# Step 1: Create 5 quantile length buckets
df['length_bucket'] = pd.qcut(df['word_count'], q=5, labels=False, duplicates='drop')

# Step 2: 95% train, 5% temp — stratified by bucket
train_df, temp_df = train_test_split(df, test_size=0.05, stratify=df['length_bucket'], random_state=42)

# Step 3: Split temp 50/50 into val and test — stratified by bucket
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['length_bucket'], random_state=42)
```

The five quantile buckets (very short / short / medium / long / very long) are defined
dynamically from the data distribution rather than hardcoded word-count thresholds.
`pd.qcut` ensures each bucket contains an equal number of sentences, so stratification
gives each split a proportional sample from every length range.

`random_state=42` is set in both splits. This is critical for reproducibility — the
exact same split can be recreated on any machine at any time by re-running the script.

### Result

| Split | Pairs | Share | EN Mean Length |
|-------|-------|-------|----------------|
| Train | 217,996 | 95.00% | 19.80 words |
| Val   | 5,737   | 2.50%  | 19.83 words |
| Test  | 5,737   | 2.50%  | 19.72 words |

The near-identical mean lengths across all three splits confirm the stratification
worked correctly. The 95/2.5/2.5 ratio is a deliberate choice for this corpus size —
with 229K pairs, even 2.5% gives 5,737 evaluation examples, which is more than enough
for statistically reliable BLEU and chrF scores. Giving more data to training is
valuable when training from scratch without pretrained weights.

### Note on the Earlier 80/10/10 Script

`02_split_dataset.py` uses a simple 80/10/10 random split without stratification. This
was an early prototype script and was superseded by the stratified approach in
`src/preprocess.py`. The current pipeline uses `src/preprocess.py` exclusively.

---

## 7. SentencePiece BPE Tokenizer

`src/tokenizer.py` trains a shared Byte Pair Encoding (BPE) tokenizer on the training
split using the SentencePiece library.

### Why a Shared Vocabulary

A shared vocabulary means the same 32,000 tokens cover both Amharic and English. The
alternative — two separate vocabularies — doubles the embedding table size and removes
the possibility of the model learning cross-lingual subword alignments (e.g., named
entities like `ኢትዮጵያ` / `ethiopia` sharing representational proximity in embedding
space). Shared vocabularies are standard practice in multilingual NMT.

### Why BPE Over Wordpiece or Unigram

BPE is deterministic and reproducible — given the same training text, it always produces
the same vocabulary. It handles Amharic's agglutinative morphology well because it
naturally discovers common morpheme boundaries through frequency-based merging.
For Ge'ez script specifically, BPE tends to produce sensible subword splits that align
with Amharic morphological boundaries (prefixes, roots, suffixes) without requiring any
language-specific rules.

### Why `character_coverage=0.9995`

SentencePiece's `character_coverage` parameter controls what fraction of unique
characters in the training corpus must be covered by the vocabulary. The default of
0.9995 is intentionally high for this project. Amharic's Ge'ez script has 517 syllabic
characters (fidels) — if coverage is set too low, rare Ge'ez characters will be
mapped to `<unk>`, making it impossible for the model to learn their pronunciation or
semantics. 0.9995 ensures virtually all Ge'ez characters are represented as atomic
tokens before any BPE merges happen.

### Vocabulary Composition

The EDA revealed the following breakdown of the 32,000 token vocabulary:

| Token Type | Count | Share |
|------------|-------|-------|
| Ge'ez script tokens | 20,195 | 63.1% |
| Latin script tokens | 11,395 | 35.6% |
| Digit tokens | 344 | 1.1% |
| Other / special | ~66 | 0.2% |

63% Ge'ez vs 35.6% Latin is reasonable for a corpus where both languages are present.
Amharic's syllabic script naturally generates more unique character combinations than
alphabetic English, so it claims more vocabulary share even though English contributes
roughly the same number of training sentences.

### Special Token Pinning

```python
spm.SentencePieceTrainer.train(
    ...
    pad_id=0, pad_piece="<pad>",
    unk_id=1, unk_piece="<unk>",
    bos_id=2, bos_piece="<s>",
    eos_id=3, eos_piece="</s>",
)
```

The IDs for the four special tokens are hardcoded to 0–3 and must match the constants
in `src/config.py` (`PAD_IDX=0`, `UNK_IDX=1`, `BOS_IDX=2`, `EOS_IDX=3`). These IDs
are referenced directly in the model's loss computation (`ignore_index=0`), in the
inference decoder loop (stopping at EOS ID 3), and in the collate function for batch
padding. Any mismatch between the tokenizer and config would silently corrupt training.

### Token Length Statistics (from EDA)

| Metric | Amharic tokens | English tokens |
|--------|---------------|----------------|
| Mean | 19.75 | 22.34 |
| Median | 17 | 20 |
| Max | 135 | 133 |
| 95th percentile | 42 | 49 |

English tokenizes into more tokens than Amharic on average despite shorter word count.
This is the fertility effect — Amharic words tend to be multi-syllable morphological
units that BPE encodes as 1–2 tokens each, while English words, especially after
lowercasing, also tokenize compactly. The median of 17–20 tokens is well within
`MAX_LENGTH=128`, confirming the length filter and tokenizer are well calibrated.

### Note on the Earlier NLLB Tokenizer Script

`04_05_tokenize.py` used Facebook's NLLB-200 tokenizer from HuggingFace. This was
an experimental early script exploring fine-tuning a pretrained multilingual model.
The project direction changed to training a Transformer from scratch, making a custom
BPE tokenizer the correct choice. `04_05_tokenize.py` is no longer part of the
active pipeline.

---

## 8. V2 Data Expansion Pipeline

A second data collection and filtering pass was conducted after the initial corpus was
built. Four new sources were evaluated: GlobalVoices, GNOME, OpenSubtitles, and TED2020.

### Filtering Criteria

62,941 raw pairs were imported and filtered against these rules:

| Rule | Rejected | Reason |
|------|----------|--------|
| Too short (< min word threshold) | 24,463 | Short strings carry no translation signal |
| Duplicate within new sources | 16,973 | Cross-source deduplication |
| Amharic missing Ge'ez characters | 7,178 | Wrong language or encoding failure |
| Amharic mostly Latin (>threshold%) | 4,840 | Garbled or untranslated Amharic |
| Exact overlap with legacy corpus | 1,472 | Already in `final_dataset.csv` |
| Extreme length ratio | 342 | Likely misalignment |
| Empty text | 70 | Blank rows |
| English contains Ge'ez | 3 | Column swap |

**Overall acceptance rate: 12.07%** (7,600 out of 62,941 pairs accepted).

### Acceptance Rate by Source

| Source | Imported | Accepted | Rate |
|--------|----------|----------|------|
| TED2020 | 1,040 | 991 | **95.3%** |
| OpenSubtitles | 3,011 | 2,326 | **77.3%** |
| GlobalVoices | 1,780 | 380 | 21.3% |
| GNOME | 57,110 | 3,903 | 6.8% |

TED2020 and OpenSubtitles have very high acceptance rates, reflecting their clean
alignment and sentence-level structure. GNOME's 6.8% rate is expected — GNOME
localisation strings are very short UI snippets (menu items, error messages, button
labels), and the majority fail the minimum length filter.

### Current Status

The 7,600 accepted pairs are stored in `data/v2/verified_additions.csv` with full
source provenance metadata (`source`, `source_am_file`, `source_en_file`,
`source_line`). They have **not yet been merged** into the main training corpus. A
200-pair random sample (`data/v2/human_review_sample.csv`) was created for manual
quality review before the merge decision is made.

---

## 9. EDA Findings Summary

The full EDA is implemented in `eda.py` and visualized in `eda_plots.py`. Key findings:

### Corpus Health

- **229,472 pairs, zero missing values** — the cleaning pipeline was thorough
- **99.99% Ge'ez script coverage** — only 15 rows have no Ge'ez characters
- **Average Ge'ez purity: 75.39%** — about 25% of characters in the Amharic column
  are non-Ge'ez (spaces, punctuation, and a small number of digits), which is normal
  for real-world text

### Length Distribution

- English: mean 20.1 words, median 18 words, max 100 words
- Amharic: mean 14.2 words, median 12 words, max 94 words
- Length ratio (Amharic/English): mean 0.75, median 0.71 — consistent and narrow
  standard deviation (0.30), indicating good alignment quality

### Domain Distribution

| Domain | Coverage |
|--------|----------|
| Religious / Bible | 30.1% |
| Family / Relationships | 11.2% |
| News / Politics | 10.9% |
| Daily Life / Conversational | 8.0% |
| Legal / Government | 7.1% |
| Nature / Agriculture | 6.5% |
| Health / Medical | 4.3% |
| Education / Science | 3.7% |
| Software / UI | 1.5% |
| Other / Unclassified | 41.4% |

Religious text dominates at 30%. The 41.4% "unclassified" is not missing data — it
reflects sentences that don't match any domain probe keyword, most of which are still
Bible/JW300 text using less keyword-dense vocabulary.

---

## 10. Strengths of This Pipeline

**Thorough source-specific cleaning.** Each source received a dedicated cleaning script
tailored to its specific noise patterns. This is significantly better than applying one
generic filter to everything and hoping for the best.

**Stratified splitting by sentence length.** This is the correct approach for NMT
datasets. It guarantees that evaluation scores reflect performance across the full
difficulty spectrum, not just the easy short sentences.

**Ge'ez-aware tokenizer configuration.** Setting `character_coverage=0.9995` and
training a shared BPE vocabulary with hardcoded special token IDs is technically sound
and follows best practices for low-resource multilingual NMT.

**Reproducibility.** `random_state=42` in both split stages, deterministic BPE
training, and versioned file paths mean the entire pipeline is reproducible from scratch.

**Source provenance tracking.** The V2 pipeline records which source file and line
number each accepted pair came from. This enables future debugging and selective
re-filtering if a specific source turns out to have quality problems.

**Multi-level deduplication.** Deduplication happens within each source's cleaning
script and again globally in `merge_datasets.py`. This two-level approach catches both
within-source and cross-source duplicates.

---

## 11. Known Limitations and Risks

**Domain skew.** The corpus is dominated by religious text (OPUS-100/JW300 + Bible).
The model will translate formal Amharic well but will struggle with conversational,
colloquial, or technical Amharic. Only 1,009 conversational pairs (0.44% of the corpus)
exist from the Rush Hour subtitles.

**OPUS-100 quality variance.** Web-crawled parallel text has inherently variable
quality. OPUS-100's Amharic-English pair quality is not uniformly high — some alignments
are at the paragraph level rather than sentence level, and some pairs are semantically
related but not direct translations. The cleaning filters remove the most obvious noise
but cannot catch subtle misalignments.

**English lowercase normalization is irreversible.** Lowercasing removes proper noun
case and the distinction between `I` (pronoun) and `i` (other). For most NMT use cases
this is acceptable, but it means the API output will return lowercase English, which
may need post-processing (re-capitalization) before being shown to end users.

**Shared vocabulary trade-off.** A 32K shared vocabulary gives Ge'ez 63% of tokens and
Latin 35.6%. With a larger vocabulary (64K) both scripts could be more fully covered.
The current size is a memory/performance compromise for the 93M-parameter model.

**The tokenizer encodes `'T'` as `<unk>`.** During EDA, the tokenization of
`"The dog bit the man."` showed `'T'` mapping to ID 1 (`<unk>`). This is harmless for
training because English is lowercased before being fed to the model. However, if the
API ever receives unnormalized text and the tokenizer is called directly without
preprocessing, uppercase letters will silently become unknown tokens. The API's
preprocessing step should apply `normalize_english()` before tokenization.

**V2 data not yet merged.** 7,600 high-quality new pairs — especially 991 TED2020 and
2,326 OpenSubtitles pairs — are sitting unused in `data/v2/`. These would directly
improve conversational and general-domain performance.

---

## 12. Recommendations

**Merge v2 data before the next training run.** The 991 TED2020 pairs (talks, general
knowledge) and 2,326 OpenSubtitles pairs (conversational, informal) are the most
valuable additions. They directly address the domain skew problem. A simple
`pd.concat` + deduplication in `merge_datasets.py` is all that is needed, followed by
re-running `src/preprocess.py` and `src/tokenizer.py`.

**Apply post-processing recapitalization in the API.** Add a simple sentence-initial
capitalization step in `api/main.py` after decoding. The model will always output
lowercase English — at minimum, capitalizing the first letter of each sentence improves
readability of the API response.

**Consider domain tags as a future improvement.** Adding a source domain tag (e.g.,
`<RELIGIOUS>`, `<CONVERSATIONAL>`, `<NEWS>`) as a prefix token at training time would
allow the model to be conditioned on domain at inference, improving output quality for
specific use cases. This is a standard technique in multilingual and domain-adapted NMT.

**Increase vocabulary size if retraining.** If the corpus grows significantly after v2
and further data collection, consider training a 64K vocabulary tokenizer. The Ge'ez
script's 517 syllabic characters need more vocabulary budget to achieve good subword
coverage for rare morphological forms.

---

*Document generated from direct code review of: `src/preprocess.py`, `src/tokenizer.py`,
`clean_bible.py`, `clean_opus.py`, `clean_rushhour.py`, `clean_csv_source.py`,
`merge_datasets.py`, `eda.py`, `domain_analysis.py`, and `data/v2/import_report.json`.*
