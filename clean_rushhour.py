"""
Clean rush_hour_parallel_pairs.xlsx → rush_hour_clean.csv
Output: 2-column (amharic, english) with clean words only.

Pass 1 — Structural cleanup:
  1. Strip speaker labels from both sides  (e.g. "CARTER: ", "ሰው: ", "MAN 1 :")
  2. Remove leading / trailing ellipsis artifacts ("...", "…")
  3. Remove rows flagged CHECK (length ratio < 0.5 or > 2.0)
  4. Remove rows where either side is too short (≤ 2 words)
  5. Remove rows where Amharic is >30% Latin characters (garbled MT)

Pass 2 — Deep text cleanup (words only):
  6. Remove all numbers and digit sequences
  7. Strip stray Latin characters / words from Amharic column
  8. Strip leading dashes from subtitle turn-taking ("-word" → "word")
  9. Remove all remaining non-word punctuation except basic sentence enders
 10. Collapse multiple spaces, re-strip

Pass 3 — Final filter:
 11. Drop pairs that became too short after cleaning (≤ 2 words either side)
 12. Drop exact duplicates on either column
"""
import re
import pandas as pd

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_excel('rush_hour_parallel_pairs.xlsx')
print(f"Loaded: {len(df)} rows")
original_count = len(df)

# ═══════════════════════════════════════════════════════════════
# PASS 1 — Structural cleanup
# ═══════════════════════════════════════════════════════════════

def strip_speaker_label_en(text):
    """Remove ALL-CAPS speaker labels anywhere in string: 'CARTER: ', 'MAN 1 : '"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\b[A-Z][A-Z\s\-]+(\s*\d+)?\s*:\s*', '', text)
    return text.strip()

def strip_speaker_label_am(text):
    """Remove any word+optional number+colon labels anywhere: 'ሰው:', 'ሰው 2:', 'ካርተር:'"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[\w\u1200-\u137F]+(\s+\d+)?\s*:\s*', '', text)
    return text.strip()

def strip_ellipsis(text):
    """Remove leading/trailing ellipsis subtitle artifacts"""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r'^[\.…]+\s*', '', text)
    text = re.sub(r'\s*[\.…]+$', '', text)
    return text.strip()

def is_majority_latin(text):
    """True if >30% of alphabetic chars are Latin — garbled Amharic"""
    if not isinstance(text, str):
        return True
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return True
    latin = sum(1 for c in alpha if c.isascii())
    return latin / len(alpha) > 0.30

# Apply Pass 1
df['english'] = df['english'].apply(strip_speaker_label_en)
df['amharic'] = df['amharic'].apply(strip_speaker_label_am)
df['english'] = df['english'].apply(strip_ellipsis)
df['amharic'] = df['amharic'].apply(strip_ellipsis)

before = len(df)
df = df[df['quality_flag'] != 'CHECK'].copy()
print(f"[Pass 1] Dropped {before - len(df)} rows — bad length ratio")

df['en_words'] = df['english'].apply(lambda x: len(str(x).split()))
df['am_words'] = df['amharic'].apply(lambda x: len(str(x).split()))
before = len(df)
df = df[(df['en_words'] > 2) & (df['am_words'] > 1)].copy()
print(f"[Pass 1] Dropped {before - len(df)} rows — too short")

before = len(df)
df = df[~df['amharic'].apply(is_majority_latin)].copy()
print(f"[Pass 1] Dropped {before - len(df)} rows — majority Latin in Amharic")

# ═══════════════════════════════════════════════════════════════
# PASS 2 — Deep text cleanup
# ═══════════════════════════════════════════════════════════════

def clean_english(text):
    """
    Clean English to words only:
    - Remove numbers and digits
    - Remove leading dashes (subtitle turn-taking artifact)
    - Remove special characters — keep only letters, spaces, basic punctuation
    - Collapse whitespace
    """
    if not isinstance(text, str):
        return ""
    # Remove numbers (standalone digits and digit groups)
    text = re.sub(r'\b\d+\b', '', text)
    text = re.sub(r'\d+', '', text)
    # Remove leading dashes and hyphens used for turn-taking
    text = re.sub(r'^[-\s]+', '', text)
    # Remove dashes mid-sentence that are turn-taking (space-dash-space pattern)
    text = re.sub(r'\s*-\s*', ' ', text)
    # Keep only letters, spaces, apostrophe (contractions), and basic punctuation
    text = re.sub(r"[^a-zA-Z\s'.,!?]", ' ', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def clean_amharic(text):
    """
    Clean Amharic to words only:
    - Remove all Latin characters and words (stray untranslated tokens)
    - Remove numbers and digits
    - Remove leading dashes
    - Keep only Ge'ez script characters, Ethiopic punctuation, and spaces
    - Collapse whitespace
    """
    if not isinstance(text, str):
        return ""
    # Remove all ASCII/Latin characters (including stray English words)
    text = re.sub(r'[a-zA-Z]+', ' ', text)
    # Remove all digit characters
    text = re.sub(r'\d+', ' ', text)
    # Remove leading dashes
    text = re.sub(r'^[-\s]+', '', text)
    # Remove dashes used for turn-taking
    text = re.sub(r'\s*-\s*', ' ', text)
    # Keep only Amharic (Ge'ez block U+1200–U+137F),
    # Ethiopic supplement (U+1380–U+139F),
    # Ethiopic punctuation (። ፡ ፣ ፤ ፥ ፦ ፧),
    # and common punctuation/spaces
    text = re.sub(r'[^\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF\s።፡፣፤፥፦፧.,!?]', ' ', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

df['english'] = df['english'].apply(clean_english)
df['amharic'] = df['amharic'].apply(clean_amharic)
print(f"[Pass 2] Deep text cleanup applied")

# ═══════════════════════════════════════════════════════════════
# PASS 3 — Final filter after cleanup
# ═══════════════════════════════════════════════════════════════

# Recount words after cleanup
df['en_words'] = df['english'].apply(lambda x: len(str(x).split()))
df['am_words'] = df['amharic'].apply(lambda x: len(str(x).split()))

before = len(df)
df = df[(df['en_words'] > 2) & (df['am_words'] > 1)].copy()
print(f"[Pass 3] Dropped {before - len(df)} rows — too short after deep clean")

before = len(df)
df = df[(df['english'].str.strip().str.len() > 0) & (df['amharic'].str.strip().str.len() > 0)]
df = df.drop_duplicates(subset=['amharic'])
df = df.drop_duplicates(subset=['english'])
print(f"[Pass 3] Dropped {before - len(df)} rows — empty or duplicate")

# ── Final output ──────────────────────────────────────────────────────────────
df = df[['amharic', 'english']].reset_index(drop=True)

output_csv = 'rush_hour_clean.csv'
df.to_csv(output_csv, index=False, encoding='utf-8-sig')

print(f"\n{'='*50}")
print(f"Original rows  : {original_count}")
print(f"Final rows     : {len(df)}")
print(f"Total removed  : {original_count - len(df)}")
print(f"Saved to       : {output_csv}")
print(f"{'='*50}")

print("\n=== Sample of final clean pairs ===")
for i, row in df.sample(min(12, len(df)), random_state=42).iterrows():
    print(f"\n  EN: {row['english']}")
    print(f"  AM: {row['amharic']}")
