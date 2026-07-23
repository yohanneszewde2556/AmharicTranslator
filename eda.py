"""
Comprehensive EDA for Amharic-English NMT Project
Analyzes raw corpus, processed splits, tokenizer statistics, and v2 candidate data
"""

import pandas as pd
import numpy as np
import re
import json
import os
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from src.config import DATA_VERSION
except ImportError:
    DATA_VERSION = "v2"

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 80)
print(f"AMHARIC-ENGLISH NMT: COMPREHENSIVE EDA (DATA_VERSION='{DATA_VERSION}')")
print("=" * 80)

# ============================================================================
# 1. RAW CORPUS ANALYSIS
# ============================================================================
raw_filename = f"final_dataset_{DATA_VERSION}.csv" if DATA_VERSION else "final_dataset.csv"
raw_path = os.path.join("data", "raw", raw_filename)
if not os.path.exists(raw_path):
    raw_path = "data/raw/final_dataset.csv"

print("\n" + "=" * 80)
print(f"1. RAW CORPUS ANALYSIS ({raw_path})")
print("=" * 80)
if os.path.exists(raw_path):
    df_raw = pd.read_csv(raw_path)
    
    print(f"\n📊 Dataset Shape: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")
    print(f"Columns: {df_raw.columns.tolist()}")
    
    # Missing values
    print("\n🔍 Missing Values:")
    missing = df_raw.isnull().sum()
    for col in df_raw.columns:
        if missing[col] > 0:
            print(f"  {col}: {missing[col]:,} ({missing[col]/len(df_raw)*100:.2f}%)")
        else:
            print(f"  {col}: None")
    
    # Word count statistics
    print("\n📏 Word Count Statistics:")
    df_raw['am_word_count'] = df_raw['amharic'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
    df_raw['en_word_count'] = df_raw['english'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
    
    print("\nAmharic:")
    print(f"  Mean:   {df_raw['am_word_count'].mean():.2f} words")
    print(f"  Median: {df_raw['am_word_count'].median():.0f} words")
    print(f"  Std:    {df_raw['am_word_count'].std():.2f}")
    print(f"  Min:    {df_raw['am_word_count'].min()}")
    print(f"  Max:    {df_raw['am_word_count'].max()}")
    print(f"  25th percentile: {df_raw['am_word_count'].quantile(0.25):.0f}")
    print(f"  75th percentile: {df_raw['am_word_count'].quantile(0.75):.0f}")
    
    print("\nEnglish:")
    print(f"  Mean:   {df_raw['en_word_count'].mean():.2f} words")
    print(f"  Median: {df_raw['en_word_count'].median():.0f} words")
    print(f"  Std:    {df_raw['en_word_count'].std():.2f}")
    print(f"  Min:    {df_raw['en_word_count'].min()}")
    print(f"  Max:    {df_raw['en_word_count'].max()}")
    print(f"  25th percentile: {df_raw['en_word_count'].quantile(0.25):.0f}")
    print(f"  75th percentile: {df_raw['en_word_count'].quantile(0.75):.0f}")
    
    # Length ratio analysis
    df_raw['length_ratio'] = df_raw['am_word_count'] / (df_raw['en_word_count'] + 1e-6)
    print("\n📐 Length Ratio (Amharic/English):")
    print(f"  Mean:   {df_raw['length_ratio'].mean():.2f}")
    print(f"  Median: {df_raw['length_ratio'].median():.2f}")
    print(f"  Std:    {df_raw['length_ratio'].std():.2f}")
    
    # Character count statistics
    df_raw['am_char_count'] = df_raw['amharic'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
    df_raw['en_char_count'] = df_raw['english'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
    
    print("\n📝 Character Count Statistics:")
    print(f"Amharic: Mean={df_raw['am_char_count'].mean():.1f}, Max={df_raw['am_char_count'].max()}")
    print(f"English: Mean={df_raw['en_char_count'].mean():.1f}, Max={df_raw['en_char_count'].max()}")
    
    # Ge'ez script analysis
    def has_geez(text):
        """Check if text contains Ge'ez script characters (U+1200 to U+137F)"""
        if not isinstance(text, str):
            return False
        return bool(re.search(r'[\u1200-\u137F]', text))
    
    def count_geez(text):
        """Count Ge'ez characters in text"""
        if not isinstance(text, str):
            return 0
        return len(re.findall(r'[\u1200-\u137F]', text))
    
    df_raw['has_geez'] = df_raw['amharic'].apply(has_geez)
    df_raw['geez_count'] = df_raw['amharic'].apply(count_geez)
    df_raw['geez_purity'] = df_raw['geez_count'] / (df_raw['am_char_count'] + 1e-6)
    
    print("\n🔤 Ge'ez Script Analysis:")
    print(f"  Rows with Ge'ez script: {df_raw['has_geez'].sum():,} ({df_raw['has_geez'].sum()/len(df_raw)*100:.2f}%)")
    print(f"  Rows without Ge'ez: {(~df_raw['has_geez']).sum():,} ({(~df_raw['has_geez']).sum()/len(df_raw)*100:.2f}%)")
    print(f"  Average Ge'ez purity: {df_raw['geez_purity'].mean():.2%}")
    
    # Check for potential quality issues
    print("\n⚠️  Quality Check:")
    very_short_am = (df_raw['am_word_count'] < 3).sum()
    very_short_en = (df_raw['en_word_count'] < 3).sum()
    very_long_am = (df_raw['am_word_count'] > 50).sum()
    very_long_en = (df_raw['en_word_count'] > 50).sum()
    extreme_ratio = ((df_raw['length_ratio'] < 0.3) | (df_raw['length_ratio'] > 3.0)).sum()
    
    print(f"  Very short Amharic (<3 words): {very_short_am:,} ({very_short_am/len(df_raw)*100:.2f}%)")
    print(f"  Very short English (<3 words): {very_short_en:,} ({very_short_en/len(df_raw)*100:.2f}%)")
    print(f"  Very long Amharic (>50 words): {very_long_am:,} ({very_long_am/len(df_raw)*100:.2f}%)")
    print(f"  Very long English (>50 words): {very_long_en:,} ({very_long_en/len(df_raw)*100:.2f}%)")
    print(f"  Extreme length ratio (<0.3 or >3.0): {extreme_ratio:,} ({extreme_ratio/len(df_raw)*100:.2f}%)")
    
    # Sample pairs
    print("\n📝 Sample Pairs (Random):")
    for i, row in df_raw.sample(3, random_state=42).iterrows():
        print(f"\n  Amharic: {row['amharic']}")
        print(f"  English: {row['english']}")
        print(f"  Lengths: AM={row['am_word_count']} words, EN={row['en_word_count']} words")
    
else:
    print(f"\n❌ File not found: {raw_path}")

# ============================================================================
# 2. PROCESSED SPLITS ANALYSIS
# ============================================================================
proc_dir = os.path.join("data", "processed", DATA_VERSION) if DATA_VERSION else os.path.join("data", "processed")
print("\n\n" + "=" * 80)
print(f"2. PROCESSED SPLITS ANALYSIS ({proc_dir})")
print("=" * 80)

splits = {
    'train': os.path.join(proc_dir, "train.csv"),
    'val': os.path.join(proc_dir, "val.csv"),
    'test': os.path.join(proc_dir, "test.csv")
}

split_stats = {}
for split_name, split_path in splits.items():
    if os.path.exists(split_path):
        df_split = pd.read_csv(split_path)
        
        # Calculate statistics
        df_split['am_word_count'] = df_split['amharic'].apply(lambda x: len(str(x).split()))
        df_split['en_word_count'] = df_split['english'].apply(lambda x: len(str(x).split()))
        
        split_stats[split_name] = {
            'count': len(df_split),
            'am_mean': df_split['am_word_count'].mean(),
            'en_mean': df_split['en_word_count'].mean(),
            'am_max': df_split['am_word_count'].max(),
            'en_max': df_split['en_word_count'].max()
        }
        
        print(f"\n📊 {split_name.upper()} Split:")
        print(f"  Size: {len(df_split):,} pairs")
        print(f"  Amharic: mean={df_split['am_word_count'].mean():.2f} words, max={df_split['am_word_count'].max()}")
        print(f"  English: mean={df_split['en_word_count'].mean():.2f} words, max={df_split['en_word_count'].max()}")
    else:
        print(f"\n❌ {split_name.upper()} split not found: {split_path}")

# Summary comparison
if split_stats:
    print("\n📈 Split Size Distribution:")
    total = sum(s['count'] for s in split_stats.values())
    for split_name, stats in split_stats.items():
        percentage = stats['count'] / total * 100
        print(f"  {split_name.capitalize()}: {stats['count']:,} pairs ({percentage:.2f}%)")

# ============================================================================
# 3. TOKENIZER ANALYSIS
# ============================================================================
print("\n\n" + "=" * 80)
print(f"3. TOKENIZER ANALYSIS (SentencePiece BPE in {proc_dir})")
print("=" * 80)

tokenizer_model = os.path.join(proc_dir, "am_en_bpe.model")
tokenizer_vocab = os.path.join(proc_dir, "am_en_bpe.vocab")

if os.path.exists(tokenizer_model) and os.path.exists(tokenizer_vocab):
    try:
        import sentencepiece as spm
        
        sp = spm.SentencePieceProcessor()
        sp.load(tokenizer_model)
        
        print(f"\n🔧 Tokenizer Configuration:")
        print(f"  Vocabulary size: {sp.vocab_size():,}")
        print(f"  PAD token: '{sp.id_to_piece(0)}' (ID: 0)")
        print(f"  UNK token: '{sp.id_to_piece(1)}' (ID: 1)")
        print(f"  BOS token: '{sp.id_to_piece(2)}' (ID: 2)")
        print(f"  EOS token: '{sp.id_to_piece(3)}' (ID: 3)")
        
        # Load vocabulary file to analyze token types
        vocab_tokens = []
        with open(tokenizer_vocab, 'r', encoding='utf-8') as f:
            for line in f:
                token = line.split('\t')[0]
                vocab_tokens.append(token)
        
        # Categorize tokens
        geez_tokens = [t for t in vocab_tokens if re.search(r'[\u1200-\u137F]', t)]
        latin_tokens = [t for t in vocab_tokens if re.search(r'[a-zA-Z]', t) and not re.search(r'[\u1200-\u137F]', t)]
        digit_tokens = [t for t in vocab_tokens if re.search(r'\d', t)]
        special_tokens = [t for t in vocab_tokens if t.startswith('<') or t.startswith('▁')]
        
        print(f"\n📚 Vocabulary Composition:")
        print(f"  Ge'ez script tokens: {len(geez_tokens):,} ({len(geez_tokens)/len(vocab_tokens)*100:.2f}%)")
        print(f"  Latin script tokens: {len(latin_tokens):,} ({len(latin_tokens)/len(vocab_tokens)*100:.2f}%)")
        print(f"  Digit tokens: {len(digit_tokens):,} ({len(digit_tokens)/len(vocab_tokens)*100:.2f}%)")
        print(f"  Special/subword tokens: {len(special_tokens):,} ({len(special_tokens)/len(vocab_tokens)*100:.2f}%)")
        
        # Test tokenization on sample texts
        test_samples = {
            'Amharic (simple)': 'ውሻው ሰውን ነከሰ።',
            'English (simple)': 'The dog bit the man.',
            'Amharic (complex)': 'በእነዚያ ቀናት መጓዝ የበለጠ አስቸጋሪ ነበር።',
            'English (complex)': 'Traveling was more difficult in those days.'
        }
        
        print(f"\n🧪 Tokenization Examples:")
        for desc, text in test_samples.items():
            tokens = sp.encode_as_pieces(text)
            ids = sp.encode_as_ids(text)
            print(f"\n  {desc}:")
            print(f"    Text: {text}")
            print(f"    Tokens: {tokens}")
            print(f"    Token count: {len(tokens)}")
            print(f"    IDs (first 10): {ids[:10]}")
        
        # Token length statistics on train split
        if os.path.exists(splits['train']):
            print(f"\n📊 Token Length Statistics (Train Split: {splits['train']}):")
            df_train = pd.read_csv(splits['train'])
            
            # Sample for speed (analyze 10,000 random pairs)
            sample_size = min(10000, len(df_train))
            df_sample = df_train.sample(sample_size, random_state=42)
            
            am_token_lengths = []
            en_token_lengths = []
            
            for _, row in df_sample.iterrows():
                am_tokens = sp.encode_as_ids(str(row['amharic']))
                en_tokens = sp.encode_as_ids(str(row['english']))
                am_token_lengths.append(len(am_tokens))
                en_token_lengths.append(len(en_tokens))
            
            print(f"  (Analyzed {sample_size:,} random training pairs)")
            print(f"\n  Amharic token lengths:")
            print(f"    Mean:   {np.mean(am_token_lengths):.2f} tokens")
            print(f"    Median: {np.median(am_token_lengths):.0f} tokens")
            print(f"    Max:    {np.max(am_token_lengths)}")
            print(f"    95th percentile: {np.percentile(am_token_lengths, 95):.0f}")
            
            print(f"\n  English token lengths:")
            print(f"    Mean:   {np.mean(en_token_lengths):.2f} tokens")
            print(f"    Median: {np.median(en_token_lengths):.0f} tokens")
            print(f"    Max:    {np.max(en_token_lengths)}")
            print(f"    95th percentile: {np.percentile(en_token_lengths, 95):.0f}")
            
            # Check how many sequences exceed MAX_LENGTH=128
            am_exceeds = sum(1 for l in am_token_lengths if l > 128)
            en_exceeds = sum(1 for l in en_token_lengths if l > 128)
            print(f"\n  ⚠️  Sequences exceeding MAX_LENGTH=128:")
            print(f"    Amharic: {am_exceeds} ({am_exceeds/sample_size*100:.2f}%)")
            print(f"    English: {en_exceeds} ({en_exceeds/sample_size*100:.2f}%)")
            
    except ImportError:
        print("\n❌ sentencepiece package not installed. Run: pip install sentencepiece")
    except Exception as e:
        print(f"\n❌ Error loading tokenizer: {e}")
else:
    print(f"\n❌ Tokenizer files not found")
    print(f"  Model: {tokenizer_model}")
    print(f"  Vocab: {tokenizer_vocab}")

# ============================================================================
# 4. V2 CANDIDATE DATA ANALYSIS
# ============================================================================
print("\n\n" + "=" * 80)
print("4. V2 CANDIDATE DATA ANALYSIS (data/v2/)")
print("=" * 80)

import_report_path = "data/v2/import_report.json"
verified_additions_path = "data/v2/verified_additions.csv"

if os.path.exists(import_report_path):
    with open(import_report_path, 'r') as f:
        import_report = json.load(f)
    
    print("\n📥 Import Report Summary:")
    print(f"  Total rows imported: {import_report['imported_rows']:,}")
    print(f"  Accepted rows: {import_report['accepted_rows']:,} ({import_report['accepted_rows']/import_report['imported_rows']*100:.2f}%)")
    print(f"  Rejected rows: {import_report['rejected_rows']:,} ({import_report['rejected_rows']/import_report['imported_rows']*100:.2f}%)")
    
    print("\n📊 Sources:")
    for source, count in import_report['sources'].items():
        print(f"  {source}: {count:,} rows")
    
    print("\n✅ Accepted by Source:")
    for source, count in import_report['accepted_by_source'].items():
        acceptance_rate = count / import_report['sources'][source] * 100
        print(f"  {source}: {count:,} ({acceptance_rate:.1f}% acceptance)")
    
    print("\n❌ Rejection Reasons:")
    rejection_reasons = import_report['rejections_by_reason']
    total_rejections = sum(rejection_reasons.values())
    for reason, count in sorted(rejection_reasons.items(), key=lambda x: x[1], reverse=True):
        print(f"  {reason}: {count:,} ({count/total_rejections*100:.1f}%)")
else:
    print(f"\n❌ Import report not found: {import_report_path}")

if os.path.exists(verified_additions_path):
    df_v2 = pd.read_csv(verified_additions_path)
    
    print(f"\n📋 Verified Additions Dataset:")
    print(f"  Shape: {df_v2.shape[0]:,} rows × {df_v2.shape[1]} columns")
    print(f"  Columns: {df_v2.columns.tolist()}")
    
    # Length analysis
    df_v2['am_word_count'] = df_v2['amharic'].apply(lambda x: len(str(x).split()))
    df_v2['en_word_count'] = df_v2['english'].apply(lambda x: len(str(x).split()))
    
    print(f"\n📏 V2 Length Statistics:")
    print(f"  Amharic: mean={df_v2['am_word_count'].mean():.2f}, median={df_v2['am_word_count'].median():.0f}, max={df_v2['am_word_count'].max()}")
    print(f"  English: mean={df_v2['en_word_count'].mean():.2f}, median={df_v2['en_word_count'].median():.0f}, max={df_v2['en_word_count'].max()}")
    
    # Source distribution
    if 'source' in df_v2.columns:
        print(f"\n📊 Source Distribution in Verified Additions:")
        source_counts = df_v2['source'].value_counts()
        for source, count in source_counts.items():
            print(f"  {source}: {count:,} ({count/len(df_v2)*100:.1f}%)")
    
    # Compare to main corpus
    if os.path.exists(raw_path):
        print(f"\n🔄 Comparison: V2 vs Main Corpus:")
        print(f"  V2 size: {len(df_v2):,} pairs")
        print(f"  Main corpus size: {len(df_raw):,} pairs")
        print(f"  V2 as % of main: {len(df_v2)/len(df_raw)*100:.2f}%")
        print(f"  Potential merged size: {len(df_raw) + len(df_v2):,} pairs")
else:
    print(f"\n❌ Verified additions not found: {verified_additions_path}")

# ============================================================================
# 5. OVERALL SUMMARY
# ============================================================================
print("\n\n" + "=" * 80)
print("5. OVERALL SUMMARY")
print("=" * 80)

print("\n📊 Dataset Overview:")
if os.path.exists(raw_path):
    print(f"  ✓ Raw corpus: {len(df_raw):,} pairs")
if split_stats:
    total_processed = sum(s['count'] for s in split_stats.values())
    print(f"  ✓ Processed splits: {total_processed:,} pairs (train/val/test)")
if os.path.exists(verified_additions_path):
    print(f"  ✓ V2 candidate additions: {len(df_v2):,} pairs")
if os.path.exists(tokenizer_model):
    print(f"  ✓ Tokenizer: SentencePiece BPE with {sp.vocab_size():,} tokens")

print("\n🎯 Key Insights:")
if os.path.exists(raw_path):
    print(f"  • Average sentence length: {df_raw['en_word_count'].mean():.1f} words (English)")
    print(f"  • Ge'ez script coverage: {df_raw['has_geez'].sum()/len(df_raw)*100:.1f}% of Amharic sentences")
    print(f"  • Length ratio (Am/En): {df_raw['length_ratio'].mean():.2f} (mean)")
if os.path.exists(tokenizer_model) and 'am_token_lengths' in locals():
    print(f"  • Average token length: {np.mean(am_token_lengths):.1f} (Amharic), {np.mean(en_token_lengths):.1f} (English)")
    exceeds_pct = (am_exceeds + en_exceeds) / (2 * sample_size) * 100
    print(f"  • Sequences >128 tokens: {exceeds_pct:.2f}% (will be truncated in training)")

print("\n⚠️  Data Characteristics:")
print("  • Dataset is heavily skewed toward Bible/JW300 religious text")
print("  • Formal/religious Amharic will perform better than conversational")
print("  • Consider merging v2 candidates for improved domain coverage")

print("\n" + "=" * 80)
print("EDA COMPLETE")
print("=" * 80)
