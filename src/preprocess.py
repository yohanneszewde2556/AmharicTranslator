import pandas as pd
import re
from sklearn.model_selection import train_test_split
import os

def normalize_amharic(text):
    if not isinstance(text, str):
        return ""
    # Convert '::' to Ethiopic full stop '።'
    text = re.sub(r':{2,}', '።', text)
    # Convert remaining single ':' to Ethiopic word space '፡'
    text = text.replace(':', '፡')
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_english(text):
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = str(text).lower()
    # Remove bizarre non-ASCII characters to keep the vocab clean
    text = re.sub(r'[^a-z0-9\s.,!?\'"-]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def run_preprocessing(input_path, output_dir):
    print(f"Loading raw dataset from {input_path}...")
    try:
        df = pd.read_csv(input_path, engine="c", on_bad_lines='skip')
    except Exception as e:
        print(f"Error reading {input_path}: {e}")
        return
    
    # Safely find the column names
    try:
        am_col = [col for col in df.columns if 'am' in col.lower()][0]
        en_col = [col for col in df.columns if 'en' in col.lower()][0]
    except IndexError:
        print(f"Could not automatically find am/en columns. Columns are: {df.columns}")
        return
        
    print(f"Using Amharic column: '{am_col}' and English column: '{en_col}'")
    
    print("Normalizing text (this might take a moment)...")
    df['amharic_clean'] = df[am_col].apply(normalize_amharic)
    df['english_clean'] = df[en_col].apply(normalize_english)
    
    # Drop rows that are empty after cleaning
    df = df[(df['amharic_clean'].str.len() > 0) & (df['english_clean'].str.len() > 0)].copy()
    
    print("Calculating lengths for quantile stratification...")
    # Stratify based on word count of the english translation
    df['word_count'] = df['english_clean'].apply(lambda x: len(x.split()))
    
    # Filter out absurdly long sentences that will break a standard Seq2Seq Transformer OOM
    df = df[df['word_count'] <= 100].copy()
    
    # Create quantiles (5 buckets: very short, short, medium, long, very long)
    df['length_bucket'] = pd.qcut(df['word_count'], q=5, labels=False, duplicates='drop')
    
    print("Splitting dataset with Stratified Sampling...")
    # First split: 95% train, 5% temp
    train_df, temp_df = train_test_split(df, test_size=0.05, stratify=df['length_bucket'], random_state=42)
    # Second split: 50/50 of temp (2.5% each for Val and Test)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['length_bucket'], random_state=42)
    
    columns_to_keep = ['amharic_clean', 'english_clean']
    
    train_df = train_df[columns_to_keep].rename(columns={'amharic_clean': 'amharic', 'english_clean': 'english'})
    val_df = val_df[columns_to_keep].rename(columns={'amharic_clean': 'amharic', 'english_clean': 'english'})
    test_df = test_df[columns_to_keep].rename(columns={'amharic_clean': 'amharic', 'english_clean': 'english'})
    
    # Save processed files
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    print("\n--- Summary ---")
    print(f"Train size: {len(train_df)}")
    print(f"Val size:   {len(val_df)}")
    print(f"Test size:  {len(test_df)}")
    print(f"\nFiles saved in {output_dir}")
    print("Preprocessing Phase 1 completed successfully!")

if __name__ == "__main__":
    raw_csv = "data/raw/final_dataset.csv"
    processed_dir = "data/processed"
    if not os.path.exists(raw_csv):
        print(f"FATAL WARNING: {raw_csv} not found.")
        print("Please ensure the CSV file is located in the data/raw/ directory.")
    else:
        run_preprocessing(raw_csv, processed_dir)
