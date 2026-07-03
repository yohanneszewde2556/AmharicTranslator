import pandas as pd
from sklearn.model_selection import train_test_split
import os

def split_dataset(file_path: str):
    """
    Splits the Amharic-English dataset into Train (80%), Validation (10%), and Test (10%).
    Ensures reproducibility with a fixed random seed.
    """
    print(f"Loading {file_path} for splitting...")
    try:
        # Load the cleaned dataset
        df = pd.read_csv(file_path, engine="c", on_bad_lines='skip')
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # To ensure high-quality splits, let's just make sure there are no NaN values 
    # left over from any previous steps that were skipped.
    df = df.dropna()
    df = df.drop_duplicates()
    
    total_size = len(df)
    print(f"Total clean sentence pairs: {total_size}")

    # Step 1: Split off 20% for the temporary testing/validation pool.
    # The remaining 80% becomes the Training set.
    # We use random_state=42 to ensure every time you run this, you get the exact same split.
    train_df, temp_df = train_test_split(df, test_size=0.20, random_state=42)
    
    # Step 2: Split the 20% temp pool into two equal halves (10% validation, 10% test)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)
    
    print("\n--- Split Sizes ---")
    print(f"Train set:      {len(train_df)} ({len(train_df)/total_size*100:.1f}%)")
    print(f"Validation set: {len(val_df)}  ({len(val_df)/total_size*100:.1f}%)")
    print(f"Test set:       {len(test_df)}  ({len(test_df)/total_size*100:.1f}%)")
    
    # Save the splits to new CSV files
    print("\nSaving splits to CSV files...")
    train_df.to_csv("train.csv", index=False)
    val_df.to_csv("val.csv", index=False)
    test_df.to_csv("test.csv", index=False)
    
    print("Files saved successfully! (train.csv, val.csv, test.csv)")
    
if __name__ == "__main__":
    file_path = "final_dataset.csv"
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}. Please make sure you have the expected dataset.")
    else:
        split_dataset(file_path)
