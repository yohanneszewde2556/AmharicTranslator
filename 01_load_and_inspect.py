import pandas as pd
import os

def load_and_inspect_dataset(file_path: str):
    """
    Loads and inspects the Amharic-English translation dataset.
    
    Args:
        file_path (str): Path to the dataset CSV file.
    """
    print(f"Loading dataset from {file_path}...\n")
    
    # 1. Load the dataset
    # We use engine="python" or default engine="c". If you expect a massive dataset, "c" is faster.
    # Handling potential bad lines with on_bad_lines='skip' is a good defensive practice if the CSV has parsing issues.
    try:
        df = pd.read_csv(file_path, engine="c", on_bad_lines='skip')
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

    # 2. Detect column names automatically
    print("--- Column Information ---")
    columns = df.columns.tolist()
    print(f"Detected columns: {columns}\n")
    
    # Assuming standard parallel dataset columns if not named explicitly, e.g., 'am' and 'en'
    # But let's verify what columns we have in the dataframe
    
    # 3. Verify missing values
    print("--- Missing Values ---")
    missing_values = df.isnull().sum()
    print(missing_values)
    
    if missing_values.sum() > 0:
        print("Dropping rows with missing values...")
        df = df.dropna()
        print(f"New dataset size after dropping missing values: {len(df)}\n")
    else:
        print("No missing values found!\n")

    # 4. Remove duplicates
    print("--- Duplicates ---")
    initial_len = len(df)
    duplicates_count = df.duplicated().sum()
    print(f"Number of exact duplicate rows: {duplicates_count}")
    
    if duplicates_count > 0:
        print("Dropping duplicates...")
        df = df.drop_duplicates()
        print(f"Removed {initial_len - len(df)} duplicate rows.\n")
    else:
        print("No duplicates found!\n")
        
    # 5. Display dataset statistics
    print("--- Dataset Statistics ---")
    print(f"Total pure sentence pairs remaining: {len(df)}")
    print(df.describe(include='all'))
    print("\n")

    # 6. Show a few sample sentence pairs
    print("--- Sample Sentence Pairs ---")
    # Using sample() to show random rows guarantees a diverse look at the data
    samples = df.sample(n=min(5, len(df)), random_state=42)
    for idx, row in samples.iterrows():
        print(f"Pair {idx}:")
        for col in columns:
            print(f"  {col}: {row[col]}")
        print("-" * 40)
        
    return df

if __name__ == "__main__":
    # Ensure this matches your actual dataset path
    file_path = "final_dataset.csv"
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}. Please check the path.")
    else:
        df = load_and_inspect_dataset(file_path)
