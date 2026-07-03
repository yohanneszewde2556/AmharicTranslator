from datasets import load_dataset, DatasetDict
import os

def load_hf_dataset():
    """
    Loads local CSV splits into a Hugging Face DatasetDict.
    This creates an object holding train, val, and test data together,
    which plays very nicely with the Hugging Face Trainer later.
    """
    # Define our data files. You must have run 02_split_dataset.py first!
    data_files = {
        "train": "train.csv",
        "validation": "val.csv",
        "test": "test.csv"
    }

    # Verify files exist locally
    for split, path in data_files.items():
        if not os.path.exists(path):
            print(f"Error: Could not find {path}. Did you run Step 2?")
            return None

    print("Loading CSV files into Hugging Face DatasetDict...")
    
    # Hugging Face's load_dataset handles massive CSVs blazingly fast.
    dataset = load_dataset("csv", data_files=data_files)
    
    print("\n--- Successful Conversion ---")
    # This will print the structure of our DatasetDict
    print(dataset)
    
    # Let's inspect column names and the very first row of the train set to verify!
    print("\n--- Sample Row (Train Set Row 0) ---")
    print(dataset["train"][0])
    
    return dataset

if __name__ == "__main__":
    dataset = load_hf_dataset()
    # Notice we don't save this! When working with HF, we usually just load 
    # directly from the CSV into memory (or cache) dynamically in our main training script later.
