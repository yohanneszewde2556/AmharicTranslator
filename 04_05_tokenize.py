from datasets import load_dataset
from transformers import NllbTokenizerFast
import os

def prepare_tokenized_dataset():
    """
    Step 4 & 5: Configures the Tokenizer and Tokenizes the datasets.
    """
    model_checkpoint = "facebook/nllb-200-distilled-600M"
    
    print("--- Step 4: Loading Tokenizer ---")
    # For NLLB, language codes follow the BCP-47 standard loosely format. 
    # Amharic is "amh_Ethi", English is "eng_Latn".
    # This ensures the tokenizer uses the correct language-specific vocabulary rules.
    tokenizer = NllbTokenizerFast.from_pretrained(
        model_checkpoint, 
        src_lang="amh_Ethi", 
        tgt_lang="eng_Latn"
    )
    print("Tokenizer loaded successfully.")

    # Load the CSV datasets dynamically
    print("\nLoading dataset splits (from Step 3)...")
    raw_datasets = load_dataset(
        "csv", 
        data_files={"train": "train.csv", "validation": "val.csv", "test": "test.csv"}
    )
    
    # Let's see what column names we are working with
    # Assuming CSV has 'amharic' and 'english' columns (or 'am' / 'en').
    # We will grab the first row to automatically detect column names if needed.
    columns = list(raw_datasets["train"].features.keys())
    print(f"Detected columns: {columns}")
    
    # We will assume the first column is source (Amharic) and second is target (English)
    # E.g., if you have 'am' and 'en', am is src, en is tgt.
    src_col = columns[0]
    tgt_col = columns[1]
    print(f"Using '{src_col}' as Source (Amharic) and '{tgt_col}' as Target (English).")

    # The maximum length of our tokens. 
    # Amharic text and English text rarely exceed 128 words per sentence.
    # Capping at 128 tightly bounds our RAM consumption.
    max_length = 128

    def preprocess_function(examples):
        # 1. Tokenize Amharic (inputs)
        inputs = [str(ex) for ex in examples[src_col]]
        targets = [str(ex) for ex in examples[tgt_col]]
        
        # Tokenize the input sequence
        model_inputs = tokenizer(inputs, max_length=max_length, truncation=True)

        # 2. Tokenize English (labels)
        # In seq2seq models, the "target" text needs to be processed with text_target parameter
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_length, truncation=True)
            
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("\n--- Step 5: Tokenizing Dataset (This might take a minute) ---")
    # Using python's multiprocessing with map via `batched=True`
    # This heavily speeds up tokenization.
    tokenized_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names, # Remove text, keep only numbers
        desc="Running tokenizer on dataset splits"
    )

    print("\n--- Tokenization Complete ---")
    print(tokenized_datasets)
    
    # Look at one sample to ensure "input_ids" and "labels" format
    sample = tokenized_datasets["train"][0]
    print("\nSample tokenized row keys:", sample.keys())
    print(f"Lengths -> input_ids: {len(sample['input_ids'])}, labels: {len(sample['labels'])}")

    # We save to disk! This gives us a folder we can efficiently load during training without re-tokenizing.
    save_path = "nllb_tokenized_dataset"
    print(f"\nSaving tokenized dataset to {save_path}...")
    tokenized_datasets.save_to_disk(save_path)
    print("Done! You're ready for modeling.")
    
    return tokenized_datasets

if __name__ == "__main__":
    prepare_tokenized_dataset()
