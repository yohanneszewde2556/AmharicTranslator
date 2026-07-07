import sentencepiece as spm
import pandas as pd
import os

def train_tokenizer(train_csv_path, output_dir, vocab_size=32000):
    text_file_path = os.path.join(output_dir, "spm_train.txt")
    model_prefix = os.path.join(output_dir, "am_en_bpe")
    
    print(f"Loading {train_csv_path} to extract raw text...")
    df = pd.read_csv(train_csv_path)
    
    print(f"Creating raw text dump for SentencePiece at {text_file_path}...")
    with open(text_file_path, "w", encoding="utf-8") as f:
        # Write Amharic lines
        for text in df['amharic'].dropna():
            f.write(str(text) + "\n")
        # Write English lines
        for text in df['english'].dropna():
            f.write(str(text) + "\n")
            
    print("\nBeginning SentencePiece training... (This will take a few minutes)")
    # Train BPE using the specified parameters in our implementation plan
    spm.SentencePieceTrainer.train(
        input=text_file_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=0.9995,
        pad_id=0, pad_piece="<pad>",
        unk_id=1, unk_piece="<unk>",
        bos_id=2, bos_piece="<s>",
        eos_id=3, eos_piece="</s>",
        user_defined_symbols=""
    )
    
    print("\nTraining complete! Tokenizer saved to:")
    print(f" - {model_prefix}.model")
    print(f" - {model_prefix}.vocab")
    
    # ----------------------------------------------------
    # Validate the tokenizer logic
    # ----------------------------------------------------
    print("\n--- Validating Tokenizer ---")
    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")
    
    # Testing an Amharic string
    test_am = "ውሻው ሰውን ነከሰ።"
    encoded_am = sp.encode_as_pieces(test_am)
    print(f"\nOriginal (AM): {test_am}")
    print(f"Tokenized pieces: {encoded_am}")
    print(f"IDs: {sp.encode_as_ids(test_am)}")
    
    # Testing an English string
    test_en = "The dog bit the man."
    encoded_en = sp.encode_as_pieces(test_en)
    print(f"\nOriginal (EN): {test_en}")
    print(f"Tokenized pieces: {encoded_en}")
    print(f"IDs: {sp.encode_as_ids(test_en)}")
    
    # Clean up the huge temporary text dump so it doesn't waste user's storage
    if os.path.exists(text_file_path):
        os.remove(text_file_path)
        print(f"\nCleaned up temporary txt file: {text_file_path}")
        
    print("\nPhase 2 Complete! Model ready for Transformer.")

if __name__ == "__main__":
    import sys
    train_csv = "data/processed/train.csv"
    output_dir = "data/processed"
    
    if not os.path.exists(train_csv):
        print(f"Error: {train_csv} not found.")
        sys.exit(1)
        
    train_tokenizer(train_csv, output_dir)
