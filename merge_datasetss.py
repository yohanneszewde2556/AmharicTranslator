import pandas as pd

# Load all datasets
print("Loading datasets...")

original = pd.read_excel("AmharicDataset.xlsx")
original.columns = ["amharic", "oromo", "english"]
original = original[["amharic", "english"]]
print("Original:", original.shape)

opus = pd.read_excel("opus100_clean.xlsx")
print("Opus-100:", opus.shape)

flores = pd.read_excel("flores_am_en.xlsx")
print("FLORES:", flores.shape)

bible = pd.read_excel("bible_clean.xlsx")
print("Bible:", bible.shape)

csv_source = pd.read_excel("csv_source_clean.xlsx")
print("CSV Source:", csv_source.shape)

# Combine all
combined = pd.concat([original, opus, flores, bible, csv_source], ignore_index=True)
print("\nCombined before cleaning:", combined.shape)

# Final clean
combined = combined.dropna()
combined = combined[combined["amharic"].str.strip() != ""]
combined = combined[combined["english"].str.strip() != ""]
combined = combined.drop_duplicates(subset=["amharic"])
combined = combined.drop_duplicates(subset=["english"])
combined = combined.reset_index(drop=True)

print("Final clean shape:", combined.shape)
print("\nSample:")
print(combined.sample(5))

# Save both formats
combined.to_csv("final_dataset.csv", index=False)
print("\nSaved as final_dataset.csv")

combined.to_excel("final_dataset.xlsx", index=False)
print("Saved as final_dataset.xlsx")