from datasets import load_dataset
import pandas as pd

print("Downloading dataset...")

dataset = load_dataset("Helsinki-NLP/opus-100", "am-en")

print("Dataset loaded:", dataset)

# Convert to dataframe
train_data = dataset["train"]
df_new = pd.DataFrame({
    "amharic": [item["translation"]["am"] for item in train_data],
    "english": [item["translation"]["en"] for item in train_data]
})

print("Shape:", df_new.shape)
print(df_new.head())

# Save it
df_new.to_excel("opus100_am_en.xlsx", index=False)
print("Saved as opus100_am_en.xlsx")