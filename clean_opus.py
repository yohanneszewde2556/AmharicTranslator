import pandas as pd
import re

# Load
df = pd.read_excel("opus100_am_en.xlsx")
print("Original shape:", df.shape)

# 1. Drop missing values
df = df.dropna()
print("After dropping missing:", df.shape)

# 2. Add length columns
df["amharic_len"] = df["amharic"].apply(lambda x: len(str(x).split()))
df["english_len"] = df["english"].apply(lambda x: len(str(x).split()))

# 3. Filter out too short (less than 3 words) and too long (more than 50 words)
df = df[df["amharic_len"] >= 3]
df = df[df["english_len"] >= 3]
df = df[df["amharic_len"] <= 50]
df = df[df["english_len"] <= 50]
print("After length filter:", df.shape)

# 4. Remove rows where amharic column contains only latin characters
# (means it's not actually Amharic)
def is_amharic(text):
    amharic_chars = re.findall(r'[\u1200-\u137F]', str(text))
    return len(amharic_chars) > 0

df = df[df["amharic"].apply(is_amharic)]
print("After removing non-Amharic rows:", df.shape)

# 5. Remove rows with underscores in the middle of words (noise)
df = df[~df["amharic"].str.contains(r'\w_\w', regex=True)]
df = df[~df["english"].str.contains(r'\w_\w', regex=True)]
print("After removing underscore noise:", df.shape)

# 6. Drop duplicates
df = df.drop_duplicates(subset=["amharic"])
df = df.drop_duplicates(subset=["english"])
print("After dropping duplicates:", df.shape)

# 7. Reset index and drop length columns
df = df.drop(columns=["amharic_len", "english_len"])
df = df.reset_index(drop=True)

# Preview
print("\nSample of clean data:")
print(df.head(10))

# Save
df.to_excel("opus100_clean.xlsx", index=False)
print("\nSaved as opus100_clean.xlsx")
