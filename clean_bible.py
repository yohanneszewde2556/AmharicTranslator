import pandas as pd
import re

df = pd.read_excel("new_source.xlsx")
print("Original shape:", df.shape)

# Add length columns
df["amharic_len"] = df["amharic"].apply(lambda x: len(str(x).split()))
df["english_len"] = df["english"].apply(lambda x: len(str(x).split()))

# Remove empty rows
df = df[df["amharic_len"] > 0]
df = df[df["english_len"] > 0]
print("After removing empty:", df.shape)

# Filter sentence length (between 3 and 100 words)
df = df[df["amharic_len"] >= 3]
df = df[df["english_len"] >= 3]
df = df[df["amharic_len"] <= 100]
df = df[df["english_len"] <= 100]
print("After length filter:", df.shape)

# Remove bracket artifacts like [to be] [from] [be]
df["english"] = df["english"].apply(lambda x: re.sub(r'\[.*?\]', '', str(x)).strip())

# Remove duplicate spaces left after bracket removal
df["english"] = df["english"].apply(lambda x: re.sub(r' +', ' ', x).strip())

# Remove duplicates
df = df.drop_duplicates(subset=["amharic"])
df = df.drop_duplicates(subset=["english"])

# Drop length columns
df = df.drop(columns=["amharic_len", "english_len"])
df = df.reset_index(drop=True)

print("Final clean shape:", df.shape)
print("\nSample after cleaning:")
print(df.head())

# Save
df.to_excel("bible_clean.xlsx", index=False)
print("\nSaved as bible_clean.xlsx")

# Fix missing spaces after bracket removal
df["english"] = df["english"].apply(lambda x: re.sub(r'([a-z])([A-Z])', r'\1 \2', str(x)))
df["english"] = df["english"].apply(lambda x: re.sub(r',([a-zA-Z])', r', \1', str(x)))
df["english"] = df["english"].apply(lambda x: re.sub(r' +', ' ', x).strip())