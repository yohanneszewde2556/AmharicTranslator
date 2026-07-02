import pandas as pd
import re

df = pd.read_csv("new_source2.csv")
df.columns = ["amharic", "english"]
print("Original shape:", df.shape)

# Drop missing
df = df.dropna()
print("After dropping missing:", df.shape)

# Add length columns
df["amharic_len"] = df["amharic"].apply(lambda x: len(str(x).split()))
df["english_len"] = df["english"].apply(lambda x: len(str(x).split()))

# Filter length (between 3 and 100 words)
df = df[df["amharic_len"] >= 3]
df = df[df["english_len"] >= 3]
df = df[df["amharic_len"] <= 100]
df = df[df["english_len"] <= 100]
print("After length filter:", df.shape)

# Remove rows starting with numbers
df = df[~df["amharic"].str.match(r'^\d+')]
df = df[~df["english"].str.match(r'^\d+')]
print("After removing number-start rows:", df.shape)

# Remove rows with Amharic in english column
def has_amharic(text):
    return bool(re.search(r'[\u1200-\u137F]', str(text)))

df = df[~df["english"].apply(has_amharic)]
print("After removing mismatched rows:", df.shape)

# Drop duplicates
df = df.drop_duplicates(subset=["amharic"])
df = df.drop_duplicates(subset=["english"])
print("After removing duplicates:", df.shape)

# Drop length columns
df = df.drop(columns=["amharic_len", "english_len"])
df = df.reset_index(drop=True)

print("\nSample:")
print(df.head(10))

# Save
df.to_excel("csv_source_clean.xlsx", index=False)
print("\nSaved as csv_source_clean.xlsx")
# Fix missing spaces in english
import re
df["english"] = df["english"].apply(lambda x: re.sub(r'([a-z])([A-Z])', r'\1 \2', str(x)))
df["english"] = df["english"].apply(lambda x: re.sub(r',([a-zA-Z])', r', \1', str(x)))
df["english"] = df["english"].apply(lambda x: re.sub(r' +', ' ', x).strip())

# Remove rows containing only dates/numbers/names
df = df[~df["english"].str.match(r'^[\w\s]+ (january|february|march|april|may|june|july|august|september|october|november|december) \d+ \d+', case=False)]
df = df[~df["amharic"].str.contains(r'^\w+ የካቲት|ጥቅምት|ህዳር', regex=True)]

print("After final cleaning:", df.shape)