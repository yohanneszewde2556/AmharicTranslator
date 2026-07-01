import pandas as pd

# Load the dataset
df = pd.read_excel("opus100_am_en.xlsx")

# Column names
print("Columns:", df.columns.tolist())

# Shape
print("Shape:", df.shape)

# First 5 rows
print(df.head())

# Missing values
print("\nMissing values:")
print(df.isnull().sum())

# Sentence lengths
df["amharic_len"] = df["amharic"].apply(lambda x: len(str(x).split()))
df["english_len"] = df["english"].apply(lambda x: len(str(x).split()))

print("\nAverage word count:")
print("Amharic:", df["amharic_len"].mean().round(2))
print("English:", df["english_len"].mean().round(2))

print("\nShortest sentences:")
print("Amharic:", df["amharic_len"].min())
print("English:", df["english_len"].min())

print("\nLongest sentences:")
print("Amharic:", df["amharic_len"].max())
print("English:", df["english_len"].max())

# Check very short sentences
print("\nSingle word sentences:")
print(df[df["amharic_len"] == 1][["amharic", "english"]].head(10))

# Check very long sentences
print("\nLongest Amharic sentences:")
print(df.nlargest(5, "amharic_len")[["amharic", "english"]])