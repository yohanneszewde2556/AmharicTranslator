import pandas as pd

# Load the dataset
df = pd.read_excel("AmharicDataset.xlsx")

# How many rows and columns?
print("Shape:", df.shape)

# What are the column names?
print("Columns:", df.columns.tolist())

# First 5 rows
print(df.head())

# Any missing values?
print("Missing values:")

# Rename columns to simpler names
df.columns = ["amharic", "oromo", "english"]

print(df.head())
print("Shape:", df.shape)
# Average sentence length (in words) for each language
df["amharic_len"] = df["amharic"].apply(lambda x: len(str(x).split()))
df["oromo_len"] = df["oromo"].apply(lambda x: len(str(x).split()))
df["english_len"] = df["english"].apply(lambda x: len(str(x).split()))

print("Average word count per language:")
print("Amharic:", df["amharic_len"].mean().round(2))
print("Oromo:  ", df["oromo_len"].mean().round(2))
print("English:", df["english_len"].mean().round(2))

print("\nShortest sentences:")
print("Amharic:", df["amharic_len"].min())
print("Oromo:  ", df["oromo_len"].min())
print("English:", df["english_len"].min())

print("\nLongest sentences:")
print("Amharic:", df["amharic_len"].max())
print("Oromo:  ", df["oromo_len"].max())
print("English:", df["english_len"].max())



# Check shortest sentences
print("\nSingle word Amharic sentences:")
print(df[df["amharic_len"] == 1][["amharic", "oromo", "english"]].head(10))

print("\nSingle word Oromo sentences:")
print(df[df["oromo_len"] == 1][["amharic", "oromo", "english"]].head(10))