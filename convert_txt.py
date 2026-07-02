import pandas as pd

# Load both files
with open("new_source.am", "r", encoding="utf-8") as f:
    amharic = f.read().splitlines()

with open("new_source.en", "r", encoding="utf-8") as f:
    english = f.read().splitlines()

# Check line counts match
print("Amharic lines:", len(amharic))
print("English lines:", len(english))

# Preview first 5
for i in range(5):
    print(f"AM: {amharic[i]}")
    print(f"EN: {english[i]}")
    print("---")

# Convert to dataframe
df = pd.DataFrame({
    "amharic": amharic,
    "english": english
})

print("Shape:", df.shape)
df.to_excel("new_source.xlsx", index=False)
print("Saved as new_source.xlsx")
# Quality check
df["amharic_len"] = df["amharic"].apply(lambda x: len(str(x).split()))
df["english_len"] = df["english"].apply(lambda x: len(str(x).split()))

print("Average word count:")
print("Amharic:", df["amharic_len"].mean().round(2))
print("English:", df["english_len"].mean().round(2))

print("\nMissing values:")
print(df.isnull().sum())

print("\nShortest sentences:")
print("Amharic:", df["amharic_len"].min())
print("English:", df["english_len"].min())

print("\nLongest sentences:")
print("Amharic:", df["amharic_len"].max())
print("English:", df["english_len"].max())