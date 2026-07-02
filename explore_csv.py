import pandas as pd

df = pd.read_csv("new_source2.csv")

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())