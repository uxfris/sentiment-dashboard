import pandas as pd

df = pd.read_csv("data/reviews.csv")

print("=== Shape ===")
print(df.shape)  # rows, columns

print("\n=== First 3 rows ===")
print(df.head(3))

print("\n=== Label distribution ===")
print(df["true_label"].value_counts())  # how many positive vs negative

print("\n=== Sample review ===")
sample = df.iloc[0]
print(f"Title: {sample['title']}")
print(f"Review: {sample['review_text'][:200]}...")  # first 200 chars
print(f"Label: {'Positive' if sample['true_label'] == 1 else 'Negative'}")

print("\n=== Average review length ===")
df["review_length"] = df["review_text"].str.len()
print(f"{df['review_length'].mean():.0f} characters")