from datasets import load_dataset
import pandas as pd
import os

print("Downloading assets...")
dataset = load_dataset("SetFit/amazon_polarity", split="test")  # ~400k rows, we use 'test' split (smaller)

#Convert to Pandas DataFrame
df = pd.DataFrame(dataset)

df = df[["title", "text", "label"]].rename(columns={
    "title": "title",
    "text": "review_text",
    "label": "true_label" # 0 = negative, 1 = positive
})

# Ensure the output directory exists
os.makedirs("data", exist_ok=True)

# Save a sample of 1000 rows so it's fast to work with
df_sample = df.sample(n=1000, random_state=42).reset_index(drop=True)
df_sample.to_csv("data/reviews.csv", index=False)

print(f"Saved {len(df_sample)} reviews to data/reviews.csv")
print(df_sample.head())