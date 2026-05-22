from app.model import predict_sentiment
from app.cleaner import clean_text
import pandas as pd

# --- Test 1: Single predictions ---
print("=== Single Predictions ===")
tests = [
    "This product is absolutely amazing, best purchase I've ever made!",
    "Complete waste of money. Broke after one day. Never buying again.",
    "It's okay, nothing special but gets the job done.",  # ambiguous case
]

for text in tests:
    cleaned = clean_text(text)
    result = predict_sentiment(cleaned)
    print(f"\nText:  {text[:60]}...")
    print(f"Label: {result['label']}  |  Confidence: {result['score']:.2%}")

# --- Test 2: Run on real CSV data ---
print("\n=== Running on Real Reviews ===")
df = pd.read_csv("data/reviews.csv")
df_sample = df.head(5)  # just first 5 rows

for _, row in df_sample.iterrows():
    cleaned = clean_text(row["review_text"])
    result = predict_sentiment(cleaned)
    true = "POSITIVE" if row["true_label"] == 1 else "NEGATIVE"
    match = "✅" if result["label"] == true else "❌"
    print(f"{match} Predicted: {result['label']:8s} | Actual: {true:8s} | Score: {result['score']:.2%}")