from transformers import pipeline
import torch

DEVICE = 0 if torch.cuda.is_available() else -1

print(f"Using: {'GPU' if DEVICE==0 else 'CPU'}")

# Load pre-trained sentiment pipeline
# This will download ~270MB on first run — normal
sentiment_pipeline = pipeline(
    task="text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=DEVICE
)

def predict_sentiment(text: str) -> dict:
    """
    Takes a raw text string, returns sentiment labels and confidence score.

    Returns:
    {
        "label": "POSITIVE" or "NEGATIVE"
        "score": float between 0 and 1 (confidence)
    }
    """
    # HuggingFace pipelines truncate automatically at 512 tokens
    result = sentiment_pipeline(text, truncation=True, max_length=512)

    return {
        "label": result[0]["label"],
        "score": round(result[0]["score"], 4)
    }