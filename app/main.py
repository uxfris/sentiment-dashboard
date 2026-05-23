from fastapi import FastAPI, HTTPException
from httpx import request
from pydantic import BaseModel
from app.cleaner import clean_text
from app.model import predict_sentiment
import time

app = FastAPI(
    title="Sentiment Analysis API",
    description="Real-time sentiment analysis on product reviews",
    version="1.0.0"
)

# --- Request & Response Schemas ---

class ReviewRequest(BaseModel):
    text: str
    source: str = "manual"   # where the review came from, default "manual"

class SentimentResponse(BaseModel):
    original_text: str
    cleaned_text: str
    label: str           # "POSITIVE" or "NEGATIVE"
    score: float         #  confidence, 0 to 1
    source: str
    processed_at: float  # unix timestamp

# --- Endpoints ---

@app.get("/")
def root():
    return {"status": "ok", "message": "Sentiment API is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict", response_model=SentimentResponse)
def predict(request: ReviewRequest):
    # Validate input isn't empty
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text canonot be empty")

    # Clean and run inference
    cleaned = clean_text(request.text)
    result = predict_sentiment(cleaned)

    return SentimentResponse(
        original_text=request.text,
        cleaned_text=cleaned,
        label=result["label"],
        score=result["score"],
        source=request.source,
        processed_at=time.time()
    )

@app.post("/predict/batch")
def predict_batch(requests: list[ReviewRequest]):
    """
    Accepts a list of reviews and returns predictions for all of them.
    Useful for processing CSV data in chunks.
    """
    if len(requests) > 50:
        raise HTTPException(status_code=400, detail="Batch size cannot exceed 50")

    results = []
    for req in requests:
        cleaned = clean_text(req.text)
        result = predict_sentiment(cleaned)
        results.append({
            "original_text": req.text,
            "label": result["label"],
            "score": result["score"],
            "source": req.source,
            "processed_at": time.time()
        })

    return {"count": len(results), "results": results}