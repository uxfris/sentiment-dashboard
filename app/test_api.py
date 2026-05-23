import requests

BASE_URL = "http://localhost:8000"

# Test 1: Health check
response = requests.get(f"{BASE_URL}/health")
print("Health:", response.json())

# Test 2: Single prediction
payload = {"text": "Worst purchase of my life. Total garbage.", "source": "test"}
response = requests.post(f"{BASE_URL}/predict", json=payload)
result = response.json()
print(f"\nSingle Prediction:")
print(f"  Label: {result['label']}")
print(f"  Score: {result['score']:.2%}")

# Test 3: Batch prediction
batch_payload = [
    {"text": "Five stars, would buy again!", "source": "batch_test"},
    {"text": "Stopped working after a week.", "source": "batch_test"},
    {"text": "Average product, average price.", "source": "batch_test"},
]
response = requests.post(f"{BASE_URL}/predict/batch", json=batch_payload)
batch_result = response.json()
print(f"\nBatch Predictions ({batch_result['count']} reviews):")
for r in batch_result["results"]:
    print(f"  {r['label']:8s} ({r['score']:.2%}) — {r['original_text'][:40]}")