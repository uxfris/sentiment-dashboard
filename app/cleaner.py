import re

def clean_text(text: str) -> str:
    """
    Basic text cleaning for review data
    """
    if not isinstance(text, str):
        return ""

     # Lowercase
    text = text.lower()
    
    # Remove HTML tags (common in scraped reviews)
    text = re.sub(r"<[^>]+>", "", text)
    
    # Remove special characters, keep letters/numbers/spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

if __name__ == "__main__":
     sample = "  This is <b>AMAZING</b>!!! Best product ever... 10/10 👍  "
     print(f"Before:    {sample}")
     print(f"after:     {clean_text(sample)}")