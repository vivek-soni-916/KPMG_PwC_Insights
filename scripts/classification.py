from transformers import pipeline
import pandas as pd

classifier = pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-large-zeroshot-v1")

CANDIDATE_LABELS = [
    "Finance", "Technology", "Healthcare", "Retail", "Real Estate",
    "Energy", "Telecom", "Manufacturing", "Sustainability", "Policy"
]

def classify_domain(text: str) -> str:
    """Classify the domain of an article using zero-shot classification."""
    # Truncating the text to 512 tokens (max input size for DeBERTa v3)
    truncated_text = text[:512]  # Optionally, consider chunking for longer text
    result = classifier(truncated_text, candidate_labels=CANDIDATE_LABELS)
    return result['labels'][0]

def apply_domain_classification(df: pd.DataFrame) -> pd.DataFrame:
    """Apply classification across dataframe."""
    df['domain'] = df['clean_content'].apply(classify_domain)
    return df
