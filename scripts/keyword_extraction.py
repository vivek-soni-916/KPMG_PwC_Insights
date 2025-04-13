from keybert import KeyBERT
import pandas as pd

# Initialize KeyBERT model
kw_model = KeyBERT(model='all-MiniLM-L6-v2')

def extract_keywords(text: str) -> list:
    """Extract top keywords using KeyBERT."""
    if not text or len(text.split()) < 3:  # Skip empty or too short texts
        return []
    
    # Extract the top 5 keywords
    keywords = kw_model.extract_keywords(text, top_n=5, stop_words='english')
    
    # Extract only the keyword texts
    return [kw[0] for kw in keywords]

def apply_keyword_extraction(df: pd.DataFrame) -> pd.DataFrame:
    """Add keyword list column to dataframe."""
    # Apply the keyword extraction function to each row in the 'clean_content' column
    df['keywords'] = df['clean_content'].apply(extract_keywords)
    return df
