from transformers import pipeline
import pandas as pd

# Initialize the summarization pipeline
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def summarize_text(text: str) -> str:
    """Summarize content to extract insights."""
    # Check if text has enough words, if not return the original text
    if len(text.split()) < 50:
        return text
    
    # If the text length is greater than 1024 tokens, split it into chunks
    if len(text) > 1024:
        # Split the text into chunks of size 1024
        text_chunks = [text[i:i+1024] for i in range(0, len(text), 1024)]
        summarized_chunks = [summarizer(chunk, max_length=150, min_length=30, do_sample=False)[0]['summary_text'] for chunk in text_chunks]
        return " ".join(summarized_chunks)  # Join the summarized chunks
    else:
        # Summarize the text directly if it's within the limit
        return summarizer(text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']

def summarize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply summarization on clean content."""
    df['summary'] = df['clean_content'].apply(summarize_text)
    return df
