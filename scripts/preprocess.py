import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# # Download resources if not already
# nltk.download('punkt')
# nltk.download('stopwords')

def clean_text(text: str) -> str:
    """
    Clean and normalize the text.
    - Lowercase
    - Remove URLs, HTML tags, numbers, punctuation
    - Remove stopwords
    """
    if pd.isna(text):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove numbers and punctuation
    text = re.sub(r'[\d]', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Join tokens back to string
    cleaned_text = " ".join(tokens)

    return cleaned_text

def preprocess_dataframe(df: pd.DataFrame, text_col: str = 'content') -> pd.DataFrame:
    """
    Apply text cleaning to a specified column in the dataframe.
    Adds a new column 'clean_content'.
    """

    print(f"[INFO] Preprocessing text from column: {text_col}")
    df['clean_content'] = df[text_col].astype(str).apply(clean_text)
    return df
