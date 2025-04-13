import pandas as pd
from preprocessing import preprocess_dataframe
from classification import apply_domain_classification
from summarization import summarize_dataframe
from keyword_extraction import apply_keyword_extraction
from clustering import vectorize_and_cluster

def run_nlp_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the entire NLP pipeline step-by-step.
    """
    df = preprocess_dataframe(df, text_col='clean_content')
    df = apply_domain_classification(df)
    df = summarize_dataframe(df)
    df = apply_keyword_extraction(df)
    df, embeddings, optimal_k = vectorize_and_cluster(df, column='clean_content')
    
    print(f"[✓] Optimal number of clusters determined: {optimal_k}")
    return df

def main(input_path: str, output_path: str):
    """
    Load data, run NLP pipeline, and save results.
    """
    print("[...] Loading data")
    df = pd.read_csv(input_path)

    print("[...] Running NLP pipeline")
    df_processed = run_nlp_pipeline(df)

    print(f"[...] Saving output to {output_path}")
    df_processed.to_csv(output_path, index=False)

    print(f"[✓] All steps completed successfully.")

if __name__ == "__main__":
    input_file = "cleaned_data/combined_insights_data.csv"
    output_file = "output/output_analyzed_df.csv"
    main(input_file, output_file)
