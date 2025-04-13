# import pandas as pd
# from preprocessing import preprocess_dataframe
# from classification import apply_domain_classification
# from summarization import summarize_dataframe
# from keyword_extraction import apply_keyword_extraction

# def main(input_path: str, output_path: str):
#     # Load data
#     df = pd.read_csv(input_path)

#     # Step-by-step NLP pipeline
#     df = preprocess_dataframe(df, text_col='content')
#     df = apply_domain_classification(df)
#     df = summarize_dataframe(df)
#     df = apply_keyword_extraction(df)

#     # Save output
#     df.to_excel(output_path, index=False)
#     print(f"[✓] Processed file saved to {output_path}")

# if __name__ == "__main__":
#     input_file = "../data/input_combined_df.csv"
#     output_file = "../data/output_analyzed_df.xlsx"
#     main(input_file, output_file)


import pandas as pd
import os
import logging
from preprocessing import preprocess_dataframe
from classification import apply_domain_classification
from summarization import summarize_dataframe
from keyword_extraction import apply_keyword_extraction
from clustering import vectorize_and_cluster  # Import the clustering function

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(input_path: str, output_path: str, n_clusters: int = 5):
    try:
        # Load data
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        df = pd.read_csv(input_path)
        logging.info(f"Loaded data from {input_path}.")

        # Step-by-step NLP pipeline
        logging.info("Starting preprocessing...")
        df = preprocess_dataframe(df, text_col='content')
        logging.info("Preprocessing completed.")

        logging.info("Starting domain classification...")
        df = apply_domain_classification(df)
        logging.info("Domain classification completed.")

        logging.info("Starting summarization...")
        df = summarize_dataframe(df)
        logging.info("Summarization completed.")

        logging.info("Starting keyword extraction...")
        df = apply_keyword_extraction(df)
        logging.info("Keyword extraction completed.")
        
        # # Apply clustering
        # logging.info(f"Starting clustering with {n_clusters} clusters...")
        # df, embeddings = vectorize_and_cluster(df, column="summary", n_clusters=n_clusters)
        # logging.info("Clustering completed.")
        
        # Save output
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        df.to_excel(output_path, index=False)
        logging.info(f"Processed file saved to {output_path}.")

    except Exception as e:
        logging.error(f"Error occurred: {e}")

if __name__ == "__main__":
    input_file = "../data/input_combined_df.csv"
    output_file = "../data/output_analyzed_with_clustering.xlsx"
    main(input_file, output_file)
