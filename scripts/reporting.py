import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter
import plotly.express as px

def plot_industry_distribution(df: pd.DataFrame):
    """Visualize industry distribution by firm."""
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='domain', hue='source')
    plt.xticks(rotation=45)
    plt.title("Industry Distribution by Source")
    plt.tight_layout()
    plt.savefig("reports/industry_distribution.png")
    plt.close()

def plot_top_keywords(df: pd.DataFrame, top_n=10):
    """Visualize most frequent keywords across firms."""
    keyword_counts = {"KPMG": Counter(), "PWC": Counter()}
    for _, row in df.iterrows():
        keyword_counts[row['source']].update(row['keywords'])

    for source in keyword_counts:
        common_keywords = keyword_counts[source].most_common(top_n)
        keywords, counts = zip(*common_keywords)
        fig = px.bar(x=keywords, y=counts, title=f"Top {top_n} Keywords - {source}")
        fig.write_image(f"reports/keywords_{source}.png")

def plot_theme_clusters(df: pd.DataFrame):
    """Plot cluster count per source."""
    cluster_data = df.groupby(['source', 'cluster']).size().unstack(fill_value=0)
    cluster_data.plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.title("Cluster Distribution by Source")
    plt.xlabel("Source")
    plt.ylabel("Number of Articles")
    plt.legend(title="Cluster")
    plt.tight_layout()
    plt.savefig("reports/cluster_distribution.png")
    plt.close()

def generate_report(df: pd.DataFrame):
    """Main function to run all reporting visualizations."""
    os.makedirs("reports", exist_ok=True)
    print("[...] Generating report visualizations")

    plot_industry_distribution(df)
    plot_top_keywords(df)
    plot_theme_clusters(df)

    print("[✓] Report assets saved in reports/")

if __name__ == "__main__":
    input_file = "output\output_analyzed_df.csv"
    df = pd.read_csv(input_file)
    generate_report(df)
