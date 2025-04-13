import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def load_and_compare(path):
    df = pd.read_csv(path)
    kpmg_insights = df[df['source'] == 'KPMG']['business_insights'].dropna()
    pwc_insights = df[df['source'] == 'PWC']['business_insights'].dropna()

    # Combine all insights into keyword list
    def flatten(insights):
        return [word.strip().lower() for ins in insights for word in ins.split() if len(word) > 3]

    kpmg_words = flatten(kpmg_insights)
    pwc_words = flatten(pwc_insights)

    # Count and show comparison
    kpmg_count = Counter(kpmg_words)
    pwc_count = Counter(pwc_words)

    return kpmg_count, pwc_count

def plot_wordcloud(counter, title):
    wc = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(counter)
    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation="bilinear")
    plt.title(title)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    kpmg, pwc = load_and_compare("output/output_with_insights.csv")
    plot_wordcloud(kpmg, "KPMG - Business Insights WordCloud")
    plot_wordcloud(pwc, "PwC - Business Insights WordCloud")
