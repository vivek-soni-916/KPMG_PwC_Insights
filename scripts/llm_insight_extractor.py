import pandas as pd
from transformers import pipeline
from tqdm import tqdm

# Load lightweight open-source LLM
llm = pipeline("text2text-generation", model="google/flan-t5-base", device=-1)

# Optimized prompt template
def make_prompt(summary):
    return (
        "Read the following article summary and extract key business insights. "
        "Focus on:\n\n"
        "- Business strategies or market trends\n"
        "- Emerging technologies or innovations\n"
        "- Regulatory or economic impacts\n"
        "- Industry challenges or opportunities\n"
        "- Any notable recommendations or findings\n\n"
        "Respond in 3–5 concise bullet points using clear business language.\n\n"
        f"Summary:\n{summary}\n\nBusiness Insights:"
    )

# Load data
df = pd.read_csv("cleaned_data/combined_insights_data.csv")

# Apply LLM to extract insights
insights = []
for summary in tqdm(df['summary'].fillna("").tolist(), desc="Extracting Insights"):
    prompt = make_prompt(summary)
    result = llm(prompt, max_new_tokens=150, do_sample=False)[0]['generated_text'].strip()
    insights.append(result)

# Add to DataFrame
df['business_insights'] = insights

# Save or explore
df.to_csv("outputs/with_business_insights.csv", index=False)
print("Business insights extracted and saved to 'outputs/with_business_insights.csv'")
