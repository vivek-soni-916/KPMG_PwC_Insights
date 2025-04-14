# NLP Pipeline for Business Insights Extraction

This project implements a comprehensive NLP pipeline to process, classify, summarize, extract keywords, and cluster articles from KPMG and PwC. The primary goal is to extract key business insights and present them in a structured format for further analysis.

## Project Overview

The NLP pipeline follows a series of steps:
1. **Data Preprocessing**: Clean and normalize the text.
2. **Domain Classification**: Classify the domain of the article into predefined categories.
3. **Summarization**: Generate concise summaries for the content.
4. **Keyword Extraction**: Extract key terms that capture the essence of the article.
5. **Clustering**: Group articles based on content similarity using unsupervised learning techniques.

The pipeline utilizes state-of-the-art models from Hugging Face and other libraries for each step.

## Flow Description

1. **Preprocessing**: The content is cleaned by removing URLs, HTML tags, numbers, punctuation, and stopwords. The text is then tokenized and normalized to prepare it for further processing.

2. **Domain Classification**: The domain of each article is classified into predefined categories using a **DeBERTa v3-large model** for zero-shot classification. The following categories are considered:
   - Finance
   - Technology
   - Healthcare
   - Retail
   - Real Estate
   - Energy
   - Telecom
   - Manufacturing
   - Sustainability
   - Policy

3. **Summarization**: The articles are summarized using **DistilBART**, a transformer-based model. If the text length exceeds the model's input limit (1024 tokens), it is split into chunks and each chunk is summarized independently.

4. **Keyword Extraction**: **KeyBERT** is used to extract the top 5 keywords from each article. These keywords provide a quick overview of the most relevant terms associated with the content.

5. **Clustering**: The content is vectorized using **SentenceTransformers**. The embeddings are then clustered using **KMeans** to group similar articles together. The optimal number of clusters is determined using the elbow method.

## LLM Models Used

- **DeBERTa v3-large**: Used for zero-shot domain classification in the "apply_domain_classification" function.
- **DistilBART**: Used for summarizing articles in the "summarize_text" function.
- **KeyBERT**: Used for keyword extraction in the "extract_keywords" function.
- **SentenceTransformers (all-MiniLM-L6-v2)**: Used for generating embeddings and performing clustering in the "vectorize_and_cluster" function.

## Key Themes and Insights

### **KPMG – Key Themes and Insights**
**Top keywords**: industries, renewables, sector, logistics, aviation, hotels, luxury, builders

📝 **Insights**:

- **Industry Diversification**: KPMG's content covers a wide array of industries including aviation, logistics, hospitality, and consumer appliances, highlighting a strong focus on cross-sectoral insights.
  
- **Focus on Renewable Energy**: Keywords like renewables, energy, and emerging suggest KPMG is emphasizing the transition towards green and sustainable energy solutions.
  
- **Real Estate & Hospitality Emphasis**: Several articles delve into residential/commercial real estate and luxury hotel chains, reflecting interest in property development trends and customer experiences.
  
- **Aviation & Logistics Innovation**: Repeated mentions of airlines, aviation, and logistics point to KPMG’s attention on infrastructure and mobility challenges or transformation.

### **PwC – Key Themes and Insights**
**Top keywords**: investments, healthcare, healthtech, ventures, families, generation, agent

📝 **Insights**:

- **Investment-Centric Perspective**: Terms like investments, invest, and ventures show PwC’s reports are focused on investment strategies and portfolio diversification.
  
- **Healthcare & Healthtech Focus**: There's a clear interest in the growth of the healthcare industry, especially tech-driven healthcare solutions (healthtech).
  
- **Wealth Transfer & Generational Planning**: PwC emphasizes topics such as intergenerational wealth, planning for families, and financial literacy across generations.
  
- **Advisory & Personalization**: Keywords related to agent, agentrecommends, and agentassists indicate a trend toward personalized advisory or tech-assisted consulting models.

## Requirements

To run the pipeline, you'll need to install the following libraries:
- `pandas`
- `transformers`
- `sentence-transformers`
- `keybert`
- `sklearn`
- `nltk`
- `torch`

You can install all required libraries using:

```bash
pip install -r requirements.txt
