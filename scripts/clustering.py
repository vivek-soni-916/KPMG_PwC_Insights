# from sentence_transformers import SentenceTransformer
# from sklearn.cluster import KMeans

# def vectorize_and_cluster(df, column="cleaned_content", n_clusters=5):
#     model = SentenceTransformer('all-MiniLM-L6-v2')
#     embeddings = model.encode(df[column].tolist(), show_progress_bar=True)
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#     df["cluster"] = kmeans.fit_predict(embeddings)
#     return df, embeddings


from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

def vectorize_and_cluster(df, column="cleaned_content", n_clusters=5, use_mini_batch=False, visualize=False):
    # Initialize SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Generate embeddings for the text data
    embeddings = model.encode(df[column].tolist(), show_progress_bar=True)
    
    # Choose KMeans or MiniBatchKMeans for better performance on larger datasets
    if use_mini_batch:
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    
    # Apply KMeans clustering to the embeddings
    df["cluster"] = kmeans.fit_predict(embeddings)
    
    # Optionally, visualize the clusters using PCA for dimensionality reduction
    if visualize:
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(embeddings)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=df["cluster"], cmap="viridis", marker='o')
        plt.title("Clustering Visualization")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.colorbar()
        plt.show()
    
    return df, embeddings

# Example usage:
# df = pd.DataFrame({
#     'cleaned_content': ["text 1", "text 2", "text 3", ...]
# })
# df, embeddings = vectorize_and_cluster(df, column="cleaned_content", n_clusters=5, visualize=True)
