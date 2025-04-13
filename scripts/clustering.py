from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np

def calculate_optimal_clusters(embeddings, max_clusters=10):
    """Determine the optimal number of clusters using the elbow method."""
    wcss = []
    for i in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(embeddings)
        wcss.append(kmeans.inertia_)

    x = np.arange(2, len(wcss) + 2)
    y = np.array(wcss)
    line_vec = np.array([x[-1] - x[0], y[-1] - y[0]])
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
    point_vecs = np.vstack((x - x[0], y - y[0])).T
    distances = np.abs(np.cross(line_vec, point_vecs)) / np.linalg.norm(line_vec)
    optimal_idx = np.argmax(distances)
    return x[optimal_idx]

def vectorize_and_cluster(df, column="clean_content"):
    """Vectorize using SentenceTransformer and cluster using optimal K."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df[column].tolist(), show_progress_bar=True)

    optimal_k = calculate_optimal_clusters(embeddings)
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    df["cluster"] = kmeans.fit_predict(embeddings)

    return df, embeddings, optimal_k
