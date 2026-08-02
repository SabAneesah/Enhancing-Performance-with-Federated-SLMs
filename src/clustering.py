from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def build_client_signature(shard, max_samples=20):
    texts = [str(context) for context, _ in shard[:max_samples]]
    return " ".join(texts) if texts else ""

def cluster_clients(client_shards, num_clusters=2, seed=42):
    num_clients = len(client_shards)
    num_clusters = max(1, min(num_clusters, num_clients))
    signatures = [build_client_signature(s) for s in client_shards]

    if num_clusters <= 1 or all(sig.strip() == "" for sig in signatures):
        assignment = [0] * num_clients
    else:
        try:
            vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
            X = vectorizer.fit_transform(signatures)
            km = KMeans(n_clusters=num_clusters, random_state=seed, n_init=10)
            assignment = km.fit_predict(X).tolist()
            if len(set(assignment)) < 2 and num_clusters > 1:
                raise ValueError("KMeans collapsed every client into a single cluster")
        except Exception as e:
            print(f"  TF-IDF/KMeans clustering failed or degenerated ({e}); "
                  f"falling back to round-robin cluster assignment.")
            assignment = [i % num_clusters for i in range(num_clients)]

    clusters = {}
    for client_idx, cluster_id in enumerate(assignment):
        clusters.setdefault(cluster_id, []).append(client_idx)
    clusters = {new_id: members for new_id, (_, members) in enumerate(sorted(clusters.items()))}
    return clusters