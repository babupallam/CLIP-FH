import numpy as np

def evaluate_rank(similarity_matrix, query_ids, gallery_ids, topk=[1, 5, 10]):
    """
    Evaluate using CMC and mAP metrics.

    Args:
        similarity_matrix: cosine similarity [n_query, n_gallery]
        query_ids: ground truth labels for query set
        gallery_ids: ground truth labels for gallery set

    Returns:
        dict of Rank-1, Rank-5, Rank-10, mAP
    """

    # Convert tensors to NumPy arrays
    sim = similarity_matrix.cpu().numpy()
    q_ids = query_ids.cpu().numpy()
    g_ids = gallery_ids.cpu().numpy()

    num_query = sim.shape[0]  # Number of query samples
    cmc = np.zeros(len(topk))  # Cumulative Matching Characteristic at each k
    ap = []  # List to store average precision for each query

    for i in range(num_query):
        scores = sim[i]  # Similarity scores for this query
        indices = np.argsort(scores)[::-1]  # Sort gallery by descending similarity
        matches = (g_ids[indices] == q_ids[i])  # True/False match array

        # CMC
        for j, k in enumerate(topk):
            if np.any(matches[:k]):  # Check if a correct match appears in top-k
                cmc[j] += 1

        # mAP
        relevant = matches.nonzero()[0]  # Indices of correct matches
        if relevant.size > 0:
            # Calculate precision at each correct match rank
            precision_at_i = [(i + 1) / (rank + 1) for i, rank in enumerate(relevant)]
            ap.append(np.mean(precision_at_i))  # Average of all precision values
        else:
            ap.append(0)  # No matches, precision is zero

    # Average CMC and mAP over all queries
    cmc /= num_query
    rank_metrics = {f"Rank-{k}": cmc[i] for i, k in enumerate(topk)}
    map_metric = {"mAP": np.mean(ap)}

    # Merge and return all metrics as a dictionary
    return {**rank_metrics, **map_metric}
