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
    sim = similarity_matrix.cpu().numpy()
    q_ids = query_ids.cpu().numpy()
    g_ids = gallery_ids.cpu().numpy()

    num_query = sim.shape[0]
    cmc = np.zeros(len(topk))
    ap = []

    for i in range(num_query):
        scores = sim[i]
        indices = np.argsort(scores)[::-1]
        matches = (g_ids[indices] == q_ids[i])

        # CMC
        for j, k in enumerate(topk):
            if np.any(matches[:k]):
                cmc[j] += 1

        # mAP
        relevant = matches.nonzero()[0]
        if relevant.size > 0:
            precision_at_i = [(i + 1) / (rank + 1) for i, rank in enumerate(relevant)]
            ap.append(np.mean(precision_at_i))
        else:
            ap.append(0)

    cmc /= num_query
    rank_metrics = {f"Rank-{k}": cmc[i] for i, k in enumerate(topk)}
    map_metric = {"mAP": np.mean(ap)}

    return {**rank_metrics, **map_metric}
