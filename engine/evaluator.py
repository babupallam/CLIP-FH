import numpy as np

def evaluate_rank(similarity_matrix, query_ids, gallery_ids, topk=[1, 5, 10]):
    """
    Evaluate using CMC and mAP metrics, excluding distractor gallery IDs.

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
        query_id = q_ids[i]

        # ✅ Step 1: Filter gallery indices that have the same ID as query
        valid_indices = np.where(g_ids == query_id)[0]

        # ✅ Step 2: If there are no matching IDs in the gallery, skip
        if len(valid_indices) == 0:
            continue

        # ✅ Step 3: Get scores only for valid (non-distractor) gallery entries
        valid_scores = scores[valid_indices]
        sorted_idx = np.argsort(valid_scores)[::-1]
        matches = (g_ids[valid_indices[sorted_idx]] == query_id)

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
    return {
        f"Rank-{k}": cmc[i] for i, k in enumerate(topk)
    } | {
        "mAP": np.mean(ap)
    }
