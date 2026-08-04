"""Personalised PageRank"""
import numpy as np


def run(graph, restart, n_out, damping, weights=None):
    out = np.zeros((n_out, restart.shape[1]), dtype=np.float32)
    for b in range(restart.shape[1]):
        pr = graph.personalized_pagerank(
            damping=damping, reset=restart[:, b].astype(np.float64).tolist(), weights=weights)
        out[:, b] = np.asarray(pr[:n_out], dtype=np.float32)
    return out
def rank_of(scores, gold_ids):
    order = np.argsort(-scores)
    rank = np.empty(len(scores), dtype=np.int64)
    rank[order] = np.arange(len(scores))
    hits = [int(rank[g]) for g in gold_ids if 0 <= g < len(scores)]
    return min(hits) + 1 if hits else None
