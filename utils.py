
import random
import numpy as np
import torch
import torch.distributed as dist
from sklearn.metrics import ndcg_score

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_mrr_at_k(predictions, labels, k):
    mrr = 0.0
    for pred, label in zip(predictions, labels):
        for i, p in enumerate(pred[:k]):
            if p == label:
                mrr += 1.0 / (i + 1)
                break
    return mrr / len(labels)

def compute_hits_at_k(predictions, labels, k):
    hits = 0
    for pred, label in zip(predictions, labels):
        if label in pred[:k]:
            hits += 1
    return hits / len(labels)

def compute_ndcg_at_k(predictions, labels, k):
    all_labels = []
    all_scores = []
    for pred, label in zip(predictions, labels):
        relevance = [1 if p == label else 0 for p in pred[:k]]
        scores = list(range(k, 0, -1))
        all_labels.append(np.asarray(relevance).reshape(1, -1))
        all_scores.append(np.asarray(scores).reshape(1, -1))
    ndcgs = [ndcg_score(l, s, k=k) for l, s in zip(all_labels, all_scores)]
    return np.mean(ndcgs)
