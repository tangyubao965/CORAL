"""Retrieval metrics for CORAL.

Default metric presets:
- MS MARCO 320K: MRR@3, Hits@1, Hits@10, Hits@100
- NQ 320K: MRR@1, Hits@1, Hits@10, MRR@20
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Sequence, Set, Tuple

Qrels = Mapping[str, Mapping[str, float]]
Ranking = Mapping[str, Sequence[str]]

MSMARCO320K_METRICS = ["MRR@3", "Hits@1", "Hits@10", "Hits@100"]
NQ320K_METRICS = ["MRR@1", "Hits@1", "Hits@10", "MRR@20"]


def metric_preset(dataset: str) -> List[str]:
    name = dataset.lower().replace("-", "").replace("_", "")
    if name in {"msmarco", "msmarco320k"}:
        return MSMARCO320K_METRICS
    if name in {"nq", "nq320k", "naturalquestions"}:
        return NQ320K_METRICS
    raise ValueError(f"Unknown dataset preset: {dataset}. Please pass --metrics explicitly.")


def parse_metric(metric: str) -> Tuple[str, int]:
    if "@" not in metric:
        raise ValueError(f"Metric should be like MRR@3 or Hits@10, got {metric}")
    name, k = metric.split("@", 1)
    return name.strip().lower(), int(k)


def mrr_at_k(ranked_docs: Sequence[str], relevant_docs: Set[str], k: int) -> float:
    if not relevant_docs:
        return 0.0
    for rank, doc_id in enumerate(ranked_docs[:k], start=1):
        if str(doc_id) in relevant_docs:
            return 1.0 / rank
    return 0.0


def hits_at_k(ranked_docs: Sequence[str], relevant_docs: Set[str], k: int) -> float:
    if not relevant_docs:
        return 0.0
    return float(any(str(doc_id) in relevant_docs for doc_id in ranked_docs[:k]))


def recall_at_k(ranked_docs: Sequence[str], relevant_docs: Set[str], k: int) -> float:
    if not relevant_docs:
        return 0.0
    return len(set(map(str, ranked_docs[:k])) & relevant_docs) / len(relevant_docs)


def dcg_at_k(ranked_docs: Sequence[str], rel_map: Mapping[str, float], k: int) -> float:
    score = 0.0
    for rank, doc_id in enumerate(ranked_docs[:k], start=1):
        rel = float(rel_map.get(str(doc_id), 0.0))
        if rel > 0:
            score += (2.0 ** rel - 1.0) / math.log2(rank + 1)
    return score


def ndcg_at_k(ranked_docs: Sequence[str], rel_map: Mapping[str, float], k: int) -> float:
    ideal_rels = sorted([float(v) for v in rel_map.values() if float(v) > 0], reverse=True)
    if not ideal_rels:
        return 0.0
    ideal = 0.0
    for rank, rel in enumerate(ideal_rels[:k], start=1):
        ideal += (2.0 ** rel - 1.0) / math.log2(rank + 1)
    if ideal == 0:
        return 0.0
    return dcg_at_k(ranked_docs, rel_map, k) / ideal


def evaluate_query(
    ranked_docs: Sequence[str],
    rel_map: Mapping[str, float],
    metrics: Sequence[str],
    min_rel: float = 1.0,
) -> Dict[str, float]:
    relevant_docs = {str(doc_id) for doc_id, rel in rel_map.items() if float(rel) >= min_rel}
    ranked_docs = [str(doc_id) for doc_id in ranked_docs]
    scores = {}

    for metric in metrics:
        name, k = parse_metric(metric)
        if name == "mrr":
            scores[metric] = mrr_at_k(ranked_docs, relevant_docs, k)
        elif name in {"hit", "hits", "h"}:
            scores[metric] = hits_at_k(ranked_docs, relevant_docs, k)
        elif name == "recall":
            scores[metric] = recall_at_k(ranked_docs, relevant_docs, k)
        elif name == "ndcg":
            scores[metric] = ndcg_at_k(ranked_docs, rel_map, k)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    return scores


def evaluate_run(
    qrels: Qrels,
    rankings: Ranking,
    metrics: Sequence[str],
    min_rel: float = 1.0,
) -> Dict[str, float]:
    totals = {metric: 0.0 for metric in metrics}
    num_queries = 0

    for qid, rel_map in qrels.items():
        qid = str(qid)
        query_scores = evaluate_query(rankings.get(qid, []), rel_map, metrics, min_rel=min_rel)
        for metric, value in query_scores.items():
            totals[metric] += value
        num_queries += 1

    if num_queries == 0:
        return {metric: 0.0 for metric in metrics}
    return {metric: totals[metric] / num_queries for metric in metrics}


def per_query_scores(
    qrels: Qrels,
    rankings: Ranking,
    metric: str,
    min_rel: float = 1.0,
) -> Dict[str, float]:
    return {
        str(qid): evaluate_query(rankings.get(str(qid), []), rel_map, [metric], min_rel=min_rel)[metric]
        for qid, rel_map in qrels.items()
    }
