"""
Group-based evaluation metrics for retrieval systems.
"""

import numpy as np
from typing import List, Dict, Any


def calculate_group_coverage(retrieved: List[str], gold_groups: List[List[str]], k: int) -> float:
    """Calculate proportion of information groups covered in top-k results."""
    if not gold_groups:
        return 0.0
    
    retrieved_k = set(retrieved[:k])
    covered_groups = 0
    
    for group in gold_groups:
        if any(chunk_id in retrieved_k for chunk_id in group):
            covered_groups += 1
    
    return covered_groups / len(gold_groups)


def calculate_perfect_match(retrieved: List[str], gold_groups: List[List[str]], k: int) -> float:
    """Binary metric: 1.0 if all groups covered, 0.0 otherwise."""
    coverage = calculate_group_coverage(retrieved, gold_groups, k)
    return 1.0 if coverage == 1.0 else 0.0


def calculate_group_mrr(retrieved: List[str], gold_groups: List[List[str]]) -> float:
    """Calculate Mean Reciprocal Rank across all groups."""
    if not gold_groups:
        return 0.0
    
    group_ranks = []
    
    for group in gold_groups:
        for rank, chunk_id in enumerate(retrieved, 1):
            if chunk_id in group:
                group_ranks.append(1.0 / rank)
                break
        else:
            group_ranks.append(0.0)
    
    return sum(group_ranks) / len(gold_groups)


def calculate_group_ndcg(retrieved: List[str], gold_groups: List[List[str]], k: int) -> float:
    """Calculate Group NDCG at k."""
    if not gold_groups:
        return 0.0
    
    retrieved_k = retrieved[:k]
    
    dcg = 0.0
    matched_groups = set()
    
    for i, chunk_id in enumerate(retrieved_k):
        for group_idx, group in enumerate(gold_groups):
            if chunk_id in group and group_idx not in matched_groups:
                dcg += 1.0 / np.log2(i + 2)
                matched_groups.add(group_idx)
                break
    
    idcg = 0.0
    for i in range(min(len(gold_groups), k)):
        idcg += 1.0 / np.log2(i + 2)
    
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_query(retrieved: List[str], gold_chunks: List[str], gold_groups: List[List[str]], k: int = 10) -> Dict[str, Any]:
    """Evaluate retrieval performance for a single query."""
    return {
        f'coverage@{k}': calculate_group_coverage(retrieved, gold_groups, k),
        f'perfect_match@{k}': calculate_perfect_match(retrieved, gold_groups, k),
        f'ndcg@{k}': calculate_group_ndcg(retrieved, gold_groups, k),
        'mrr': calculate_group_mrr(retrieved, gold_groups)
    }


def evaluate_model(search_results: List[List], queries: List[Dict[str, Any]], k: int = 10) -> Dict[str, float]:
    """Evaluate overall model performance across all queries."""
    all_metrics = []
    
    for i, query in enumerate(queries):
        retrieved = [r.chunk_id for r in search_results[i]]
        gold_chunks = query['gold_chunk_ids']
        gold_groups = query['gold_chunk_groups']
        
        metrics = evaluate_query(retrieved, gold_chunks, gold_groups, k)
        all_metrics.append(metrics)
    
    avg_metrics = {}
    if all_metrics:
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    return avg_metrics
