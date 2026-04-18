#!/usr/bin/env python3
"""
Ranking utility functions for comparing validation method quality against GOLD standard
"""

import math
import logging
from typing import List, Dict, Tuple
from scipy import stats
import numpy as np
from rare_entities import AtomicInfo

logger = logging.getLogger(__name__)


def generate_ranking_from_scores(atomic_info_list: List[AtomicInfo]) -> List[str]:
    """
    Generate ranking based on overall_confidence scores
    
    Returns:
        List[str]: Atomic info content ordered by overall_confidence (highest first)
    """
    if not atomic_info_list:
        return []
    
    # Sort by overall_confidence in descending order (best first)
    sorted_atomic = sorted(
        atomic_info_list, 
        key=lambda x: x.overall_confidence, 
        reverse=True
    )
    
    # Return content as ranking order
    return [atomic.content for atomic in sorted_atomic]


def generate_chunk_based_rankings(atomic_info_list: List[AtomicInfo]) -> Dict[str, List[str]]:
    """
    Generate chunk-based rankings where each chunk gets its own independent ranking
    
    Args:
        atomic_info_list: List of AtomicInfo objects with chunk_id
        
    Returns:
        Dict[str, List[str]]: chunk_id -> ranking (atomic info content ordered by overall_confidence)
    """
    if not atomic_info_list:
        return {}
    
    # Group atomic info by chunk_id
    chunk_groups = {}
    for atomic_info in atomic_info_list:
        chunk_id = atomic_info.chunk_id
        if chunk_id not in chunk_groups:
            chunk_groups[chunk_id] = []
        chunk_groups[chunk_id].append(atomic_info)
    
    # Generate ranking for each chunk independently
    chunk_rankings = {}
    for chunk_id, chunk_atomic_list in chunk_groups.items():
        # Sort by overall_confidence in descending order (best first)
        sorted_atomic = sorted(
            chunk_atomic_list,
            key=lambda x: x.overall_confidence,
            reverse=True
        )
        # Store ranking as list of content strings
        chunk_rankings[chunk_id] = [atomic.content for atomic in sorted_atomic]
    
    logger.info(f"Generated chunk-based rankings for {len(chunk_rankings)} chunks")
    return chunk_rankings


def generate_chunk_based_rankings_by_rank_average(
    atomic_info_list: List[AtomicInfo],
    rrf_k: int = 0,
    print_scores: bool = False,
) -> Dict[str, List[str]]:
    """
    Generate chunk-based rankings using rank-averaged scoring method:
    1. Rank items by each score dimension (validity, completeness, specificity, clarity, questionability)  
    2. Convert ranks to scores (1/position)
    3. Average the rank-based scores 
    4. Generate final ranking from averaged rank-scores
    
    This method normalizes score scales and reduces dimension-specific bias.
    
    Args:
        atomic_info_list: List of AtomicInfo objects with chunk_id and all 5 score dimensions
        
    Returns:
        Dict[str, List[str]]: chunk_id -> ranking (atomic info content ordered by rank-averaged scores)
    """
    if not atomic_info_list:
        return {}
    
    # Group atomic info by chunk_id
    chunk_groups = {}
    for atomic_info in atomic_info_list:
        chunk_id = atomic_info.chunk_id
        if chunk_id not in chunk_groups:
            chunk_groups[chunk_id] = []
        chunk_groups[chunk_id].append(atomic_info)
    
    # Generate ranking for each chunk independently using rank averaging
    chunk_rankings = {}
    for chunk_id, chunk_atomic_list in chunk_groups.items():
        if len(chunk_atomic_list) == 1:
            # Single item - no ranking needed
            chunk_rankings[chunk_id] = [chunk_atomic_list[0].content]
            continue
            
        # Step 1: Create rankings for each score dimension
        validity_ranked = sorted(chunk_atomic_list, key=lambda x: x.validity_score, reverse=True)
        completeness_ranked = sorted(chunk_atomic_list, key=lambda x: x.completeness_score, reverse=True)  
        specificity_ranked = sorted(chunk_atomic_list, key=lambda x: x.specificity_score, reverse=True)
        clarity_ranked = sorted(chunk_atomic_list, key=lambda x: x.clarity_score, reverse=True)
        questionability_ranked = sorted(chunk_atomic_list, key=lambda x: x.questionability_score, reverse=True)
        
        # Step 2: Create position mappings for each dimension
        validity_positions = {item.content: pos + 1 for pos, item in enumerate(validity_ranked)}
        completeness_positions = {item.content: pos + 1 for pos, item in enumerate(completeness_ranked)}
        specificity_positions = {item.content: pos + 1 for pos, item in enumerate(specificity_ranked)}
        clarity_positions = {item.content: pos + 1 for pos, item in enumerate(clarity_ranked)}
        questionability_positions = {item.content: pos + 1 for pos, item in enumerate(questionability_ranked)}
        
        # Step 3: Calculate rank-based scores (RRF: 1/(position + rrf_k)) and average them
        final_scores = {}
        for atomic_info in chunk_atomic_list:
            content = atomic_info.content
            
            # Convert ranks to scores (RRF)
            validity_rank_score = 1.0 / (validity_positions[content] + rrf_k)
            completeness_rank_score = 1.0 / (completeness_positions[content] + rrf_k)
            specificity_rank_score = 1.0 / (specificity_positions[content] + rrf_k)
            clarity_rank_score = 1.0 / (clarity_positions[content] + rrf_k)
            questionability_rank_score = 1.0 / (questionability_positions[content] + rrf_k)
            
            # Average the rank-based scores
            avg_rank_score = (validity_rank_score + completeness_rank_score + specificity_rank_score + clarity_rank_score + questionability_rank_score) / 5.0
            final_scores[content] = avg_rank_score
        
        # Step 4: Create final ranking based on averaged rank-scores
        sorted_by_avg_rank = sorted(
            chunk_atomic_list,
            key=lambda x: final_scores[x.content],
            reverse=True
        )
        
        # Store ranking as list of content strings
        chunk_rankings[chunk_id] = [atomic.content for atomic in sorted_by_avg_rank]

        # Optional: print RRF scores in ranking order (one line per chunk)
        if print_scores:
            scores_sorted = [final_scores[item.content] for item in sorted_by_avg_rank]
            print(" ".join(f"{s:.6f}" for s in scores_sorted))
        
        # Debug logging
        logger.debug(f"Chunk {chunk_id} rank-averaged ranking:")
        for i, atomic in enumerate(sorted_by_avg_rank[:3]):  # Top 3
            content_short = atomic.content[:50] + "..." if len(atomic.content) > 50 else atomic.content
            score = final_scores[atomic.content]
            logger.debug(f"  {i+1}. Score: {score:.3f} - {content_short}")
    
    logger.info(f"Generated rank-averaged chunk-based rankings for {len(chunk_rankings)} chunks")
    return chunk_rankings


def generate_chunk_based_rankings_by_minmax_scaling(atomic_info_list: List[AtomicInfo]) -> Dict[str, List[str]]:
    """
    Generate chunk-based rankings using MinMax scaling method:
    1. Apply MinMax scaling to each score dimension (validity, completeness, specificity, clarity, questionability)
    2. Average the scaled scores
    3. Generate final ranking from averaged scaled scores
    
    This method normalizes score ranges and ensures fair weighting across dimensions.
    
    Args:
        atomic_info_list: List of AtomicInfo objects with chunk_id and all 5 score dimensions
        
    Returns:
        Dict[str, List[str]]: chunk_id -> ranking (atomic info content ordered by minmax-scaled scores)
    """
    if not atomic_info_list:
        return {}
    
    # Group atomic info by chunk_id
    chunk_groups = {}
    for atomic_info in atomic_info_list:
        chunk_id = atomic_info.chunk_id
        if chunk_id not in chunk_groups:
            chunk_groups[chunk_id] = []
        chunk_groups[chunk_id].append(atomic_info)
    
    # Generate ranking for each chunk independently using MinMax scaling
    chunk_rankings = {}
    for chunk_id, chunk_atomic_list in chunk_groups.items():
        if len(chunk_atomic_list) == 1:
            # Single item - no scaling needed
            chunk_rankings[chunk_id] = [chunk_atomic_list[0].content]
            continue
            
        # Step 1: Extract all scores for each dimension
        validity_scores = [item.validity_score for item in chunk_atomic_list]
        completeness_scores = [item.completeness_score for item in chunk_atomic_list]
        specificity_scores = [item.specificity_score for item in chunk_atomic_list]
        clarity_scores = [item.clarity_score for item in chunk_atomic_list]
        questionability_scores = [item.questionability_score for item in chunk_atomic_list]
        
        # Step 2: Calculate min/max for each dimension
        validity_min, validity_max = min(validity_scores), max(validity_scores)
        completeness_min, completeness_max = min(completeness_scores), max(completeness_scores)
        specificity_min, specificity_max = min(specificity_scores), max(specificity_scores)
        clarity_min, clarity_max = min(clarity_scores), max(clarity_scores)
        questionability_min, questionability_max = min(questionability_scores), max(questionability_scores)
        
        # Step 3: Apply MinMax scaling and calculate averaged scaled scores
        final_scores = {}
        for atomic_info in chunk_atomic_list:
            content = atomic_info.content
            
            # MinMax scaling: (score - min) / (max - min), handle division by zero
            if validity_max == validity_min:
                validity_scaled = 1.0  # All items have same score
            else:
                validity_scaled = (atomic_info.validity_score - validity_min) / (validity_max - validity_min)
                
            if completeness_max == completeness_min:
                completeness_scaled = 1.0  # All items have same score
            else:
                completeness_scaled = (atomic_info.completeness_score - completeness_min) / (completeness_max - completeness_min)
                
            if specificity_max == specificity_min:
                specificity_scaled = 1.0  # All items have same score
            else:
                specificity_scaled = (atomic_info.specificity_score - specificity_min) / (specificity_max - specificity_min)
            
            if clarity_max == clarity_min:
                clarity_scaled = 1.0  # All items have same score
            else:
                clarity_scaled = (atomic_info.clarity_score - clarity_min) / (clarity_max - clarity_min)
            
            if questionability_max == questionability_min:
                questionability_scaled = 1.0  # All items have same score
            else:
                questionability_scaled = (atomic_info.questionability_score - questionability_min) / (questionability_max - questionability_min)
            
            # Average the scaled scores
            avg_scaled_score = (validity_scaled + completeness_scaled + specificity_scaled + clarity_scaled + questionability_scaled) / 5.0
            final_scores[content] = avg_scaled_score
        
        # Step 4: Create final ranking based on averaged scaled scores
        sorted_by_scaled_avg = sorted(
            chunk_atomic_list,
            key=lambda x: final_scores[x.content],
            reverse=True
        )
        
        # Store ranking as list of content strings
        chunk_rankings[chunk_id] = [atomic.content for atomic in sorted_by_scaled_avg]
        
        # Debug logging
        logger.debug(f"Chunk {chunk_id} MinMax-scaled ranking:")
        for i, atomic in enumerate(sorted_by_scaled_avg[:3]):  # Top 3
            content_short = atomic.content[:50] + "..." if len(atomic.content) > 50 else atomic.content
            score = final_scores[atomic.content]
            logger.debug(f"  {i+1}. Scaled Score: {score:.3f} - {content_short}")
    
    logger.info(f"Generated MinMax-scaled chunk-based rankings for {len(chunk_rankings)} chunks")
    return chunk_rankings


def comprehensive_chunk_based_comparison(
    method_chunk_rankings: Dict[str, List[str]], 
    gold_chunk_rankings: Dict[str, List[str]], 
    method_name: str = "Method"
) -> Dict[str, float]:
    """
    Perform comprehensive chunk-based comparison between method rankings and gold standard
    Each chunk is compared independently, then results are averaged
    
    Args:
        method_chunk_rankings: Dict of chunk_id -> ranking generated by validation method
        gold_chunk_rankings: Dict of chunk_id -> gold standard ranking  
        method_name: Name of the method for logging
        
    Returns:
        Dict with averaged ranking metrics across all chunks
    """
    if not method_chunk_rankings or not gold_chunk_rankings:
        logger.warning(f"Empty chunk rankings provided for {method_name}")
        return {
            "kendall_tau": 0.0,
            "spearman_correlation": 0.0,
            "top_1_accuracy": 0.0,
            "top_3_accuracy": 0.0,
            "top_5_accuracy": 0.0,
            "ndcg_3": 0.0,
            "ranking_distance": 1.0,
            "overall_quality_score": 0.0,
            "chunks_compared": 0
        }
    
    # Find common chunks between method and gold rankings
    common_chunks = set(method_chunk_rankings.keys()) & set(gold_chunk_rankings.keys())
    
    if not common_chunks:
        logger.warning(f"No common chunks found between method and gold rankings for {method_name}")
        return {
            "kendall_tau": 0.0,
            "spearman_correlation": 0.0,
            "top_1_accuracy": 0.0,
            "top_3_accuracy": 0.0,
            "top_5_accuracy": 0.0,
            "ndcg_3": 0.0,
            "ranking_distance": 1.0,
            "overall_quality_score": 0.0,
            "chunks_compared": 0
        }
    
    # Calculate metrics for each chunk independently
    chunk_results = []
    for chunk_id in common_chunks:
        method_ranking = method_chunk_rankings[chunk_id]
        gold_ranking = gold_chunk_rankings[chunk_id]
        
        # Skip chunks with insufficient data
        if len(method_ranking) < 2 or len(gold_ranking) < 2:
            logger.debug(f"Skipping chunk {chunk_id} - insufficient data")
            continue
            
        # Calculate comprehensive comparison for this chunk
        chunk_comparison = comprehensive_ranking_comparison(
            method_ranking, gold_ranking, f"{method_name}-{chunk_id}"
        )
        chunk_results.append(chunk_comparison)
    
    if not chunk_results:
        logger.warning(f"No valid chunk comparisons completed for {method_name}")
        return {
            "kendall_tau": 0.0,
            "spearman_correlation": 0.0,
            "top_1_accuracy": 0.0,
            "top_3_accuracy": 0.0,
            "top_5_accuracy": 0.0,
            "ndcg_3": 0.0,
            "ranking_distance": 1.0,
            "overall_quality_score": 0.0,
            "chunks_compared": 0
        }
    
    # Average results across all chunks
    avg_results = {
        "kendall_tau": sum(r["kendall_tau"] for r in chunk_results) / len(chunk_results),
        "spearman_correlation": sum(r["spearman_correlation"] for r in chunk_results) / len(chunk_results),
        "top_1_accuracy": sum(r["top_1_accuracy"] for r in chunk_results) / len(chunk_results),
        "top_3_accuracy": sum(r["top_3_accuracy"] for r in chunk_results) / len(chunk_results),
        "top_5_accuracy": sum(r["top_5_accuracy"] for r in chunk_results) / len(chunk_results),
        "ndcg_3": sum(r["ndcg_3"] for r in chunk_results) / len(chunk_results),
        "ranking_distance": sum(r["ranking_distance"] for r in chunk_results) / len(chunk_results),
        "overall_quality_score": sum(r["overall_quality_score"] for r in chunk_results) / len(chunk_results),
        "chunks_compared": len(chunk_results)
    }
    
    logger.info(f"Chunk-based comparison for {method_name}: {len(chunk_results)} chunks compared")
    return avg_results


def calculate_kendall_tau(ranking1: List[str], ranking2: List[str]) -> float:
    """
    Calculate Kendall's Tau correlation between two rankings
    Measures the ordinal association between two measured quantities
    
    Args:
        ranking1: First ranking (list of items in order)
        ranking2: Second ranking (list of items in order)
        
    Returns:
        float: Kendall's Tau correlation coefficient (-1 to 1)
    """
    if not ranking1 or not ranking2:
        return 0.0
        
    # Create position dictionaries
    pos1 = {item: idx for idx, item in enumerate(ranking1)}
    pos2 = {item: idx for idx, item in enumerate(ranking2)}
    
    # Only consider items present in both rankings
    common_items = set(ranking1) & set(ranking2)
    if len(common_items) < 2:
        return 0.0
    
    common_list = list(common_items)
    
    # Create position arrays for common items
    positions1 = [pos1[item] for item in common_list]
    positions2 = [pos2[item] for item in common_list]
    
    try:
        tau, p_value = stats.kendalltau(positions1, positions2)
        return tau if not math.isnan(tau) else 0.0
    except Exception as e:
        logger.warning(f"Error calculating Kendall's Tau: {e}")
        return 0.0


def calculate_spearman_correlation(ranking1: List[str], ranking2: List[str]) -> float:
    """
    Calculate Spearman's rank correlation between two rankings
    Measures how well the relationship between rankings can be described by monotonic function
    
    Args:
        ranking1: First ranking (list of items in order)
        ranking2: Second ranking (list of items in order)
        
    Returns:
        float: Spearman correlation coefficient (-1 to 1)
    """
    if not ranking1 or not ranking2:
        return 0.0
    
    # Create position dictionaries
    pos1 = {item: idx for idx, item in enumerate(ranking1)}
    pos2 = {item: idx for idx, item in enumerate(ranking2)}
    
    # Only consider items present in both rankings
    common_items = set(ranking1) & set(ranking2)
    if len(common_items) < 2:
        return 0.0
    
    common_list = list(common_items)
    
    # Create position arrays for common items
    positions1 = [pos1[item] for item in common_list]
    positions2 = [pos2[item] for item in common_list]
    
    try:
        rho, p_value = stats.spearmanr(positions1, positions2)
        return rho if not math.isnan(rho) else 0.0
    except Exception as e:
        logger.warning(f"Error calculating Spearman correlation: {e}")
        return 0.0


def calculate_top_k_accuracy(ranking1: List[str], ranking2: List[str], k: int = 5) -> float:
    """
    Calculate Top-K accuracy between two rankings
    Measures what fraction of top-K items in ranking1 are also in top-K of ranking2
    
    Args:
        ranking1: First ranking (reference)
        ranking2: Second ranking (comparison)
        k: Number of top items to consider
        
    Returns:
        float: Top-K accuracy (0 to 1)
    """
    if not ranking1 or not ranking2 or k <= 0:
        return 0.0
    
    # Get top-K items from each ranking
    top_k1 = set(ranking1[:min(k, len(ranking1))])
    top_k2 = set(ranking2[:min(k, len(ranking2))])
    
    if not top_k1:
        return 0.0
    
    # Calculate intersection ratio
    intersection = top_k1 & top_k2
    accuracy = len(intersection) / len(top_k1)
    
    return accuracy


def calculate_ndcg(ranking1: List[str], ranking2: List[str], k: int = 3) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG)
    Standard information retrieval metric for ranking quality
    
    Args:
        ranking1: Gold standard ranking (reference)
        ranking2: Method ranking (comparison)
        k: Number of top positions to consider
        
    Returns:
        float: NDCG score (0 to 1)
    """
    if not ranking1 or not ranking2 or k <= 0:
        return 0.0
    
    # Create relevance scores based on gold standard positions
    gold_positions = {item: len(ranking1) - idx for idx, item in enumerate(ranking1)}
    
    def dcg(ranking: List[str]) -> float:
        """Calculate Discounted Cumulative Gain"""
        dcg_score = 0.0
        for i, item in enumerate(ranking[:k]):
            if item in gold_positions:
                # Relevance based on gold standard position (higher position = more relevant)
                relevance = gold_positions[item]
                dcg_score += relevance / math.log2(i + 2)  # i+2 because log2(1) = 0
        return dcg_score
    
    # Calculate DCG for the method ranking
    method_dcg = dcg(ranking2)
    
    # Calculate Ideal DCG (best possible ordering)
    ideal_dcg = dcg(ranking1)
    
    # Normalize
    if ideal_dcg == 0:
        return 0.0
    
    ndcg_score = method_dcg / ideal_dcg
    return min(1.0, ndcg_score)  # Cap at 1.0


def calculate_ranking_distance(ranking1: List[str], ranking2: List[str]) -> float:
    """
    Calculate normalized ranking distance (Spearman's footrule distance)
    Lower values indicate more similar rankings
    
    Args:
        ranking1: First ranking
        ranking2: Second ranking
        
    Returns:
        float: Normalized distance (0 to 1), where 0 = identical rankings
    """
    if not ranking1 or not ranking2:
        return 1.0
    
    # Create position dictionaries
    pos1 = {item: idx for idx, item in enumerate(ranking1)}
    pos2 = {item: idx for idx, item in enumerate(ranking2)}
    
    # Only consider items present in both rankings
    common_items = set(ranking1) & set(ranking2)
    if not common_items:
        return 1.0
    
    # Calculate sum of position differences
    total_distance = sum(abs(pos1[item] - pos2[item]) for item in common_items)
    
    # Maximum possible distance for normalization
    n = len(common_items)
    max_distance = n * (n - 1) / 2 if n > 1 else 1
    
    # Normalize to 0-1 range
    normalized_distance = total_distance / max_distance
    return min(1.0, normalized_distance)


def comprehensive_ranking_comparison(
    method_ranking: List[str], 
    gold_ranking: List[str], 
    method_name: str = "Method"
) -> Dict[str, float]:
    """
    Perform comprehensive comparison between method ranking and gold standard
    
    Args:
        method_ranking: Ranking generated by validation method
        gold_ranking: Gold standard ranking
        method_name: Name of the method for logging
        
    Returns:
        Dict with all ranking metrics
    """
    if not method_ranking or not gold_ranking:
        logger.warning(f"Empty rankings provided for {method_name}")
        return {
            "kendall_tau": 0.0,
            "spearman_correlation": 0.0,
            "top_1_accuracy": 0.0,
            "top_3_accuracy": 0.0,
            "top_5_accuracy": 0.0,
            "ndcg_3": 0.0,
            "ranking_distance": 1.0,
            "overall_quality_score": 0.0
        }
    
    # Calculate all metrics
    kendall_tau = calculate_kendall_tau(method_ranking, gold_ranking)
    spearman_corr = calculate_spearman_correlation(method_ranking, gold_ranking)
    top_1_acc = calculate_top_k_accuracy(gold_ranking, method_ranking, k=1)
    top_3_acc = calculate_top_k_accuracy(gold_ranking, method_ranking, k=3)
    top_5_acc = calculate_top_k_accuracy(gold_ranking, method_ranking, k=5)
    ndcg_3 = calculate_ndcg(gold_ranking, method_ranking, k=3)
    ranking_dist = calculate_ranking_distance(method_ranking, gold_ranking)
    
    # Calculate overall quality score (weighted average)
    # Higher weight on Top-K accuracy and NDCG as they're most relevant for RAG
    overall_score = (
        0.15 * kendall_tau + 0.15 * spearman_corr +  # Correlation measures
        0.30 * top_1_acc + 0.25 * top_3_acc + 0.15 * top_5_acc +  # Top-K accuracy (high weight)
        0.25 * ndcg_3 +                              # NDCG (high weight)
        0.0 * (1 - ranking_dist)                     # Distance (inverted)
    )
    
    # Ensure all values are in proper range
    overall_score = max(0.0, min(1.0, overall_score))
    
    results = {
        "kendall_tau": kendall_tau,
        "spearman_correlation": spearman_corr,
        "top_1_accuracy": top_1_acc,
        "top_3_accuracy": top_3_acc,
        "top_5_accuracy": top_5_acc,
        "ndcg_3": ndcg_3,
        "ranking_distance": ranking_dist,
        "overall_quality_score": overall_score
    }
    
    logger.info(f"[{method_name}] Ranking Quality Metrics: "
               f"Kendall={kendall_tau:.3f}, Spearman={spearman_corr:.3f}, "
               f"Top-3={top_3_acc:.3f}, Top-5={top_5_acc:.3f}, "
               f"NDCG@3={ndcg_3:.3f}, Overall={overall_score:.3f}")
    
    return results


def print_ranking_comparison_table(results: Dict[str, Dict[str, float]]):
    """
    Print a nicely formatted comparison table for multiple methods
    
    Args:
        results: Dict with method names as keys and metric dicts as values
    """
    if not results:
        print("No ranking results to display")
        return
    
    # Get all metric names
    metrics = list(next(iter(results.values())).keys())
    methods = list(results.keys())
    
    print("\n📊 RANKING QUALITY COMPARISON TABLE")
    print("=" * 80)
    
    # Header
    print(f"{'Metric':<20}", end="")
    for method in methods:
        print(f"{method:>15}", end="")
    print()
    
    print("-" * 80)
    
    # Rows for each metric
    metric_names = {
        "kendall_tau": "Kendall's Tau",
        "spearman_correlation": "Spearman Corr",
        "top_1_accuracy": "Top-1 Accuracy",
        "top_3_accuracy": "Top-3 Accuracy",
        "top_5_accuracy": "Top-5 Accuracy", 
        "ndcg_3": "NDCG@3",
        "ranking_distance": "Distance",
        "overall_quality_score": "Overall Score"
    }
    
    for metric in metrics:
        display_name = metric_names.get(metric, metric)
        print(f"{display_name:<20}", end="")
        
        for method in methods:
            value = results[method][metric]
            print(f"{value:>15.3f}", end="")
        print()
    
    print("-" * 80)
    
    # Find best method for each metric
    print("\n🏆 WINNERS BY METRIC:")
    for metric in ["overall_quality_score", "top_1_accuracy", "top_3_accuracy", "top_5_accuracy", "ndcg_3"]:
        if metric in metrics:
            best_method = max(methods, key=lambda m: results[m][metric])
            best_score = results[best_method][metric]
            display_name = metric_names.get(metric, metric)
            print(f"   {display_name}: {best_method} ({best_score:.3f})")
