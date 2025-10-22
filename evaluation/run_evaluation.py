#!/usr/bin/env python3
"""RARE Retrieval Evaluation System."""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

DEFAULT_DATASET_NAME = "finance_eval_dataset"

from evaluation.models import (
    search_single,
    search,
    get_simple_name,
    SearchResult,
    set_cache_dir,
)
from evaluation.metrics import evaluate_model, evaluate_query


def load_dataset(input_file: Optional[str] = None):
    """Load evaluation dataset from JSON file."""
    base_dir = Path(__file__).parent
    root_dir = base_dir.parent
    dataset_dir = root_dir / "dataset"
    candidates = []

    if input_file:
        provided = Path(input_file)
        if provided.is_absolute() and provided.exists():
            candidates.append(provided)
        else:
            if not provided.is_absolute():
                candidates.append(base_dir / provided)
                candidates.append(root_dir / provided)
                candidates.append(dataset_dir / provided)
            candidates.append(dataset_dir / provided.name)
    else:
        candidates.append(dataset_dir / f"{DEFAULT_DATASET_NAME}.json")

    dataset_file = None
    for candidate in candidates:
        if candidate.exists():
            dataset_file = candidate
            break

    if not dataset_file:
        checked_paths = ", ".join(str(path) for path in candidates)
        print(f"ERROR: Dataset file not found. Checked: {checked_paths}")
        return None

    with open(dataset_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    return data


def check_model(model_type: str) -> bool:
    """Check if model is available."""
    
    if model_type == "bm25":
        return True
    
    elif "openai" in model_type:
        return os.getenv("OPENAI_API_KEY") is not None
    
    elif model_type == "bge_multivector":
        return True
    
    else:  # HuggingFace
        return True


def print_progress_bar(current: int, total: int, prefix: str = "Progress", 
                      suffix: str = "Complete", length: int = 50):
    """Print a professional progress bar."""
    percent = (current / total) * 100
    filled_length = int(length * current // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    
    sys.stdout.write(f'\r{prefix} |{bar}| {percent:.1f}% {suffix} ({current}/{total})')
    sys.stdout.flush()
    
    if current == total:
        print()


def run_evaluation(models: List[str], dataset: Dict, device: str = "cpu", use_batch: bool = True, batch_size: int = 128,
                   query_type: str = "original", top_k: int = 10):
    """Execute evaluation for specified models."""
    
    corpus = dataset['corpus']
    queries = dataset['queries']
    
    query_mode_display = {
        "original": "Standard",
    }
    
    print(f"Dataset: {len(queries)} queries, {len(corpus)} documents")
    print(f"Query Mode: {query_mode_display.get(query_type, query_type)}")
    print(f"Models: {', '.join([get_simple_name(m) for m in models])}")
    print(f"Batch processing: Enabled")
    print("-" * 60)
    
    results = {}
    results_per_model = {}  # Store search results for group analysis
    
    for model_idx, model_type in enumerate(models):
        model_name = get_simple_name(model_type)
        print(f"\n[{model_idx + 1}/{len(models)}] Evaluating {model_name}")
        
        # Decomposed query evaluation is currently disabled. Any request for decomposed modes falls back to original queries.
        if query_type != "original":
            print(f"WARNING: Query type '{query_type}' is disabled. Falling back to original queries.")
            query_type = "original"
        
        if query_type == "original":
            # Standard query processing
            if use_batch:
                # Batch processing for all models
                query_texts = [query['question'] for query in queries]
                print(f"Processing all {len(query_texts)} queries in batch mode...")
                
                search_results = search(model_type, corpus, query_texts, top_k=top_k, batch_size=batch_size)
                print(f"Batch processing completed for {model_name}")
                
            else:
                # Individual processing (when batch is disabled)
                search_results = []
                
                for i, query in enumerate(queries):
                    # Update progress bar
                    print_progress_bar(
                        i + 1, 
                        len(queries), 
                        prefix=f"{model_name} Evaluation",
                        suffix="queries processed"
                    )
                    
                    query_results = search_single(model_type, corpus, query['question'], top_k=top_k, batch_size=batch_size)
                    search_results.append(query_results)
        
        # Store search results for group analysis
        results_per_model[model_type] = search_results
        
        # Calculate metrics
        metrics = evaluate_model(search_results, queries, top_k)
        results[model_type] = metrics
        
        print(f"{model_name} evaluation completed")
    
    return results, results_per_model


def evaluate_by_group_size(results_per_model: Dict, queries: List[Dict], top_k: int = 10) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Evaluate models by required group size (1-4 groups)."""
    
    # Group queries by required group count
    queries_by_group_size = {1: [], 2: [], 3: [], 4: []}
    for i, query in enumerate(queries):
        group_count = len(query['gold_chunk_groups'])
        if group_count in queries_by_group_size:
            queries_by_group_size[group_count].append((i, query))
    
    # Evaluate each model for each group size
    group_results = {}
    for model_type, search_results in results_per_model.items():
        group_results[model_type] = {}
        
        for group_size in [1, 2, 3, 4]:
            if not queries_by_group_size[group_size]:
                continue
                
            group_metrics = []
            for query_idx, query in queries_by_group_size[group_size]:
                retrieved = [r.chunk_id for r in search_results[query_idx]]
                gold_chunks = query['gold_chunk_ids']
                gold_groups = query['gold_chunk_groups']
                
                metrics = evaluate_query(retrieved, gold_chunks, gold_groups, top_k)
                group_metrics.append(metrics)
            
            # Average metrics for this group size
            if group_metrics:
                avg_metrics = {}
                for key in group_metrics[0].keys():
                    avg_metrics[key] = sum(m[key] for m in group_metrics) / len(group_metrics)
                group_results[model_type][group_size] = avg_metrics
    
    return group_results


def print_results(results: Dict, models_order: List[str], queries: List[Dict], results_per_model: Dict, top_k: int = 10) -> Dict:
    """Print evaluation results in formatted tables with group analysis. Returns group results."""
    
    # Define metrics with coverage first
    all_metrics = [f'coverage@{top_k}', f'perfect_match@{top_k}', f'ndcg@{top_k}', 'mrr']
    metric_names = {f'perfect_match@{top_k}': f'Top@{top_k}', f'coverage@{top_k}': f'Coverage@{top_k}', f'ndcg@{top_k}': f'NDCG@{top_k}', 'mrr': 'MRR'}
    
    print("\n" + "=" * 100)
    print("RARE RETRIEVAL EVALUATION RESULTS")
    print("=" * 100)
    
    def print_single_table(title: str, table_results: Dict[str, Dict[str, float]], query_count: int = None):
        """Print a single results table."""
        print(f"\n{title}")
        if query_count:
            print(f"({query_count} queries)")
        print("-" * 85)
        
        # Header
        print(f"{'Model':<25} ", end="")
        for metric in all_metrics:
            display_name = metric_names.get(metric, metric)
            print(f"{display_name:>14} ", end="")
        print()
        print("-" * 85)
        
        # Rows in input order
        for model_type in models_order:
            if model_type in table_results:
                model_name = get_simple_name(model_type)
                print(f"{model_name:<25} ", end="")
                for metric in all_metrics:
                    value = table_results[model_type].get(metric, 0.0) * 100  # Convert to percentage
                    print(f"{value:>14.2f} ", end="")
                print()
    
    # 1. Overall Results
    print_single_table("Overall Results", results, len(queries))
    
    # 2. Group-based Analysis
    group_results = evaluate_by_group_size(results_per_model, queries, top_k)
    
    # Count queries by group size
    group_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for query in queries:
        group_count = len(query['gold_chunk_groups'])
        if group_count in group_counts:
            group_counts[group_count] += 1
    
    # Print group-specific tables
    for group_size in [1, 2, 3, 4]:
        if group_counts[group_size] > 0:
            group_data = {}
            for model_type in models_order:
                if model_type in group_results and group_size in group_results[model_type]:
                    group_data[model_type] = group_results[model_type][group_size]
            
            if group_data:
                print_single_table(
                    f"{group_size}-Hop Results", 
                    group_data, 
                    group_counts[group_size]
                )
    
    print("\n" + "=" * 100)
    
    return group_results


def print_method_comparison_tables(models: List[str], dataset_name: str, top_k: int = 10):
    """Print comparison tables showing 4 query methods per model."""
    results_dir = Path(__file__).parent / "results"
    
    # Map query types to display names
    method_names = {
        'original': 'Original',
    }

    # Only original queries are currently supported
    method_types = ['original']
    
    # Load all result data (new format: per-model files)
    all_method_data = {}
    
    for method in method_types:
        all_method_data[method] = {'overall': {}, 'hop_results': {}}
        
        # Load data for each model from separate files
        for model in models:
            if method == 'original':
                filename = f"evaluation_results_{model}_top{top_k}_{dataset_name}.json"
            else:
                filename = f"evaluation_results_{method}_{model}_top{top_k}_{dataset_name}.json"
            
            result_file = results_dir / filename
            
            if result_file.exists():
                try:
                    with open(result_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Merge model data into method data
                    if 'overall' in data and model in data['overall']:
                        all_method_data[method]['overall'][model] = data['overall'][model]
                    
                    if 'hop_results' in data and model in data['hop_results']:
                        all_method_data[method]['hop_results'][model] = data['hop_results'][model]
                        
                except Exception as e:
                    print(f"Warning: Error loading {filename}: {e}")
                    continue
    
    if not all_method_data:
        print("No method comparison data available.")
        return
    
    print("\n" + "=" * 100)
    print("METHOD COMPARISON BY MODEL")
    print("=" * 100)
    
    for model_type in models:
        model_name = get_simple_name(model_type)
        
        # Check if model exists in any method
        model_exists = False
        for method_data in all_method_data.values():
            if 'overall' in method_data and model_type in method_data['overall']:
                model_exists = True
                break
        
        if not model_exists:
            continue
        
        print(f"\n{model_name} - Query Method Comparison")
        print("-" * 85)
        
        # Overall Results
        print("\nOverall Results")
        print("-" * 85)
        print(f"{'Method':<15} {f'Coverage@{top_k}':>14} {f'Top@{top_k}':>14} {f'NDCG@{top_k}':>14} {'MRR':>14}")
        print("-" * 85)
        
        # Track best method for overall
        best_coverage = 0
        best_method = None
        
        for method in method_types:
            if method in all_method_data and 'overall' in all_method_data[method] and model_type in all_method_data[method]['overall']:
                metrics = all_method_data[method]['overall'][model_type]
                coverage = metrics.get(f'coverage@{top_k}', 0.0) * 100
                top10 = metrics.get(f'perfect_match@{top_k}', 0.0) * 100  
                ndcg = metrics.get(f'ndcg@{top_k}', 0.0) * 100
                mrr = metrics.get('mrr', 0.0) * 100
                
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_method = method
                
                method_display = method_names[method]
                print(f"{method_display:<15} {coverage:>13.2f} {top10:>13.2f} {ndcg:>13.2f} {mrr:>13.2f}")
            else:
                print(f"{method_names[method]:<15} {'N/A':>13} {'N/A':>13} {'N/A':>13} {'N/A':>13}")
        
        # Hop Results
        for hop in [1, 2, 3, 4]:
            print(f"\n{hop}-Hop Results")
            print("-" * 85)
            print(f"{'Method':<15} {f'Coverage@{top_k}':>14} {f'Top@{top_k}':>14} {f'NDCG@{top_k}':>14} {'MRR':>14}")
            print("-" * 85)
            
            for method in method_types:
                if (method in all_method_data and 
                    'hop_results' in all_method_data[method] and
                    model_type in all_method_data[method]['hop_results'] and
                    str(hop) in all_method_data[method]['hop_results'][model_type]):
                    
                    metrics = all_method_data[method]['hop_results'][model_type][str(hop)]
                    coverage = metrics.get(f'coverage@{top_k}', 0.0) * 100
                    top10 = metrics.get(f'perfect_match@{top_k}', 0.0) * 100  
                    ndcg = metrics.get(f'ndcg@{top_k}', 0.0) * 100
                    mrr = metrics.get('mrr', 0.0) * 100
                    
                    method_display = method_names[method]
                    print(f"{method_display:<15} {coverage:>13.2f} {top10:>13.2f} {ndcg:>13.2f} {mrr:>13.2f}")
                else:
                    print(f"{method_names[method]:<15} {'N/A':>13} {'N/A':>13} {'N/A':>13} {'N/A':>13}")
        
        # Show best method summary
        if best_method and len(all_method_data) > 1 and 'original' in all_method_data:
            original_coverage = all_method_data['original']['overall'][model_type].get(f'coverage@{top_k}', 0.0) * 100
            improvement = best_coverage - original_coverage
            print(f"\nBest method: {method_names[best_method]} (+{improvement:.2f}% vs Original)")
    
    print("\n" + "=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="RARE Retrieval Model Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        python run_evaluation.py --models bm25
        python run_evaluation.py --models bm25 --input-file custom_dataset.json
        python run_evaluation.py --models all --device cuda
        python run_evaluation.py --models all --input-file test.json --cache-dir test_cache --top-k 5
        """)
    
    parser.add_argument("--models", nargs="+", default=["bm25"], 
                       help="Models to evaluate (bm25, openai_large, bge_m3, bge_multivector, qwen3_0.6b, qwen3_4b, qwen3_8b, gemma_embedding, jina_v4, e5_large, e5_mistral_7b, all)")
    parser.add_argument("--device", default="cpu", 
                       help="Device for HuggingFace models (cpu/cuda)")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size for HuggingFace models (default: 16). OpenAI models use fixed batch size of 128.")
    parser.add_argument("--query-type", choices=["original"], default="original",
                       help="Type of queries to use. Decomposed query evaluation is temporarily disabled.")
    parser.add_argument("--input-file", type=str,
                       help="Custom input dataset file (e.g., finance_eval_dataset.json)")
    parser.add_argument("--cache-dir", type=str,
                       help="Custom cache directory path (default: auto-generated from input filename)")
    parser.add_argument("--compare-methods", action="store_true",
                       help="Compare 4 query methods for specified models (no evaluation, just show comparison)")
    parser.add_argument("--top-k", type=int, default=10,
                       help="Number of top results to retrieve for each  query (default: 10)")
    
    args = parser.parse_args()
    
    # Load dataset
    dataset = load_dataset(input_file=args.input_file)
    if not dataset:
        return 1

    dataset_name = Path(args.input_file).stem if args.input_file else DEFAULT_DATASET_NAME
    
    # Set cache directory
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
        if not cache_dir.is_absolute():
            cache_dir = Path(__file__).parent / cache_dir
    else:
        cache_dir = Path(__file__).parent / "cache" / dataset_name
    
    set_cache_dir(str(cache_dir))
    print(f"Using cache directory: {cache_dir}")
    
    # Handle method comparison mode
    if args.compare_methods:
        # Extract dataset name from input file
        if args.input_file:
            dataset_name = Path(args.input_file).stem
        else:
            dataset_name = DEFAULT_DATASET_NAME
        
        # Process model list for comparison
        available_models = ["bm25", "openai_large", "bge_m3", "bge_multivector", "qwen3_0.6b", "qwen3_4b", "qwen3_8b", "gemma_embedding", "jina_v4", "e5_large", "e5_mistral_7b"]
        
        if "all" in args.models:
            compare_models = available_models
        else:
            compare_models = args.models
        
        # Filter to available models (but don't check model availability since we're just reading results)
        valid_compare_models = [model for model in compare_models if model in available_models]
        
        if not valid_compare_models:
            print("ERROR: No valid models specified for comparison.")
            return 1
        
        print(f"METHOD COMPARISON MODE")
        print(f"Dataset: {dataset_name}")
        print(f"Models: {', '.join([get_simple_name(m) for m in valid_compare_models])}")
        print("-" * 60)
        
        # Show method comparison tables
        print_method_comparison_tables(valid_compare_models, dataset_name, args.top_k)
        return 0
    
    # Process model list
    available_models = ["bm25", "openai_large", "bge_m3", "bge_multivector", "qwen3_0.6b", "qwen3_4b", "qwen3_8b", "gemma_embedding", "jina_v4", "e5_large", "e5_mistral_7b"]
    
    if "all" in args.models:
        models = available_models
    else:
        models = args.models
    
    # Filter available models
    valid_models = []
    for model in models:
        if model in available_models and check_model(model):
            valid_models.append(model)
        else:
            print(f"WARNING: {model} unavailable (skipped)")
    
    if not valid_models:
        print("ERROR: No available models found.")
        return 1
    
    # Run evaluation  
    use_batch = True
    
    # Original single query type processing only
    results, results_per_model = run_evaluation(
        valid_models, dataset, args.device, use_batch, args.batch_size,
        query_type=args.query_type, top_k=args.top_k
    )
    
    # Print results (with model order, queries, and per-model results for group analysis)
    group_results = print_results(results, valid_models, dataset['queries'], results_per_model, args.top_k)
    
    # Save results per model (each model gets its own file)
    dataset_suffix = ""
    if dataset_name != DEFAULT_DATASET_NAME:
        dataset_suffix = f"_{dataset_name}"
    
    # Save each model separately
    for model in valid_models:
        if model in results and model in group_results:
            model_combined_results = {
                'overall': {model: results[model]},
                'hop_results': {model: group_results[model]}
            }
            
            output_filename = f"evaluation_results_{model}_top{args.top_k}{dataset_suffix}.json"
            output_file = Path(__file__).parent / "results" / output_filename
            output_file.parent.mkdir(exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(model_combined_results, f, indent=2, ensure_ascii=False)
            
            print(f"Results saved: {output_file}")
    
    # Write combined latest results for visualize_results.py
    combined_overall = {}
    combined_hop_results = {}
    for model in valid_models:
        if model in results and model in group_results:
            combined_overall[model] = results[model]
            combined_hop_results[model] = group_results[model]
    combined_results = {
        'overall': combined_overall,
        'hop_results': combined_hop_results
    }
    combined_file = Path(__file__).parent / "results" / "evaluation_results.json"
    with open(combined_file, 'w', encoding='utf-8') as f:
        json.dump(combined_results, f, indent=2, ensure_ascii=False)
    print(f"Combined results saved: {combined_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())