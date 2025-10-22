#!/usr/bin/env python3

import os
import sys
import json
import argparse
import random
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rare_core.rare_orchestration_service import RareOrchestrator
from rare_entities import AtomicInfo, STEP5_SETTINGS

def main():
    parser = argparse.ArgumentParser(description="RARE Step 5: Embedding Generation")
    parser.add_argument("--input-file", default=STEP5_SETTINGS.input_file,
                        help="Input file from Step 4")
    parser.add_argument("--output-dir", default=STEP5_SETTINGS.output_dir,
                        help="Output directory for embeddings")
    parser.add_argument("--batch-size", type=int, default=STEP5_SETTINGS.batch_size, help="Batch size for embedding generation")
    parser.add_argument("--limit", type=int, default=STEP5_SETTINGS.limit, help="Limit for testing")
    parser.add_argument("--similarity-threshold", default=STEP5_SETTINGS.similarity_threshold,
                        help="Similarity threshold for step6 precomputation")
    parser.add_argument("--auto-threshold-batch-size", type=int, default=STEP5_SETTINGS.auto_threshold_batch_size,
                        help="Batch size for auto threshold calculation")
    parser.add_argument("--embedding-only", action="store_true",
                        help="Generate embeddings only")
    parser.add_argument("--similarity-only", action="store_true",
                        help="Compute similarities only")
    parser.add_argument("--similarity-batch-size", type=int, default=STEP5_SETTINGS.similarity_batch_size,
                        help="Batch size for similarity computation")
    
    args = parser.parse_args()
    
    if args.embedding_only and args.similarity_only:
        print("Error: embedding-only and similarity-only are mutually exclusive")
        return
    
    similarity_threshold = "auto" if args.similarity_threshold == "auto" else float(args.similarity_threshold)
    
    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_file.exists():
        print("Error: Input file not found")
        return
    
    with open(input_file, 'r', encoding='utf-8') as f:
        atomic_info_data = json.load(f)
    
    if not isinstance(atomic_info_data, dict):
        print("Error: Invalid input format")
        return
    
    if args.limit:
        info_items = list(atomic_info_data.items())
        if len(info_items) > args.limit:
            info_items = random.sample(info_items, args.limit)
        atomic_info_data = dict(info_items)
    
    atomic_info_map = {}
    total_items = 0
    for chunk_id, atomic_list in atomic_info_data.items():
        atomic_info_objects = []
        for atomic_dict in atomic_list:
            atomic_info = AtomicInfo(
                atomic_info_id=atomic_dict.get('id'),
                content=atomic_dict.get('content'),
                chunk_id=atomic_dict.get('chunk_id')
            )
            atomic_info_objects.append(atomic_info)
        
        if atomic_info_objects:
            atomic_info_map[chunk_id] = atomic_info_objects
            total_items += len(atomic_info_objects)
    
    orchestrator = RareOrchestrator()
    orchestrator.output_dir = output_dir
    
    embeddings_data = orchestrator.step5_build_embeddings(
        atomic_info_map=atomic_info_map,
        embedding_batch_size=args.batch_size,
        similarity_threshold=similarity_threshold,
        auto_threshold_batch_size=args.auto_threshold_batch_size,
        embedding_only=args.embedding_only,
        similarity_only=args.similarity_only,
        similarity_batch_size=args.similarity_batch_size
    )
    
    print(f"Generated embeddings for {total_items} atomic items. Cost: ${embeddings_data['cost_info']['total_cost_usd']:.6f}")

if __name__ == "__main__":
    main()