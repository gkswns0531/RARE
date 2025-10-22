#!/usr/bin/env python3

import os
import sys
import json
import argparse
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rare_core.rare_orchestration_service import RareOrchestrator
from rare_entities import STEP6_SETTINGS, RedundancyMapping

def main():
    parser = argparse.ArgumentParser(description="RARE Step 6: Redundancy Detection")
    parser.add_argument("--embeddings-file", default=STEP6_SETTINGS.embeddings_file,
                        help="Input file from Step 5")
    parser.add_argument("--output-dir", default=STEP6_SETTINGS.output_dir,
                        help="Output directory for redundancy mapping")
    parser.add_argument("--model", default=STEP6_SETTINGS.model, help="LLM model for processing")
    parser.add_argument("--language", default=STEP6_SETTINGS.language, help="Language for processing")
    parser.add_argument("--max-workers", type=int, default=STEP6_SETTINGS.max_workers,
                        help="Maximum number of parallel workers")
    parser.add_argument("--max-similar-items", type=int, default=STEP6_SETTINGS.max_similar_items,
                        help="Maximum number of similar items per atomic info")
    parser.add_argument("--top-k-per-chunk", type=int, default=STEP6_SETTINGS.top_k_per_chunk,
                        help="Top-k atomic info per chunk to use for redundancy detection (default: 3)")
    parser.add_argument("--max-chunks", type=int, default=STEP6_SETTINGS.max_chunks,
                        help="Maximum number of chunks to process for redundancy detection")
    
    args = parser.parse_args()
    
    embeddings_file = Path(args.embeddings_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not embeddings_file.exists():
        print("Error: Embeddings file not found")
        return

    orchestrator = RareOrchestrator()
    orchestrator.output_dir = output_dir
    
    redundancy_mapping = orchestrator.step6_redundancy_detection(
        atomic_info_map={},  # Loaded from embeddings pkl file
        embeddings_data=str(embeddings_file),
        language=args.language,
        model=args.model,
        max_workers=args.max_workers,
        max_similar_items=args.max_similar_items,
        top_k_per_chunk=args.top_k_per_chunk,
        max_chunks=args.max_chunks
    )
    
    output_file = output_dir / "redundancy_mapping.json"
    
    redundancy_dict = {}
    for key, mapping in redundancy_mapping.items():
        redundancy_dict[key] = {
            "atomic_info_id": mapping.atomic_info_id,
            "content": mapping.content,
            "chunk_id": mapping.chunk_id,
            "redundant_items": mapping.redundant_items,
            "similarity_scores": mapping.similarity_scores
        }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(redundancy_dict, f, ensure_ascii=False, indent=2)
    
    unique_count = sum(1 for mapping in redundancy_mapping.values() 
                      if mapping.redundant_items == ["unique"])
    redundant_count = len(redundancy_mapping) - unique_count
    
    print(f"Processed {len(redundancy_mapping)} items. Unique: {unique_count}, Redundant: {redundant_count} ({redundant_count/len(redundancy_mapping)*100:.1f}%)")

if __name__ == "__main__":
    main()