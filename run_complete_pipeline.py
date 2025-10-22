#!/usr/bin/env python3
"""
RARE Step-by-Step Pipeline Runner
Supports running individual steps or complete pipeline
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add RARE to path
sys.path.insert(0, str(Path(__file__).parent))

from rare_core.rare_orchestration_service import run_rare_pipeline
from rare_const import DEFAULT_SIMILARITY_THRESHOLD
from rare_entities import DEFAULT_MODEL, LANGUAGE, DEFAULT_TARGET_COUNT, STEP7_SETTINGS


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )
    
    # Disable verbose external library logs
    for logger_name in ['openai', 'httpx', 'httpcore', 'pdfminer', 'pdfplumber']:
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    logging.getLogger('rare_core.rare_cost_tracker_service').setLevel(logging.WARNING)


def main():
    parser = argparse.ArgumentParser(description="RARE Step-by-Step Pipeline - PDF to Redundancy-Aware Benchmark")
    
    # Input options  
    parser.add_argument("--input", type=str, help="Input PDF file or folder path")
    
    # Step selection
    parser.add_argument("--steps", type=str, nargs='+', 
                       default=["parsing", "chunking", "atomic_info_extraction", "atomic_info_selection", "embedding_similarity", "redundancy_detection", "data_generation"],
                       help="Steps to execute (default: all steps)")
    
    # Pipeline options
    parser.add_argument("--target-count", type=int, default=DEFAULT_TARGET_COUNT,
                       help=f"Number of evaluation items to generate (default: {DEFAULT_TARGET_COUNT})")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                       help=f"LLM model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--language", type=str, default="English",
                       help="Language for generation (default: English)")
    parser.add_argument("--similarity-threshold", type=float, default=DEFAULT_SIMILARITY_THRESHOLD,
                       help=f"Similarity threshold for redundancy detection (default: {DEFAULT_SIMILARITY_THRESHOLD})")
    
    # Chunking options
    parser.add_argument("--chunk-size", type=int, default=512,
                       help="Chunk size in tokens (default: 512)")
    parser.add_argument("--chunk-overlap", type=int, default=0,
                       help="Chunk overlap in tokens (default: 0)")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default="outputs",
                       help="Output directory for intermediate results (default: outputs)")
    
    # Performance options
    parser.add_argument("--max-workers", type=int, default=1024,
                       help="Number of parallel workers for atomic info extraction (default: 1024)")
    parser.add_argument("--top-k-per-doc", type=int, default=None,
                       help="Only process top-k atomic info per document for redundancy detection (default: None = all)")
    parser.add_argument("--top-k-per-chunk", type=int, default=3,
                       help="Step 6: Number of redundancy detection targets per chunk (default: 3)")
    
    # Step 7 options
    parser.add_argument("--step7-generation-model", type=str, default=STEP7_SETTINGS.generation_model,
                        help="Step 7 generation model")
    parser.add_argument("--step7-filter-model", type=str, default=STEP7_SETTINGS.filter_model,
                        help="Step 7 logical filter model")
    parser.add_argument("--step7-validation-model", type=str, default=STEP7_SETTINGS.validation_model,
                        help="Step 7 validation model")
    # parser.add_argument("--step7-paraphrase-model", type=str, default=STEP7_SETTINGS.generation_model,
    #                     help="Step 7 paraphrase model")
    parser.add_argument("--step7-answerability-model", type=str, default=STEP7_SETTINGS.answerability_model,
                        help="Step 7 answerability model")
    parser.add_argument("--step7-num-information", type=int, default=STEP7_SETTINGS.num_information,
                        help="Information per question for Step 7")
    parser.add_argument("--step7-num-questions", type=int, default=STEP7_SETTINGS.num_questions,
                        help="Candidate questions per sample for Step 7")
    parser.add_argument("--step7-num-samples", type=int, default=STEP7_SETTINGS.num_samples,
                        help="Number of samples to generate in Step 7")
    parser.add_argument("--step7-input-pool-size", type=int, default=STEP7_SETTINGS.input_pool_size,
                        help="Input pool size for Step 7 LLM selection")
    parser.add_argument("--step7-max-workers", type=int, default=STEP7_SETTINGS.max_workers,
                        help="Max workers for Step 7")
    # parser.add_argument("--step7-limit", type=int, default=None,
    #                     help="Limit number of redundancy groups for Step 7")

    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    print("RARE Step-by-Step Pipeline")
    print("=" * 60)
    print(f"Steps to execute: {', '.join(args.steps)}")
    if args.input:
        print(f"Input: {args.input}")
    print(f"Output directory: {args.output_dir}")
    
    # Check API key for steps that need it
    api_required_steps = ["atomic_info", "redundancy", "evaluation", "embedding"]
    if any(step in args.steps for step in api_required_steps):
        if not os.getenv("OPENAI_API_KEY"):
            print("OPENAI_API_KEY environment variable not set")
            print("Please set your OpenAI API key:")
            print("export OPENAI_API_KEY='your-api-key'")
            return 1
    
    try:
        # Run pipeline with selected steps
        result = run_rare_pipeline(
            input_path=args.input,
            target_count=args.target_count,
            model=args.model,
            language=args.language,
            similarity_threshold=args.similarity_threshold,
            api_key=os.getenv("OPENAI_API_KEY"),
            output_dir=args.output_dir,
            steps=args.steps,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            max_workers=args.max_workers,
            top_k_per_document=args.top_k_per_doc,
            top_k_per_chunk=args.top_k_per_chunk,
            step7_generation_model=args.step7_generation_model,
            step7_filter_model=args.step7_filter_model,
            step7_validation_model=args.step7_validation_model,
            # step7_paraphrase_model=args.step7_paraphrase_model,
            step7_answerability_model=args.step7_answerability_model,
            step7_num_information=args.step7_num_information,
            step7_num_questions=args.step7_num_questions,
            step7_num_samples=args.step7_num_samples,
            step7_input_pool_size=args.step7_input_pool_size,
            step7_max_workers=args.step7_max_workers,
            # step7_limit=args.step7_limit,
        )
        
        # Final Summary
        print("\nPipeline Completed Successfully")
        print("=" * 60)
        
        print(f"Results Summary:")
        stats = result.statistics
        cost = result.cost_summary
        
        print(f"Results Summary:")
        print(f"- Documents processed: {stats.get('total_documents', 0)}")
        print(f"- Total atomic info: {stats.get('total_atomic_info', 0)}")
        print(f"- Unique atomic info: {stats.get('unique_atomic_info', 0)}")
        print(f"- Redundant atomic info: {stats.get('redundant_atomic_info', 0)}")
        print(f"- Evaluation items generated: {stats.get('generated_evaluation_items', 0)}")
        print(f"- Total cost: ${cost.total_cost_usd:.6f} USD")
        print(f"- Total API calls: {cost.total_calls}")
        print(f"- Results saved in: {args.output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"Pipeline Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
