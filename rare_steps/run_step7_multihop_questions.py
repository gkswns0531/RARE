#!/usr/bin/env python3

import os
import sys
import argparse
import json
from pathlib import Path


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rare_core.rare_orchestration_service import RareOrchestrator
from rare_entities import RedundancyMapping, STEP7_SETTINGS


def _load_redundancy_mapping(file_path: Path) -> dict:
    with open(file_path, "r", encoding="utf-8") as fp:
        raw_data = json.load(fp)
    mapping = {}
    for key, data in raw_data.items():
        mapping[key] = RedundancyMapping(
            atomic_info_id=data["atomic_info_id"],
            content=data["content"],
            chunk_id=data["chunk_id"],
            redundant_items=data.get("redundant_items", ["unique"]),
            similarity_scores=data.get("similarity_scores", {}),
        )
    return mapping


def _load_chunk_data(chunk_file: Path) -> dict:
    if not chunk_file.exists():
        return {}
    with open(chunk_file, "r", encoding="utf-8") as fp:
        chunks = json.load(fp)
    if isinstance(chunks, dict):
        chunks = list(chunks.values())
    if not isinstance(chunks, list):
        return {}

    chunk_data = {}
    for chunk in chunks:
        file_name = chunk.get("file_name", "")
        page_no = chunk.get("page_no", 0)
        sub_text_index = str(chunk.get("sub_text_index", "1"))
        if not file_name or not page_no:
            continue
        base_name = file_name.replace(".pdf", "")
        chunk_id = f"{base_name}_page{page_no:03d}_chunk{sub_text_index.zfill(3)}"
        chunk_data[chunk_id] = {
            "source_title": base_name,
            "content": chunk.get("content", ""),
            "page_number": page_no,
        }
    return chunk_data


def main():
    parser = argparse.ArgumentParser(description="RARE Step 7: Multi-hop Question Generation")
    parser.add_argument(
        "--input-file",
        default=STEP7_SETTINGS.input_file,
        help="Input file from Step 6 (redundancy mapping)",
    )
    parser.add_argument(
        "--chunk-file",
        default=STEP7_SETTINGS.chunk_file,
        help="Chunk metadata file from Step 2",
    )
    parser.add_argument(
        "--output-dir",
        default=STEP7_SETTINGS.output_dir,
        help="Output directory for generated questions",
    )
    parser.add_argument("--generation-model", default=STEP7_SETTINGS.generation_model, help="Model for question generation")
    parser.add_argument("--filter-model", default=STEP7_SETTINGS.filter_model, help="Model for logical filtering")
    parser.add_argument("--validation-model", default=STEP7_SETTINGS.validation_model, help="Model for quality validation")
    parser.add_argument("--answerability-model", default=STEP7_SETTINGS.answerability_model, help="Model for answerability checks")
    parser.add_argument("--language", default=STEP7_SETTINGS.language, help="Language for processing")
    parser.add_argument("--num-information", type=int, default=STEP7_SETTINGS.num_information, help="Information units per question")
    parser.add_argument("--num-questions", type=int, default=STEP7_SETTINGS.num_questions, help="Candidate questions per sample")
    parser.add_argument("--num-sample", type=int, default=STEP7_SETTINGS.num_samples, help="Target number of samples to generate")
    parser.add_argument("--max-workers", type=int, default=STEP7_SETTINGS.max_workers, help="Maximum parallel workers")
    parser.add_argument("--input-pool-size", type=int, default=STEP7_SETTINGS.input_pool_size, help="Pool size given to the LLM")
    parser.add_argument("--output-questions", type=int, default=STEP7_SETTINGS.output_questions, help="Questions generated per pool")
    parser.add_argument("--legacy-mode", action="store_true", help="Use legacy HotPot pipeline")
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        print("Error: Input file not found")
        return
    
    redundancy_mapping = _load_redundancy_mapping(input_path)
    chunk_data = _load_chunk_data(Path(args.chunk_file))

    orchestrator = RareOrchestrator()
    orchestrator.output_dir = output_dir
    
    result = orchestrator.step7_multihop_question_generation(
        redundancy_mapping=redundancy_mapping,
        chunk_data=chunk_data,
        num_information=args.num_information,
        num_sample=args.num_sample,
        num_questions=args.output_questions,
        input_pool_size=args.input_pool_size,
        generation_model=args.generation_model,
        filter_model=args.filter_model,
        validation_model=args.validation_model,
        answerability_model=args.answerability_model,
        language=args.language,
        max_workers=args.max_workers,
        legacy_mode=args.legacy_mode,
    )

    generated_count = len(result.get("generated_samples", []))
    legacy_count = len(result.get("legacy_questions", []))
    if args.legacy_mode:
        print(f"Legacy pipeline produced {legacy_count} questions. Output directory: {output_dir}")
    else:
        print(f"Generated {generated_count} samples. Output directory: {output_dir}")
        

if __name__ == "__main__":
    main()
