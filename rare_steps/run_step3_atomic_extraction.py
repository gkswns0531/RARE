#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

from rare_entities import STEP3_SETTINGS
from rare_core.rare_orchestration_service import RareOrchestrator

def extract_doc_title(chunk_id):
    parts = chunk_id.split('_')
    title_parts = []
    for part in parts:
        if part.startswith('page') and part[4:].isdigit():
            break
        title_parts.append(part)
    return '_'.join(title_parts).upper() if title_parts else ''


def main():
    parser = argparse.ArgumentParser(description="RARE Step 3: Atomic Information Extraction")
    parser.add_argument("--input-file", default=STEP3_SETTINGS.input_file,
                        help="Input file from Step 2")
    parser.add_argument("--output-dir", default=STEP3_SETTINGS.output_dir,
                        help="Output directory for atomic information")
    parser.add_argument("--model", default=STEP3_SETTINGS.model, help="Model for extraction")
    parser.add_argument("--workers", type=int, default=STEP3_SETTINGS.workers, help="Number of parallel workers")
    parser.add_argument("--language", default=STEP3_SETTINGS.language, help="Document language")
    parser.add_argument("--limit", type=int, default=STEP3_SETTINGS.limit, help="Limit chunks for testing")

    args = parser.parse_args()

    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_file.exists():
        print("Error: Input file not found")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        text_chunks = json.load(f)

    if args.limit:
        text_chunks = text_chunks[:args.limit]

    orchestrator = RareOrchestrator()
    orchestrator.output_dir = output_dir

    atomic_info_map = orchestrator.step3_atomic_info_extraction(
        chunks=text_chunks, language=args.language, model=args.model, max_workers=args.workers
    )

    serializable_data = {}
    for chunk_id, atomic_info_list in atomic_info_map.items():
        doc_title_prefix = extract_doc_title(chunk_id)
        serializable_data[chunk_id] = []
        for atomic_info in atomic_info_list:
            serializable_data[chunk_id].append({
                'id': atomic_info.atomic_info_id,
                'content': atomic_info.content,
                'chunk_id': atomic_info.chunk_id,
                'document_title': doc_title_prefix
            })

    output_file = output_dir / "atomic_info.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, ensure_ascii=False, indent=2)

    total_extractions = sum(len(info_list) for info_list in atomic_info_map.values())
    cost = orchestrator.llm_client.get_total_cost_usd()
    print(f"Extracted {total_extractions} atomic info items. Cost: ${cost:.4f}")


if __name__ == "__main__":
    main()
