#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

from rare_entities import STEP2_SETTINGS
from rare_core.rare_orchestration_service import RareOrchestrator


def main():
    parser = argparse.ArgumentParser(description="RARE Step 2: Text Chunking")
    parser.add_argument(
        "--input-file",
        default=STEP2_SETTINGS.input_file,
        help="Input file from Step 1",
    )
    parser.add_argument(
        "--output-dir",
        default=STEP2_SETTINGS.output_dir,
        help="Output directory for text chunks",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=STEP2_SETTINGS.chunk_size,
        help="Chunk size in tokens",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=STEP2_SETTINGS.overlap,
        help="Overlap between chunks",
    )

    args = parser.parse_args()

    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_file.exists():
        print("Error: Input file not found")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        parsed_documents = json.load(f)

    orchestrator = RareOrchestrator()
    orchestrator.output_dir = output_dir

    text_chunks = orchestrator.step2_document_chunking(parsed_texts=parsed_documents)

    output_file = output_dir / "text_chunks.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(text_chunks, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(text_chunks)} text chunks")


if __name__ == "__main__":
    main()
