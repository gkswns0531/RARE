#!/usr/bin/env python3

import os
import sys
import json
import argparse
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rare_entities import STEP1_SETTINGS
from rare_core.rare_orchestration_service import RareOrchestrator


def main():
    parser = argparse.ArgumentParser(description="RARE Step 1: PDF Parsing")
    parser.add_argument(
        "--input-dir",
        default=STEP1_SETTINGS.input_dir,
        help="Input directory containing PDF files",
    )
    parser.add_argument(
        "--output-dir",
        default=STEP1_SETTINGS.output_dir,
        help="Output directory for parsed documents",
    )
    parser.add_argument(
        "--language",
        default=STEP1_SETTINGS.language,
        help="Document language",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = list(input_dir.glob("*.pdf"))
    if not pdf_files:
        print("Error: No PDF files found")
        return

    orchestrator = RareOrchestrator()
    orchestrator.output_dir = output_dir

    parsed_documents = orchestrator.step1_document_parsing(input_path=str(input_dir))

    output_file = output_dir / "parsed_documents.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(parsed_documents, f, ensure_ascii=False, indent=2)

    print(
        f"Parsed {len(parsed_documents)} documents. Cost: ${orchestrator.llm_client.get_total_cost_usd():.4f}"
    )


if __name__ == "__main__":
    main()
