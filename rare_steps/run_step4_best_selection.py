#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

from rare_entities import STEP4_SETTINGS
from rare_core.rare_orchestration_service import RareOrchestrator
from rare_entities import AtomicInfo


def main():
    parser = argparse.ArgumentParser(description="RARE Step 4: Best Information Selection")
    parser.add_argument("--input-file", default=STEP4_SETTINGS.input_file,
                        help="Input file from Step 3")
    parser.add_argument("--output-dir", default=STEP4_SETTINGS.output_dir,
                        help="Output directory for selected information")
    parser.add_argument("--model", default=STEP4_SETTINGS.model, help="Model for validation")
    parser.add_argument("--workers", type=int, default=STEP4_SETTINGS.workers, help="Number of parallel workers")
    parser.add_argument("--language", default=STEP4_SETTINGS.language, help="Document language")
    parser.add_argument("--limit", type=int, default=STEP4_SETTINGS.limit, help="Limit for testing")
    parser.add_argument("--logical-filtering-model", default=STEP4_SETTINGS.logical_filtering_model,
                        help="Model for completeness filtering")
    parser.add_argument("--skip-logical-filtering", action="store_true",
                        help="Skip logical consistency filtering stage")

    args = parser.parse_args()
    enable_logical_filtering = not args.skip_logical_filtering

    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_file.exists():
        print("Error: Input file not found")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        atomic_info_data = json.load(f)

    chunks_file = Path("outputs/step2/text_chunks.json")
    if not chunks_file.exists():
        print("Error: Chunks file not found")
        return

    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    atomic_info_map = {}
    for chunk_id, atom_list in atomic_info_data.items():
        atomic_info_objects = []
        for atom_dict in atom_list:
            atomic_info = AtomicInfo(
                atomic_info_id=atom_dict['id'],
                content=atom_dict['content'],
                chunk_id=atom_dict['chunk_id']
            )
            atomic_info.document_title = atom_dict.get('document_title', '')
            atomic_info_objects.append(atomic_info)
        atomic_info_map[chunk_id] = atomic_info_objects

    if args.limit:
        info_items = list(atomic_info_map.items())[:args.limit]
        atomic_info_map = dict(info_items)

    orchestrator = RareOrchestrator()
    orchestrator.output_dir = output_dir

    selected_atomic_info_map = orchestrator.step4_best_info_selection(
        atomic_info_map=atomic_info_map,
        chunks=chunks,
        language=args.language,
        model=args.model,
        max_workers=args.workers,
        enable_logical_filtering=enable_logical_filtering,
        logical_filtering_model=args.logical_filtering_model
    )

    output_file = output_dir / "selected_atomic_info.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(selected_atomic_info_map, f, ensure_ascii=False, indent=2)

    # Check if RRF validation succeeded
    validation_ranking = selected_atomic_info_map.get("validation_ranking_by_chunk", {})
    if not validation_ranking:
        print("RRF validation produced no results - all items filtered out during validation phase")

    # Save atomic info format for Step 5
    atomic_info_output_file = output_dir / "atomic_info_filtered.json"
    atomic_info_dict = {}

    for chunk_id, selected_contents in validation_ranking.items():
        original_chunk_atomics = atomic_info_map.get(chunk_id, [])
        chunk_atomic_list = []

        for content in selected_contents:
            for atomic_info in original_chunk_atomics:
                if atomic_info.content.strip() == content.strip():
                    doc_title = "_".join(chunk_id.split("_")[:-2]) if "_" in chunk_id else chunk_id

                    chunk_atomic_list.append({
                        "id": atomic_info.atomic_info_id,
                        "content": atomic_info.content,
                        "chunk_id": atomic_info.chunk_id,
                        "document_title": doc_title
                    })
                    break

        if chunk_atomic_list:
            atomic_info_dict[chunk_id] = chunk_atomic_list

    with open(atomic_info_output_file, 'w', encoding='utf-8') as f:
        json.dump(atomic_info_dict, f, ensure_ascii=False, indent=2)

    original_count = sum(len(info_list) for info_list in atomic_info_map.values())
    selected_count = sum(len(info_list) for info_list in atomic_info_dict.values())

    # Save failed logical consistency cases (only when filtering is enabled)
    if enable_logical_filtering:
        failed_logical_consistency = {}

        # Calculate failed cases by finding items not in the selected results
        for chunk_id, original_atomics in atomic_info_map.items():
            selected_contents = validation_ranking.get(chunk_id, [])
            selected_contents_set = set(content.strip() for content in selected_contents)

            failed_items = []
            for atomic_info in original_atomics:
                if atomic_info.content.strip() not in selected_contents_set:
                    doc_title = "_".join(chunk_id.split("_")[:-2]) if "_" in chunk_id else chunk_id
                    failed_items.append({
                        "id": atomic_info.atomic_info_id,
                        "content": atomic_info.content,
                        "chunk_id": atomic_info.chunk_id,
                        "document_title": doc_title,
                        "failure_reason": "logical_consistency_filtering"
                    })

            if failed_items:
                failed_logical_consistency[chunk_id] = failed_items

        if failed_logical_consistency:
            failed_output_file = output_dir / "atomic_info_failed_logical_consistency.json"
            with open(failed_output_file, 'w', encoding='utf-8') as f:
                json.dump(failed_logical_consistency, f, ensure_ascii=False, indent=2)

    cost = orchestrator.llm_client.get_total_cost_usd()
    selection_rate = (selected_count / original_count * 100) if original_count > 0 else 0

    print(f"Selected {selected_count}/{original_count} atomic info ({selection_rate:.1f}%). Cost: ${cost:.4f}")
    if enable_logical_filtering:
        failed_count = original_count - selected_count
        pass_rate = (selected_count / original_count * 100) if original_count > 0 else 0
        print(f"Logical Consistency Pass Rate: {pass_rate:.1f}% ({selected_count} passed, {failed_count} failed)")
    else:
        print("Logical consistency filtering skipped.")


if __name__ == "__main__":
    main()