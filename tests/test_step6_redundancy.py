import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from rare_core.rare_orchestration_service import RareOrchestrator
from rare_entities import AtomicInfo


def write_step5_outputs(output_dir: Path, atomic_infos: list[AtomicInfo], valid_items: dict[str, list[dict]]) -> Path:
    """Write the two files Step 6 reads, in the flat layout run_complete_pipeline.py produces."""
    embeddings_file = output_dir / "step4_embeddings.pkl"
    with open(embeddings_file, "wb") as f:
        pickle.dump(
            {
                "embeddings": np.zeros((len(atomic_infos), 2)),
                "search_documents": [],
                "all_atomic_info": atomic_infos,
            },
            f,
        )
    similarity_data = {
        ai.atomic_info_id: {
            "target_content": ai.content,
            "target_chunk_id": ai.chunk_id,
            "valid_items": valid_items.get(ai.atomic_info_id, []),
        }
        for ai in atomic_infos
    }
    with open(output_dir / "precomputed_similarities.pkl", "wb") as f:
        pickle.dump(similarity_data, f)
    return embeddings_file


def run_step6(orchestrator: RareOrchestrator, embeddings_file: Path, top_k_per_chunk: int | None = None) -> dict:
    return orchestrator.step6_redundancy_detection(
        atomic_info_map={},
        embeddings_data=str(embeddings_file),
        language="English",
        model="gpt5_nano",
        max_workers=2,
        max_similar_items=512,
        top_k_per_chunk=top_k_per_chunk,
    )


TARGET = AtomicInfo(content="NVIDIA revenue was $60.9 billion in fiscal 2024.",
                    chunk_id="report_page001_chunk001", atomic_info_id="report_page001_chunk001_atomic_000")
COMPARISONS = [
    {"atomic_info_id": "report_page002_chunk001_atomic_000", "content": "Fiscal 2024 revenue reached $60.9 billion.",
     "chunk_id": "report_page002_chunk001", "score": 0.9},
    {"atomic_info_id": "report_page003_chunk001_atomic_000", "content": "Data center revenue grew 217%.",
     "chunk_id": "report_page003_chunk001", "score": 0.6},
]


def test_failed_redundancy_judgment_is_not_recorded_as_unique(
    orchestrator: RareOrchestrator, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_call_api(prompt: str, model: str = "", max_retries: int = 3, validate_json: bool = True) -> str:
        raise Exception("rate limited")

    monkeypatch.setattr(orchestrator.llm_client, "call_api", failing_call_api)
    embeddings_file = write_step5_outputs(orchestrator.output_dir, [TARGET], {TARGET.atomic_info_id: COMPARISONS})

    mapping = run_step6(orchestrator, embeddings_file)

    assert TARGET.atomic_info_id not in mapping


def test_incomplete_redundancy_judgment_is_not_recorded_as_unique(
    orchestrator: RareOrchestrator, monkeypatch: pytest.MonkeyPatch
) -> None:
    def short_call_api(prompt: str, model: str = "", max_retries: int = 3, validate_json: bool = True) -> str:
        return json.dumps([{"comparison_id": 1, "reasoning": "different", "is_redundant": False}])

    monkeypatch.setattr(orchestrator.llm_client, "call_api", short_call_api)
    embeddings_file = write_step5_outputs(orchestrator.output_dir, [TARGET], {TARGET.atomic_info_id: COMPARISONS})

    mapping = run_step6(orchestrator, embeddings_file)

    assert TARGET.atomic_info_id not in mapping


def test_complete_redundancy_judgment_records_redundant_items(
    orchestrator: RareOrchestrator, monkeypatch: pytest.MonkeyPatch
) -> None:
    def complete_call_api(prompt: str, model: str = "", max_retries: int = 3, validate_json: bool = True) -> str:
        return json.dumps([
            {"comparison_id": 1, "reasoning": "same fact", "is_redundant": True},
            {"comparison_id": 2, "reasoning": "different", "is_redundant": False},
        ])

    monkeypatch.setattr(orchestrator.llm_client, "call_api", complete_call_api)
    embeddings_file = write_step5_outputs(orchestrator.output_dir, [TARGET], {TARGET.atomic_info_id: COMPARISONS})

    mapping = run_step6(orchestrator, embeddings_file)

    assert mapping[TARGET.atomic_info_id].redundant_items == ["report_page002_chunk001_atomic_000"]


def test_top_k_per_chunk_follows_step4_ranking_in_flat_output_dir(orchestrator: RareOrchestrator) -> None:
    first_extracted = AtomicInfo(content="The report covers fiscal 2024.",
                                 chunk_id="report_page001_chunk001", atomic_info_id="report_page001_chunk001_atomic_000")
    best_ranked = AtomicInfo(content="NVIDIA revenue was $60.9 billion in fiscal 2024.",
                             chunk_id="report_page001_chunk001", atomic_info_id="report_page001_chunk001_atomic_001")
    step4_result = {
        "comparison_modes": {
            "threshold_filtered": {
                "atomic_info_by_chunk": {
                    "report_page001_chunk001": [
                        {"content": first_extracted.content, "atomic_info_id": first_extracted.atomic_info_id},
                        {"content": best_ranked.content, "atomic_info_id": best_ranked.atomic_info_id},
                    ]
                },
                "chunk_rankings": {"report_page001_chunk001": [best_ranked.content, first_extracted.content]},
            }
        }
    }
    with open(orchestrator.output_dir / "step4_selected_atomic_info.json", "w", encoding="utf-8") as f:
        json.dump(step4_result, f)
    embeddings_file = write_step5_outputs(orchestrator.output_dir, [first_extracted, best_ranked], {})

    mapping = run_step6(orchestrator, embeddings_file, top_k_per_chunk=1)

    assert list(mapping) == [best_ranked.atomic_info_id]
