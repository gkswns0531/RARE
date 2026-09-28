import json
from types import SimpleNamespace

import pytest

from rare_core.rare_orchestration_service import RareOrchestrator, _Step7ValidationTracker
from rare_entities import RedundancyMapping


def test_gold_chunks_include_chunks_of_redundant_items_outside_mapping(orchestrator: RareOrchestrator) -> None:
    # Step 6 judges only the top-k atomic info per chunk, but a target's redundant items
    # come from every atomic info in Step 5, so they are often missing from the mapping.
    selected_infos = [
        {
            "atomic_info_id": "report_page001_chunk001_atomic_000",
            "chunk_id": "report_page001_chunk001",
            "redundant_items": ["report_page002_chunk004_atomic_007"],
        }
    ]

    gold_chunks = orchestrator._step7_calculate_gold_chunks(selected_infos=selected_infos, redundancy_mapping={})

    assert gold_chunks == [["report_page001_chunk001", "report_page002_chunk004"]]


def test_redundancy_count_includes_redundant_items_outside_mapping(orchestrator: RareOrchestrator) -> None:
    redundancy_mapping = {
        "report_page001_chunk001_atomic_000": RedundancyMapping(
            atomic_info_id="report_page001_chunk001_atomic_000",
            content="NVIDIA revenue was $60.9 billion in fiscal 2024.",
            chunk_id="report_page001_chunk001",
            redundant_items=["report_page002_chunk004_atomic_007"],
        )
    }

    diverse_items = orchestrator._step7_prepare_diverse_pool(redundancy_mapping=redundancy_mapping, num_information=1)

    assert diverse_items[0]["redundancy_count"] == 1


POOL_ITEMS = [
    {
        "atomic_info_id": "report_page001_chunk001_atomic_000",
        "content": "Jen-Hsun Huang is the CEO of NVIDIA.",
        "chunk_id": "report_page001_chunk001",
        "redundant_items": ["unique"],
        "similarity_scores": {},
        "redundancy_count": 0,
    },
    {
        "atomic_info_id": "report_page002_chunk001_atomic_000",
        "content": "In 2023, Jen-Hsun Huang's total compensation was $21,356,924.",
        "chunk_id": "report_page002_chunk001",
        "redundant_items": ["unique"],
        "similarity_scores": {},
        "redundancy_count": 0,
    },
]
CHUNK_DATA = {
    "report_page001_chunk001": {"source_title": "report", "content": "Jen-Hsun Huang is the CEO of NVIDIA."},
    "report_page002_chunk001": {"source_title": "report", "content": "Total compensation was $21,356,924."},
}
ARGS = SimpleNamespace(
    input_pool_size=2,
    num_information=2,
    output_questions=1,
    max_workers=2,
    generation_model="gpt5",
    filter_model="gpt5_nano",
    validation_model="gpt5_nano",
    answerability_model="gpt5_nano",
    language="English",
)


def test_question_is_rejected_when_logical_filtering_cannot_run(
    orchestrator: RareOrchestrator, monkeypatch: pytest.MonkeyPatch
) -> None:
    def call_api(prompt: str, model: str = "", max_retries: int = 3, validate_json: bool = True) -> str:
        if "# Provided Information Pool" in prompt:
            return json.dumps({"generated_questions": [{
                "question_id": 1,
                "selected_items": [1, 2],
                "reasoning": "two hops",
                "generated_question": "How much was NVIDIA's CEO paid in 2023?",
                "generated_answer": "$21,356,924",
            }]})
        if "expert reading comprehension analyst" in prompt:
            return json.dumps([{
                "question_id": 1,
                "reasoning": "both chunks needed",
                "generated_answer": "$21,356,924",
                "answerability_check": "pass",
            }])
        raise Exception("filter model unavailable")

    monkeypatch.setattr(orchestrator.llm_client, "call_api", call_api)
    tracker = _Step7ValidationTracker()

    result = orchestrator._step7_generate_single_sample(
        "llm_sample_001", POOL_ITEMS, {}, CHUNK_DATA, ARGS, tracker
    )

    assert result is None
    assert tracker.stats["contextual_independence"]["failed"] == 1
