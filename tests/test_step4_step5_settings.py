import json
from pathlib import Path

import numpy as np
import pytest

import rare_core.rare_orchestration_service as orchestration_service
from rare_core.rare_orchestration_service import RareOrchestrator, run_rare_pipeline
from rare_entities import AtomicInfo


def test_step4_scores_with_full_document_title(orchestrator: RareOrchestrator, monkeypatch: pytest.MonkeyPatch) -> None:
    seen_titles = []

    def validate(atomic_list: list[AtomicInfo], doc_content: str, doc_title: str, language: str,
                 model: str) -> list[AtomicInfo]:
        seen_titles.append(doc_title)
        return atomic_list

    monkeypatch.setattr(orchestrator, "_validate_separate_chunk_atomic_info", validate)
    chunk_id = "ANNUAL_REPORT_NVIDIA_2024_page085_chunk002"
    atomic_info_map = {chunk_id: [AtomicInfo(content="In 2023, total compensation was $21,356,924.",
                                             chunk_id=chunk_id, atomic_info_id=f"{chunk_id}_atomic_000")]}

    orchestrator.step4_best_info_selection(
        atomic_info_map=atomic_info_map,
        chunks=[],
        language="English",
        model="gpt5_nano",
        max_workers=1,
        enable_logical_filtering=False,
    )

    assert seen_titles == ["ANNUAL_REPORT_NVIDIA_2024"]


def test_pipeline_passes_similarity_threshold_to_step5(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    received = {}

    def step5(self: RareOrchestrator, atomic_info_map: dict, **kwargs: object) -> dict:
        received.update(kwargs)
        return {}

    monkeypatch.setattr(RareOrchestrator, "step5_build_embeddings", step5)
    monkeypatch.setattr(RareOrchestrator, "load_step_result", lambda self, pattern: {})

    run_rare_pipeline(steps=["embedding_similarity"], similarity_threshold=0.7, output_dir=str(tmp_path))

    assert received["similarity_threshold"] == 0.7


class FakeSearchClient:
    """Stands in for the OpenAI embedding client with fixed 2-d embeddings."""

    def __init__(self, documents: list[dict], api_key: str | None = None, batch_size: int = 1024) -> None:
        self.documents = documents
        self.doc_embeddings = np.array([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]])

    def get_usage_stats(self) -> dict:
        return {"total_cost_usd": 0.0, "total_tokens_used": 0, "model": "fake"}


def test_step5_records_the_threshold_it_used(orchestrator: RareOrchestrator, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(orchestration_service, "SearchClient", FakeSearchClient)
    atomic_info_map = {
        f"report_page00{i}_chunk001": [AtomicInfo(content=f"fact {i}", chunk_id=f"report_page00{i}_chunk001",
                                                  atomic_info_id=f"report_page00{i}_chunk001_atomic_000")]
        for i in range(1, 4)
    }

    orchestrator.step5_build_embeddings(atomic_info_map, similarity_threshold="auto")

    metadata = json.loads((orchestrator.output_dir / "step4_embedding_data.json").read_text())
    # Mean of the three pairwise cosine similarities: (0.6 + 0.0 + 0.8) / 3
    assert metadata["similarity_threshold"] == pytest.approx(1.4 / 3)
