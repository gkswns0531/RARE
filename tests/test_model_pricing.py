import pytest

from rare_const import (
    DEFAULT_MODEL,
    DEFAULT_STEP7_ANSWERABILITY_MODEL,
    DEFAULT_STEP7_FILTER_MODEL,
    DEFAULT_STEP7_GENERATION_MODEL,
    DEFAULT_STEP7_VALIDATION_MODEL,
    MODEL_PRICING,
)
from rare_core.rare_cost_tracker_service import CostTracker

DEFAULT_MODELS = [
    DEFAULT_MODEL,
    DEFAULT_STEP7_GENERATION_MODEL,
    DEFAULT_STEP7_FILTER_MODEL,
    DEFAULT_STEP7_VALIDATION_MODEL,
    DEFAULT_STEP7_ANSWERABILITY_MODEL,
]


@pytest.mark.parametrize("model", DEFAULT_MODELS)
def test_default_models_have_their_own_pricing(model: str) -> None:
    # A model missing from MODEL_PRICING is silently billed as gpt5_nano, under-reporting cost
    assert model in MODEL_PRICING


@pytest.mark.parametrize("model", DEFAULT_MODELS)
def test_default_models_are_not_scheduled_for_removal(model: str) -> None:
    # gpt-5, gpt-5-mini and gpt-5-nano are removed from the API on 2026-12-11
    assert model not in {"gpt5", "gpt5_mini", "gpt5_nano", "gpt-5", "gpt-5-mini", "gpt-5-nano"}


@pytest.mark.parametrize(
    ("model", "input_per_1m", "output_per_1m"),
    [
        # Standard short-context rates from developers.openai.com, checked 2026-09-28
        ("gpt-6-astra", 10.0, 50.0),
        ("gpt-6-sol", 2.0, 10.0),
        ("gpt-6-luna", 0.1, 0.5),
        ("gpt-5.6-sol", 4.0, 20.0),
        ("gpt-5.6-terra", 2.0, 12.0),
        ("gpt-5.6-luna", 0.2, 1.2),
        ("gpt-5", 1.25, 10.0),
        ("gpt-5-mini", 0.25, 2.0),
        ("gpt-5-nano", 0.05, 0.4),
        ("gpt-4.1", 2.0, 8.0),
        ("gpt-4.1-mini", 0.4, 1.6),
        ("gpt-4.1-nano", 0.1, 0.4),
        ("gpt-4o", 2.5, 10.0),
        ("gpt-4o-mini", 0.15, 0.6),
    ],
)
def test_cost_matches_official_rates(model: str, input_per_1m: float, output_per_1m: float) -> None:
    tracker = CostTracker()

    assert tracker.calculate_cost(1_000_000, 0, model) == pytest.approx(input_per_1m)
    assert tracker.calculate_cost(0, 1_000_000, model) == pytest.approx(output_per_1m)
