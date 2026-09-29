"""Тесты Части 2: прогноз потребности и рекомендация закупки"""

import json
import pytest

from spa_assistant import config
from spa_assistant.catalog import load_catalog
from spa_assistant.config import DATASET_PATH
from spa_assistant.forecast import (
    build_model,
    compute_confidence,
    detect_level_shift,
    forecast_demand,
    round_order_qty,
)


@pytest.fixture
def history() -> dict:
    with open(DATASET_PATH, encoding="utf-8") as f:
        return json.load(f)["history"]


@pytest.mark.parametrize(
    ("net", "pack", "moq", "expected"),
    [
        (0.0, 5, 10, 0.0),
        (-3.0, 5, 10, 0.0),
        (3.0, 5, 10, 10.0),
        (10.0, 5, 10, 10.0),
        (10.01, 5, 10, 15.0),
        (12.0, 5, 10, 15.0),
        (101.0, 50, 100, 150.0),
        (4.2, 1, 5, 5.0),
        (33.59, 1, 5, 34.0),
        (150.0, 100, 200, 200.0),
    ],
)
def test_round_order_qty(net: float, pack: float, moq: float, expected: float) -> None:
    assert round_order_qty(net, pack, moq) == expected


def test_gap_is_excluded_not_zero() -> None:
    model = build_model([6.0, 6.0, None, 6.0, 6.0, 6.0, 6.0, 6.0])
    assert model.missing_weeks == [2]
    assert model.intercept == pytest.approx(6.0)


def test_level_shift_detected() -> None:
    points = list(enumerate([3.0, 3.2, 3.1, 3.3, 3.0, 7.0, 7.2, 7.1, 7.3]))
    assert detect_level_shift(points) == 5


def test_trend_is_not_level_shift(history: dict) -> None:
    points = list(enumerate(history["weekly_consumption"]["OIL-001"]))
    assert detect_level_shift(points) is None


def test_single_spike_is_clipped_not_shift() -> None:
    model = build_model([5.0, 5.1, 4.9, 5.0, 5.2, 20.0, 5.0, 4.8, 5.1, 5.0])
    assert model.shift_week is None
    assert model.outliers == 1


def test_confidence_drops_with_short_history() -> None:
    long_conf, _ = compute_confidence(build_model([5.0, 5.2, 4.9, 5.1] * 3))
    short_conf, _ = compute_confidence(build_model([5.0, 5.2, 4.9]))
    assert short_conf < long_conf


def test_confidence_drops_with_volatility() -> None:
    stable, _ = compute_confidence(build_model([5.0, 5.2, 4.9, 5.1] * 3))
    volatile, _ = compute_confidence(build_model([2.0, 8.0, 3.0, 7.0] * 3))
    assert volatile < stable


def test_scrb_uses_new_level_and_needs_clarification(history: dict) -> None:
    result = forecast_demand(history, "SCRB-020", 30, load_catalog()["SCRB-020"])
    assert result["avg_daily_consumption"] == pytest.approx(7.225 / 7, abs=0.01)
    assert result["needs_clarification"] is True
    assert result["deficit_risk"] is True


@pytest.mark.parametrize("sku", ["OIL-001", "SCRB-020", "WRAP-030"])
@pytest.mark.parametrize("horizon", [30, 90])
def test_result_consistency(history: dict, sku: str, horizon: int) -> None:
    item = load_catalog()[sku]
    result = forecast_demand(history, sku, horizon, item)
    qty = result["recommended_qty"]
    assert qty % item["pack_size"] == 0
    assert qty == 0 or qty >= item["min_order_qty"]
    assert result["estimated_cost"] == pytest.approx(qty * item["price"])
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["current_stock"] == history["current_stock"][sku]
    assert result == forecast_demand(history, sku, horizon, item)


def test_demand_multiplier_scales_forecast(history: dict) -> None:
    item = load_catalog()["OIL-001"]
    base = forecast_demand(history, "OIL-001", 30, item)
    boosted = forecast_demand(
        history, "OIL-001", 30, {**item, "demand_multiplier": 1.2}
    )
    assert boosted["forecast_demand"] == pytest.approx(
        base["forecast_demand"] * 1.2, abs=0.01
    )


def test_no_order_when_stock_is_enough(history: dict) -> None:
    result = forecast_demand(history, "WRAP-030", 30, load_catalog()["WRAP-030"])
    assert result["recommended_qty"] == 0.0
    assert result["estimated_cost"] == 0.0


def test_unknown_sku_raises(history: dict) -> None:
    with pytest.raises(ValueError):
        forecast_demand(history, "OIL-002", 30, load_catalog()["OIL-002"])


def test_threshold_is_in_config() -> None:
    assert 0.0 < config.CONFIDENCE_THRESHOLD < 1.0
