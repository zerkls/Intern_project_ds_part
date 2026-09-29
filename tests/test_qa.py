"""Тесты Части 3: разбор вопроса, уточнения и происхождение чисел в ответе"""

import itertools
import json
import re
import pytest

from spa_assistant import config
from spa_assistant.catalog import load_catalog
from spa_assistant.config import DATASET_PATH
from spa_assistant.forecast import forecast_demand
from spa_assistant.qa import (
    answer_question,
    extract_budget_limit,
    extract_multiplier,
    extract_period,
    find_sku,
)


@pytest.fixture(scope="module")
def dataset() -> dict:
    with open(DATASET_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("на три месяца", 90),
        ("в ближайшие 14 дней", 14),
        ("на квартал", 90),
        ("на полгода", 180),
        ("за месяц", 30),
        ("на год", 365),
        ("на 2 недели", 14),
        ("истёк срок годности", None),
        ("Заказать масло", None),
    ],
)
def test_extract_period(text: str, expected: int | None) -> None:
    assert extract_period(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("при лимите 200 тысяч", 200_000.0),
        ("не более 1,5 млн", 1_500_000.0),
        ("в пределах 50000", 50_000.0),
        ("бюджет на квартал", None),
    ],
)
def test_extract_budget_limit(text: str, expected: float | None) -> None:
    assert extract_budget_limit(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("если загрузка вырастет на 20%", 1.2),
        ("если загрузка снизится на 10%", 0.9),
        ("сколько масла уйдёт", None),
    ],
)
def test_extract_multiplier(text: str, expected: float | None) -> None:
    result = extract_multiplier(text)
    assert result == pytest.approx(expected) if expected else result is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("сколько масла закупить", ["OIL-001", "OIL-002"]),
        ("ароматическое масло", ["OIL-002"]),
        ("закупка oil 001", ["OIL-001"]),
        ("альгинатной маски", ["WRAP-030"]),
        ("погода в Сочи", []),
    ],
)
def test_find_sku(text: str, expected: list[str]) -> None:
    assert find_sku(text, load_catalog()) == expected


@pytest.mark.parametrize(
    ("question", "reason"),
    [
        ("Сколько масла закупить на три месяца?", "missing_params"),
        ("Заказать масло", "missing_params"),
        ("Сколько это будет стоить?", "ambiguous_intent"),
        ("Какая погода в Сочи на выходных?", "out_of_domain"),
        ("Когда привезут заказ от поставщика?", "unsupported_topic"),
    ],
)
def test_unknown_instead_of_guess(dataset: dict, question: str, reason: str) -> None:
    parsed, confidence, text = answer_question(question, {"history": dataset["history"]})
    assert parsed["intent"] == "unknown"
    assert parsed["reason"] == reason
    assert confidence == 0.0
    without_codes = re.sub(r"[A-Z]+-\d{3}", " ", text).replace("14 дней", "")
    assert not re.search(r"\d", without_codes)


@pytest.mark.parametrize(
    ("question", "intent", "sku", "period"),
    [
        ("Что нужно заказать в ближайшие 14 дней?", "reorder_list", None, 14),
        ("Какой бюджет закупок на квартал?", "budget", None, 90),
        ("Что закончится до следующей поставки в Сочи?", "deficit_risk", None, None),
        ("Какие партии сгорят в этом месяце?", "expiry_risk", None, 30),
        ("Насколько подорожало ароматическое масло?", "price_dynamics", "OIL-002", None),
        ("Посчитай закупку скраба на полгода при лимите 200 тысяч",
         "forecast_purchase", "SCRB-020", 180),
        ("Сколько стоит закупить всё, что в риске дефицита?", "deficit_risk", None, None),
    ],
)
def test_intent_and_entities(
    dataset: dict, question: str, intent: str, sku: str | None, period: int | None
) -> None:
    parsed, _, _ = answer_question(question, {"history": dataset["history"]})
    assert (parsed["intent"], parsed["sku"], parsed["period_days"]) == (intent, sku, period)


def test_purchase_numbers_come_from_forecast(dataset: dict) -> None:
    history = dataset["history"]
    parsed, _, text = answer_question(
        "Посчитай закупку скраба на полгода при лимите 200 тысяч", {"history": history}
    )
    expected = forecast_demand(history, "SCRB-020", 180, load_catalog()["SCRB-020"])
    assert f"{expected['recommended_qty']:g} кг" in text
    assert f"{expected['estimated_cost']:,.2f}".replace(",", " ") in text
    assert parsed["budget_limit"] == 200_000.0


def _numbers(text: str) -> list[float]:
    text = re.sub(r"\d{4}-\d{2}-\d{2}|[A-Z]+-\d{3}", " ", text)
    raw = re.findall(r"\d{1,3}(?: \d{3})+(?:\.\d+)?|\d+(?:\.\d+)?", text)
    return [float(n.replace(" ", "")) for n in raw]


def _allowed_numbers(history: dict, parsed: dict) -> set[float]:
    catalog = load_catalog()
    allowed: set[float] = {config.CONFIDENCE_THRESHOLD, config.DEFAULT_COVER_DAYS}
    allowed |= set(range(len(catalog) + 1))
    for item in catalog.values():
        allowed |= {v for v in item.values() if isinstance(v, (int, float))}
    for block in ("current_stock", "incoming_qty"):
        allowed |= set(history[block].values())
    for key in ("period_days", "budget_limit"):
        if parsed.get(key):
            allowed.add(parsed[key])
    horizons = {config.DEFAULT_COVER_DAYS, parsed.get("period_days") or 0} - {0}
    costs = []
    for sku in history["weekly_consumption"]:
        for horizon in horizons:
            params = {**catalog[sku], "demand_multiplier": parsed.get("demand_multiplier") or 1}
            r = forecast_demand(history, sku, horizon, params)
            allowed |= {v for v in r.values() if isinstance(v, (int, float))}
            allowed |= set(r["level_shift"] or [])
            costs.append(r["estimated_cost"])
            limit = parsed.get("budget_limit")
            if limit:
                allowed |= {abs(limit - r["estimated_cost"])}
                price, pack = catalog[sku]["price"], catalog[sku]["pack_size"]
                affordable = int(limit / price / pack) * pack
                allowed |= {affordable, round(affordable * price, 2)}
    for n in range(1, len(costs) + 1):
        allowed |= {round(sum(c), 2) for c in itertools.combinations(costs, n)}
    return allowed


def test_every_number_in_answer_is_traceable(dataset: dict) -> None:
    history = dataset["history"]
    for question in dataset["questions"]:
        parsed, confidence, text = answer_question(question, {"history": history})
        answer_part = text.split("Как посчитано:")[0]
        allowed = _allowed_numbers(history, parsed) | {confidence}
        for number in _numbers(answer_part):
            assert any(abs(number - a) < 0.006 for a in allowed), (question, number)


def test_llm_disabled_by_default_runs_without_keys(dataset: dict) -> None:
    parsed, _, _ = answer_question("Какой бюджет закупок на квартал?",
                                   {"history": dataset["history"]})
    assert parsed["parser"] == "rules"


def test_llm_falls_back_to_rules(dataset: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    import spa_assistant.llm as llm

    monkeypatch.setattr(llm, "parse_with_llm", lambda *args: (None, "нет сети"))
    parsed, _, _ = answer_question(
        "Какой бюджет закупок на квартал?",
        {"history": dataset["history"], "use_llm": True},
    )
    assert parsed["parser"] == "rules"
    assert parsed["llm_error"] == "нет сети"
    assert parsed["intent"] == "budget"


LLM_REPLY = {
    "intent": "forecast_purchase", "intent_confidence": 0.9,
    "second_intent": "budget", "second_confidence": 0.1,
    "sku_candidates": ["OIL-001", "XXX-999"], "location": "", "period_days": 90,
    "budget_limit": 0, "demand_multiplier": 1,
}


def test_llm_output_is_validated() -> None:
    from spa_assistant.llm import _to_raw

    raw = _to_raw(LLM_REPLY, load_catalog())
    assert raw["sku_candidates"] == ["OIL-001"]
    assert raw["period_days"] == 90
    assert raw["budget_limit"] is None
    assert raw["scores"]["forecast_purchase"] == 0.9


@pytest.mark.parametrize("intent", ["unknown", "неизвестно", 0, ""])
def test_llm_unknown_intent_gives_zero_scores(intent: object) -> None:
    from spa_assistant.llm import _to_raw

    raw = _to_raw({**LLM_REPLY, "intent": intent, "second_intent": ""}, load_catalog())
    assert sum(raw["scores"].values()) == 0


def test_llm_missing_fields_mean_not_specified() -> None:
    from spa_assistant.llm import _to_raw

    raw = _to_raw({"intent": "budget", "intent_confidence": 1, "second_intent": ""},
                  load_catalog())
    assert raw["scores"]["budget"] == 1
    assert (raw["sku_candidates"], raw["period_days"], raw["location"]) == ([], None, None)


def test_llm_wrong_types_are_rejected() -> None:
    from spa_assistant.llm import _to_raw

    with pytest.raises(ValueError):
        _to_raw({**LLM_REPLY, "period_days": "квартал"}, load_catalog())
    with pytest.raises(ValueError):
        _to_raw({**LLM_REPLY, "sku_candidates": "OIL-001"}, load_catalog())


def test_llm_without_credentials_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    from spa_assistant.llm import parse_with_llm
    from spa_assistant.qa import INTENTS

    monkeypatch.delenv("GIGACHAT_CREDENTIALS", raising=False)
    raw, error = parse_with_llm("Какой бюджет на квартал?", load_catalog(), INTENTS)
    assert raw is None
    assert "GIGACHAT_CREDENTIALS" in error


def _ask_hybrid(
    dataset: dict, monkeypatch: pytest.MonkeyPatch, question: str, reply: dict
) -> dict:
    import spa_assistant.llm as llm

    raw = llm._to_raw(reply, load_catalog())
    monkeypatch.setattr(llm, "parse_with_llm", lambda *args: (raw, None))
    parsed, _, _ = answer_question(question, {"history": dataset["history"], "use_llm": True})
    return parsed


def test_hybrid_does_not_let_llm_guess_sku(
    dataset: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    reply = {"intent": "forecast_purchase", "intent_confidence": 1, "second_intent": "",
             "sku_candidates": ["OIL-001"], "period_days": 90}
    parsed = _ask_hybrid(
        dataset, monkeypatch,
        "Сколько масла закупить на три месяца и сколько это будет стоить?", reply,
    )
    assert parsed["intent"] == "unknown"
    assert parsed["sku_candidates"] == ["OIL-001", "OIL-002"]


def test_hybrid_keeps_confident_rules_intent(
    dataset: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    reply = {"intent": "reorder_list", "intent_confidence": 1, "second_intent": "",
             "location": "Сочи"}
    parsed = _ask_hybrid(
        dataset, monkeypatch, "Что закончится до следующей поставки в Сочи?", reply,
    )
    assert (parsed["parser"], parsed["intent"]) == ("hybrid:rules", "deficit_risk")
    assert parsed["llm_intent"] == "reorder_list"


def test_hybrid_uses_llm_when_rules_do_not_know_phrase(
    dataset: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    question = "Прикинь, во что обойдётся снабжение на квартал"
    rules_parsed, _, _ = answer_question(question, {"history": dataset["history"]})
    assert rules_parsed["reason"] == "out_of_domain"

    reply = {"intent": "budget", "intent_confidence": 0.9, "second_intent": "",
             "period_days": 90}
    parsed = _ask_hybrid(dataset, monkeypatch, question, reply)
    assert (parsed["parser"], parsed["intent"], parsed["status"]) == (
        "hybrid:llm", "budget", "answered",
    )


def test_hybrid_keeps_unsupported_topic_message(
    dataset: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    reply = {"intent": "неизвестно", "intent_confidence": 0, "second_intent": "неизвестно"}
    parsed = _ask_hybrid(dataset, monkeypatch, "Когда привезут заказ от поставщика?", reply)
    assert parsed["reason"] == "unsupported_topic"
