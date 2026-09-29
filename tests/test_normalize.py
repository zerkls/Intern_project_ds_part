"""Тесты Части 1: разбор дат, единиц, операций, SKU"""

import json
import pytest

from spa_assistant.catalog import load_catalog
from spa_assistant.config import DATASET_PATH
from spa_assistant.normalize import (
    extract_date,
    extract_operation,
    extract_quantity,
    match_sku_by_name,
    normalize_movement,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("05.03.2026", "2026-03-05"),
        ("07.03.26", "2026-03-07"),
        ("1 марта 2026 г.", "2026-03-01"),
        ("12 мая 2026", "2026-05-12"),
        ("03/06/26", "2026-06-03"),
        ("2026-03-08", "2026-03-08"),
        ("32.13.2026", None),
        ("без даты", None),
    ],
)
def test_extract_date(text: str, expected: str | None) -> None:
    assert extract_date(text)[0] == expected


@pytest.mark.parametrize(
    ("text", "sku", "qty", "unit"),
    [
        ("450 мл", "OIL-001", 0.45, "л"),
        ("3,5кг", "WRAP-030", 3.5, "кг"),
        ("250 г", "SCRB-020", 0.25, "кг"),
        ("2 канистры по 5 л", "OIL-001", 10.0, "л"),
        ("4 уп. по 50 пар", "CONS-051", 200.0, "пар"),
        ("−120 шт", "CONS-052", -120.0, "шт"),
    ],
)
def test_extract_quantity(text: str, sku: str, qty: float, unit: str) -> None:
    assert extract_quantity(text, load_catalog()[sku]) == (qty, unit)


def test_incompatible_unit_is_not_converted() -> None:
    assert extract_quantity("2 кг", load_catalog()["CONS-052"]) == (2.0, "кг")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("приход", "receipt"),
        ("Расход", "consume"),
        ("списание", "writeoff"),
        ("Возврат", "return"),
        ("Корректировка", "correction"),
        ("перемещение", None),
    ],
)
def test_extract_operation(text: str, expected: str | None) -> None:
    assert extract_operation(text) == expected


@pytest.mark.parametrize("spelling", ["OIL-001", "oil 001", "«oil 001»", "oil-001"])
def test_sku_spellings(spelling: str) -> None:
    assert normalize_movement(f"расход {spelling} 1 л")["sku"] == "OIL-001"


def test_match_by_name_and_ambiguity() -> None:
    catalog = load_catalog()
    assert match_sku_by_name("тапочек одноразовых", catalog) == "CONS-051"
    assert match_sku_by_name("одноразовые", catalog) is None


def test_missing_fields_are_none() -> None:
    result = normalize_movement("что-то непонятное")
    assert set(result) == {
        "date", "sku", "location", "operation", "qty", "unit", "batch", "doc_no",
    }
    assert all(value is None for value in result.values())


EXPECTED = {
    "M1": ("2026-03-05", "OIL-001", "MS-01", "receipt", 10.0, "л", None, "НК-345"),
    "M2": ("2026-03-01", "OIL-001", "MS-01", "consume", 0.45, "л",
           "B-OIL-001-012", None),
    "M3": ("2026-06-03", "SCRB-020", "Сочи", "writeoff", 1.2, "кг", None, None),
    "M4": ("2026-03-07", "WRAP-030", "MS-02", "consume", 3.5, "кг",
           "B-WRAP-030-004", None),
    "M5": ("2026-03-12", "OIL-002", None, "return", 2.0, "л", None, None),
    "M6": ("2026-03-08", "CONS-051", "MS-01", "consume", 48.0, "пар", None, None),
    "M7": ("2026-03-15", "CONS-052", "MS-02", "correction", -120.0, "шт",
           None, None),
    "M8": (None, "CONS-051", "Красная Поляна", "receipt", 200.0, "пар",
           None, None),
}


def _movements() -> list[dict[str, str]]:
    with open(DATASET_PATH, encoding="utf-8") as f:
        return json.load(f)["movements"]


@pytest.mark.parametrize("record", _movements(), ids=lambda r: r["id"])
def test_dataset_records(record: dict[str, str]) -> None:
    keys = ("date", "sku", "location", "operation", "qty", "unit", "batch", "doc_no")
    result = normalize_movement(record["text"])
    assert tuple(result[k] for k in keys) == EXPECTED[record["id"]]
