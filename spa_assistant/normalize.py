"""Часть 1. Нормализация свободных текстовых записей движения товара"""

import re
from datetime import date
from typing import Any

from spa_assistant import config
from spa_assistant.catalog import load_catalog

MINUS_CHARS = "-−–"
_MINUS_CLASS = r"\-−–"

_RE_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_RE_DOT_DATE = re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4}|\d{2})\b")
_RE_SLASH_DATE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4}|\d{2})\b")
_RE_TEXT_DATE = re.compile(
    r"\b(\d{1,2})\s+([а-яё]+)\s+(\d{4})(?:\s*г\.?)?", re.IGNORECASE
)
_RE_BATCH = re.compile(
    r"(?:парт(?:ия)?\.?\s*)?\b(B-[A-Z]+-\d{3}-\d{3})\b", re.IGNORECASE
)
_RE_DOC = re.compile(r"(?:\b([А-ЯЁ]{1,4}-\d+)\b|№\s*(\S+))")
_RE_SKU = re.compile(r"\b([A-Za-z]{2,5})[\s\-_]?(\d{3})\b")
_RE_WAREHOUSE = re.compile(r"\b(MS)-(\d{2})\b", re.IGNORECASE)

_NUM = rf"[{_MINUS_CLASS}]?\d+(?:[.,]\d+)?"
_UNIT = "|".join(sorted(map(re.escape, config.UNITS), key=len, reverse=True))
_CONTAINER = "|".join(config.CONTAINER_WORDS)
_RE_PACKED_QTY = re.compile(
    rf"(?<![\w{_MINUS_CLASS}])({_NUM})\s*(?:{_CONTAINER})[а-яё]*\.?\s+по\s+"
    rf"(\d+(?:[.,]\d+)?)\s*({_UNIT})(?![а-яё])",
    re.IGNORECASE,
)
_RE_PLAIN_QTY = re.compile(
    rf"(?<![\w{_MINUS_CLASS}])({_NUM})\s*({_UNIT})(?![а-яё])", re.IGNORECASE
)


def _to_float(raw: str) -> float:
    """Преобразовать число с запятой и/или юникодным минусом в float"""
    cleaned = raw.replace(",", ".")
    for ch in MINUS_CHARS:
        cleaned = cleaned.replace(ch, "-")
    return float(cleaned)


def _full_year(raw: str) -> int:
    """Дополнить двузначный год до четырёхзначного"""
    year = int(raw)
    return year + config.TWO_DIGIT_YEAR_BASE if len(raw) == 2 else year


def _month_from_word(word: str) -> int | None:
    """Номер месяца по русскому слову ("марта" -> 3), по самой длинной основе"""
    word = word.lower()
    best: tuple[int, int] | None = None
    for stem, month in config.RU_MONTHS.items():
        if word.startswith(stem) and (best is None or len(stem) > best[0]):
            best = (len(stem), month)
    return best[1] if best else None


def _cut(text: str, match: re.Match[str]) -> str:
    """Заменить найденный фрагмент пробелом, чтобы он не мешал дальше"""
    return text[: match.start()] + " " + text[match.end():]


def extract_date(text: str) -> tuple[str | None, str]:
    """Найти дату в одном из поддерживаемых форматов
    Возвращает ISO-строку (или None) и текст без найденной даты
    """
    if m := _RE_ISO_DATE.search(text):
        y, mo, d = int(m[1]), int(m[2]), int(m[3])
    elif m := _RE_DOT_DATE.search(text):
        d, mo, y = int(m[1]), int(m[2]), _full_year(m[3])
    elif m := _RE_SLASH_DATE.search(text):
        a, b, y = int(m[1]), int(m[2]), _full_year(m[3])
        mo, d = (a, b) if config.SLASH_DATE_ORDER == "MDY" else (b, a)
    elif (m := _RE_TEXT_DATE.search(text)) and _month_from_word(m[2]):
        d, mo, y = int(m[1]), _month_from_word(m[2]), int(m[3])
    else:
        return None, text
    try:
        iso = date(y, mo, d).isoformat()
    except ValueError:
        return None, text
    return iso, _cut(text, m)


def extract_operation(text: str) -> str | None:
    """Привести тип операции, записанный словами, к каноническому значению"""
    lowered = text.lower()
    for stem, operation in config.OPERATION_STEMS.items():
        if stem in lowered:
            return operation
    return None


def extract_location(text: str) -> tuple[str | None, str]:
    """Найти склад (MS-01) или известный объект по названию"""
    if m := _RE_WAREHOUSE.search(text):
        return f"{m[1].upper()}-{m[2]}", _cut(text, m)
    words = re.findall(r"[а-яё]+", text.lower())
    for key, name in config.KNOWN_LOCATIONS.items():
        if all(any(w.startswith(s) for w in words) for s in key.split()):
            return name, text
    return None, text


def _stems(text: str) -> set[str]:
    """Грубый стемминг: обрезать слова до фиксированной длины"""
    return {
        w[: config.NAME_STEM_LEN]
        for w in re.findall(r"[а-яё]+", text.lower())
        if len(w) >= config.NAME_MIN_WORD_LEN
    }


def match_sku_by_name(text: str, catalog: dict[str, dict[str, Any]]) -> str | None:
    """Сопоставить позицию по названию, если в тексте нет артикула"""
    text_stems = _stems(text)
    scores: dict[str, float] = {}
    for sku, item in catalog.items():
        name_stems = _stems(re.sub(r"\(.*?\)", "", item["name"]))
        if name_stems:
            scores[sku] = len(name_stems & text_stems) / len(name_stems)
    if not scores:
        return None
    best = max(scores.values())
    leaders = [sku for sku, s in scores.items() if s == best]
    if best >= config.NAME_MATCH_MIN_SCORE and len(leaders) == 1:
        return leaders[0]
    return None


def extract_sku(
    text: str, catalog: dict[str, dict[str, Any]]
) -> tuple[str | None, str]:
    """Найти артикул в любом написании или сопоставить по названию"""
    for m in _RE_SKU.finditer(text):
        candidate = f"{m[1].upper()}-{m[2]}"
        if candidate in catalog:
            return candidate, _cut(text, m)
    return match_sku_by_name(text, catalog), text


def extract_quantity(
    text: str, item: dict[str, Any] | None
) -> tuple[float | None, str | None]:
    """Извлечь количество и привести его к базовой единице позиции"""
    if m := _RE_PACKED_QTY.search(text):
        qty = _to_float(m[1]) * _to_float(m[2])
        unit = m[3].lower()
    elif m := _RE_PLAIN_QTY.search(text):
        qty = _to_float(m[1])
        unit = m[2].lower()
    else:
        return None, None

    if item is None:
        return round(qty, config.QTY_PRECISION), unit

    base_unit = item["unit"]
    dim, factor = config.UNITS[unit]
    base_dim, base_factor = config.UNITS[base_unit]
    if dim != base_dim:
        return round(qty, config.QTY_PRECISION), unit
    return round(qty * factor / base_factor, config.QTY_PRECISION), base_unit


def normalize_movement(text: str) -> dict[str, Any]:
    """Разобрать текстовую запись движения товара в структурированный вид"""
    catalog = load_catalog()
    rest = text

    iso_date, rest = extract_date(rest)

    batch = None
    if m := _RE_BATCH.search(rest):
        batch = m[1].upper()
        rest = _cut(rest, m)

    doc_no = None
    if m := _RE_DOC.search(rest):
        doc_no = m[1] or m[2]
        rest = _cut(rest, m)

    location, rest = extract_location(rest)
    sku, rest = extract_sku(rest, catalog)
    operation = extract_operation(rest)
    qty, unit = extract_quantity(rest, catalog.get(sku) if sku else None)

    return {
        "date": iso_date,
        "sku": sku,
        "location": location,
        "operation": operation,
        "qty": qty,
        "unit": unit,
        "batch": batch,
        "doc_no": doc_no,
    }
