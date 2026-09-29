"""Конфигурация: словари синонимов, единицы измерения и пороги.

Все константы, влияющие на разбор и расчёты, собраны здесь,
чтобы в коде не было «магических чисел».
"""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
CATALOG_PATH: Path = PROJECT_ROOT / "catalog.json"
DATASET_PATH: Path = PROJECT_ROOT / "dataset.json"

TWO_DIGIT_YEAR_BASE: int = 2000

SLASH_DATE_ORDER: str = "DMY"

RU_MONTHS: dict[str, int] = {
    "январ": 1, "феврал": 2, "март": 3, "апрел": 4, "ма": 5, "июн": 6,
    "июл": 7, "август": 8, "сентябр": 9, "октябр": 10, "ноябр": 11,
    "декабр": 12,
}

OPERATION_STEMS: dict[str, str] = {
    "приход": "receipt",
    "поступлен": "receipt",
    "расход": "consume",
    "израсходов": "consume",
    "списан": "writeoff",
    "возврат": "return",
    "корректир": "correction",
}

KNOWN_LOCATIONS: dict[str, str] = {
    "сочи": "Сочи",
    "красн поля": "Красная Поляна",
}

UNITS: dict[str, tuple[str, float]] = {
    "мл": ("volume", 0.001),
    "л": ("volume", 1.0),
    "г": ("mass", 0.001),
    "гр": ("mass", 0.001),
    "кг": ("mass", 1.0),
    "шт": ("count", 1.0),
    "пар": ("count", 1.0),
    "пары": ("count", 1.0),
}

CONTAINER_WORDS: tuple[str, ...] = (
    "канистр", "уп", "упак", "короб", "флакон", "бутыл", "пачк", "мешк",
)

QTY_PRECISION: int = 4

NAME_STEM_LEN: int = 5
NAME_MATCH_MIN_SCORE: float = 0.75
NAME_MIN_WORD_LEN: int = 3
