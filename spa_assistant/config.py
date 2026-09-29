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

DAYS_PER_WEEK: int = 7

SHIFT_MIN_WEEKS: int = 4
SHIFT_MIN_RATIO: float = 1.5

OUTLIER_MAD_SCALE: float = 1.4826
OUTLIER_Z_THRESHOLD: float = 3.5

TREND_MIN_WEEKS: int = 6
TREND_MIN_R2: float = 0.5

CONF_FULL_HISTORY_WEEKS: int = 8
CONF_ZERO_AT_CV: float = 0.5
CONFIDENCE_THRESHOLD: float = 0.6

STOCKOUT_SEARCH_DAYS: int = 365
MONEY_PRECISION: int = 2
REPORT_PRECISION: int = 2

INTENT_PATTERNS: dict[str, list[tuple[str, float]]] = {
    "forecast_purchase": [
        (r"закуп", 1.0),
        (r"заказать|закаж", 1.0),
        (r"уйд[её]т|понадоб|потребност|израсход", 1.0),
        (r"сколько\s+(это\s+)?(будет\s+)?сто(ит|ить)|стоимост", 1.0),
        (r"\bвс[её]\b", -1.0),
    ],
    "reorder_list": [
        (r"\bчто\b[^?]*(заказ|закуп|докуп)", 2.0),
        (r"список\s+(заказ|закуп)", 2.0),
    ],
    "budget": [
        (r"бюджет", 2.0),
        (r"план\w*\s+закуп", 2.0),
        (r"сколько\s+(это\s+)?(будет\s+)?сто(ит|ить)|стоимост", 1.0),
    ],
    "deficit_risk": [
        (r"дефицит", 2.0),
        (r"риск\w*\s+дефицит", 1.0),
        (r"законч|кончит|не хватит|хватит ли", 2.0),
        (r"остал(ось|ся|ись)|остат", 2.0),
    ],
    "expiry_risk": [
        (r"сгор|срок\w*\s+годност|истека|просроч", 2.0),
        (r"парти", 1.0),
    ],
    "price_dynamics": [
        (r"подорож|подешев|динамик\w*\s+цен|рост\w*\s+цен", 2.0),
        (r"\bцен", 1.0),
    ],
}

INTENT_DESCRIPTIONS: dict[str, str] = {
    "forecast_purchase": "расчёт закупки конкретной позиции на период",
    "reorder_list": "список того, что пора заказать",
    "budget": "бюджет закупок на период",
    "deficit_risk": "позиции с риском дефицита и остатки",
    "expiry_risk": "партии с риском списания по сроку годности",
    "price_dynamics": "изменение цен",
}

INTENT_REQUIRED: dict[str, tuple[str, ...]] = {
    "forecast_purchase": ("sku", "period_days"),
    "reorder_list": ("period_days",),
    "budget": ("period_days",),
    "deficit_risk": (),
    "expiry_risk": (),
    "price_dynamics": (),
}

INTENT_GAP_THRESHOLD: float = 0.25

QUESTION_STEM_LEN: int = 4
QUESTION_MIN_WORD_LEN: int = 4

NUMBER_WORDS: dict[str, int] = {
    "один": 1, "одну": 1, "одного": 1, "два": 2, "две": 2, "двух": 2,
    "три": 3, "трёх": 3, "трех": 3, "четыре": 4, "пять": 5, "шесть": 6,
    "семь": 7, "восемь": 8, "девять": 9, "десять": 10, "двенадцать": 12,
}

PERIOD_UNITS: list[tuple[str, int]] = [
    (r"полгод\w*|полугоди\w*", 180),
    (r"квартал\w*", 90),
    (r"год(а|у|ом)?|лет", 365),
    (r"месяц\w*", 30),
    (r"недел\w*", 7),
    (r"дн(ей|я)|день", 1),
]

BUDGET_MULTIPLIERS: dict[str, float] = {"тыс": 1_000.0, "к": 1_000.0, "млн": 1_000_000.0}

UNSUPPORTED_TOPICS: dict[str, str] = {
    r"привез|доставк|когда\s+(придёт|приедет|поступит)": (
        "В данных нет дат поставок: известно только количество товара в пути. "
        "Могу подсказать, на сколько хватит запаса и когда нужно сделать заказ."
    ),
}

COMPARISON_PATTERN: str = r"почему|по сравнению|прошл\w+"

DEFAULT_COVER_DAYS: int = 30

LLM_MODEL: str = "GigaChat-2"
LLM_MAX_TOKENS: int = 512

LLM_ERROR_MAX_LEN: int = 300
