"""Загрузка справочника товаров"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from spa_assistant.config import CATALOG_PATH


@lru_cache(maxsize=None)
def load_catalog(path: Path = CATALOG_PATH) -> dict[str, dict[str, Any]]:
    """Прочитать catalog.json и вернуть словарь {sku: карточка товара}"""
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    return {item["sku"]: item for item in items}
