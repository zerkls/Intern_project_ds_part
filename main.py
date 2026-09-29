"""CLI для прогона пайплайна на тестовых данных"""

import argparse
import json
from spa_assistant.config import DATASET_PATH
from spa_assistant.normalize import normalize_movement

FIELDS = ("date", "sku", "location", "operation", "qty", "unit", "batch", "doc_no")


def run_normalize() -> None:
    """Прогнать normalize_movement на записях M1–M8 и вывести таблицу"""
    with open(DATASET_PATH, encoding="utf-8") as f:
        movements = json.load(f)["movements"]
    print("| id | " + " | ".join(FIELDS) + " |")
    print("|" + "---|" * (len(FIELDS) + 1))
    for record in movements:
        result = normalize_movement(record["text"])
        cells = ["—" if result[k] is None else str(result[k]) for k in FIELDS]
        print(f"| {record['id']} | " + " | ".join(cells) + " |")


def main() -> None:
    parser = argparse.ArgumentParser(description="ИИ-помощник по закупкам спа")
    parser.add_argument("part", choices=["normalize"], help="какую часть запустить")
    args = parser.parse_args()
    if args.part == "normalize":
        run_normalize()


if __name__ == "__main__":
    main()
