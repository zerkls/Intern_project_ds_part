"""CLI для прогона пайплайна на тестовых данных"""

import argparse
import json
from spa_assistant.catalog import load_catalog
from spa_assistant.config import DATASET_PATH
from spa_assistant.forecast import forecast_demand
from spa_assistant.normalize import normalize_movement

FIELDS = ("date", "sku", "location", "operation", "qty", "unit", "batch", "doc_no")
HORIZONS = (30, 90)
FORECAST_FIELDS = (
    "horizon", "avg_daily_consumption", "forecast_demand", "current_stock",
    "incoming_qty", "safety_stock", "reorder_point", "recommended_qty",
    "estimated_cost", "stockout_date", "order_date", "deficit_risk",
    "confidence", "needs_clarification",
)


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


def run_forecast(explain: bool) -> None:
    """Прогнать forecast_demand по позициям из истории на 30 и 90 дней"""
    with open(DATASET_PATH, encoding="utf-8") as f:
        history = json.load(f)["history"]
    catalog = load_catalog()
    print("| sku | " + " | ".join(FORECAST_FIELDS) + " |")
    print("|" + "---|" * (len(FORECAST_FIELDS) + 1))
    explanations = []
    for sku in history["weekly_consumption"]:
        for horizon in HORIZONS:
            result = forecast_demand(history, sku, horizon, catalog[sku])
            row = [str(horizon)] + [str(result[k]) for k in FORECAST_FIELDS[1:]]
            print(f"| {sku} | " + " | ".join(row) + " |")
            explanations.append(f"{sku}, {horizon} дн.\n{result['explanation']}")
    if explain:
        print("\n" + "\n\n".join(explanations))


def main() -> None:
    parser = argparse.ArgumentParser(description="ИИ-помощник по закупкам спа")
    parser.add_argument(
        "part", choices=["normalize", "forecast"], help="какую часть запустить"
    )
    parser.add_argument(
        "--explain", action="store_true", help="вывести текстовые объяснения"
    )
    args = parser.parse_args()
    if args.part == "normalize":
        run_normalize()
    elif args.part == "forecast":
        run_forecast(args.explain)


if __name__ == "__main__":
    main()
