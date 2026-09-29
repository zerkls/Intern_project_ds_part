"""Часть 2. Прогноз потребности и рекомендация закупки"""

import math
import statistics
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from spa_assistant import config

REQUIRED_PARAMS: tuple[str, ...] = (
    "pack_size", "min_order_qty", "safety_stock_days", "lead_time_days", "price",
)


@dataclass
class DemandModel:
    """Модель недельного расхода: уровень, тренд и диагностика ряда"""

    intercept: float
    slope: float
    last_week: int
    weeks_used: int
    weeks_window: int
    missing_weeks: list[int]
    shift_week: int | None
    shift_before: float | None
    shift_after: float | None
    outliers: int
    r2: float
    cv: float

    def weekly_rate(self, position: float) -> float:
        """Ожидаемый недельный расход в точке position шкалы недель"""
        return max(0.0, self.intercept + self.slope * position)

    def daily_rates(self, days: int, multiplier: float) -> list[float]:
        """Прогноз расхода по дням, начиная со дня после as_of"""
        start = self.last_week + 0.5
        return [
            self.weekly_rate(start + (d + 0.5) / config.DAYS_PER_WEEK)
            / config.DAYS_PER_WEEK * multiplier
            for d in range(days)
        ]


def detect_level_shift(points: list[tuple[int, float]]) -> int | None:
    """Найти индекс начала нового устойчивого уровня спроса или None"""
    n = len(points)
    for split in range(config.SHIFT_MIN_WEEKS, n - config.SHIFT_MIN_WEEKS + 1):
        before = [v for _, v in points[:split]]
        after = [v for _, v in points[split:]]
        mean_before = statistics.mean(before)
        ratio = math.inf if mean_before == 0 else statistics.mean(after) / mean_before
        grew = min(after) > max(before) and ratio >= config.SHIFT_MIN_RATIO
        fell = max(after) < min(before) and ratio <= 1 / config.SHIFT_MIN_RATIO
        if grew or fell:
            return split
    return None


def clip_outliers(
    points: list[tuple[int, float]]
) -> tuple[list[tuple[int, float]], int]:
    """Ограничить выбросы по модифицированному z-score (медиана и MAD)"""
    values = [v for _, v in points]
    median = statistics.median(values)
    mad = statistics.median(abs(v - median) for v in values)
    if mad == 0:
        return points, 0
    bound = config.OUTLIER_Z_THRESHOLD * config.OUTLIER_MAD_SCALE * mad
    clipped = [(i, min(max(v, median - bound), median + bound)) for i, v in points]
    count = sum(1 for (_, a), (_, b) in zip(points, clipped) if a != b)
    return clipped, count


def fit_trend(points: list[tuple[int, float]]) -> tuple[float, float, float]:
    """Прямая по МНК: возвращает (intercept, slope, r2)"""
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    mean_x, mean_y = statistics.mean(xs), statistics.mean(ys)
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx == 0:
        return mean_y, 0.0, 0.0
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / sxx
    intercept = mean_y - slope * mean_x
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - intercept - slope * x) ** 2 for x, y in points)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return intercept, slope, r2


def build_model(weekly: list[float | None]) -> DemandModel | None:
    """Очистить ряд (пропуски, сдвиг уровня, выбросы) и подобрать модель"""
    points = [(i, float(v)) for i, v in enumerate(weekly) if v is not None]
    if not points:
        return None

    shift_week = shift_before = shift_after = None
    split = detect_level_shift(points)
    if split is not None:
        shift_week = points[split][0]
        shift_before = statistics.mean(v for _, v in points[:split])
        shift_after = statistics.mean(v for _, v in points[split:])
        points = points[split:]

    window_start = points[0][0]
    missing = [i for i in range(window_start, len(weekly)) if weekly[i] is None]
    points, outliers = clip_outliers(points)

    intercept, slope, r2 = fit_trend(points)
    mean_y = statistics.mean(v for _, v in points)
    if len(points) < config.TREND_MIN_WEEKS or r2 < config.TREND_MIN_R2:
        intercept, slope = mean_y, 0.0

    residuals = [y - intercept - slope * x for x, y in points]
    if len(points) > 1 and mean_y > 0:
        cv = statistics.stdev(residuals) / mean_y
    else:
        cv = config.CONF_ZERO_AT_CV

    return DemandModel(
        intercept=intercept,
        slope=slope,
        last_week=len(weekly) - 1,
        weeks_used=len(points),
        weeks_window=len(weekly) - window_start,
        missing_weeks=missing,
        shift_week=shift_week,
        shift_before=shift_before,
        shift_after=shift_after,
        outliers=outliers,
        r2=r2,
        cv=cv,
    )


def compute_confidence(model: DemandModel) -> tuple[float, dict[str, float]]:
    """Уверенность = длина истории * стабильность расхода * полнота ряда"""
    factors = {
        "history": min(1.0, model.weeks_used / config.CONF_FULL_HISTORY_WEEKS),
        "stability": max(0.0, 1 - model.cv / config.CONF_ZERO_AT_CV),
        "completeness": 1 - len(model.missing_weeks) / model.weeks_window,
    }
    return round(math.prod(factors.values()), config.REPORT_PRECISION), factors


def round_order_qty(net_need: float, pack_size: float, min_order_qty: float) -> float:
    """Округлить потребность вверх до минимальной партии и кратности упаковки"""
    if net_need <= 0:
        return 0.0
    qty = max(round(net_need, config.QTY_PRECISION), min_order_qty)
    packs = math.ceil(round(qty / pack_size, config.QTY_PRECISION))
    return float(packs * pack_size)


def _first_day(rates: list[float], stop: float) -> int | None:
    """Номер дня (с 1), к концу которого накопленный расход превысит stop"""
    total = 0.0
    for day, rate in enumerate(rates, start=1):
        total += rate
        if total > stop:
            return day
    return None


def _date_after(as_of: date, days: int | None) -> str | None:
    """Дата в ISO-формате, отстоящая от as_of на days дней"""
    return None if days is None else (as_of + timedelta(days=days)).isoformat()


def _fmt(value: float) -> str:
    """Число для текста объяснения"""
    return f"{value:.{config.REPORT_PRECISION}f}"


def forecast_demand(
    history: dict[str, Any], sku: str, horizon_days: int, params: dict[str, Any]
) -> dict[str, Any]:
    """Спрогнозировать потребность и рассчитать рекомендуемую закупку

    history - блок истории из dataset.json (as_of, weekly_consumption,
    current_stock, incoming_qty); params - карточка позиции из справочника,
    можно добавить demand_multiplier для сценария роста загрузки
    """
    missing_params = [k for k in REQUIRED_PARAMS if k not in params]
    if missing_params:
        raise ValueError(f"Не хватает параметров позиции: {missing_params}")
    if horizon_days <= 0:
        raise ValueError("horizon_days должен быть положительным")
    weekly = history.get("weekly_consumption", {}).get(sku)
    if weekly is None:
        raise ValueError(f"Нет истории расхода по {sku}")
    if sku not in history.get("current_stock", {}):
        raise ValueError(f"Нет данных об остатке по {sku}")

    as_of = date.fromisoformat(history["as_of"])
    unit = params.get("unit", "")
    current_stock = float(history["current_stock"][sku])
    incoming_qty = float(history.get("incoming_qty", {}).get(sku, 0.0))
    multiplier = float(params.get("demand_multiplier", 1.0))

    model = build_model(weekly)
    if model is None:
        return {
            "avg_daily_consumption": None, "forecast_demand": None,
            "current_stock": current_stock, "incoming_qty": incoming_qty,
            "safety_stock": None, "reorder_point": None, "recommended_qty": None,
            "estimated_cost": None, "stockout_date": None, "order_date": None,
            "deficit_risk": None, "trend_per_week": None, "level_shift": None,
            "confidence": 0.0, "needs_clarification": True,
            "explanation": f"По {sku} нет ни одной недели с данными о расходе.",
        }

    lead_days = int(params["lead_time_days"])
    rates = model.daily_rates(
        max(horizon_days, lead_days, config.STOCKOUT_SEARCH_DAYS), multiplier
    )
    forecast = sum(rates[:horizon_days])
    avg_daily = forecast / horizon_days
    lead_demand = sum(rates[:lead_days])
    lead_daily = lead_demand / lead_days
    safety_stock = lead_daily * params["safety_stock_days"]
    reorder_point = lead_demand + safety_stock

    available = current_stock + incoming_qty
    net_need = forecast + safety_stock - available
    recommended = round_order_qty(net_need, params["pack_size"], params["min_order_qty"])
    cost = round(recommended * params["price"], config.MONEY_PRECISION)

    search = rates[:config.STOCKOUT_SEARCH_DAYS]
    stockout_date = _date_after(as_of, _first_day(search, available))
    stockout_no_incoming = _date_after(as_of, _first_day(search, current_stock))
    if available <= reorder_point:
        order_date = as_of.isoformat()
    else:
        order_date = _date_after(as_of, _first_day(search, available - reorder_point))

    arrival_if_ordered_now = as_of + timedelta(days=lead_days)
    deficit_risk = (
        stockout_date is not None
        and date.fromisoformat(stockout_date) <= arrival_if_ordered_now
    )

    confidence, factors = compute_confidence(model)
    needs_clarification = confidence < config.CONFIDENCE_THRESHOLD

    lines = [
        f"История: {len(weekly)} нед. по {as_of.isoformat()}, "
        f"в расчёте {model.weeks_used} нед.",
    ]
    if model.missing_weeks:
        weeks = ", ".join(str(w + 1) for w in model.missing_weeks)
        lines.append(
            f"Пропущены данные за неделю {weeks}: неделя исключена из расчёта, "
            f"а не считается нулевым расходом."
        )
    if model.shift_week is not None:
        lines.append(
            f"С недели {model.shift_week + 1} расход устойчиво изменился: "
            f"{_fmt(model.shift_before)} -> {_fmt(model.shift_after)} {unit}/нед. "
            f"Это считается новым уровнем спроса, прогноз строится только по "
            f"неделям после сдвига."
        )
    if model.outliers:
        lines.append(f"Скорректировано выбросов: {model.outliers}.")
    if model.slope:
        lines.append(
            f"Метод: линейный тренд (R²={_fmt(model.r2)}), расход растёт на "
            f"{_fmt(model.slope)} {unit}/нед. каждую неделю; текущий уровень "
            f"{_fmt(model.weekly_rate(model.last_week))} {unit}/нед."
        )
    else:
        if model.weeks_used < config.TREND_MIN_WEEKS:
            reason = (
                f"недель меньше {config.TREND_MIN_WEEKS}, тренд не оценивается"
            )
        else:
            reason = f"тренд незначим, R²={_fmt(model.r2)}"
        lines.append(
            f"Метод: среднее без тренда ({reason}), "
            f"уровень {_fmt(model.intercept)} {unit}/нед."
        )
    if multiplier != 1.0:
        lines.append(f"Спрос умножен на коэффициент загрузки {_fmt(multiplier)}.")
    lines += [
        f"Прогноз на {horizon_days} дн.: {_fmt(forecast)} {unit} "
        f"(в среднем {_fmt(avg_daily)} {unit}/день).",
        f"Страховой запас: {params['safety_stock_days']} дн. × "
        f"{_fmt(lead_daily)} {unit}/день = {_fmt(safety_stock)} {unit}.",
        f"Точка заказа: расход за срок поставки {lead_days} дн. "
        f"({_fmt(lead_demand)} {unit}) + страховой запас = {_fmt(reorder_point)} {unit}.",
        f"Потребность: прогноз {_fmt(forecast)} + страховой запас "
        f"{_fmt(safety_stock)} − остаток {_fmt(current_stock)} − в пути "
        f"{_fmt(incoming_qty)} = {_fmt(net_need)} {unit}.",
    ]
    if recommended:
        lines.append(
            f"Округление вверх до минимальной партии {params['min_order_qty']} {unit} "
            f"и упаковки {params['pack_size']} {unit}: {_fmt(recommended)} {unit}; "
            f"стоимость {_fmt(recommended)} × {_fmt(params['price'])} = {_fmt(cost)}."
        )
    else:
        lines.append("Закупка на этот период не нужна: запаса хватает.")
    lines.append(
        f"Запаса вместе с поставкой в пути хватит до {stockout_date or 'конца года'}; "
        f"без поставки в пути - до {stockout_no_incoming or 'конца года'}. "
        f"Заказать не позднее {order_date or 'конца года'}."
    )
    if deficit_risk:
        lines.append(
            f"Риск дефицита: запас закончится {stockout_date}, а заказ, "
            f"сделанный сегодня, придёт не раньше "
            f"{arrival_if_ordered_now.isoformat()}."
        )
    lines.append(
        f"Уверенность {_fmt(confidence)} = длина истории {_fmt(factors['history'])} "
        f"× стабильность {_fmt(factors['stability'])} × полнота "
        f"{_fmt(factors['completeness'])}."
    )
    if needs_clarification:
        lines.append(
            f"Требуется уточнение: уверенность ниже порога "
            f"{_fmt(config.CONFIDENCE_THRESHOLD)}."
        )

    precision = config.REPORT_PRECISION
    return {
        "avg_daily_consumption": round(avg_daily, precision),
        "forecast_demand": round(forecast, precision),
        "current_stock": current_stock,
        "incoming_qty": incoming_qty,
        "safety_stock": round(safety_stock, precision),
        "reorder_point": round(reorder_point, precision),
        "recommended_qty": recommended,
        "estimated_cost": cost,
        "stockout_date": stockout_date,
        "order_date": order_date,
        "deficit_risk": deficit_risk,
        "trend_per_week": round(model.slope, precision),
        "level_shift": (
            None if model.shift_week is None
            else [round(model.shift_before, precision), round(model.shift_after, precision)]
        ),
        "confidence": confidence,
        "needs_clarification": needs_clarification,
        "explanation": "\n".join(lines),
    }
