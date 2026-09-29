"""Часть 3. Разбор вопроса пользователя и ответ с объяснением расчёта"""

import math
import re
from datetime import date, timedelta
from typing import Any

from spa_assistant import config
from spa_assistant.catalog import load_catalog
from spa_assistant.forecast import forecast_demand
from spa_assistant.normalize import extract_location

INTENTS: tuple[str, ...] = tuple(config.INTENT_PATTERNS)

_RE_SKU_CODE = re.compile(r"\b([A-Za-z]{2,5})[\s\-_]?(\d{3})\b")
_NUMBER = r"\d+(?:[.,]\d+)?"
_NUMBER_WORD = "|".join(config.NUMBER_WORDS)
_RE_BUDGET_LIMIT = re.compile(
    rf"(?:лимит\w*|не более|не больше|в пределах|бюджет\w*\s+до)\s*(?:в\s+)?"
    rf"({_NUMBER})\s*({'|'.join(config.BUDGET_MULTIPLIERS)})?",
    re.IGNORECASE,
)
_RE_MULTIPLIER = re.compile(
    rf"(вырас|вырост|увелич|рост|снизи|снижен|уменьш|упад)\w*\s+на\s+({_NUMBER})\s*%",
    re.IGNORECASE,
)
_DECREASE_STEMS = ("сниз", "сниж", "уменьш", "упад")


def score_intents(question: str) -> dict[str, float]:
    """Сумма весов сработавших шаблонов по каждому интенту"""
    lowered = question.lower()
    return {
        intent: max(0.0, sum(w for p, w in patterns if re.search(p, lowered)))
        for intent, patterns in config.INTENT_PATTERNS.items()
    }


def find_sku(question: str, catalog: dict[str, dict[str, Any]]) -> list[str]:
    """Кандидаты SKU: по артикулу, иначе по словам названия"""
    codes = [f"{m[1].upper()}-{m[2]}" for m in _RE_SKU_CODE.finditer(question)]
    found = [c for c in codes if c in catalog]
    if found:
        return found[:1]

    def stems(text: str) -> set[str]:
        """Основы слов текста для сравнения с названиями"""
        return {
            w[: config.QUESTION_STEM_LEN]
            for w in re.findall(r"[а-яё]+", text.lower())
            if len(w) >= config.QUESTION_MIN_WORD_LEN
        }

    question_stems = stems(question)
    scores = {sku: len(stems(item["name"]) & question_stems) for sku, item in catalog.items()}
    best = max(scores.values(), default=0)
    if best == 0:
        return []
    return [sku for sku, score in scores.items() if score == best]


def extract_period(question: str) -> int | None:
    """Период в днях: "на три месяца", "14 дней", "квартал", "полгода" """
    lowered = question.lower()
    for unit_pattern, unit_days in config.PERIOD_UNITS:
        m = re.search(
            rf"(?:\b({_NUMBER}|{_NUMBER_WORD})\s+)?\b(?:{unit_pattern})\b", lowered
        )
        if not m:
            continue
        if m[1] is None:
            count = 1.0
        elif m[1] in config.NUMBER_WORDS:
            count = float(config.NUMBER_WORDS[m[1]])
        else:
            count = float(m[1].replace(",", "."))
        return int(count * unit_days)
    return None


def extract_budget_limit(question: str) -> float | None:
    """Бюджетный лимит: "при лимите 200 тысяч" -> 200000"""
    m = _RE_BUDGET_LIMIT.search(question)
    if not m:
        return None
    value = float(m[1].replace(",", "."))
    return value * config.BUDGET_MULTIPLIERS.get((m[2] or "").lower(), 1.0)


def extract_multiplier(question: str) -> float | None:
    """Коэффициент загрузки: "вырастет на 20%" -> 1.2"""
    m = _RE_MULTIPLIER.search(question)
    if not m:
        return None
    share = float(m[2].replace(",", ".")) / 100
    return 1 - share if m[1].lower().startswith(_DECREASE_STEMS) else 1 + share


def parse_rules(question: str, catalog: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Разбор вопроса правилами и ключевыми словами (работает без API-ключей)"""
    location, _ = extract_location(question)
    unsupported = next(
        (p for p in config.UNSUPPORTED_TOPICS if re.search(p, question.lower())), None
    )
    return {
        "scores": score_intents(question),
        "unsupported_topic": unsupported,
        "sku_candidates": find_sku(question, catalog),
        "location": location,
        "period_days": extract_period(question),
        "budget_limit": extract_budget_limit(question),
        "demand_multiplier": extract_multiplier(question),
    }


def intent_gap(scores: dict[str, float]) -> float:
    """Разрыв между первым и вторым интентом в долях от суммы весов"""
    total = sum(scores.values())
    if total == 0:
        return 0.0
    top, second = (sorted(scores.values(), reverse=True) + [0.0])[:2]
    return (top - second) / total


def merge_parses(
    rules_raw: dict[str, Any], llm_raw: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    """Гибрид: интент правил, если они уверены, иначе модели; позиции — объединение"""
    rules_confident = intent_gap(rules_raw["scores"]) >= config.INTENT_GAP_THRESHOLD
    source = "rules" if rules_confident else "llm"
    candidates = rules_raw["sku_candidates"] + [
        s for s in llm_raw["sku_candidates"] if s not in rules_raw["sku_candidates"]
    ]
    merged = {
        key: rules_raw[key] if rules_raw[key] is not None else llm_raw[key]
        for key in ("location", "period_days", "budget_limit", "demand_multiplier")
    }
    merged.update(
        scores=rules_raw["scores"] if rules_confident else llm_raw["scores"],
        sku_candidates=candidates,
        unsupported_topic=rules_raw.get("unsupported_topic"),
    )
    return merged, f"hybrid:{source}"


def resolve(raw: dict[str, Any], parser: str) -> dict[str, Any]:
    """Выбрать интент по разрыву между первым и вторым, проверить параметры"""
    scores = raw["scores"]
    total = sum(scores.values())
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_intent, top_score = ranked[0]
    share = top_score / total if total else 0.0
    gap = intent_gap(scores)

    candidates = raw["sku_candidates"]
    parsed: dict[str, Any] = {
        "intent": top_intent,
        "sku": candidates[0] if len(candidates) == 1 else None,
        "sku_candidates": candidates,
        "location": raw["location"],
        "period_days": raw["period_days"],
        "budget_limit": raw["budget_limit"],
        "demand_multiplier": raw["demand_multiplier"],
        "intent_scores": {k: round(v, config.REPORT_PRECISION) for k, v in ranked if v},
        "intent_confidence": round(share, config.REPORT_PRECISION),
        "missing": [],
        "reason": None,
        "parser": parser,
    }
    if total == 0 and raw.get("unsupported_topic"):
        parsed.update(
            intent="unknown", reason="unsupported_topic",
            unsupported_topic=raw["unsupported_topic"],
        )
    elif total == 0:
        parsed.update(intent="unknown", reason="out_of_domain")
    elif gap < config.INTENT_GAP_THRESHOLD:
        parsed.update(
            intent="unknown", reason="ambiguous_intent",
            candidate_intents=[ranked[0][0], ranked[1][0]],
        )
    else:
        missing = [k for k in config.INTENT_REQUIRED[top_intent] if parsed[k] is None]
        if missing:
            parsed.update(
                intent="unknown", reason="missing_params",
                candidate_intents=[top_intent], missing=missing,
            )
    return parsed


def _money(value: float) -> str:
    """Сумма с разделителем разрядов"""
    return f"{value:,.{config.MONEY_PRECISION}f}".replace(",", " ")


def _qty(value: float, unit: str) -> str:
    """Количество с единицей измерения"""
    return f"{value:g} {unit}"


def clarification(parsed: dict[str, Any], catalog: dict[str, dict[str, Any]]) -> str:
    """Уточняющий вопрос вместо догадки"""
    if parsed["reason"] == "unsupported_topic":
        return config.UNSUPPORTED_TOPICS[parsed["unsupported_topic"]]
    if parsed["reason"] == "out_of_domain":
        options = "; ".join(config.INTENT_DESCRIPTIONS.values())
        return (
            "Я отвечаю только на вопросы о складе и закупках, а этот вопрос к ним "
            f"не относится или я его не распознал. Могу помочь с такими темами: {options}."
        )
    if parsed["reason"] == "ambiguous_intent":
        first, second = (config.INTENT_DESCRIPTIONS[i] for i in parsed["candidate_intents"])
        return f"Уточните, что вы хотите узнать: {first} или {second}?"
    questions = []
    if "sku" in parsed["missing"]:
        if parsed["sku_candidates"]:
            options = "; ".join(
                f"{sku} — {catalog[sku]['name']}" for sku in parsed["sku_candidates"]
            )
            questions.append(f"Уточните позицию: {options}.")
        else:
            questions.append("Уточните, о какой позиции идёт речь.")
    if "period_days" in parsed["missing"]:
        questions.append("Уточните период: например, месяц, квартал или 14 дней.")
    return " ".join(questions)


class Answerer:
    """Формирует ответы только из результатов forecast_demand и справочника"""

    def __init__(self, history: dict[str, Any], catalog: dict[str, dict[str, Any]]) -> None:
        """Запомнить данные и разделить позиции на имеющие историю и нет"""
        self.history = history
        self.catalog = catalog
        self.as_of = date.fromisoformat(history["as_of"])
        self.tracked = [s for s in history["weekly_consumption"] if s in catalog]
        self.untracked = [s for s in catalog if s not in self.tracked]
        self.used: list[tuple[float, float]] = []

    def forecast(self, sku: str, days: int, multiplier: float | None) -> dict[str, Any]:
        """Расчёт Части 2 с запоминанием его уверенности"""
        params = {**self.catalog[sku], "demand_multiplier": multiplier or 1.0}
        result = forecast_demand(self.history, sku, days, params)
        self.used.append((result["confidence"], result["estimated_cost"]))
        return result

    def data_confidence(self) -> float:
        """Уверенность расчётов: средняя, взвешенная по стоимости позиций"""
        if not self.used:
            return 1.0
        total_cost = sum(cost for _, cost in self.used)
        if total_cost == 0:
            return min(conf for conf, _ in self.used)
        return sum(conf * cost for conf, cost in self.used) / total_cost

    def scope(self, parsed: dict[str, Any]) -> list[str]:
        """Позиции, по которым считать ответ"""
        return [parsed["sku"]] if parsed["sku"] else self.tracked

    def notes(self, parsed: dict[str, Any], skus: list[str]) -> list[str]:
        """Общие оговорки: объект, позиции без истории"""
        lines = []
        if parsed["location"]:
            lines.append(
                f"Объект «{parsed['location']}»: остатки и расход в данных не разделены "
                "по объектам, поэтому расчёт сделан по общим данным."
            )
        if not parsed["sku"] and self.untracked:
            lines.append(
                "Нет истории расхода, в расчёт не вошли: " + ", ".join(self.untracked) + "."
            )
        return lines

    def item_line(self, sku: str, r: dict[str, Any]) -> str:
        """Строка по позиции с исходными показателями"""
        item = self.catalog[sku]
        unit = item["unit"]
        return (
            f"- {sku} ({item['name']}): закупить {_qty(r['recommended_qty'], unit)} "
            f"на {_money(r['estimated_cost'])}; остаток {_qty(r['current_stock'], unit)}, "
            f"в пути {_qty(r['incoming_qty'], unit)}, прогноз {_qty(r['forecast_demand'], unit)}, "
            f"цена {_money(item['price'])} за {unit}, заказать до {r['order_date']}."
        )

    def limit_lines(self, cost: float, limit: float | None) -> list[str]:
        """Сравнение стоимости с бюджетным лимитом"""
        if limit is None:
            return []
        if cost <= limit:
            return [f"Укладывается в лимит {_money(limit)}, остаётся {_money(limit - cost)}."]
        return [f"Превышает лимит {_money(limit)} на {_money(cost - limit)}."]

    def forecast_purchase(self, parsed: dict[str, Any]) -> tuple[str, str]:
        """Закупка конкретной позиции на период"""
        sku, days = parsed["sku"], parsed["period_days"]
        if sku not in self.tracked:
            return "no_data", f"По {sku} нет истории расхода, прогноз посчитать нельзя."
        item = self.catalog[sku]
        r = self.forecast(sku, days, parsed["demand_multiplier"])
        lines = [
            f"{item['name']} ({sku}) на {days} дн.: закупить "
            f"{_qty(r['recommended_qty'], item['unit'])} на сумму {_money(r['estimated_cost'])}. "
            f"Прогноз расхода за период: {_qty(r['forecast_demand'], item['unit'])}. "
            f"Заказать не позднее {r['order_date']}."
        ]
        limit = parsed["budget_limit"]
        lines += self.limit_lines(r["estimated_cost"], limit)
        if limit is not None and r["estimated_cost"] > limit:
            affordable = (
                math.floor(limit / item["price"] / item["pack_size"]) * item["pack_size"]
            )
            if affordable < item["min_order_qty"]:
                lines.append("В лимит не укладывается даже минимальная партия.")
            else:
                lines.append(
                    f"В пределах лимита можно заказать {_qty(affordable, item['unit'])} "
                    f"на {_money(affordable * item['price'])} (с учётом упаковки "
                    f"{item['pack_size']} {item['unit']}); это меньше расчётной потребности."
                )
        lines += self.notes(parsed, [sku])
        lines += ["", "Как посчитано:", r["explanation"]]
        return "answered", "\n".join(lines)

    def budget(self, parsed: dict[str, Any], question: str) -> tuple[str, str]:
        """Бюджет закупок на период по всем позициям или одной"""
        days = parsed["period_days"]
        skus = [s for s in self.scope(parsed) if s in self.tracked]
        if not skus:
            return "no_data", f"По {parsed['sku']} нет истории расхода, бюджет посчитать нельзя."
        results = {s: self.forecast(s, days, parsed["demand_multiplier"]) for s in skus}
        total = sum(r["estimated_cost"] for r in results.values())
        lines = [f"Бюджет закупок на {days} дн.: {_money(total)}."]
        lines += [self.item_line(s, r) for s, r in results.items()]
        lines += self.limit_lines(total, parsed["budget_limit"])
        if re.search(config.COMPARISON_PATTERN, question.lower()):
            lines.append(
                "Плана прошлого периода в данных нет, поэтому сравнить с ним нельзя. "
                "Что увеличивает текущий расчёт:"
            )
            lines += self.drivers(results)
        lines += self.notes(parsed, skus)
        return "answered", "\n".join(lines)

    def drivers(self, results: dict[str, dict[str, Any]]) -> list[str]:
        """Причины роста потребности из расчёта Части 2"""
        lines = []
        for sku, r in results.items():
            unit = self.catalog[sku]["unit"]
            if r["level_shift"]:
                before, after = r["level_shift"]
                lines.append(
                    f"- {sku}: расход вырос скачком с {before} до {after} {unit}/нед."
                )
            elif r["trend_per_week"] and r["trend_per_week"] > 0:
                lines.append(
                    f"- {sku}: расход растёт на {r['trend_per_week']} {unit}/нед. каждую неделю."
                )
        return lines or ["- роста спроса по истории не видно."]

    def reorder_list(self, parsed: dict[str, Any]) -> tuple[str, str]:
        """Что нужно заказать в ближайшие N дней"""
        window = parsed["period_days"]
        deadline = self.as_of + timedelta(days=window)
        cover = config.DEFAULT_COVER_DAYS
        due = {}
        for sku in self.scope(parsed):
            if sku not in self.tracked:
                continue
            r = self.forecast(sku, cover, parsed["demand_multiplier"])
            if r["order_date"] and date.fromisoformat(r["order_date"]) <= deadline:
                due[sku] = r
        if not due:
            lines = [f"До {deadline.isoformat()} заказывать ничего не нужно."]
        else:
            total = sum(r["estimated_cost"] for r in due.values())
            lines = [
                f"До {deadline.isoformat()} нужно заказать {len(due)} поз. "
                f"на {_money(total)} (объём рассчитан на {cover} дн.):"
            ]
            lines += [self.item_line(s, r) for s, r in due.items()]
        lines += self.notes(parsed, list(due))
        return "answered", "\n".join(lines)

    def deficit_risk(self, parsed: dict[str, Any]) -> tuple[str, str]:
        """Риск дефицита и остатки"""
        cover = parsed["period_days"] or config.DEFAULT_COVER_DAYS
        skus = self.scope(parsed)
        if parsed["sku"] and parsed["sku"] not in self.tracked:
            item = self.catalog[parsed["sku"]]
            stock = self.history["current_stock"].get(parsed["sku"])
            text = f"По {parsed['sku']} нет истории расхода."
            if stock is not None:
                text += f" Остаток: {_qty(stock, item['unit'])}."
            return "no_data", text
        results = {
            s: self.forecast(s, cover, parsed["demand_multiplier"])
            for s in skus if s in self.tracked
        }
        if parsed["sku"]:
            sku, r = next(iter(results.items()))
            unit = self.catalog[sku]["unit"]
            lines = [
                f"{sku} ({self.catalog[sku]['name']}): остаток {_qty(r['current_stock'], unit)}, "
                f"в пути {_qty(r['incoming_qty'], unit)}. Запаса хватит до {r['stockout_date']}.",
                "Риск дефицита есть: запас закончится раньше, чем придёт заказ, сделанный сегодня."
                if r["deficit_risk"] else "Риска дефицита нет.",
            ]
        else:
            risky = {s: r for s, r in results.items() if r["deficit_risk"]}
            if not risky:
                lines = ["Позиций с риском дефицита нет."]
            else:
                total = sum(r["estimated_cost"] for r in risky.values())
                lines = [
                    f"Риск дефицита: {len(risky)} поз. Запас закончится раньше, чем "
                    f"придёт заказ, сделанный сегодня. Закупка на {cover} дн.: {_money(total)}."
                ]
                for s, r in risky.items():
                    lines.append(
                        self.item_line(s, r)[:-1] + f"; запас кончится {r['stockout_date']}."
                    )
        lines += self.notes(parsed, skus)
        return "answered", "\n".join(lines)

    def expiry_risk(self, parsed: dict[str, Any]) -> tuple[str, str]:
        """Риск списания по сроку годности"""
        return "no_data", (
            "В данных нет партий со сроками годности, поэтому оценить риск списания "
            "нельзя. Для расчёта нужны: номер партии, остаток по партии и срок годности."
        )

    def price_dynamics(self, parsed: dict[str, Any]) -> tuple[str, str]:
        """Изменение цен"""
        text = "В данных нет истории цен, поэтому изменение цены посчитать нельзя."
        if parsed["sku"]:
            item = self.catalog[parsed["sku"]]
            text += (
                f" Текущая цена {parsed['sku']} ({item['name']}) по справочнику: "
                f"{_money(item['price'])} за {item['unit']}."
            )
        return "no_data", text


def answer_question(question: str, context: dict[str, Any]) -> tuple[dict, float, str]:
    """Разобрать вопрос, посчитать ответ по данным Части 2 и объяснить его

    context: history (блок истории из dataset.json), catalog (необязательно,
    по умолчанию catalog.json), use_llm (гибридный разбор: правила и GigaChat)
    """
    catalog = context.get("catalog") or load_catalog()
    raw, parser = parse_rules(question, catalog), "rules"
    llm_raw, llm_error = None, None
    if context.get("use_llm"):
        from spa_assistant.llm import parse_with_llm

        llm_raw, llm_error = parse_with_llm(question, catalog, INTENTS)
        if llm_raw is not None:
            raw, parser = merge_parses(raw, llm_raw)
    parsed = resolve(raw, parser)
    parsed["llm_error"] = llm_error
    parsed["llm_intent"] = (
        max(llm_raw["scores"], key=llm_raw["scores"].get)
        if llm_raw and any(llm_raw["scores"].values()) else None
    )

    if parsed["intent"] == "unknown":
        parsed.update(status="clarification", needs_clarification=True)
        return parsed, 0.0, clarification(parsed, catalog)

    answerer = Answerer(context["history"], catalog)
    handler = getattr(answerer, parsed["intent"])
    if parsed["intent"] == "budget":
        status, text = handler(parsed, question)
    else:
        status, text = handler(parsed)

    confidence = round(
        min(parsed["intent_confidence"], answerer.data_confidence()),
        config.REPORT_PRECISION,
    )
    low = confidence < config.CONFIDENCE_THRESHOLD
    if status == "answered" and low and "Требуется уточнение" not in text:
        text += (
            f"\n\nТребуется уточнение: уверенность {confidence} ниже порога "
            f"{config.CONFIDENCE_THRESHOLD}. Проверьте исходные данные по позициям "
            "с короткой или нестабильной историей."
        )
    parsed.update(status=status, needs_clarification=status != "answered" or low)
    return parsed, confidence, text
