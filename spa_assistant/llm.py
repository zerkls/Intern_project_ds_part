"""Необязательный разбор вопроса через GigaChat

Модель только извлекает интент и сущности в JSON. Числа в ответ она не
пишет: ответ собирается кодом из результатов Части 2. При любой ошибке
(нет пакета, ключа, сети, невалидный JSON) возвращается None, и
answer_question переходит на правила.

Настройки подключения библиотека gigachat читает из переменных окружения:
GIGACHAT_CREDENTIALS (ключ авторизации), GIGACHAT_SCOPE,
GIGACHAT_CA_BUNDLE_FILE или GIGACHAT_VERIFY_SSL_CERTS.
"""

import json
import os
import re
from typing import Any

from spa_assistant import config

SYSTEM_PROMPT = """Ты разбираешь вопросы управляющего спа-салоном о складе и закупках.
Верни только параметры запроса, ничего не считай и не придумывай.

Интенты:
{intents}

Позиции справочника:
{catalog}

Ответь одним JSON-объектом без пояснений, строго с такими полями:
{{
  "intent": один из [{intent_names}],
  "intent_confidence": число от 0 до 1,
  "second_intent": второй по вероятности интент из того же списка или "", если его нет,
  "second_confidence": число от 0 до 1,
  "sku_candidates": список артикулов из справочника, о которых идёт речь;
      если название подходит к нескольким позициям, перечисли все;
      если позиция не названа, пустой список,
  "location": объект ("Сочи", "Красная Поляна", "MS-01") или "",
  "period_days": период в днях, только если он явно назван
      (месяц = 30, квартал = 90, полгода = 180, год = 365), иначе 0,
  "budget_limit": бюджетный лимит в рублях, если назван, иначе 0,
  "demand_multiplier": множитель спроса ("загрузка вырастет на 20%" = 1.2), иначе 1
}}
Если вопрос не про склад и закупки, верни "intent": "unknown" с уверенностью 0.
"""

_RE_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def _unit_interval(value: Any) -> float:
    """Число, ограниченное отрезком [0, 1]"""
    return min(1.0, max(0.0, float(value)))


def _to_raw(data: dict[str, Any], catalog: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Привести ответ модели к формату parse_rules

    Отсутствующее поле означает «не указано». Интент не из списка
    («неизвестно», 0, "") получает нулевой вес, поэтому вопрос вне домена
    уходит в уточнение. Значения неверного типа вызывают ошибку.
    """
    scores = {name: 0.0 for name in config.INTENT_PATTERNS}
    intent, second = data.get("intent"), data.get("second_intent")
    if intent in scores:
        scores[intent] = _unit_interval(data.get("intent_confidence", 0))
    if second in scores and second != intent:
        scores[second] = _unit_interval(data.get("second_confidence", 0))

    candidates = data.get("sku_candidates") or []
    if not isinstance(candidates, list):
        raise ValueError(f"sku_candidates не список: {candidates!r}")
    period = int(data.get("period_days") or 0)
    limit = float(data.get("budget_limit") or 0)
    multiplier = float(data.get("demand_multiplier") or 1)
    return {
        "scores": scores,
        "sku_candidates": [s for s in candidates if s in catalog],
        "location": str(data.get("location") or "") or None,
        "period_days": period if period > 0 else None,
        "budget_limit": limit if limit > 0 else None,
        "demand_multiplier": multiplier if multiplier != 1 else None,
    }


def parse_with_llm(
    question: str, catalog: dict[str, dict[str, Any]], intents: tuple[str, ...]
) -> tuple[dict[str, Any] | None, str | None]:
    """Разобрать вопрос моделью GigaChat; при неудаче (None, причина)"""
    if not os.environ.get("GIGACHAT_CREDENTIALS"):
        return None, "не задан GIGACHAT_CREDENTIALS"
    try:
        from gigachat import Chat, GigaChat, Messages, MessagesRole
    except ImportError:
        return None, "не установлен пакет gigachat"

    system = SYSTEM_PROMPT.format(
        intents="\n".join(f"- {k}: {v}" for k, v in config.INTENT_DESCRIPTIONS.items()),
        catalog="\n".join(f"- {s}: {item['name']}" for s, item in catalog.items()),
        intent_names=", ".join(intents),
    )
    payload = Chat(
        messages=[
            Messages(role=MessagesRole.SYSTEM, content=system),
            Messages(role=MessagesRole.USER, content=question),
        ],
        max_tokens=config.LLM_MAX_TOKENS,
    )
    try:
        with GigaChat(model=config.LLM_MODEL) as client:
            response = client.chat(payload)
    except Exception as exc:
        return None, f"ошибка запроса: {type(exc).__name__}: {exc}"[:config.LLM_ERROR_MAX_LEN]

    choice = response.choices[0]
    if choice.finish_reason != "stop":
        return None, f"ответ не завершён: finish_reason={choice.finish_reason}"
    content = choice.message.content or ""
    match = _RE_JSON_OBJECT.search(content)
    if not match:
        return None, f"в ответе нет JSON: {content}"[:config.LLM_ERROR_MAX_LEN]
    try:
        return _to_raw(json.loads(match.group()), catalog), None
    except (TypeError, ValueError, KeyError, AttributeError) as exc:
        reason = f"ответ не прошёл проверку: {type(exc).__name__}: {exc}; {match.group()}"
        return None, reason[:config.LLM_ERROR_MAX_LEN]
