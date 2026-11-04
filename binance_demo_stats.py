#!/usr/bin/env python3
"""Статистика торговли на Binance USDⓈ-M Futures **Demo Trading** (не Testnet).

Источники в документации Binance (developers.binance.com, USDⓈ-M Futures):
  * REST demo:   https://demo-fapi.binance.com  (ключи — только с demo.binance.com)
  * GET /fapi/v1/userTrades : weight 5, limit<=1000, startTime..endTime <= 7 дней,
                              fromId нельзя вместе со startTime/endTime, история ~3 месяца
  * GET /fapi/v1/income     : weight 30, limit<=1000, история ~3 месяца
  * Лимиты: REQUEST_WEIGHT (на IP, 1 минута), читаются из exchangeInfo в рантайме;
    429 -> пауза, повтор 429 -> 418 (бан IP на 2 мин ... 3 дня) -> нельзя долбить дальше.

Лимиты Binance используются только на SAFETY_FACTOR (50%).

Зависимости:  pip install ccxt pandas python-dotenv tenacity jinja2
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, Iterator

import ccxt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from jinja2 import Environment
from tenacity import before_sleep_log, retry, retry_if_exception, stop_after_attempt

log = logging.getLogger("demo_stats")

# --------------------------------------------------------------------------- #
# Константы
# --------------------------------------------------------------------------- #
DAY_MS = 86_400_000
WEEK_MS = 7 * DAY_MS            # максимум окна startTime..endTime у userTrades
PAGE_LIMIT = 1000               # максимум limit у userTrades / income
HISTORY_DAYS = 90               # Binance хранит ~3 месяца

SAFETY_FACTOR = 0.5             # используем не более 50% от лимитов Binance
FALLBACK_WEIGHT_LIMIT_1M = 2400 # если exchangeInfo не отдал rateLimits

# Веса эндпоинтов (по документации). Реальный расход всё равно берётся из
# заголовка X-MBX-USED-WEIGHT-1M, эти числа нужны только для "упреждающей" паузы.
W_USER_TRADES = 5
W_INCOME = 30
W_SYMBOL_CONFIG = 5             # GET /fapi/v1/symbolConfig (плечо и режим маржи; в ccxt cost=5)
W_POSITION_RISK = 5             # GET /fapi/v3/positionRisk (в ccxt cost=1 — занижен, берём из доки)

QUOTE_ASSETS = {"USDT", "USDC"}  # комиссии в других активах (BNB) в сумму не входят
MAX_ORDERS_IN_HTML = 1000        # полная история — в CSV
MAX_POSITIONS_IN_HTML = 1000
MARGIN_MODE_RU = {"CROSSED": "Кросс-маржа", "ISOLATED": "Изолированная"}


# --------------------------------------------------------------------------- #
# Работа с лимитами Binance
# --------------------------------------------------------------------------- #
def _retryable(exc: BaseException) -> bool:
    """429 и сетевые/5xx ошибки — повторяем. 418 (бан IP) — НЕТ."""
    if isinstance(exc, ccxt.RateLimitExceeded):      # HTTP 429 / -1003
        return True
    if isinstance(exc, ccxt.DDoSProtection):         # HTTP 418 и прочий бан/WAF
        return False
    return isinstance(exc, ccxt.NetworkError)        # таймауты, 503 и т.п. (read-only — безопасно)


def _backoff(retry_state) -> float:
    exc = retry_state.outcome.exception()
    if isinstance(exc, ccxt.RateLimitExceeded):
        return 60 - (time.time() % 60) + 1           # ждём новую минутную квоту веса
    return min(2 ** retry_state.attempt_number, 30)  # 2, 4, 8 ... c


@retry(
    retry=retry_if_exception(_retryable),
    wait=_backoff,
    stop=stop_after_attempt(4),
    before_sleep=before_sleep_log(log, logging.WARNING),
    reraise=True,
)
def _request(fn: Callable[[dict], object], params: dict):
    return fn(params)


class WeightGovernor:
    """Ограничитель REQUEST_WEIGHT: равномерный темп + контроль по заголовку Binance."""

    def __init__(self, exchange: ccxt.Exchange, limit_1m: int) -> None:
        self.ex = exchange
        self.budget = max(int(limit_1m * SAFETY_FACTOR), 1)
        self.used = 0
        self._next_ts = 0.0
        log.info("Лимит веса Binance: %d/мин, рабочий бюджет (x%.1f): %d/мин",
                 limit_1m, SAFETY_FACTOR, self.budget)

    def _sync_used(self) -> None:
        headers = {str(k).lower(): v for k, v in (self.ex.last_response_headers or {}).items()}
        value = headers.get("x-mbx-used-weight-1m")
        if value is not None:
            self.used = int(value)

    def _pace(self, cost: int) -> None:
        wait = self._next_ts - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        if self.used + cost > self.budget:
            pause = 60 - (time.time() % 60) + 1      # окно веса — календарная минута
            log.info("Бюджет веса исчерпан (%d/%d), пауза %.0f c", self.used, self.budget, pause)
            time.sleep(pause)
            self.used = 0
        self._next_ts = time.monotonic() + cost * 60 / self.budget

    def call(self, fn: Callable[[dict], object], params: dict, cost: int):
        self._pace(cost)
        try:
            return _request(fn, params)
        finally:
            self._sync_used()


# --------------------------------------------------------------------------- #
# Загрузка данных
# --------------------------------------------------------------------------- #
def paged(fetch: Callable[[int], list[dict]], start: int, uid: Callable[[dict], object]) -> list[dict]:
    """Постраничная выгрузка по времени (fromId с startTime/endTime использовать нельзя).

    Курсор — время последней записи (включительно), дубли отсекаются по uid:
    так не теряются записи с одинаковым timestamp на границе страницы.
    """
    seen: set = set()
    out: list[dict] = []
    cursor = start
    while True:
        rows = fetch(cursor)
        fresh = [r for r in rows if uid(r) not in seen]
        seen.update(uid(r) for r in fresh)
        out.extend(fresh)
        if len(rows) < PAGE_LIMIT:
            return out
        last = int(rows[-1]["time"])
        cursor = last if fresh else last + 1         # страховка от зацикливания


def week_windows(times: Iterable[int], end: int) -> Iterator[tuple[int, int]]:
    """Минимальный набор 7-дневных окон, покрывающих все моменты активности."""
    ts = sorted(set(times))
    i = 0
    while i < len(ts):
        stop = min(ts[i] + WEEK_MS - 1, end)
        yield ts[i], stop
        i = bisect_right(ts, stop)


def tile_windows(start: int, end: int) -> Iterator[tuple[int, int]]:
    """Сплошное покрытие диапазона 7-дневными окнами (когда активность неизвестна)."""
    t = start
    while t <= end:
        yield t, min(t + WEEK_MS - 1, end)
        t += WEEK_MS


def discover_activity(gov: WeightGovernor, start: int, end: int) -> dict[str, list[int]]:
    """symbol -> моменты активности, по income (раньше искали по открытым позициям — это неверно)."""
    for income_type in ("COMMISSION", "REALIZED_PNL"):   # комиссия есть у каждого филла
        rows = paged(
            lambda cur, t=income_type: gov.call(
                gov.ex.fapiPrivateGetIncome,
                {"incomeType": t, "startTime": cur, "endTime": end, "limit": PAGE_LIMIT},
                W_INCOME,
            ),
            start,
            uid=lambda r: (r.get("tranId"), r.get("incomeType"), r.get("tradeId"), r.get("time")),
        )
        activity: dict[str, list[int]] = defaultdict(list)
        for r in rows:
            if r.get("symbol"):
                activity[r["symbol"]].append(int(r["time"]))
        if activity:
            return activity
        log.info("income/%s пуст, пробуем следующий тип", income_type)
    return {}


def fetch_fills(gov: WeightGovernor, windows: dict[str, list[tuple[int, int]]]) -> list[dict]:
    fills: list[dict] = []
    total = sum(len(w) for w in windows.values())
    done = 0
    for symbol in sorted(windows):
        before = len(fills)
        for w_start, w_end in windows[symbol]:
            done += 1
            fills += paged(
                lambda cur, s=symbol, e=w_end: gov.call(
                    gov.ex.fapiPrivateGetUserTrades,
                    {"symbol": s, "startTime": cur, "endTime": e, "limit": PAGE_LIMIT},
                    W_USER_TRADES,
                ),
                w_start,
                uid=lambda r: r["id"],
            )
        log.info("%-14s сделок: %-6d (запросов окон: %d/%d)", symbol, len(fills) - before, done, total)
    return fills


def get_weight_limit(ex: ccxt.Exchange) -> int:
    info = _request(ex.fapiPublicGetExchangeInfo, {})
    for r in info.get("rateLimits", []):
        if (r.get("rateLimitType") == "REQUEST_WEIGHT" and r.get("interval") == "MINUTE"
                and int(r.get("intervalNum", 1)) == 1):
            return int(r["limit"])
    return FALLBACK_WEIGHT_LIMIT_1M


def fetch_symbol_config(gov: WeightGovernor) -> dict[str, dict]:
    """Плечо и режим маржи по символам. Это ТЕКУЩИЕ настройки — API не хранит их на момент сделки."""
    rows = gov.call(gov.ex.fapiPrivateGetSymbolConfig, {}, W_SYMBOL_CONFIG)
    return {r["symbol"]: r for r in rows}


POSITION_NUM_COLS = ["positionAmt", "entryPrice", "markPrice", "unRealizedProfit", "notional", "liquidationPrice"]


def fetch_open_positions(gov: WeightGovernor) -> pd.DataFrame:
    """Открытые позиции и нереализованный PnL. v3 отдаёт только символы с позицией/ордерами."""
    rows = gov.call(gov.ex.fapiPrivateV3GetPositionRisk, {}, W_POSITION_RISK)
    df = pd.DataFrame(rows).reindex(columns=["symbol", "positionSide", *POSITION_NUM_COLS])
    df[POSITION_NUM_COLS] = df[POSITION_NUM_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    df = df[df["positionAmt"] != 0].copy()           # убираем символы, где есть только открытые ордера
    entry_value = (df["positionAmt"].abs() * df["entryPrice"]).replace(0, float("nan"))
    df["pnl_pct"] = (df["unRealizedProfit"] / entry_value * 100).fillna(0.0)   # % от стоимости входа (без плеча)
    return df.sort_values("unRealizedProfit", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Аналитика
# --------------------------------------------------------------------------- #
def build_fills_frame(rows: list[dict], tz: str = "UTC") -> pd.DataFrame:
    df = pd.DataFrame(rows).drop_duplicates(subset=["symbol", "id"])
    if "positionSide" not in df:
        df["positionSide"] = "BOTH"
    for col in ("price", "qty", "quoteQty", "realizedPnl", "commission"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["time"] = pd.to_datetime(df["time"].astype("int64"), unit="ms", utc=True).dt.tz_convert(tz)
    df["fee_usdt"] = df["commission"].where(df["commissionAsset"].isin(QUOTE_ASSETS), 0.0)
    return df.sort_values("time").reset_index(drop=True)


def build_orders_frame(fills: pd.DataFrame) -> pd.DataFrame:
    """Один ордер = одна запись (раньше каждый частичный филл считался отдельной сделкой)."""
    orders = fills.groupby(["symbol", "orderId"], as_index=False).agg(
        time=("time", "min"), side=("side", "first"), fills=("id", "count"),
        qty=("qty", "sum"), realized=("realizedPnl", "sum"), fee=("fee_usdt", "sum"),
    )
    orders["net"] = orders["realized"] - orders["fee"]
    return orders.sort_values("time").reset_index(drop=True)


def build_position_history(fills: pd.DataFrame, config: dict[str, dict] | None = None) -> tuple[pd.DataFrame, dict]:
    """«История позиций» как в вебе Binance. Публичного эндпоинта нет — восстанавливаем из филлов.

    Позиция живёт от выхода из нуля до возврата в ноль (отдельно по symbol + positionSide).
    В one-way режиме филл может перевернуть позицию — он делится на закрывающую и открывающую части
    (комиссия — пропорционально, realizedPnl целиком относится к закрывающей части).
    Entry/Avg close — средневзвешенные без комиссий, Max OI — максимум размера за жизнь позиции,
    Closing PnL = Σ realizedPnl, Net = Closing PnL − комиссии.
    """
    config = config or {}
    rows, orphan_fills, still_open = [], 0, 0
    for (symbol, pside), g in fills.groupby(["symbol", "positionSide"], sort=False):
        open_dir = {"LONG": 1, "SHORT": -1}.get(pside)       # в hedge-режиме направление открытия фиксировано
        cur = None
        for f in g.sort_values(["time", "id"]).itertuples(index=False):
            direction = 1 if f.side == "BUY" else -1
            left, pnl_left, tol = f.qty, f.realizedPnl, f.qty * 1e-9
            if cur is None and (pnl_left != 0 or (open_dir is not None and direction != open_dir)):
                orphan_fills += 1                             # закрытие позиции, открытой раньше окна истории
                continue
            while left > tol:
                if cur is None:
                    cur = dict(dir=direction, open_time=f.time, open_qty=0.0, open_notional=0.0, close_qty=0.0,
                               close_notional=0.0, size=0.0, max_size=0.0, pnl=0.0, fee=0.0)
                if direction == cur["dir"]:                   # увеличение позиции
                    cur["open_qty"] += left
                    cur["open_notional"] += left * f.price
                    cur["size"] += left
                    cur["max_size"] = max(cur["max_size"], cur["size"])
                    cur["fee"] += f.fee_usdt * left / f.qty
                    left = 0.0
                else:                                         # уменьшение / закрытие
                    closing = min(left, cur["size"])
                    cur["close_qty"] += closing
                    cur["close_notional"] += closing * f.price
                    cur["pnl"] += pnl_left
                    pnl_left = 0.0
                    cur["fee"] += f.fee_usdt * closing / f.qty
                    cur["size"] -= closing
                    left -= closing
                    if cur["size"] <= cur["max_size"] * 1e-9:  # вернулись в ноль -> позиция закрыта
                        cfg = config.get(symbol, {})
                        lev = float(cfg["leverage"]) if cfg.get("leverage") else np.nan
                        entry = cur["open_notional"] / cur["open_qty"]
                        net = cur["pnl"] - cur["fee"]
                        margin = entry * cur["max_size"] / lev
                        rows.append(dict(
                            symbol=symbol, side="LONG" if cur["dir"] > 0 else "SHORT", position_mode=pside,
                            margin_mode=cfg.get("marginType"), leverage=lev,
                            open_time=cur["open_time"], close_time=f.time,
                            entry_price=entry, avg_close_price=cur["close_notional"] / cur["close_qty"],
                            max_qty=cur["max_size"], closed_qty=cur["close_qty"],
                            closing_pnl=cur["pnl"], fee=cur["fee"], net=net,
                            roi=net / margin * 100 if margin == margin and margin else np.nan,
                        ))
                        cur = None
        still_open += cur is not None
    cols = ["symbol", "side", "position_mode", "margin_mode", "leverage", "open_time", "close_time", "entry_price",
            "avg_close_price", "max_qty", "closed_qty", "closing_pnl", "fee", "net", "roi"]
    hist = pd.DataFrame(rows, columns=cols).sort_values("close_time").reset_index(drop=True)
    return hist, {"orphan_fills": orphan_fills, "still_open": still_open}


def compute_stats(orders: pd.DataFrame, fills: pd.DataFrame) -> dict:
    closing = orders[orders["realized"] != 0]        # открывающие ордера PnL не имеют
    wins, losses = closing[closing["net"] > 0], closing[closing["net"] < 0]
    wins_gross = closing[closing["realized"] > 0]    # без учёта комиссии (как «закрытые сделки» в UI Binance)
    gross_win, gross_loss = wins["net"].sum(), losses["net"].sum()
    equity = orders["net"].cumsum()
    peak = equity.cummax().clip(lower=0)
    return {
        "realized": float(orders["realized"].sum()),
        "fees": float(orders["fee"].sum()),
        "net": float(orders["net"].sum()),
        "orders": len(orders),
        "fills": len(fills),
        "closing": len(closing),
        "win_rate": len(wins) / len(closing) * 100 if len(closing) else 0.0,
        "win_rate_gross": len(wins_gross) / len(closing) * 100 if len(closing) else 0.0,
        "wins": len(wins),
        "wins_gross": len(wins_gross),
        "profit_factor": (gross_win / abs(gross_loss)) if gross_loss else (float("inf") if gross_win > 0 else 0.0),
        "avg_win": float(wins["net"].mean()) if len(wins) else 0.0,
        "avg_loss": float(losses["net"].mean()) if len(losses) else 0.0,
        "max_drawdown": float((equity - peak).min()) if len(equity) else 0.0,
        "foreign_fee_fills": int((~fills["commissionAsset"].isin(QUOTE_ASSETS)).sum()),
    }


def by_symbol(orders: pd.DataFrame) -> pd.DataFrame:
    is_close = orders["realized"] != 0
    g = orders.assign(closes=is_close, wins=is_close & (orders["net"] > 0)).groupby("symbol")
    out = g.agg(net=("net", "sum"), realized=("realized", "sum"), fee=("fee", "sum"),
                orders=("orderId", "count"), closes=("closes", "sum"), wins=("wins", "sum")).reset_index()
    out["win_rate"] = (out["wins"] / out["closes"].where(out["closes"] > 0) * 100).fillna(0.0)
    return out.sort_values("net", ascending=False)


def by_day(orders: pd.DataFrame) -> pd.DataFrame:
    daily = (orders.assign(date=orders["time"].dt.strftime("%Y-%m-%d"))
             .groupby("date", as_index=False)["net"].sum())
    daily["cum"] = daily["net"].cumsum()
    return daily


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
TEMPLATE = """<!DOCTYPE html>
<html lang="ru"><head><meta charset="UTF-8">
<title>Binance Demo Futures Stats</title>
<script src="https://cdn.tailwindcss.com/3.4.17"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
 table.sortable th{cursor:pointer;user-select:none}
 table.sortable th:hover{background:#f9fafb}
 table.sortable th::after{content:' ↕';opacity:.3}
</style></head>
<body class="bg-gray-100 p-6 font-sans"><div class="max-w-7xl mx-auto">
{% macro clr(v) %}{{ 'text-green-600' if v >= 0 else 'text-red-600' }}{% endmacro %}
{% macro money(v) %}{{ '{:,.2f}'.format(v) }}{% endmacro %}
<h1 class="text-3xl font-bold text-gray-800 mb-2">📊 Binance Futures — Demo Trading</h1>
<p class="text-sm text-gray-500 mb-6">Символов: {{ symbols|length }} · Ордеров: {{ s.orders }} (филлов: {{ s.fills }})
 · Период: {{ period }} · Время: {{ tz }} · Сформировано: {{ generated }}</p>
{% if s.foreign_fee_fills %}
<p class="mb-4 p-3 rounded bg-yellow-100 text-yellow-800 text-sm">⚠ {{ s.foreign_fee_fills }} филлов с комиссией не в USDT/USDC
 (например BNB) — в «Комиссии» и Net PnL они не учтены.</p>{% endif %}
<div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4 mb-8">
 {% for label, val, color in cards %}
 <div class="bg-white p-4 rounded-lg shadow"><p class="text-sm text-gray-500">{{ label }}</p>
 <p class="text-xl font-bold {{ color }}">{{ val }}</p></div>
 {% endfor %}
</div>
<div class="bg-white p-6 rounded-lg shadow mb-8"><h2 class="text-xl font-semibold mb-4">Открытые позиции
 <span class="text-sm font-normal text-gray-500">(нереализованный PnL по mark price на момент запуска)</span></h2>
{% if positions %}<div class="overflow-x-auto"><table class="min-w-full sortable text-sm">
<thead><tr class="bg-gray-50">
 <th class="px-4 py-2 text-left">Символ</th><th class="px-4 py-2 text-left">Сторона</th>
 <th class="px-4 py-2 text-right">Объём</th><th class="px-4 py-2 text-right">Вход</th>
 <th class="px-4 py-2 text-right">Mark</th><th class="px-4 py-2 text-right">Notional</th>
 <th class="px-4 py-2 text-right">Ликвидация</th><th class="px-4 py-2 text-right">Unrealized PnL</th>
 <th class="px-4 py-2 text-right">% от входа</th></tr></thead><tbody>
{% for p in positions %}<tr class="border-t hover:bg-gray-50">
 <td class="px-4 py-2" data-v="{{ p.symbol }}">{{ p.symbol }}</td>
 <td class="px-4 py-2 {{ 'text-green-600' if p.positionAmt > 0 else 'text-red-600' }}" data-v="{{ p.positionAmt }}">{{ 'LONG' if p.positionAmt > 0 else 'SHORT' }}{% if p.positionSide != 'BOTH' %} ({{ p.positionSide }}){% endif %}</td>
 <td class="px-4 py-2 text-right" data-v="{{ p.positionAmt|abs }}">{{ p.positionAmt|abs }}</td>
 <td class="px-4 py-2 text-right" data-v="{{ p.entryPrice }}">{{ p.entryPrice }}</td>
 <td class="px-4 py-2 text-right" data-v="{{ p.markPrice }}">{{ p.markPrice }}</td>
 <td class="px-4 py-2 text-right" data-v="{{ p.notional|abs }}">{{ money(p.notional|abs) }}</td>
 <td class="px-4 py-2 text-right text-gray-500" data-v="{{ p.liquidationPrice }}">{{ p.liquidationPrice if p.liquidationPrice else '—' }}</td>
 <td class="px-4 py-2 text-right {{ clr(p.unRealizedProfit) }}" data-v="{{ p.unRealizedProfit }}">{{ money(p.unRealizedProfit) }}</td>
 <td class="px-4 py-2 text-right {{ clr(p.pnl_pct) }}" data-v="{{ p.pnl_pct }}">{{ '%.2f'|format(p.pnl_pct) }}%</td></tr>
{% endfor %}</tbody></table></div>
{% elif positions_error %}<p class="text-sm text-yellow-700">Не удалось получить позиции: {{ positions_error }}</p>
{% else %}<p class="text-sm text-gray-500">Открытых позиций нет.</p>{% endif %}</div>

<div class="bg-white p-6 rounded-lg shadow mb-8"><h2 class="text-xl font-semibold mb-2">История позиций</h2>
<p class="text-xs text-gray-500 mb-3">Закрытых позиций: {{ hist_total }}{% if hist|length < hist_total %} (показаны последние {{ hist|length }}){% endif %}.
 Восстановлено из сделок (публичного API истории позиций у Binance нет). Плечо и режим маржи — <b>текущие</b> настройки символа.
 Цены без комиссий; Net PnL = Closing PnL − комиссии; ROI = Net PnL / (цена входа × макс. объём / плечо).{% if hist_info.orphan_fills %}
 Пропущено закрытий позиций, открытых раньше окна истории: {{ hist_info.orphan_fills }}.{% endif %}{% if hist_info.still_open %}
 Ещё открыто (см. блок выше): {{ hist_info.still_open }}.{% endif %}</p>
<div class="flex flex-wrap items-center gap-2 mb-3 text-sm">
 {% for d, label in [(1,'1Д'),(7,'1Н'),(30,'1М'),(0,'Всё')] %}<button type="button" data-days="{{ d }}" class="range-btn px-3 py-1 rounded border {{ 'bg-gray-800 text-white' if d == 0 else 'bg-white' }}">{{ label }}</button>{% endfor %}
 <input id="sym-filter" type="search" placeholder="Символ…" class="ml-2 px-3 py-1 rounded border">
</div>
{% if hist %}<div class="overflow-x-auto max-h-[32rem] overflow-y-auto"><table id="hist" class="min-w-full sortable text-sm">
<thead class="sticky top-0 bg-gray-50"><tr>
 <th class="px-3 py-2 text-left">Символ</th><th class="px-3 py-2 text-left">Открыта</th><th class="px-3 py-2 text-left">Закрыта</th>
 <th class="px-3 py-2 text-right">Длительность</th><th class="px-3 py-2 text-right">Closing PnL</th>
 <th class="px-3 py-2 text-right">Комиссии</th><th class="px-3 py-2 text-right">Net PnL</th><th class="px-3 py-2 text-right">ROI</th>
 <th class="px-3 py-2 text-right">Объём закрытия</th><th class="px-3 py-2 text-right">Цена входа</th>
 <th class="px-3 py-2 text-right">Сред. цена закрытия</th><th class="px-3 py-2 text-right">Макс. объём</th></tr></thead><tbody>
{% for h in hist %}<tr class="border-t hover:bg-gray-50" data-ts="{{ h.ts }}" data-sym="{{ h.symbol|lower }}">
 <td class="px-3 py-2 whitespace-nowrap" data-v="{{ h.symbol }}"><b>{{ h.symbol }}</b>
  {% if h.lev %}<span class="ml-1 px-1 rounded bg-gray-200 text-xs">{{ h.lev }}x</span>{% endif %}
  {% if h.margin_ru %}<span class="ml-1 px-1 rounded bg-green-100 text-green-700 text-xs">{{ h.margin_ru }}</span>{% endif %}
  <span class="ml-1 text-xs font-semibold {{ 'text-green-600' if h.side == 'LONG' else 'text-red-600' }}">{{ 'Лонг' if h.side == 'LONG' else 'Шорт' }}</span></td>
 <td class="px-3 py-2 whitespace-nowrap" data-v="{{ h.open_str }}">{{ h.open_str }}</td>
 <td class="px-3 py-2 whitespace-nowrap" data-v="{{ h.close_str }}">{{ h.close_str }}</td>
 <td class="px-3 py-2 text-right whitespace-nowrap" data-v="{{ h.dur_s }}">{{ h.dur_str }}</td>
 <td class="px-3 py-2 text-right {{ clr(h.closing_pnl) }}" data-v="{{ h.closing_pnl }}">{{ '%+.4f'|format(h.closing_pnl) }}</td>
 <td class="px-3 py-2 text-right text-gray-500" data-v="{{ h.fee }}">{{ '%.4f'|format(h.fee) }}</td>
 <td class="px-3 py-2 text-right font-semibold {{ clr(h.net) }}" data-v="{{ h.net }}">{{ '%+.4f'|format(h.net) }}</td>
 <td class="px-3 py-2 text-right {{ clr(h.roi_v) }}" data-v="{{ h.roi_v }}">{{ h.roi_str }}</td>
 <td class="px-3 py-2 text-right" data-v="{{ h.closed_qty }}">{{ h.closed_qty_str }}</td>
 <td class="px-3 py-2 text-right" data-v="{{ h.entry_price }}">{{ h.entry_str }}</td>
 <td class="px-3 py-2 text-right" data-v="{{ h.avg_close_price }}">{{ h.close_price_str }}</td>
 <td class="px-3 py-2 text-right" data-v="{{ h.max_qty }}">{{ h.max_qty_str }}</td></tr>
{% endfor %}</tbody></table></div>
{% else %}<p class="text-sm text-gray-500">Закрытых позиций за период нет.</p>{% endif %}</div>

<div class="bg-white p-6 rounded-lg shadow mb-8"><h2 class="text-xl font-semibold mb-4">Кумулятивный Net PnL (по дням)</h2>
 <canvas id="chart" height="100"></canvas></div>

<div class="bg-white p-6 rounded-lg shadow mb-8"><h2 class="text-xl font-semibold mb-4">По символам</h2>
<div class="overflow-x-auto"><table class="min-w-full sortable text-sm">
<thead><tr class="bg-gray-50">
 <th class="px-4 py-2 text-left">Символ</th><th class="px-4 py-2 text-right">Net PnL</th>
 <th class="px-4 py-2 text-right">Realized</th><th class="px-4 py-2 text-right">Комиссии</th>
 <th class="px-4 py-2 text-right">Ордеров</th><th class="px-4 py-2 text-right">Закрытий</th>
 <th class="px-4 py-2 text-right">Win Rate</th></tr></thead><tbody>
{% for r in sym_rows %}<tr class="border-t hover:bg-gray-50">
 <td class="px-4 py-2" data-v="{{ r.symbol }}">{{ r.symbol }}</td>
 <td class="px-4 py-2 text-right {{ clr(r.net) }}" data-v="{{ r.net }}">{{ money(r.net) }}</td>
 <td class="px-4 py-2 text-right" data-v="{{ r.realized }}">{{ money(r.realized) }}</td>
 <td class="px-4 py-2 text-right text-gray-500" data-v="{{ r.fee }}">{{ money(r.fee) }}</td>
 <td class="px-4 py-2 text-right" data-v="{{ r.orders }}">{{ r.orders }}</td>
 <td class="px-4 py-2 text-right" data-v="{{ r.closes }}">{{ r.closes }}</td>
 <td class="px-4 py-2 text-right" data-v="{{ r.win_rate }}">{{ '%.1f'|format(r.win_rate) }}%</td></tr>
{% endfor %}</tbody></table></div></div>

<div class="bg-white p-6 rounded-lg shadow"><h2 class="text-xl font-semibold mb-4">История ордеров
 <span class="text-sm font-normal text-gray-500">(последние {{ order_rows|length }} из {{ s.orders }}; полная история — в CSV)</span></h2>
<div class="overflow-x-auto max-h-96 overflow-y-auto"><table class="min-w-full sortable text-sm">
<thead class="sticky top-0 bg-gray-50"><tr>
 <th class="px-3 py-2 text-left">Время</th><th class="px-3 py-2 text-left">Символ</th>
 <th class="px-3 py-2 text-left">Сторона</th><th class="px-3 py-2 text-right">Кол-во</th>
 <th class="px-3 py-2 text-right">Филлов</th><th class="px-3 py-2 text-right">Realized</th>
 <th class="px-3 py-2 text-right">Комиссия</th><th class="px-3 py-2 text-right">Net</th></tr></thead><tbody>
{% for o in order_rows %}<tr class="border-t hover:bg-gray-50">
 <td class="px-3 py-1 whitespace-nowrap" data-v="{{ o.time }}">{{ o.time }}</td>
 <td class="px-3 py-1" data-v="{{ o.symbol }}">{{ o.symbol }}</td>
 <td class="px-3 py-1 {{ 'text-green-600' if o.side == 'BUY' else 'text-red-600' }}" data-v="{{ o.side }}">{{ o.side }}</td>
 <td class="px-3 py-1 text-right" data-v="{{ o.qty }}">{{ o.qty }}</td>
 <td class="px-3 py-1 text-right" data-v="{{ o.fills }}">{{ o.fills }}</td>
 <td class="px-3 py-1 text-right {{ clr(o.realized) }}" data-v="{{ o.realized }}">{{ '%.4f'|format(o.realized) }}</td>
 <td class="px-3 py-1 text-right text-gray-500" data-v="{{ o.fee }}">{{ '%.4f'|format(o.fee) }}</td>
 <td class="px-3 py-1 text-right {{ clr(o.net) }}" data-v="{{ o.net }}">{{ '%.4f'|format(o.net) }}</td></tr>
{% endfor %}</tbody></table></div></div></div>
<script>
document.querySelectorAll('table.sortable').forEach(t=>{
  t.querySelectorAll('th').forEach((th,i)=>th.addEventListener('click',()=>{
    const dir=th.dataset.dir==='asc'?'desc':'asc'; th.dataset.dir=dir;
    const val=r=>{const v=r.cells[i].dataset.v, n=Number(v); return (v!==''&&!isNaN(n))?n:v;};
    const rows=[...t.tBodies[0].rows];
    rows.sort((a,b)=>{const x=val(a),y=val(b),c=x>y?1:(x<y?-1:0); return dir==='asc'?c:-c;});
    rows.forEach(r=>t.tBodies[0].appendChild(r));
  }));
});
(()=>{const rows=[...document.querySelectorAll('#hist tbody tr')];let days=0,q='';
 const apply=()=>{const lim=days?Date.now()-days*864e5:0;
  rows.forEach(r=>{r.hidden=!((!lim||Number(r.dataset.ts)>=lim)&&r.dataset.sym.includes(q));});};
 document.querySelectorAll('.range-btn').forEach(b=>b.addEventListener('click',()=>{days=Number(b.dataset.days);
  document.querySelectorAll('.range-btn').forEach(x=>x.classList.toggle('bg-gray-800',x===b));
  document.querySelectorAll('.range-btn').forEach(x=>x.classList.toggle('text-white',x===b));apply();}));
 document.getElementById('sym-filter').addEventListener('input',e=>{q=e.target.value.trim().toLowerCase();apply();});})();
const d={{ daily|tojson }};
new Chart(document.getElementById('chart'),{type:'line',
  data:{labels:d.map(x=>x.date),datasets:[{label:'Net PnL',data:d.map(x=>x.cum),
   borderColor:'rgb(34,197,94)',backgroundColor:'rgba(34,197,94,0.1)',fill:true,tension:0.1,pointRadius:2}]},
  options:{responsive:true,plugins:{legend:{display:false}},scales:{y:{ticks:{callback:v=>'$'+v.toFixed(0)}}}}});
</script></body></html>
"""


def _num(v: float) -> str:
    return np.format_float_positional(v, precision=8, trim="-")


def _duration(td: pd.Timedelta) -> str:
    d, rest = divmod(int(td.total_seconds()), 86_400)
    h, rest = divmod(rest, 3_600)
    return " ".join(x for x in (f"{d}д" if d else "", f"{h}ч" if d or h else "", f"{rest // 60}мин") if x)


def _history_row(r) -> dict:
    roi_ok = r.roi == r.roi                                   # not NaN
    return {
        "symbol": r.symbol, "side": r.side, "lev": int(r.leverage) if r.leverage == r.leverage else None,
        "margin_ru": MARGIN_MODE_RU.get(r.margin_mode), "ts": int(r.close_time.timestamp() * 1000),
        "open_str": r.open_time.strftime("%Y-%m-%d %H:%M:%S"), "close_str": r.close_time.strftime("%Y-%m-%d %H:%M:%S"),
        "dur_s": int((r.close_time - r.open_time).total_seconds()), "dur_str": _duration(r.close_time - r.open_time),
        "closing_pnl": r.closing_pnl, "fee": r.fee, "net": r.net,
        "roi_v": r.roi if roi_ok else 0.0, "roi_str": f"{r.roi:+.2f}%" if roi_ok else "—",
        "closed_qty": r.closed_qty, "closed_qty_str": _num(r.closed_qty),
        "entry_price": r.entry_price, "entry_str": _num(r.entry_price),
        "avg_close_price": r.avg_close_price, "close_price_str": _num(r.avg_close_price),
        "max_qty": r.max_qty, "max_qty_str": _num(r.max_qty),
    }


def render_html(fills: pd.DataFrame, orders: pd.DataFrame, stats: dict, period: str,
                positions: pd.DataFrame | None = None, positions_error: str = "",
                history: pd.DataFrame | None = None, hist_info: dict | None = None) -> str:
    sym = by_symbol(orders)
    daily = by_day(orders)
    last_orders = orders.tail(MAX_ORDERS_IN_HTML).iloc[::-1].copy()
    last_orders["time"] = last_orders["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    pf = "∞" if stats["profit_factor"] == float("inf") else f"{stats['profit_factor']:.2f}"
    green, red = "text-green-600", "text-red-600"
    unrealized = float(positions["unRealizedProfit"].sum()) if positions is not None and len(positions) else 0.0
    cards = [
        ("Net PnL (realized − комиссии)", f"${stats['net']:,.2f}", green if stats["net"] >= 0 else red),
        ("Unrealized PnL", f"${unrealized:,.2f}" if positions is not None else "н/д",
         green if unrealized >= 0 else red),
        ("Realized PnL", f"${stats['realized']:,.2f}", green if stats["realized"] >= 0 else red),
        ("Комиссии", f"${stats['fees']:,.2f}", "text-gray-700"),
        ("Win Rate net (после комиссии)", f"{stats['win_rate']:.1f}% ({stats['wins']}/{stats['closing']})", "text-gray-800"),
        ("Win Rate gross (realizedPnl)", f"{stats['win_rate_gross']:.1f}% ({stats['wins_gross']}/{stats['closing']})", "text-gray-800"),
        ("Profit Factor", pf, "text-gray-800"),
        ("Ср. win / loss", f"{stats['avg_win']:,.2f} / {stats['avg_loss']:,.2f}", "text-gray-800"),
        ("Макс. просадка", f"${stats['max_drawdown']:,.2f}", red),
    ]
    history = history if history is not None else pd.DataFrame()
    hist_rows = [_history_row(r) for r in history.tail(MAX_POSITIONS_IN_HTML).iloc[::-1].itertuples(index=False)]
    if len(history):
        cards.insert(5, ("Win Rate (позиции, net)", f"{(history['net'] > 0).mean() * 100:.1f}% "
                         f"({int((history['net'] > 0).sum())}/{len(history)})", "text-gray-800"))
    env = Environment(autoescape=True)
    return env.from_string(TEMPLATE).render(
        s=stats, symbols=sym["symbol"].tolist(), period=period, cards=cards,
        generated=pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M"),
        positions=positions.to_dict("records") if positions is not None else None,
        hist=hist_rows, hist_total=len(history), hist_info=hist_info or {}, tz=str(fills["time"].dt.tz),
        positions_error=positions_error,
        sym_rows=sym.to_dict("records"), order_rows=last_orders.to_dict("records"),
        daily=daily.round(4).to_dict("records"),
    )


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def make_exchange() -> ccxt.Exchange:
    load_dotenv()
    key, secret = os.getenv("BINANCE_DEMO_API_KEY"), os.getenv("BINANCE_DEMO_API_SECRET")
    if not key or not secret:
        sys.exit("Не найдены BINANCE_DEMO_API_KEY / BINANCE_DEMO_API_SECRET в .env")
    ex = ccxt.binanceusdm({          # USDⓈ-M perpetual; defaultType='future' в ccxt — это поставочные контракты
        "apiKey": key,
        "secret": secret,
        "enableRateLimit": True,     # второй рубеж защиты поверх WeightGovernor
        "options": {"recvWindow": 10_000, "fetchCurrencies": False},
    })
    ex.enable_demo_trading(True)     # demo-fapi.binance.com (НЕ set_sandbox_mode — это старый testnet)
    ex.load_time_difference()        # защита от -1021 (timestamp вне recvWindow)
    return ex


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--days", type=int, default=HISTORY_DAYS, help="глубина истории, дней (Binance хранит ~90)")
    p.add_argument("--symbols", help="список через запятую, напр. BTCUSDT,ETHUSDT (по умолчанию — авто по income)")
    p.add_argument("--output", default="binance_demo_stats.html")
    p.add_argument("--tz", default="UTC", help="часовой пояс отображения, напр. Europe/Moscow (в вебе Binance — локальное время)")
    p.add_argument("--csv", help="сохранить все филлы в CSV")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    if args.days > HISTORY_DAYS:
        log.warning("Binance хранит ~3 месяца истории, --days ограничен %d", HISTORY_DAYS)
        args.days = HISTORY_DAYS

    try:
        ex = make_exchange()
        gov = WeightGovernor(ex, get_weight_limit(ex))
        end = ex.milliseconds()
        start = end - args.days * DAY_MS

        if args.symbols:
            wins = {s.strip().upper(): list(tile_windows(start, end)) for s in args.symbols.split(",") if s.strip()}
        else:
            log.info("Поиск символов с активностью через income...")
            activity = discover_activity(gov, start, end)
            wins = {s: list(week_windows(ts, end)) for s, ts in activity.items()}
        if not wins:
            log.error("Активности не найдено. Проверьте, что ключи — от demo.binance.com и там есть сделки.")
            return 1
        log.info("Символов: %d, запросов userTrades ≈ %d (вес ≈ %d)", len(wins),
                 sum(map(len, wins.values())), W_USER_TRADES * sum(map(len, wins.values())))

        rows = fetch_fills(gov, wins)

        positions, positions_error = None, ""
        try:
            positions = fetch_open_positions(gov)
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:   # отчёт по сделкам важнее — не роняем его
            positions_error = str(e)[:200]
            log.warning("Позиции не загружены: %s", positions_error)

        symbol_cfg = None
        try:
            symbol_cfg = fetch_symbol_config(gov)
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            log.warning("Плечо/режим маржи не загружены (ROI в истории позиций будет «—»): %s", str(e)[:200])
    except ccxt.AuthenticationError as e:
        log.error("Ошибка ключей: %s. Нужны ключи Demo Trading (demo.binance.com), не Testnet и не боевые.", e)
        return 1
    except (ccxt.DDoSProtection, ccxt.RateLimitExceeded) as e:  # 418 / WAF / 429 после всех повторов
        log.error("Binance ограничил доступ (418/429/WAF): %s. Остановлено, повторите позже.", e)
        return 2

    if not rows:
        log.error("Сделок не найдено.")
        return 1

    fills = build_fills_frame(rows, args.tz)
    orders = build_orders_frame(fills)
    stats = compute_stats(orders, fills)
    history, hist_info = build_position_history(fills, symbol_cfg)

    if args.csv:
        fills.to_csv(args.csv, index=False)
        log.info("CSV: %s", args.csv)
    period = f"{fills['time'].min():%Y-%m-%d} … {fills['time'].max():%Y-%m-%d}"
    Path(args.output).write_text(render_html(fills, orders, stats, period, positions, positions_error, history, hist_info), encoding="utf-8")

    pf = "∞" if stats["profit_factor"] == float("inf") else f"{stats['profit_factor']:.2f}"
    log.info("Закрытых позиций: %d (пропущено закрытий до окна: %d, ещё открыто: %d)",
             len(history), hist_info["orphan_fills"], hist_info["still_open"])
    log.info("✅ %s | ордеров %d | net $%.2f | win rate net %.1f%% / gross %.1f%% | PF %s | открытых позиций: %s",
             args.output, stats["orders"], stats["net"], stats["win_rate"], stats["win_rate_gross"], pf,
             "н/д" if positions is None else len(positions))
    return 0


if __name__ == "__main__":
    sys.exit(main())
