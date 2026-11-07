#!/usr/bin/env python3
"""Статистика торговли на Binance USDⓈ-M Futures **Demo Trading** (не Testnet).

Источники в документации Binance (developers.binance.com, USDⓈ-M Futures):
  * REST demo:   https://demo-fapi.binance.com  (ключи — только с demo.binance.com)
  * GET /fapi/v1/userTrades : weight 5, limit<=1000, история ~3 месяца; выгружаем постранично по fromId
                              (с 0 — с начала; fromId нельзя вместе со startTime/endTime, окно времени <= 7 дней)
  * GET /fapi/v1/income     : weight 30, limit<=1000, история ~3 месяца
  * Лимиты: REQUEST_WEIGHT (на IP, 1 минута), читаются из exchangeInfo в рантайме;
    429 -> пауза, повтор 429 -> 418 (бан IP на 2 мин ... 3 дня) -> нельзя долбить дальше.

Просадки позиций (MAE) считаются по РЕАЛЬНЫМ свечам Binance Futures (fapi.binance.com, публичный API,
без ключей): GET /fapi/v1/klines, вес зависит от limit (1 / 2 / 5 / 10), максимум 1500 свечей за запрос.

fills.csv — постоянный кэш: сделки дозагружаются только новые, рассчитанные просадки хранятся в колонках
dd_price / dd_time / dd_status на закрывающем филле позиции и повторно не запрашиваются.

Лимиты Binance используются только на SAFETY_FACTOR (50%) — и для demo-хоста, и для публичного mainnet-хоста.

Зависимости:  pip install ccxt pandas python-dotenv tenacity jinja2
"""
from __future__ import annotations

import argparse
import heapq
import logging
import os
import sys
import time
from pathlib import Path
from typing import Callable

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
PAGE_LIMIT = 1000               # максимум limit у userTrades / income
HISTORY_DAYS = 90               # Binance хранит ~3 месяца

SAFETY_FACTOR = 0.5             # используем не более 50% от лимитов Binance
FALLBACK_WEIGHT_LIMIT_1M = 2400 # если exchangeInfo не отдал rateLimits

# Веса эндпоинтов (по документации). Реальный расход всё равно берётся из
# заголовка X-MBX-USED-WEIGHT-1M, эти числа нужны только для "упреждающей" паузы.
W_USER_TRADES = 5
W_INCOME = 30
W_SYMBOL_CONFIG = 5             # GET /fapi/v1/symbolConfig (плечо и режим маржи; в ccxt cost=5)
W_EXCHANGE_INFO = 1
W_POSITION_RISK = 5             # GET /fapi/v3/positionRisk (в ccxt cost=1 — занижен, берём из доки)

QUOTE_ASSETS = {"USDT", "USDC"}  # комиссии в других активах (BNB) в сумму не входят
MAX_ORDERS_IN_HTML = 1000        # полная история — в CSV
MAX_POSITIONS_IN_HTML = 1000
MAX_FILLS_IN_HTML = 2000
M5_MS = 300_000                 # 5-минутная свеча
KLINE_MAX = 1500                # максимум свечей за один запрос /fapi/v1/klines
OVERLAP_MS = 10 * 60_000        # перехлёст при дозагрузке сделок (дубли отсекаются по id)
DD_COLS = ["dd_price", "dd_time", "dd_status", "dd_key"]   # кэш просадки в fills.csv: экстремум цены / время (ms) / ok|no_data / окно+сторона
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
        self.calls = 0
        self.spent = 0                               # суммарный вес по документации (оценка)
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
        self.calls += 1
        self.spent += cost
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


def discover_symbols(gov: WeightGovernor, start: int, end: int) -> set[str]:
    """Символы с любой активностью по income БЕЗ фильтра типа (COMMISSION, REALIZED_PNL, FUNDING_FEE...).

    Мейкерские сделки на demo бывают с нулевой комиссией и нулевым PnL — по одной COMMISSION их не найти.
    """
    rows = paged(
        lambda cur: gov.call(gov.ex.fapiPrivateGetIncome,
                             {"startTime": cur, "endTime": end, "limit": PAGE_LIMIT}, W_INCOME),
        start,
        uid=lambda r: (r.get("tranId"), r.get("incomeType"), r.get("tradeId"), r.get("time")),
    )
    return {r["symbol"] for r in rows if r.get("symbol")}


def fetch_fills(gov: WeightGovernor, plan: dict[str, int], rows: list[dict], done: set[str]) -> None:
    """Сделки по каждому символу постранично по fromId (id растёт по символу; fromId=0 — с начала истории).

    Не зависит от окон по времени (лимит 7 дней), комиссии и PnL, и стоит 1 запрос на символ при дозагрузке.
    rows/done изменяются «на месте», чтобы при бане/429 уже загруженное не терялось.
    """
    for symbol in sorted(plan):
        cursor, before = plan[symbol], len(rows)
        while True:
            page = gov.call(gov.ex.fapiPrivateGetUserTrades,
                            {"symbol": symbol, "fromId": cursor, "limit": PAGE_LIMIT}, W_USER_TRADES)
            rows.extend(page)
            if len(page) < PAGE_LIMIT:
                break
            cursor = max(int(r["id"]) for r in page) + 1
        done.add(symbol)
        log.info("%-14s новых сделок: %-6d (с id %d)", symbol, len(rows) - before, plan[symbol])


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
# Кэш fills.csv
# --------------------------------------------------------------------------- #
def load_fills_cache(path: Path) -> pd.DataFrame:
    """Читает fills.csv (в том числе старого формата с ISO-временем) -> сырые филлы, time = epoch ms."""
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, dtype=str)
    if df.empty or not {"symbol", "id", "time"} <= set(df.columns):
        return pd.DataFrame()
    if not df["time"].str.fullmatch(r"\d+").all():           # старый формат: время строкой с таймзоной
        df["time"] = (pd.to_datetime(df["time"], utc=True) - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(milliseconds=1)
        df.attrs["legacy"] = True
    for col in ("dd_price", "dd_time"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def merge_fills(cache: pd.DataFrame, new_rows: list[dict]) -> pd.DataFrame:
    """Кэш + новые сделки. Дубли отсекаются по (symbol, id), записи кэша (с dd_*) имеют приоритет."""
    parts = [d for d in (cache, pd.DataFrame(new_rows)) if len(d)]
    if not parts:
        return pd.DataFrame()
    raw = pd.concat(parts, ignore_index=True)
    for col in ("id", "orderId", "time"):
        raw[col] = raw[col].astype("int64")
    for col in (*DD_COLS, "src"):                      # src="id": символ выгружен полностью по fromId
        if col not in raw:
            raw[col] = np.nan
    raw["src"] = raw["src"].astype(object)
    raw = raw.drop_duplicates(["symbol", "id"], keep="first")
    return raw.sort_values(["time", "id"]).reset_index(drop=True)


def save_fills_cache(raw: pd.DataFrame, path: Path) -> None:
    """Атомарная запись: CSV нельзя править «на месте», поэтому пишем во временный файл и подменяем."""
    out = raw.copy()
    out["time_utc"] = pd.to_datetime(out["time"], unit="ms", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")
    out["fee_usdt"] = pd.to_numeric(out["commission"], errors="coerce").where(
        out["commissionAsset"].isin(QUOTE_ASSETS), 0.0)
    tmp = path.with_name(path.name + ".tmp")
    out.to_csv(tmp, index=False)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# Просадка позиций (MAE) по реальным свечам Binance Futures
# --------------------------------------------------------------------------- #
def kline_weight(limit: int) -> int:
    """Вес GET /fapi/v1/klines по документации: [1,100)=1, [100,500)=2, [500,1000]=5, >1000=10."""
    return 1 if limit < 100 else 2 if limit < 500 else 5 if limit <= 1000 else 10


def n_candles(a: int, b: int, step: int = M5_MS) -> int:
    return b // step - a // step + 1


def fetch_klines(gov: WeightGovernor, symbol: str, interval: str, start: int, end: int) -> np.ndarray:
    """Свечи [open_time, high, low] за [start, end]. startTime выравниваем на границу свечи,
    иначе Binance пропустит свечу, внутри которой лежит startTime."""
    step = {"5m": M5_MS, "1d": DAY_MS}[interval]
    start -= start % step
    limit = min(n_candles(start, end, step), KLINE_MAX)
    rows = gov.call(gov.ex.fapiPublicGetKlines,
                    {"symbol": symbol, "interval": interval, "startTime": start, "endTime": end, "limit": limit},
                    kline_weight(limit))
    return np.array([[r[0], r[2], r[3]] for r in rows], dtype=float).reshape(-1, 3)


def extreme_in(candles: np.ndarray, is_long: bool, a: int, b: int) -> tuple[float, int] | None:
    """Худшая для позиции цена (low для лонга, high для шорта) и время её 5m-свечи в окне [a, b]."""
    c = candles[(candles[:, 0] >= a - a % M5_MS) & (candles[:, 0] <= b)]
    if not len(c):
        return None
    i = int(c[:, 2].argmin() if is_long else c[:, 1].argmax())
    return float(c[i, 2] if is_long else c[i, 1]), int(c[i, 0])


def plan_groups(spans: list[tuple[int, int]]) -> list[list[int]]:
    """Объединяет близкие короткие позиции (отсортированы по началу) в один запрос свечей,
    пока это не дороже по весу, чем отдельные запросы, и укладывается в KLINE_MAX."""
    groups: list[list[int]] = []
    cur: list[int] = []
    a = b = 0
    for i, (s_i, e_i) in enumerate(spans):
        if cur:
            nb = max(b, e_i)
            merged = n_candles(a, nb)
            if merged <= KLINE_MAX and kline_weight(merged) <= kline_weight(n_candles(a, b)) + kline_weight(n_candles(s_i, e_i)):
                cur.append(i)
                b = nb
                continue
            groups.append(cur)
        cur, a, b = [i], s_i, e_i
    if cur:
        groups.append(cur)
    return groups


def long_position_extreme(gov: WeightGovernor, symbol: str, is_long: bool, a: int, b: int) -> tuple[float, int] | None:
    """Позиция длиннее одного запроса 5m: 1d -> 5m (branch & bound).

    Дневная свеча даёт нижнюю границу экстремума. Целые дни точны, крайние (неполные) дни — только граница.
    Берём день с лучшей границей: неполный — уточняем 5m-свечами его части внутри позиции,
    целый — один раз спускаемся на 5m, чтобы получить время экстремума. Промежуточный таймфрейм 1h не нужен:
    сутки на 5m — это 288 свечей, один запрос. Обычно 2–4 запроса на позицию.
    """
    days = fetch_klines(gov, symbol, "1d", a, b)
    heap = [(c[2] if is_long else -c[1], int(c[0]), False, 0.0, 0) for c in days]
    heapq.heapify(heap)
    while heap:
        _, day_start, resolved, price, ts = heapq.heappop(heap)
        if resolved:
            return price, ts
        lo, hi = max(a, day_start), min(b, day_start + DAY_MS - 1)
        ext = extreme_in(fetch_klines(gov, symbol, "5m", lo, hi), is_long, lo, hi)
        if ext:
            heapq.heappush(heap, (ext[0] if is_long else -ext[0], day_start, True, ext[0], ext[1]))
    return None


def _dd_key(r) -> str:
    """Окно и сторона, для которых посчитан экстремум: при изменении состава сделок кэш считается устаревшим."""
    return f"{int(r.open_ms)}-{int(r.close_ms)}-{r.side}"


def compute_drawdowns(gov: WeightGovernor, todo: pd.DataFrame, results: dict) -> None:
    """Заполняет results[(symbol, close_fill_id)] = (цена экстремума, время ms, статус).
    results изменяется «на месте», чтобы при бане/429 посреди работы уже посчитанное не терялось."""
    for symbol, g in todo.groupby("symbol"):
        g = g.sort_values("open_ms")
        key = lambda r: (symbol, int(r.close_fill_id))
        n5 = g.apply(lambda r: n_candles(int(r.open_ms), int(r.close_ms)), axis=1)
        short, long_ = g[n5 <= KLINE_MAX], g[n5 > KLINE_MAX]
        try:
            rows = list(short.itertuples(index=False))
            for grp in plan_groups([(int(r.open_ms), int(r.close_ms)) for r in rows]):
                a, b = int(rows[grp[0]].open_ms), max(int(rows[i].close_ms) for i in grp)
                candles = fetch_klines(gov, symbol, "5m", a, b)
                for i in grp:
                    r = rows[i]
                    ext = extreme_in(candles, r.side == "LONG", int(r.open_ms), int(r.close_ms))
                    results[key(r)] = (*ext, "ok", _dd_key(r)) if ext else (np.nan, np.nan, "no_data", _dd_key(r))
            for r in long_.itertuples(index=False):
                ext = long_position_extreme(gov, symbol, r.side == "LONG", int(r.open_ms), int(r.close_ms))
                results[key(r)] = (*ext, "ok", _dd_key(r)) if ext else (np.nan, np.nan, "no_data", _dd_key(r))
        except ccxt.BadSymbol:                      # на mainnet такого контракта нет (например, только на demo)
            log.warning("%s: на боевом Binance Futures нет свечей (символ не найден) — просадка недоступна", symbol)
            for r in g.itertuples(index=False):
                results.setdefault(key(r), (np.nan, np.nan, "no_data", _dd_key(r)))
        except ccxt.ExchangeError as e:             # не кэшируем: возможно временная/параметрическая ошибка
            log.warning("%s: ошибка запроса свечей, пропускаем (не кэшируется): %s", symbol, str(e)[:200])


def attach_drawdowns(history: pd.DataFrame, raw: pd.DataFrame, get_public_gov: Callable[[], WeightGovernor] | None,
                     force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Добавляет в историю позиций просадку: берёт из кэша fills.csv, недостающее досчитывает по свечам.
    Возвращает (история, обновлённый raw для сохранения, число новых расчётов)."""
    hist = history.copy()
    new_count = 0
    if hist.empty:
        for col in (*DD_COLS, "dd_pct", "dd_usdt", "dd_at"):
            hist[col] = pd.Series(dtype="float64")
        return hist, raw, 0

    table = raw.set_index(["symbol", "id"])
    for col in ("dd_status", "dd_key"):
        table[col] = table[col].astype(object)                      # иначе запись строк в float-колонку
    idx = pd.MultiIndex.from_arrays([hist["symbol"], hist["close_fill_id"]])
    expected = (hist["open_ms"].astype("int64").astype(str) + "-" + hist["close_ms"].astype("int64").astype(str)
                + "-" + hist["side"]).to_numpy()
    cached = table[DD_COLS].reindex(idx)
    valid = cached["dd_status"].notna().to_numpy() & (cached["dd_key"].to_numpy() == expected)
    todo = hist[~valid | force]

    results: dict = {}
    if len(todo) and get_public_gov is not None:
        log.info("Просадка: нужно рассчитать позиций: %d (из кэша: %d)", len(todo), len(hist) - len(todo))
        try:
            compute_drawdowns(get_public_gov(), todo, results)
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:        # 418/429/сеть — сохраняем уже посчитанное
            log.error("Загрузка свечей прервана: %s. Рассчитанное сохранено, остальное — при следующем запуске.", str(e)[:200])
    elif len(todo):
        log.info("Просадка: %d позиций без данных (расчёт отключён --no-dd)", len(todo))

    if results:
        new_count = len(results)
        for i, col in enumerate(DD_COLS):
            ser = pd.Series({k: v[i] for k, v in results.items()})
            table.loc[ser.index, col] = ser
        raw = table.reset_index()
    cached = table[DD_COLS].reindex(idx)
    valid = cached["dd_status"].notna().to_numpy() & (cached["dd_key"].to_numpy() == expected)
    for col in ("dd_price", "dd_time", "dd_status"):
        hist[col] = np.where(valid, cached[col].to_numpy(), np.nan)

    entry = hist["entry_price"]
    adverse = np.where(hist["side"] == "LONG", entry - hist["dd_price"], hist["dd_price"] - entry)
    pct = pd.Series(adverse / entry * 100, index=hist.index).clip(lower=0)      # цена не дошла до худшего, чем вход -> 0
    hist["dd_pct"] = -pct
    hist["dd_usdt"] = -pct / 100 * hist["peak_usdt"]                             # от стоимости пиковой позиции (оценка)
    hist["dd_at"] = pd.to_datetime(hist["dd_time"], unit="ms", utc=True).dt.tz_convert(hist["close_time"].dt.tz)
    return hist, raw, new_count


# --------------------------------------------------------------------------- #
# Аналитика
# --------------------------------------------------------------------------- #
def build_fills_frame(rows: list[dict], tz: str = "UTC") -> pd.DataFrame:
    df = pd.DataFrame(rows).drop_duplicates(subset=["symbol", "id"])
    if "positionSide" not in df:
        df["positionSide"] = "BOTH"
    df["id"] = df["id"].astype("int64")
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


def _to_ms(ts: pd.Series) -> pd.Series:
    """tz-aware datetime -> epoch ms (не зависит от внутренней единицы времени pandas)."""
    return (ts - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(milliseconds=1)


def build_position_history(fills: pd.DataFrame, config: dict[str, dict] | None = None) -> tuple[pd.DataFrame, dict]:
    """«История позиций» как в вебе Binance. Публичного эндпоинта нет — восстанавливаем из филлов.

    Позиция живёт от выхода из нуля до возврата в ноль (отдельно по symbol + positionSide).
    В one-way режиме филл может перевернуть позицию — он делится на закрывающую и открывающую части
    (комиссия — пропорционально, realizedPnl целиком относится к закрывающей части).
    Entry/Avg close — средневзвешенные без комиссий. Объём входа = Σ открывающих сделок (оборот; у закрытой
    позиции равен Σ закрывающих). Пик позиции — наибольший одновременный размер и его стоимость по цене входа,
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
                               close_notional=0.0, size=0.0, max_size=0.0, cost=0.0, peak_cost=0.0, pnl=0.0, fee=0.0)
                if direction == cur["dir"]:                   # увеличение позиции
                    cur["open_qty"] += left
                    cur["open_notional"] += left * f.price
                    cur["size"] += left
                    cur["max_size"] = max(cur["max_size"], cur["size"])
                    cur["cost"] += left * f.price                 # стоимость открытой части по цене входа
                    cur["peak_cost"] = max(cur["peak_cost"], cur["cost"])
                    cur["fee"] += f.fee_usdt * left / f.qty
                    left = 0.0
                else:                                         # уменьшение / закрытие
                    closing = min(left, cur["size"])
                    cur["close_qty"] += closing
                    cur["close_notional"] += closing * f.price
                    cur["pnl"] += pnl_left
                    pnl_left = 0.0
                    cur["fee"] += f.fee_usdt * closing / f.qty
                    cur["cost"] -= cur["cost"] * closing / cur["size"]   # метод средней цены
                    cur["size"] -= closing
                    left -= closing
                    if cur["size"] <= cur["max_size"] * 1e-9:  # вернулись в ноль -> позиция закрыта
                        cfg = config.get(symbol, {})
                        lev = float(cfg["leverage"]) if cfg.get("leverage") else np.nan
                        entry = cur["open_notional"] / cur["open_qty"]
                        net = cur["pnl"] - cur["fee"]
                        margin = cur["peak_cost"] / lev               # маржа под пиковый размер позиции
                        rows.append(dict(
                            symbol=symbol, side="LONG" if cur["dir"] > 0 else "SHORT", position_mode=pside,
                            close_fill_id=int(f.id),
                            margin_mode=cfg.get("marginType"), leverage=lev,
                            open_time=cur["open_time"], close_time=f.time,
                            entry_price=entry, avg_close_price=cur["close_notional"] / cur["close_qty"],
                            open_qty=cur["open_qty"], open_usdt=cur["open_notional"],
                            peak_qty=cur["max_size"], peak_usdt=cur["peak_cost"], closed_qty=cur["close_qty"],
                            closing_pnl=cur["pnl"], fee=cur["fee"], net=net,
                            roi=net / margin * 100 if margin == margin and margin else np.nan,
                        ))
                        cur = None
        still_open += cur is not None
    cols = ["symbol", "side", "position_mode", "close_fill_id", "margin_mode", "leverage", "open_time", "close_time", "entry_price",
            "avg_close_price", "open_qty", "open_usdt", "peak_qty", "peak_usdt", "closed_qty", "closing_pnl", "fee", "net", "roi"]
    hist = pd.DataFrame(rows, columns=cols).sort_values("close_time").reset_index(drop=True)
    hist["open_ms"] = _to_ms(hist["open_time"])
    hist["close_ms"] = _to_ms(hist["close_time"])
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

<div data-filter class="bg-white p-6 rounded-lg shadow mb-8"><h2 class="text-xl font-semibold mb-2">История позиций</h2>
<p class="text-xs text-gray-500 mb-3">Закрытых позиций: {{ hist_total }}{% if hist|length < hist_total %} (показаны последние {{ hist|length }}){% endif %}.
 Восстановлено из сделок (публичного API истории позиций у Binance нет): позиция живёт от выхода из нуля до возврата в ноль.
 При сеточной торговле внутри одной позиции много входов и выходов, поэтому два объёма — это разные величины:
 <b>Объём входа</b> — сумма всех открывающих сделок за жизнь позиции (оборот; у закрытой позиции равен сумме закрывающих),
 <b>Пик позиции</b> — наибольший размер, который держали одновременно (USDT — по цене входа этой части); это задействованный капитал,
 поэтому Объём входа может быть больше Пика, но не наоборот.
 Цены без комиссий. Net PnL = Closing PnL − комиссии. ROI = Net PnL / (Пик позиции, USDT / плечо).
 Плечо и режим маржи — <b>текущие</b> настройки символа.
 <b>Макс. просадка</b> (MAE) — худшее отклонение цены против позиции от средней цены входа за время её жизни по реальным
 5-минутным свечам Binance Futures (mainnet): лонг — по Low, шорт — по High; в USDT — от стоимости пика позиции (оценка).
 Рассчитано для {{ dd_cov[0] }} из {{ dd_cov[1] }} позиций.{% if hist_info.orphan_fills %}
 Пропущено закрытий позиций, открытых раньше окна истории: {{ hist_info.orphan_fills }}.{% endif %}{% if hist_info.still_open %}
 Ещё открыто (см. блок выше): {{ hist_info.still_open }}.{% endif %}</p>
<div class="flex flex-wrap items-center gap-2 mb-3 text-sm">
 {% for d, label in [(1,'1Д'),(7,'1Н'),(30,'1М'),(0,'Всё')] %}<button type="button" data-days="{{ d }}" class="range-btn px-3 py-1 rounded border {{ 'bg-gray-800 text-white' if d == 0 else 'bg-white' }}">{{ label }}</button>{% endfor %}
 <input type="search" placeholder="Символ…" class="sym-filter ml-2 px-3 py-1 rounded border">
</div>
{% if hist %}<div class="overflow-x-auto max-h-[32rem] overflow-y-auto"><table class="min-w-full sortable text-sm">
<thead class="sticky top-0 bg-gray-50"><tr>
 <th class="px-3 py-2 text-left">Символ</th><th class="px-3 py-2 text-left">Открыта</th><th class="px-3 py-2 text-left">Закрыта</th>
 <th class="px-3 py-2 text-right">Длительность</th><th class="px-3 py-2 text-right">Closing PnL, USDT</th>
 <th class="px-3 py-2 text-right">Комиссии, USDT</th><th class="px-3 py-2 text-right">Net PnL, USDT</th>
 <th class="px-3 py-2 text-right">ROI, %</th><th class="px-3 py-2 text-right">Макс. просадка, USDT / %</th>
 <th class="px-3 py-2 text-right">Объём входа, Coin</th><th class="px-3 py-2 text-right">Объём входа, USDT</th>
 <th class="px-3 py-2 text-right">Пик позиции, Coin</th><th class="px-3 py-2 text-right">Пик позиции, USDT</th>
 <th class="px-3 py-2 text-right">Цена входа (ср.), USDT</th><th class="px-3 py-2 text-right">Цена закрытия (ср.), USDT</th></tr></thead><tbody>
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
 <td class="px-3 py-2 text-right whitespace-nowrap {{ 'text-red-600' if h.dd_usdt_v < 0 else 'text-gray-500' }}" data-v="{{ h.dd_usdt_v }}" title="{{ h.dd_title }}">{{ h.dd_str }}<br><span class="text-xs text-gray-500">{{ h.dd_pct_str }}</span></td>
 <td class="px-3 py-2 text-right" data-v="{{ h.open_qty }}">{{ h.open_qty_str }}</td>
 <td class="px-3 py-2 text-right" data-v="{{ h.open_usdt }}">{{ money(h.open_usdt) }}</td>
 <td class="px-3 py-2 text-right" data-v="{{ h.peak_qty }}">{{ h.peak_qty_str }}</td>
 <td class="px-3 py-2 text-right" data-v="{{ h.peak_usdt }}">{{ money(h.peak_usdt) }}</td>
 <td class="px-3 py-2 text-right" data-v="{{ h.entry_price }}">{{ h.entry_str }}</td>
 <td class="px-3 py-2 text-right" data-v="{{ h.avg_close_price }}">{{ h.close_price_str }}</td></tr>
{% endfor %}</tbody></table></div>
{% else %}<p class="text-sm text-gray-500">Закрытых позиций за период нет.</p>{% endif %}</div>

<div data-filter class="bg-white p-6 rounded-lg shadow mb-8"><h2 class="text-xl font-semibold mb-2">История сделок</h2>
<p class="text-xs text-gray-500 mb-3">Исполнения как во вкладке «История сделок» Binance: одна строка — одно исполнение (у ордера их может быть несколько).
 Всего: {{ fills_total }}{% if fill_rows|length < fills_total %}, показаны последние {{ fill_rows|length }} (полный список — в fills.csv){% endif %}.
 «Объём, USDT» = цена × количество.</p>
<div class="flex flex-wrap items-center gap-2 mb-3 text-sm">
 {% for d, label in [(1,'1Д'),(7,'1Н'),(30,'1М'),(0,'Всё')] %}<button type="button" data-days="{{ d }}" class="range-btn px-3 py-1 rounded border {{ 'bg-gray-800 text-white' if d == 0 else 'bg-white' }}">{{ label }}</button>{% endfor %}
 <input type="search" placeholder="Символ…" class="sym-filter ml-2 px-3 py-1 rounded border">
</div>
{% if fill_rows %}<div class="overflow-x-auto max-h-[32rem] overflow-y-auto"><table class="min-w-full sortable text-sm">
<thead class="sticky top-0 bg-gray-50"><tr>
 <th class="px-3 py-2 text-left">№ ордера</th><th class="px-3 py-2 text-left">Время</th><th class="px-3 py-2 text-left">Тикер</th>
 <th class="px-3 py-2 text-left">Сторона</th><th class="px-3 py-2 text-right">Цена, USDT</th>
 <th class="px-3 py-2 text-right">Количество, Coin</th><th class="px-3 py-2 text-right">Объём, USDT</th>
 <th class="px-3 py-2 text-right">Комиссия, USDT</th><th class="px-3 py-2 text-left">Роль</th>
 <th class="px-3 py-2 text-right">Реализованная прибыль, USDT</th></tr></thead><tbody>
{% for f in fill_rows %}<tr class="border-t hover:bg-gray-50" data-ts="{{ f.ts }}" data-sym="{{ f.symbol|lower }}">
 <td class="px-3 py-1" data-v="{{ f.order_id }}">{{ f.order_id }}</td>
 <td class="px-3 py-1 whitespace-nowrap" data-v="{{ f.time_str }}">{{ f.time_str }}</td>
 <td class="px-3 py-1 whitespace-nowrap" data-v="{{ f.symbol }}">{{ f.symbol }} <span class="ml-1 px-1 rounded bg-gray-200 text-xs">Бессрочный</span></td>
 <td class="px-3 py-1 whitespace-nowrap {{ 'text-green-600' if f.buy else 'text-red-600' }}" data-v="{{ f.side_label }}">{{ f.side_label }}</td>
 <td class="px-3 py-1 text-right" data-v="{{ f.price }}">{{ f.price_str }}</td>
 <td class="px-3 py-1 text-right" data-v="{{ f.qty }}">{{ f.qty_str }}</td>
 <td class="px-3 py-1 text-right" data-v="{{ f.quote }}">{{ money(f.quote) }}</td>
 <td class="px-3 py-1 text-right text-gray-500" data-v="{{ f.fee }}">{{ f.fee_str }}</td>
 <td class="px-3 py-1" data-v="{{ f.role }}">{{ f.role }}</td>
 <td class="px-3 py-1 text-right {{ clr(f.realized) }}" data-v="{{ f.realized }}">{{ '%.8f'|format(f.realized) }}</td></tr>
{% endfor %}</tbody></table></div>
{% else %}<p class="text-sm text-gray-500">Сделок нет.</p>{% endif %}</div>

<div class="bg-white p-6 rounded-lg shadow mb-8"><h2 class="text-xl font-semibold mb-4">Кумулятивный Net PnL (по дням)</h2>
 <canvas id="chart" height="100"></canvas></div>

<div class="bg-white p-6 rounded-lg shadow mb-8"><div class="flex items-center justify-between mb-1">
 <h2 class="text-xl font-semibold">Просадки (по позициям)</h2>
 <div class="text-sm"><button type="button" data-mode="usdt" class="mode-btn px-3 py-1 rounded border bg-gray-800 text-white">USDT</button>
 <button type="button" data-mode="pct" class="mode-btn px-3 py-1 rounded border bg-white">%</button></div></div>
 <p class="text-xs text-gray-500 mb-3">Макс. просадка каждой закрытой позиции — те же значения, что в колонке «Макс. просадка» таблицы «История позиций», по порядку закрытия.
 <span class="text-blue-600">Синие</span> — лонг (по Low свечей), <span class="text-orange-500">оранжевые</span> — шорт (по High свечей); отсчёт от средней цены входа.</p>
 {% if chart_pos %}<canvas id="pdchart" height="90"></canvas>{% else %}<p class="text-sm text-gray-500">Нет данных по просадкам.</p>{% endif %}</div>

<div class="bg-white p-6 rounded-lg shadow mb-8"><h2 class="text-xl font-semibold mb-1">Просадки (по дням)</h2>
 <p class="text-xs text-gray-500 mb-3">Линия — просадка кумулятивного Net PnL от исторического максимума (макс. {{ '{:,.2f}'.format(s.max_drawdown) }} USDT по ордерам).
 Столбцы — худшая просадка (MAE, USDT) среди позиций, закрытых в этот день, по реальным 5m-свечам.</p>
 <canvas id="ddchart" height="90"></canvas></div>

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
document.querySelectorAll('[data-filter]').forEach(sec=>{
 const rows=[...sec.querySelectorAll('tbody tr')], btns=[...sec.querySelectorAll('.range-btn')];let days=0,q='';
 const apply=()=>{const lim=days?Date.now()-days*864e5:0;
  rows.forEach(r=>{r.hidden=!((!lim||Number(r.dataset.ts)>=lim)&&r.dataset.sym.includes(q));});};
 btns.forEach(b=>b.addEventListener('click',()=>{days=Number(b.dataset.days);
  btns.forEach(x=>{x.classList.toggle('bg-gray-800',x===b);x.classList.toggle('text-white',x===b);});apply();}));
 sec.querySelector('.sym-filter').addEventListener('input',e=>{q=e.target.value.trim().toLowerCase();apply();});
});
const d={{ daily|tojson }};
new Chart(document.getElementById('chart'),{type:'line',
  data:{labels:d.map(x=>x.date),datasets:[{label:'Net PnL',data:d.map(x=>x.cum),
   borderColor:'rgb(34,197,94)',backgroundColor:'rgba(34,197,94,0.1)',fill:true,tension:0.1,pointRadius:2}]},
  options:{responsive:true,plugins:{legend:{display:false}},scales:{y:{ticks:{callback:v=>'$'+v.toFixed(0)}}}}});
const dd={{ dd_daily|tojson }};
new Chart(document.getElementById('ddchart'),{type:'bar',
  data:{labels:dd.map(x=>x.date),datasets:[
   {type:'bar',label:'Худшая просадка позиции за день (MAE, USDT)',data:dd.map(x=>x.pos_dd),backgroundColor:'rgba(239,68,68,0.45)'},
   {type:'line',label:'Просадка Net PnL от пика (USDT)',data:dd.map(x=>x.eq_dd),borderColor:'rgb(220,38,38)',
    backgroundColor:'rgba(220,38,38,0.1)',fill:true,tension:0.1,pointRadius:2}]},
  options:{responsive:true,scales:{y:{ticks:{callback:v=>'$'+v.toFixed(2)}}}}});
const pdata={{ chart_pos|tojson }};
if(pdata.length){let mode='usdt';
 const val=x=>mode==='usdt'?x.usdt:x.pct;
 const pc=new Chart(document.getElementById('pdchart'),{type:'bar',
  data:{labels:pdata.map(x=>x.t),datasets:[{label:'Макс. просадка',data:pdata.map(val),
   backgroundColor:pdata.map(x=>x.side==='LONG'?'rgba(37,99,235,0.6)':'rgba(249,115,22,0.6)')}]},
  options:{responsive:true,plugins:{legend:{display:false},tooltip:{callbacks:{
    title:i=>{const p=pdata[i[0].dataIndex];return p.symbol+' · '+(p.side==='LONG'?'Лонг (по Low)':'Шорт (по High)');},
    label:c=>{const p=pdata[c.dataIndex];return 'Просадка: '+p.usdt.toFixed(4)+' USDT / '+p.pct.toFixed(2)+'%';},
    afterLabel:c=>{const p=pdata[c.dataIndex];return 'Закрыта: '+p.t+' · худшая цена '+p.price+' ('+p.at+')';}}}},
   scales:{x:{ticks:{maxTicksLimit:12}},y:{ticks:{callback:v=>mode==='usdt'?v.toFixed(2):v.toFixed(1)+'%'}}}}});
 const mb=[...document.querySelectorAll('.mode-btn')];
 mb.forEach(b=>b.addEventListener('click',()=>{mode=b.dataset.mode;pc.data.datasets[0].data=pdata.map(val);
  mb.forEach(x=>{x.classList.toggle('bg-gray-800',x===b);x.classList.toggle('text-white',x===b);});pc.update();}));}
</script></body></html>
"""


def _num(v: float) -> str:
    return np.format_float_positional(v, precision=8, trim="-")


def _duration(td: pd.Timedelta) -> str:
    d, rest = divmod(int(td.total_seconds()), 86_400)
    h, rest = divmod(rest, 3_600)
    return " ".join(x for x in (f"{d}д" if d else "", f"{h}ч" if d or h else "", f"{rest // 60}мин") if x)


def _dd_fields(r) -> dict:
    if r.dd_usdt != r.dd_usdt:                                # NaN: нет данных свечей
        return {"dd_usdt_v": 0.0, "dd_str": "—", "dd_pct_str": "", "dd_title": "Нет данных свечей"}
    return {"dd_usdt_v": r.dd_usdt, "dd_str": f"{r.dd_usdt:.4f}", "dd_pct_str": f"{r.dd_pct:.2f}%",
            "dd_title": f"Худшая цена {_num(r.dd_price)} · {r.dd_at:%Y-%m-%d %H:%M}"}


def drawdown_by_day(daily: pd.DataFrame, history: pd.DataFrame) -> list[dict]:
    """Дневной ряд: просадка Net PnL от пика + худшая MAE среди позиций, закрытых в этот день."""
    out = daily[["date", "cum"]].copy()
    out["eq_dd"] = out["cum"] - out["cum"].cummax().clip(lower=0)
    worst = (history.assign(date=history["close_time"].dt.strftime("%Y-%m-%d")).groupby("date")["dd_usdt"].min()
             if len(history) else pd.Series(dtype="float64"))
    out["pos_dd"] = out["date"].map(worst)
    out = out.drop(columns="cum").round(4).astype(object)
    return out.where(out.notna(), None).to_dict("records")       # NaN -> null (валидный JSON)


def _history_row(r) -> dict:
    roi_ok = r.roi == r.roi                                   # not NaN
    return {
        "symbol": r.symbol, "side": r.side, "lev": int(r.leverage) if r.leverage == r.leverage else None,
        "margin_ru": MARGIN_MODE_RU.get(r.margin_mode), "ts": int(r.close_time.timestamp() * 1000),
        "open_str": r.open_time.strftime("%Y-%m-%d %H:%M:%S"), "close_str": r.close_time.strftime("%Y-%m-%d %H:%M:%S"),
        "dur_s": int((r.close_time - r.open_time).total_seconds()), "dur_str": _duration(r.close_time - r.open_time),
        "closing_pnl": r.closing_pnl, "fee": r.fee, "net": r.net,
        "roi_v": r.roi if roi_ok else 0.0, "roi_str": f"{r.roi:+.2f}%" if roi_ok else "—",
        "open_qty": r.open_qty, "open_qty_str": _num(r.open_qty), "open_usdt": r.open_usdt,
        "peak_qty": r.peak_qty, "peak_qty_str": _num(r.peak_qty), "peak_usdt": r.peak_usdt,
        "entry_price": r.entry_price, "entry_str": _num(r.entry_price),
        "avg_close_price": r.avg_close_price, "close_price_str": _num(r.avg_close_price),
        **_dd_fields(r),
    }


def _side_label(side: str, position_side: str) -> str:
    """Подпись стороны как в UI Binance: hedge-режим — «Открыть/Закрыть лонг/шорт», one-way — «Купить/Продать»."""
    if position_side == "LONG":
        return "Открыть лонг" if side == "BUY" else "Закрыть лонг"
    if position_side == "SHORT":
        return "Открыть шорт" if side == "SELL" else "Закрыть шорт"
    return "Купить" if side == "BUY" else "Продать"


def _fill_rows(fills: pd.DataFrame) -> list[dict]:
    """Сделки (филлы) как во вкладке «История сделок» UI Binance, новые сверху."""
    f = fills.sort_values(["time", "id"], ascending=False).head(MAX_FILLS_IN_HTML)
    is_maker = f["maker"].astype(str).str.lower().eq("true") if "maker" in f else pd.Series(False, index=f.index)
    rows = []
    for r, maker in zip(f.itertuples(index=False), is_maker):
        quote = r.quoteQty if r.quoteQty else r.price * r.qty
        fee_str = f"{r.commission:.8f}" + ("" if r.commissionAsset in QUOTE_ASSETS else f" {r.commissionAsset}")
        rows.append({
            "order_id": int(r.orderId), "ts": int(r.time.timestamp() * 1000),
            "time_str": r.time.strftime("%Y-%m-%d %H:%M:%S"), "symbol": r.symbol,
            "side_label": _side_label(r.side, r.positionSide), "buy": r.side == "BUY",
            "price": r.price, "price_str": _num(r.price), "qty": r.qty, "qty_str": _num(r.qty), "quote": quote,
            "fee": r.commission, "fee_str": fee_str, "role": "Мейкер" if maker else "Тейкер", "realized": r.realizedPnl,
        })
    return rows


def _chart_pos(history: pd.DataFrame) -> list[dict]:
    """Данные графика «Просадки (по позициям)» — те же значения, что в колонке «Макс. просадка»."""
    if history.empty or "dd_usdt" not in history:
        return []
    h = history.dropna(subset=["dd_usdt"]).tail(MAX_POSITIONS_IN_HTML)
    return [{"t": r.close_time.strftime("%Y-%m-%d %H:%M"), "symbol": r.symbol, "side": r.side,
             "usdt": round(float(r.dd_usdt), 4), "pct": round(float(r.dd_pct), 2),
             "price": _num(r.dd_price), "at": r.dd_at.strftime("%Y-%m-%d %H:%M")}
            for r in h.itertuples(index=False)]


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
        dd_daily=drawdown_by_day(daily, history), chart_pos=_chart_pos(history),
        fill_rows=_fill_rows(fills), fills_total=len(fills),
        dd_cov=(int(history["dd_usdt"].notna().sum()) if "dd_usdt" in history else 0, len(history)),
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


def resolve_tz(name: str) -> str:
    """«auto» — локальный часовой пояс ОС (в вебе Binance время показывается локальное)."""
    if name != "auto":
        return name
    try:
        from tzlocal import get_localzone_name
        return get_localzone_name() or "UTC"
    except ImportError:
        log.warning("tzlocal не установлен — время в UTC (pip install tzlocal или --tz Europe/Moscow)")
        return "UTC"


def make_public_exchange() -> ccxt.Exchange:
    """Боевой fapi.binance.com, только публичные эндпоинты (ключи не нужны и не передаются)."""
    return ccxt.binanceusdm({"enableRateLimit": True, "options": {"fetchCurrencies": False}})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--days", type=int, default=HISTORY_DAYS, help="глубина истории, дней (Binance хранит ~90)")
    p.add_argument("--symbols", help="только эти символы через запятую, напр. BTCUSDT,ETHUSDT (по умолчанию — кэш + income + открытые позиции)")
    p.add_argument("--output", default="binance_demo_stats.html")
    p.add_argument("--tz", default="auto", help="часовой пояс отображения: auto (локальный, как в вебе Binance) или напр. Europe/Moscow")
    p.add_argument("--csv", default="fills.csv", help="кэш сделок и просадок (по умолчанию fills.csv)")
    p.add_argument("--rebuild", action="store_true", help="игнорировать кэш: перезагрузить сделки и пересчитать просадки")
    p.add_argument("--recompute-dd", action="store_true", help="пересчитать только просадки по свечам")
    p.add_argument("--no-dd", action="store_true", help="не запрашивать свечи (использовать только кэш)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    args.tz = resolve_tz(args.tz)
    if args.days > HISTORY_DAYS:
        log.warning("Binance хранит ~3 месяца истории, --days ограничен %d", HISTORY_DAYS)
        args.days = HISTORY_DAYS

    cache_path = Path(args.csv)
    cache = pd.DataFrame() if args.rebuild else load_fills_cache(cache_path)
    legacy = bool(cache.attrs.get("legacy"))
    cache_ids: dict[str, int] = {}                 # символ -> максимальный id сделки в кэше
    scanned: set[str] = set()                      # символы, выгруженные полностью по fromId (дальше — только дозагрузка)
    if len(cache):
        cache_ids = cache["id"].astype("int64").groupby(cache["symbol"]).max().to_dict()
        if "src" in cache:
            scanned = set(cache.loc[cache["src"] == "id", "symbol"])

    rows: list[dict] = []
    done: set[str] = set()
    positions, positions_error, symbol_cfg = None, "", None
    demo_gov, aborted = None, False

    try:
        ex = make_exchange()
        demo_gov = gov = WeightGovernor(ex, get_weight_limit(ex))
        end = ex.milliseconds()
        start = end - args.days * DAY_MS
        if len(cache):
            start = max(start, int(cache["time"].astype("int64").max()) - OVERLAP_MS)
            log.info("Кэш %s: %d сделок, %d символов", cache_path, len(cache), len(cache_ids))

        try:                                       # открытые позиции — и блок отчёта, и источник символов
            positions = fetch_open_positions(gov)
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:   # отчёт по сделкам важнее — не роняем его
            positions_error = str(e)[:200]
            log.warning("Позиции не загружены: %s", positions_error)

        if args.symbols:
            symbols = {x.strip().upper() for x in args.symbols.split(",") if x.strip()}
        else:
            log.info("Поиск символов с активностью (income) с %s UTC...",
                     pd.Timestamp(start, unit="ms").strftime("%Y-%m-%d %H:%M"))
            symbols = set(cache_ids) | discover_symbols(gov, start, end)
            if positions is not None:
                symbols |= set(positions["symbol"])
        if not symbols and not len(cache):
            log.error("Активности не найдено. Проверьте, что ключи — от demo.binance.com и там есть сделки.")
            return 1

        plan = {sym: (cache_ids[sym] + 1 if sym in scanned else 0) for sym in symbols}
        full = sum(1 for v in plan.values() if v == 0)
        log.info("Символов: %d (полная выгрузка: %d, дозагрузка: %d)", len(plan), full, len(plan) - full)
        try:
            fetch_fills(gov, plan, rows, done)
        except (ccxt.DDoSProtection, ccxt.RateLimitExceeded) as e:   # 418 / 429 после всех повторов
            aborted = True
            log.error("Binance ограничил доступ (418/429): %s. Загружено %d сделок — сохраняем и строим отчёт; "
                      "остальное при следующем запуске.", str(e)[:200], len(rows))
        if not aborted:
            try:
                symbol_cfg = fetch_symbol_config(gov)
            except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                log.warning("Плечо/режим маржи не загружены (ROI в истории позиций будет «—»): %s", str(e)[:200])
    except ccxt.AuthenticationError as e:
        log.error("Ошибка ключей: %s. Нужны ключи Demo Trading (demo.binance.com), не Testnet и не боевые.", e)
        return 1
    except (ccxt.DDoSProtection, ccxt.RateLimitExceeded) as e:
        log.error("Binance ограничил доступ (418/429/WAF): %s. Остановлено, повторите позже.", e)
        return 2

    raw = merge_fills(cache, rows)
    if raw.empty:
        log.error("Сделок не найдено.")
        return 1
    raw.loc[raw["symbol"].isin(done), "src"] = "id"
    fills = build_fills_frame(raw, args.tz)
    orders = build_orders_frame(fills)
    stats = compute_stats(orders, fills)
    history, hist_info = build_position_history(fills, symbol_cfg)

    public_govs: list[WeightGovernor] = []

    def get_public_gov() -> WeightGovernor:               # создаётся лениво — только если есть что считать
        pub = make_public_exchange()
        public_govs.append(WeightGovernor(pub, get_weight_limit(pub)))
        return public_govs[0]

    history, raw, new_dd = attach_drawdowns(history, raw, None if (args.no_dd or aborted) else get_public_gov,
                                            force=args.recompute_dd)

    if args.rebuild or legacy or new_dd or not cache_path.exists() or len(raw) != len(cache) or (done - scanned):
        save_fills_cache(raw, cache_path)
        log.info("Кэш сохранён: %s (сделок: %d, новых: %d, новых расчётов просадки: %d)",
                 cache_path, len(raw), len(raw) - len(cache), new_dd)

    period = f"{fills['time'].min():%Y-%m-%d} … {fills['time'].max():%Y-%m-%d}"
    Path(args.output).write_text(
        render_html(fills, orders, stats, period, positions, positions_error, history, hist_info), encoding="utf-8")

    pf = "∞" if stats["profit_factor"] == float("inf") else f"{stats['profit_factor']:.2f}"
    log.info("Закрытых позиций: %d (пропущено закрытий до окна: %d, ещё открыто: %d)",
             len(history), hist_info["orphan_fills"], hist_info["still_open"])
    for name, g in (("demo", demo_gov), ("mainnet klines", public_govs[0] if public_govs else None)):
        if g:
            log.info("HTTP %s: запросов %d, вес ≈ %d", name, g.calls, g.spent)
    log.info("✅ %s | ордеров %d | net $%.2f | win rate net %.1f%% / gross %.1f%% | PF %s | открытых позиций: %s",
             args.output, stats["orders"], stats["net"], stats["win_rate"], stats["win_rate_gross"], pf,
             "н/д" if positions is None else len(positions))
    return 0 if not aborted else 2


if __name__ == "__main__":
    sys.exit(main())
