#!/usr/bin/env python3
"""
place_limit_grid.py - сетка лимитных ордеров на USDT-M фьючерсах (batch + верификация).
short: лимитные SELL выше рынка        long: лимитные BUY ниже рынка
Позиционно: SYMBOL SIDE FILE [EXCHANGE] [MODE]   (SIDE - обязателен, без умолчания; регистр не важен)
Примеры:
python place_limit_grid.py BTCUSDT short Bot_levels.csv                  # binance, demo (по умолчанию)
python place_limit_grid.py BTCUSDT Long Bot_levels.csv real
python place_limit_grid.py BTCUSDT SHORT Bot_levels.csv binance real
python place_limit_grid.py BTCUSDT long Bot_levels.csv --dry-run         # только проверки
python place_limit_grid.py --symbol BTCUSDT --side long --file Bot_levels.csv --mode testnet
python place_limit_grid.py BTCUSDT short Bot_levels.csv binance demo -foc  # прежняя логика контроля первого ордера
Ключи из .env (см. .env.example). Для каждого режима - свои ключи:
real    -> BINANCE_API_KEY / BINANCE_API_SECRET
demo    -> BINANCE_DEMO_API_KEY / BINANCE_DEMO_API_SECRET      (Demo Trading, demo-fapi.binance.com)
testnet -> BINANCE_TESTNET_API_KEY / BINANCE_TESTNET_API_SECRET (testnet.binancefuture.com)
Установленные ордера пишутся в {placed|demo_placed|testnet_placed}_orders/SYMBOL_{Short|Long}_Limit_dd.mm-hh.mm.csv
Коды выхода: 0 - все ордера подтверждены на бирже; 1 - ошибка конфигурации/валидации (ничего не отправлено);
2 - часть ордеров не удалось выставить/подтвердить.
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import sys
import time
from datetime import datetime
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, TypeVar

if sys.version_info[:2] < (3, 12):
    sys.exit(f"Нужен Python >=3.12, сейчас {sys.version.split()[0]}")

import ccxt  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

# --------------------------------------------------------------------------------------
# Константы
# --------------------------------------------------------------------------------------
MAX_GRID_ORDERS = 50

# Binance USDⓈ-M POST /fapi/v1/batchOrders: максимум 5 ордеров на запрос (официальная документация).
# Если лимит изменится - правится только эта константа.
BATCH_SIZE = 5

MAX_PRICE_RATIO = Decimal(10)  # уровень не дальше чем в 10 раз от текущей цены (в обе стороны)
MAX_RETRY_ROUNDS = 3  # сколько раз повторять установку недостающих ордеров по одному
API_RETRIES = 3  # повторы для сетевых ошибок на чтении (не на размещении)

MARKET_FILL_TIMEOUT = 20.0
MARKET_POLL_INTERVAL = 0.7

MODES = ("real", "demo", "testnet")
SIDES = ("long", "short")

# short: SELL выше рынка (positionSide SHORT в hedge); long: BUY ниже рынка (positionSide LONG в hedge)
SIDE_CFG = {
    "short": {"order_side": "sell", "title": "Short", "prefix": "sg"},
    "long": {"order_side": "buy", "title": "Long", "prefix": "lg"},
}

BASE_DIR = Path(__file__).resolve().parent  # папки с результатами - рядом со скриптом
#DEFAULT_CSV = BASE_DIR / "input" / "Long26.csv"
OUTPUT_DIRS = {"real": "placed_orders", "demo": "demo_placed_orders", "testnet": "testnet_placed_orders"}
CCXT_CLASS = {"binance": "binanceusdm"}  # сюда позже добавятся другие биржи ccxt

log = logging.getLogger("limit_grid")
T = TypeVar("T")


class GridError(Exception):
    """Ошибка конфигурации или валидации: ничего не отправлено на биржу."""


# --------------------------------------------------------------------------------------
# CSV
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class CsvRow:
    line: int
    level: Decimal
    size: Decimal


def _norm_name(token: str) -> str:
    return token.strip().strip("\"'").strip().lower()


def _split(line: str, delim: str) -> list[str]:
    if delim.isspace() and delim != "\t":
        return line.split()
    return next(csv.reader([line], delimiter=delim, skipinitialspace=True), [])


def _detect_delimiter(header: str) -> str:
    """Разделитель определяется по заголовку: текст между словами Level и Size."""
    m = re.search(r"\b(level|size)\b([^A-Za-z0-9]*?)\b(level|size)\b", header, re.I)
    if not m:
        raise GridError("Заголовок CSV должен содержать колонки 'Level' и 'Size' рядом, например 'Level;Size'")
    sep = m.group(2).replace('"', "").replace("'", "")
    if "\t" in sep:
        return "\t"
    sep = sep.strip()
    if not sep:
        return " "
    if len(sep) != 1:
        raise GridError(f"Не удалось определить разделитель полей в заголовке: {header!r}")
    return sep


def _parse_decimal(token: str, what: str, line: int) -> Decimal:
    t = token.strip().strip("\"'").replace("\u00a0", "").replace(" ", "")
    if "," in t and "." in t:
        raise GridError(f"Строка {line}: {what}={token!r} - неоднозначный формат числа (и ',' и '.')")
    if t.count(",") > 1 or t.count(".") > 1:
        raise GridError(f"Строка {line}: {what}={token!r} - неверный формат числа")
    t = t.replace(",", ".")
    try:
        value = Decimal(t)
    except InvalidOperation:
        raise GridError(f"Строка {line}: {what}={token!r} - не число") from None
    if not value.is_finite() or value <= 0:
        raise GridError(f"Строка {line}: {what}={token!r} - должно быть положительным числом")
    return value


def parse_csv(path: Path) -> list[CsvRow]:
    if not path.is_file():
        raise GridError(f"Файл не найден: {path}")

    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1251")
    text = text.replace("\x00", "")

    numbered = [(i, ln) for i, ln in enumerate(text.splitlines(), 1) if ln.strip()]
    if not numbered:
        raise GridError("CSV пустой")

    (hdr_no, header), data_lines = numbered[0], numbered[1:]
    delim = _detect_delimiter(header)
    names = [_norm_name(t) for t in _split(header, delim)]

    if names.count("level") != 1 or names.count("size") != 1:
        raise GridError(f"Строка {hdr_no}: в заголовке должны быть ровно по одной колонке Level и Size, найдено: {names}")

    i_level, i_size = names.index("level"), names.index("size")
    rows: list[CsvRow] = []

    for line_no, line in data_lines:
        tokens = _split(line, delim)
        if all(not t.strip() for t in tokens):  # строки вида ";;"
            continue
        if any(t.strip() for t in tokens[len(names):]):
            raise GridError(
                f"Строка {line_no}: больше колонок, чем в заголовке ({len(names)}). "
                f"Возможно, десятичная запятая совпадает с разделителем полей"
            )
        if len(tokens) <= max(i_level, i_size):
            raise GridError(f"Строка {line_no}: не хватает колонок (нужны Level и Size)")

        lvl_tok, size_tok = tokens[i_level].strip(), tokens[i_size].strip()
        if not lvl_tok and not size_tok:
            continue
        if not lvl_tok or not size_tok:
            raise GridError(f"Строка {line_no}: заполнено только одно из полей Level/Size")

        rows.append(CsvRow(line_no, _parse_decimal(lvl_tok, "Level", line_no), _parse_decimal(size_tok, "Size", line_no)))

    if not rows:
        raise GridError("В CSV нет ни одной строки с данными")
    if len(rows) > MAX_GRID_ORDERS:
        raise GridError(f"В сетке {len(rows)} уровней, максимум {MAX_GRID_ORDERS}")

    log.info("CSV: разделитель=%r, уровней=%d", delim, len(rows))
    return rows


# --------------------------------------------------------------------------------------
# Биржа
# --------------------------------------------------------------------------------------
def env_names(exchange: str, mode: str) -> tuple[str, str]:
    infix = "" if mode == "real" else f"_{mode.upper()}"
    prefix = exchange.upper()
    return f"{prefix}{infix}_API_KEY", f"{prefix}{infix}_API_SECRET"


def build_exchange(exchange_id: str, mode: str) -> ccxt.Exchange:
    cls_name = CCXT_CLASS.get(exchange_id)
    if cls_name is None:
        raise GridError(f"Биржа {exchange_id!r} пока не поддерживается (доступно: {', '.join(CCXT_CLASS)})")

    key_var, secret_var = env_names(exchange_id, mode)
    key, secret = os.getenv(key_var, "").strip(), os.getenv(secret_var, "").strip()
    if not key or not secret:
        raise GridError(f"Для mode={mode} в .env нужны {key_var} и {secret_var}")

    ex: ccxt.Exchange = getattr(ccxt, cls_name)(
        {
            "apiKey": key,
            "secret": secret,
            "enableRateLimit": True,
            "timeout": 20_000,
            "options": {"defaultType": "swap", "adjustForTimeDifference": True},
        }
    )

    if mode == "demo":
        ex.enable_demo_trading(True)  # demo-fapi.binance.com
    elif mode == "testnet":
        ex.set_sandbox_mode(True)  # testnet.binancefuture.com

    return ex


def api_call(fn: Callable[..., T], *args: Any, tries: int = API_RETRIES, **kwargs: Any) -> T:
    """Повторяет только сетевые ошибки. Использовать для чтения, не для размещения ордеров."""
    delay = 1.0
    for attempt in range(1, tries + 1):
        try:
            return fn(*args, **kwargs)
        except ccxt.NetworkError as e:
            if attempt == tries:
                raise
            log.warning("%s: %s - повтор %d/%d через %.0fс", getattr(fn, "__name__", "api"), e, attempt, tries - 1, delay)
            time.sleep(delay)
            delay *= 2
    raise AssertionError("unreachable")


def resolve_market(ex: ccxt.Exchange, symbol_arg: str, exchange_id: str, mode: str) -> dict[str, Any]:
    sym = symbol_arg.strip().upper()
    candidates: list[dict[str, Any]] = []

    if "/" in sym:
        try:
            candidates = [ex.market(sym)]
        except ccxt.BadSymbol:
            candidates = []
    else:
        candidates = [m for m in ex.markets_by_id.get(sym, []) if m.get("swap") and m.get("linear")]

    if not candidates:
        base = re.split(r"[/:]|USDT$|USDC$", sym)[0]
        similar = sorted(m["id"] for m in ex.markets.values() if m.get("swap") and m.get("linear") and base and base in m["id"])[:10]
        hint = f" Похожие: {', '.join(similar)}." if similar else ""
        raise GridError(
            f"Символ {sym} не торгуется на {exchange_id} в режиме {mode} "
            f"(в demo/testnet символов намного меньше, чем в real; всего линейных swap: "
            f"{sum(1 for m in ex.markets.values() if m.get('swap') and m.get('linear'))}).{hint}"
        )

    market = candidates[0]
    status = (market.get("info") or {}).get("status")
    if market.get("active") is False or (status and status != "TRADING"):
        raise GridError(f"Символ {market['id']} найден, но торги сейчас недоступны (status={status})")
    if market.get("quote") != "USDT" or market.get("settle") != "USDT":
        raise GridError(f"Размер ордера задан в USDT, а {market['id']} рассчитывается в {market.get('settle')}")

    return market


# --------------------------------------------------------------------------------------
# План сетки и валидация
# --------------------------------------------------------------------------------------
@dataclass
class Spec:
    idx: int
    line: int
    cid: str
    size_usdt: Decimal
    price: Decimal
    amount: Decimal
    is_market: bool = False
    ref_price: Optional[Decimal] = None

    @property
    def notional(self) -> Decimal:
        return (self.ref_price if self.ref_price is not None else self.price) * self.amount


def fmt(d: Decimal) -> str:
    return format(d.normalize(), "f")


def _limit(market: dict[str, Any], group: str, bound: str) -> Optional[Decimal]:
    v = ((market.get("limits") or {}).get(group) or {}).get(bound)
    return Decimal(str(v)) if v is not None else None


def build_specs(
    ex: ccxt.Exchange,
    market: dict[str, Any],
    rows: list[CsvRow],
    ticker: dict[str, Any],
    run_id: str,
    allow_marketable: bool,
    side: str,
    first_order_control: bool,
) -> tuple[list[Spec], list[str], list[str]]:
    sym = market["symbol"]
    last = ticker.get("last") or ticker.get("close")
    if not last:
        raise GridError(f"Биржа не вернула текущую цену для {market['id']}")

    cur = Decimal(str(last))
    bid = Decimal(str(ticker["bid"])) if ticker.get("bid") else cur
    ask = Decimal(str(ticker["ask"])) if ticker.get("ask") else cur

    p_min, p_max = _limit(market, "price", "min"), _limit(market, "price", "max")
    a_min, a_max = _limit(market, "amount", "min"), _limit(market, "amount", "max")
    c_min = _limit(market, "cost", "min")

    errors: list[str] = []
    warnings: list[str] = []
    specs: list[Spec] = []
    seen_prices: dict[Decimal, int] = {}

    first_market = not first_order_control

    if first_market and len(rows) > 1:
        first_abs = abs(rows[0].level - cur)
        closer = next((r for r in rows[1:] if abs(r.level - cur) < first_abs), None)
        if closer is not None:
            errors.append(
                f"Первая строка CSV (Level={fmt(rows[0].level)}) не является ближайшей к текущей цене {fmt(cur)}. "
                f"Строка {closer.line} (Level={fmt(closer.level)}) ближе. "
                f"Без флага -foc/--first-order-control первый уровень обязан быть ближайшим к рынку"
            )
            return [], errors, warnings

    for n, row in enumerate(rows, 1):
        where = f"уровень #{n} (строка {row.line}, Level={fmt(row.level)})"
        is_market = first_market and n == 1

        try:
            price = Decimal(ex.price_to_precision(sym, str(row.level)))
        except ccxt.BaseError as e:
            errors.append(f"{where}: не удалось округлить цену до tickSize: {e}")
            continue

        if price <= 0:
            errors.append(f"{where}: после округления до tickSize цена равна 0")
            continue

        if price > cur * MAX_PRICE_RATIO or price < cur / MAX_PRICE_RATIO:
            errors.append(f"{where}: отличается от текущей цены {fmt(cur)} более чем в {MAX_PRICE_RATIO} раз")
            continue

        if not is_market:
            if (price <= bid) if side == "short" else (price >= ask):
                msg = (
                    f"{where}: SELL-лимит не выше текущего bid ({fmt(bid)}) - исполнится сразу по рынку"
                    if side == "short"
                    else f"{where}: BUY-лимит не ниже текущего ask ({fmt(ask)}) - исполнится сразу по рынку"
                )
                (warnings if allow_marketable else errors).append(msg + ("" if allow_marketable else " (разрешить: --allow-marketable)"))
                if not allow_marketable:
                    continue

            if p_min is not None and price < p_min or p_max is not None and price > p_max:
                errors.append(f"{where}: цена {fmt(price)} вне допустимого диапазона [{p_min}, {p_max}]")
                continue

        if price in seen_prices:
            errors.append(f"{where}: после округления совпадает с уровнем #{seen_prices[price]} (цена {fmt(price)})")
            continue

        seen_prices[price] = n

        if is_market:
            # Количество рыночного ордера считаем от актуальной цены исполнения:
            # SELL - по bid, BUY - по ask; если стакан пустой/нет данных - по last.
            ref_price = cur
            if side == "short" and bid > 0:
                ref_price = bid
            elif side == "long" and ask > 0:
                ref_price = ask

            try:
                amount = Decimal(ex.amount_to_precision(sym, str(row.size / ref_price)))  # округление вниз до stepSize
            except ccxt.BaseError as e:
                errors.append(f"{where}: размер {fmt(row.size)} USDT слишком мал для шага лота при рыночной цене {fmt(ref_price)}: {e}")
                continue

            notional = amount * ref_price

            if amount <= 0 or (a_min is not None and amount < a_min):
                errors.append(f"{where}: количество {fmt(amount)} меньше минимального {a_min}")
                continue

            if a_max is not None and amount > a_max:
                errors.append(f"{where}: количество {fmt(amount)} больше максимального {a_max}")
                continue

            if c_min is not None and notional < c_min:
                errors.append(f"{where}: стоимость ордера {fmt(notional)} USDT меньше минимальной {fmt(c_min)} USDT")
                continue

            if notional < row.size * Decimal("0.9"):
                warnings.append(f"{where}: из-за шага лота стоимость {fmt(notional)} USDT заметно меньше заданных {fmt(row.size)}")

            specs.append(
                Spec(
                    n,
                    row.line,
                    f"{SIDE_CFG[side]['prefix']}{run_id}-{n:02d}",
                    row.size,
                    price,
                    amount,
                    True,
                    ref_price,
                )
            )
            continue

        try:
            amount = Decimal(ex.amount_to_precision(sym, str(row.size / price)))  # округление вниз до stepSize
        except ccxt.BaseError as e:
            errors.append(f"{where}: размер {fmt(row.size)} USDT слишком мал для шага лота: {e}")
            continue

        notional = amount * price

        if amount <= 0 or (a_min is not None and amount < a_min):
            errors.append(f"{where}: количество {fmt(amount)} меньше минимального {a_min}")
            continue

        if a_max is not None and amount > a_max:
            errors.append(f"{where}: количество {fmt(amount)} больше максимального {a_max}")
            continue

        if c_min is not None and notional < c_min:
            errors.append(f"{where}: стоимость ордера {fmt(notional)} USDT меньше минимальной {fmt(c_min)} USDT")
            continue

        if notional < row.size * Decimal("0.9"):
            warnings.append(f"{where}: из-за шага лота стоимость {fmt(notional)} USDT заметно меньше заданных {fmt(row.size)}")

        specs.append(Spec(n, row.line, f"{SIDE_CFG[side]['prefix']}{run_id}-{n:02d}", row.size, price, amount))

    return specs, errors, warnings


def order_request(market: dict[str, Any], s: Spec, hedged: bool, side: str) -> dict[str, Any]:
    params: dict[str, Any] = {"clientOrderId": s.cid, "timeInForce": "GTC"}  # TP/SL не ставятся
    if hedged:
        params["hedged"] = True  # ccxt: sell -> positionSide=SHORT, buy -> LONG

    return {
        "symbol": market["symbol"],
        "type": "limit",
        "side": SIDE_CFG[side]["order_side"],
        "amount": float(s.amount),
        "price": float(s.price),
        "params": params,
    }


# --------------------------------------------------------------------------------------
# Размещение и проверка
# --------------------------------------------------------------------------------------
def _chunks(items: list[Spec], n: int) -> Iterable[list[Spec]]:
    for i in range(0, len(items), n):
        yield items[i : i + n]


def _err_text(e: BaseException) -> str:
    return str(e).replace("\n", " ")[:300]


def place_batches(ex: ccxt.Exchange, market: dict[str, Any], specs: list[Spec], hedged: bool, side: str) -> None:
    """Размещение пачками по BATCH_SIZE. Итог НЕ интерпретируется как истина - его проверяет verify()."""

    # Почему не ex.create_orders(): внутри он вызывает parse_orders(), который СОРТИРУЕТ результат по timestamp,
    # а у отклоненных записей ({"code":..,"msg":..}) timestamp нет - они уезжают в начало, и соответствие
    # "i-й элемент ответа = i-й ордер запроса" (гарантия Binance) теряется. Поэтому тело запроса собирает ccxt
    # (create_order_request: точность, positionSide, clientOrderId), а ответ читаем сырым и в исходном порядке.
    for chunk in _chunks(specs, BATCH_SIZE):
        batch = []
        for s in chunk:
            r = order_request(market, s, hedged, side)
            batch.append(
                ex.create_order_request(
                    r["symbol"],
                    r["type"],
                    r["side"],
                    r["amount"],
                    r["price"],
                    {**r["params"], "isAlgoOrder": False},
                )
            )

        try:
            response = ex.fapiPrivatePostBatchOrders({"batchOrders": batch})
        except ccxt.AuthenticationError:
            raise
        except ccxt.BaseError as e:
            log.warning(
                "Batch #%d-#%d: запрос завершился ошибкой (%s) - статус ордеров выяснит проверка",
                chunk[0].idx,
                chunk[-1].idx,
                _err_text(e),
            )
            continue

        if not isinstance(response, list) or len(response) != len(chunk):
            log.warning(
                "Batch #%d-#%d: неожиданный ответ %.200s - полагаемся на проверку",
                chunk[0].idx,
                chunk[-1].idx,
                response,
            )
            continue

        for s, item in zip(chunk, response):
            if isinstance(item, dict) and "orderId" in item:
                log.info("  #%02d %s @ %s принят (orderId=%s)", s.idx, fmt(s.amount), fmt(s.price), item["orderId"])
            else:
                log.warning("  #%02d %s @ %s отклонен: %s", s.idx, fmt(s.amount), fmt(s.price), item)


def place_single(ex: ccxt.Exchange, market: dict[str, Any], specs: list[Spec], hedged: bool, side: str) -> None:
    for s in specs:
        req = order_request(market, s, hedged, side)
        try:
            o = ex.create_order(req["symbol"], req["type"], req["side"], req["amount"], req["price"], req["params"])
            log.info("  #%02d повторно принят (orderId=%s)", s.idx, o.get("id"))
        except ccxt.AuthenticationError:
            raise
        except ccxt.BaseError as e:
            log.warning("  #%02d повтор не удался: %s", s.idx, _err_text(e))


@dataclass
class OrderState:
    state: str  # open | mismatch | filled | closed | absent | unknown
    detail: str = ""
    price: Optional[Decimal] = None  # реальные значения с биржи (для state == open)
    amount: Optional[Decimal] = None
    order_id: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.state == "open"


def _real(o: dict[str, Any]) -> tuple[Decimal, Decimal, Optional[str]]:
    """Цена/количество/id ордера КАК НА БИРЖЕ (строки из ответа биржи, без float-артефактов)."""
    info = o.get("info") or {}
    price = Decimal(str(info.get("price") or o["price"]))
    amount = Decimal(str(info.get("origQty") or o["amount"]))
    oid = info.get("orderId") or o.get("id")
    return price, amount, str(oid) if oid is not None else None


def verify(ex: ccxt.Exchange, market: dict[str, Any], specs: list[Spec], side: str) -> dict[str, OrderState]:
    """Источник истины - биржа: открытые ордера по символу + статус каждого отсутствующего по clientOrderId."""
    sym = market["symbol"]
    open_orders = api_call(ex.fetch_open_orders, sym)
    by_cid = {o.get("clientOrderId"): o for o in open_orders if o.get("clientOrderId")}

    result: dict[str, OrderState] = {}

    for s in specs:
        o = by_cid.get(s.cid)
        if o is not None:
            same = (
                o.get("side") == SIDE_CFG[side]["order_side"]
                and Decimal(str(o.get("price"))) == s.price
                and Decimal(str(o.get("amount"))) == s.amount
            )
            result[s.cid] = OrderState("open", "", *_real(o)) if same else OrderState(
                "mismatch",
                f"на бирже side={o.get('side')} price={o.get('price')} amount={o.get('amount')}",
            )
            continue

        try:
            o = api_call(ex.fetch_order, s.cid, sym, {"origClientOrderId": s.cid})
        except ccxt.OrderNotFound:
            result[s.cid] = OrderState("absent")
            continue
        except ccxt.BaseError as e:
            result[s.cid] = OrderState("unknown", _err_text(e))
            continue

        status = o.get("status")
        if status == "open":
            result[s.cid] = OrderState("open", "", *_real(o))
        elif status == "closed":
            result[s.cid] = OrderState("filled", f"исполнен: filled={o.get('filled')} avg={o.get('average')}")
        else:
            result[s.cid] = OrderState("closed", f"status={status} filled={o.get('filled')}")

    return result


def _dec_or_zero(value: Any) -> Decimal:
    if value is None:
        return Decimal(0)
    try:
        d = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return Decimal(0)
    return d if d.is_finite() else Decimal(0)


def _market_state_from_order(o: dict[str, Any], spec: Spec) -> OrderState:
    status = o.get("status")
    filled = _dec_or_zero(o.get("filled"))
    amount = _dec_or_zero(o.get("amount")) or spec.amount

    info = o.get("info") or {}
    avg_raw = o.get("average") or info.get("avgPrice") or o.get("price")
    oid = info.get("orderId") or o.get("id")
    order_id = str(oid) if oid is not None else None

    avg: Optional[Decimal] = None
    if avg_raw is not None:
        try:
            avg = Decimal(str(avg_raw))
        except (InvalidOperation, ValueError, TypeError):
            avg = None

    if filled > 0:
        if amount and filled >= amount:
            detail = f"исполнен полностью: filled={fmt(filled)}"
        else:
            detail = f"частично исполнен: filled={fmt(filled)} из {fmt(amount)}"
        if avg is not None:
            detail += f", avg={fmt(avg)}"
        return OrderState("filled", detail, avg, amount, order_id)

    if status in ("canceled", "expired", "rejected", "closed"):
        return OrderState("closed", f"не исполнен: status={status}", None, amount, order_id)

    if status == "open":
        return OrderState("open", "ожидает исполнения", None, amount, order_id)

    return OrderState("unknown", f"status={status}", None, amount, order_id)


def _fetch_market_order_state(ex: ccxt.Exchange, market: dict[str, Any], spec: Spec) -> OrderState:
    sym = market["symbol"]
    try:
        o = api_call(ex.fetch_order, spec.cid, sym, {"origClientOrderId": spec.cid})
    except ccxt.AuthenticationError:
        raise
    except ccxt.OrderNotFound:
        return OrderState("absent")
    except ccxt.BaseError as e:
        return OrderState("unknown", _err_text(e))

    return _market_state_from_order(o, spec)


def place_market_first(ex: ccxt.Exchange, market: dict[str, Any], spec: Spec, hedged: bool, side: str) -> OrderState:
    sym = market["symbol"]
    order_side = SIDE_CFG[side]["order_side"]
    params: dict[str, Any] = {"clientOrderId": spec.cid, "isAlgoOrder": False}
    if hedged:
        params["hedged"] = True

    log.info(
        "FOC off: первый уровень #%02d (строка %d, Level=%s) отправляется как MARKET %s amount=%s",
        spec.idx,
        spec.line,
        fmt(spec.price),
        order_side.upper(),
        fmt(spec.amount),
    )

    created: Optional[dict[str, Any]] = None
    send_error = ""

    try:
        created = ex.create_order(sym, "market", order_side, float(spec.amount), None, params)
    except ccxt.AuthenticationError:
        raise
    except ccxt.InsufficientFunds as e:
        return OrderState("closed", f"отклонен: недостаточно маржи/средств: {_err_text(e)}")
    except ccxt.BaseError as e:
        send_error = _err_text(e)
        log.warning("MARKET #%02d: ошибка отправки (%s), проверяю статус по clientOrderId", spec.idx, send_error)
        state = _fetch_market_order_state(ex, market, spec)
        if state.state == "filled":
            return state
        if state.state == "closed":
            return state
        # absent/open/unknown - продолжаем опрос, чтобы не задвоить и не потерять уже отправленный ордер
        created = None

    if isinstance(created, dict):
        state = _market_state_from_order(created, spec)
        if state.state == "filled":
            return state
        if state.state == "closed":
            return state

    deadline = time.time() + MARKET_FILL_TIMEOUT
    last_state = OrderState("unknown", send_error or "ожидание подтверждения")

    while time.time() < deadline:
        time.sleep(MARKET_POLL_INTERVAL)
        state = _fetch_market_order_state(ex, market, spec)
        last_state = state

        if state.state == "filled":
            return state
        if state.state == "closed":
            return state

    if last_state.state == "absent":
        detail = f"ордер не найден после отправки в течение {MARKET_FILL_TIMEOUT:.0f}с"
        if send_error:
            detail += f"; ошибка отправки: {send_error}"
        return OrderState("closed", detail)

    detail = f"статус не подтвержден за {MARKET_FILL_TIMEOUT:.0f}с: {last_state.state} {last_state.detail}".strip()
    if send_error:
        detail += f"; ошибка отправки: {send_error}"
    return OrderState("unknown", detail)


def run_grid(ex: ccxt.Exchange, market: dict[str, Any], specs: list[Spec], hedged: bool, side: str) -> dict[str, OrderState]:
    log.info("Размещение %d ордеров batch-запросами по %d ...", len(specs), BATCH_SIZE)
    place_batches(ex, market, specs, hedged, side)

    status: dict[str, OrderState] = {}

    for attempt in range(MAX_RETRY_ROUNDS + 1):
        time.sleep(0.5)  # дать бирже обработать конкурентно исполняемый batch
        status = verify(ex, market, specs, side)

        n_ok = sum(1 for v in status.values() if v.ok)
        log.info("Проверка на бирже: подтверждено %d из %d", n_ok, len(specs))

        # Повторяем ТОЛЬКО те, о которых биржа достоверно сказала "такого ордера нет":
        # исполненные/отмененные/неопределенные заново не ставим, чтобы не задвоить позицию.
        missing = [s for s in specs if status[s.cid].state == "absent"]
        if not missing or attempt == MAX_RETRY_ROUNDS:
            break

        log.info(
            "Повтор %d/%d: выставляю по одному %d ордеров: %s",
            attempt + 1,
            MAX_RETRY_ROUNDS,
            len(missing),
            ", ".join(f"#{s.idx:02d}" for s in missing),
        )
        place_single(ex, market, missing, hedged, side)

    return status


def write_placed_orders(symbol_id: str, side: str, mode: str, specs: list[Spec], status: dict[str, OrderState]) -> Optional[Path]:
    """Пишет SYMBOL_{Short|Long}_Limit_dd.mm-hh.mm.csv в {placed|demo_placed|testnet_placed}_orders/.
    Только ордера, РЕАЛЬНО стоящие на бирже; цена/количество - с биржи."""
    placed = [(s, status[s.cid]) for s in specs if status[s.cid].ok and status[s.cid].price is not None]
    if not placed:
        return None

    out_dir = BASE_DIR / OUTPUT_DIRS[mode]
    out_dir.mkdir(parents=True, exist_ok=True)

    base = f"{symbol_id}_{SIDE_CFG[side]['title']}_Limit_{datetime.now():%d.%m-%H.%M}"  # локальное время машины
    path, n = out_dir / f"{base}.csv", 1
    while path.exists():  # два запуска в одну минуту - не затираем файл
        n += 1
        path = out_dir / f"{base}_{n}.csv"

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["Level", "Coins", "USDT", "OrderId", "ClientOrderId"])
        for s, st in placed:
            assert st.price is not None and st.amount is not None
            w.writerow([fmt(st.price), fmt(st.amount), fmt(st.price * st.amount), st.order_id or "", s.cid])

    return path


# --------------------------------------------------------------------------------------
# Аргументы командной строки
# --------------------------------------------------------------------------------------
def resolve_args(args: argparse.Namespace) -> None:
    """Позиционно: SYMBOL SIDE FILE [EXCHANGE] [MODE]. Ключи имеют приоритет и не обязательны."""
    pos = list(args.pos)
    order = "Порядок: SYMBOL SIDE FILE [EXCHANGE] [MODE]"
    modes = ", ".join(MODES)

    if args.symbol is None and pos:
        args.symbol = pos.pop(0)
    if args.side is None and pos:
        args.side = pos.pop(0)
    if args.file is None and pos:
        args.file = pos.pop(0)

    if args.side is not None:
        raw_side, args.side = args.side, args.side.strip().lower()
        if args.side not in SIDES:
            raise GridError(f"Вторым аргументом должен быть SIDE: long или short, получено {raw_side!r}. {order}")

    if args.symbol is None or args.side is None or args.file is None:
        raise GridError("Нужны SYMBOL, SIDE и FILE: python place_limit_grid.py SYMBOL long|short FILE [EXCHANGE] [MODE]")

    if args.file.lower() in MODES or args.file.lower() in CCXT_CLASS or args.file.lower() in SIDES:
        raise GridError(f"Третьим аргументом должен быть путь к CSV, а не {args.file!r}. {order}")

    exchange, mode = None, None
    if len(pos) == 1:
        if pos[0].lower() in MODES:
            mode = pos[0]
        else:
            exchange = pos[0]
    elif len(pos) == 2:
        exchange, mode = pos
    elif len(pos) > 2:
        raise GridError(f"Лишние аргументы: {' '.join(pos[2:])}. {order}")

    if exchange is not None and exchange.lower() in MODES:
        raise GridError(f"Режим {exchange!r} должен стоять после биржи. {order}")
    if mode is not None and mode.lower() not in MODES:
        raise GridError(f"Неизвестный режим {mode!r} (допустимо: {modes}). {order}")
    if (exchange and args.exchange) or (mode and args.mode):
        raise GridError("Биржа/режим заданы и позиционно, и ключом - оставьте один способ")

    args.exchange = (exchange or args.exchange or "binance").strip().lower()
    args.mode = (mode or args.mode or "demo").strip().lower()

    if args.mode not in MODES:
        raise GridError(f"Неизвестный режим {args.mode!r} (допустимо: {modes})")

    args.symbol = args.symbol.strip().upper()
    if not args.symbol.endswith("USDT") or len(args.symbol) <= len("USDT"):
        raise GridError(f"Символ {args.symbol!r} должен оканчиваться на USDT (например BTCUSDT) - размер ордеров задан в USDT")

    if args.exchange not in CCXT_CLASS:
        raise GridError(f"Биржа {args.exchange!r} пока не поддерживается (доступно: {', '.join(CCXT_CLASS)})")


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def print_plan(specs: list[Spec]) -> None:
    print(f"\n{'#':>3} {'Level':>14} {'Qty':>14} {'Notional USDT':>14}  clientOrderId")
    for s in specs:
        mark = "  MARKET" if s.is_market else ""
        print(f"{s.idx:>3} {fmt(s.price):>14} {fmt(s.amount):>14} {fmt(s.notional):>14}  {s.cid}{mark}")
    print(f"Итого: {len(specs)} ордеров, {fmt(sum((s.notional for s in specs), Decimal(0)))} USDT\n")


def prepare(args: argparse.Namespace, run_id: str) -> tuple[ccxt.Exchange, dict[str, Any], list[Spec], bool]:
    rows = parse_csv(Path(args.file))
    ex = build_exchange(args.exchange, args.mode)

    log.info("Режим=%s, загружаю рынки %s ...", args.mode, args.exchange)
    try:
        api_call(ex.load_markets)
    except ccxt.AuthenticationError as e:
        raise GridError(f"Биржа отклонила ключи для mode={args.mode}: {_err_text(e)}") from e
    except ccxt.BaseError as e:
        raise GridError(f"Не удалось загрузить рынки в режиме {args.mode}: {_err_text(e)}") from e

    market = resolve_market(ex, args.symbol, args.exchange, args.mode)
    sym = market["symbol"]

    try:
        ticker = api_call(ex.fetch_ticker, sym)
    except ccxt.BaseError as e:
        raise GridError(f"Символ {market['id']} есть в списке, но данные по нему недоступны в режиме {args.mode}: {_err_text(e)}") from e

    try:
        lev_info = api_call(ex.fetch_leverage, sym)
        lev_key, lev_other = ("shortLeverage", "longLeverage") if args.side == "short" else ("longLeverage", "shortLeverage")
        leverage = lev_info.get(lev_key) or lev_info.get(lev_other)
        margin_mode = lev_info.get("marginMode")
    except ccxt.AuthenticationError as e:
        raise GridError(f"Биржа отклонила ключи для mode={args.mode}: {_err_text(e)}") from e
    except ccxt.BaseError as e:
        raise GridError(f"Не удалось получить плечо для {market['id']}: {_err_text(e)}") from e

    if not leverage:
        raise GridError(f"Биржа не вернула плечо для {market['id']}")

    try:
        hedged = bool(api_call(ex.fetch_position_mode, sym).get("hedged"))
    except ccxt.BaseError as e:
        raise GridError(f"Не удалось определить режим позиций (hedge/one-way): {_err_text(e)}") from e

    log.info(
        "%s %s: last=%s bid=%s ask=%s | плечо=%sx (%s) | режим позиций=%s",
        market["id"],
        args.side,
        ticker.get("last"),
        ticker.get("bid"),
        ticker.get("ask"),
        leverage,
        margin_mode,
        "hedge" if hedged else "one-way",
    )

    specs, errors, warnings = build_specs(
        ex,
        market,
        rows,
        ticker,
        run_id,
        args.allow_marketable,
        args.side,
        args.first_order_control,
    )

    # Уже стоящие ордера на тех же уровнях - вероятно, сетка уже выставлена (защита от дублей при перезапуске)
    if specs:
        existing = api_call(ex.fetch_open_orders, sym)
        grid_prices = {s.price: s for s in specs}
        for o in existing:
            if o.get("side") == SIDE_CFG[args.side]["order_side"] and o.get("price") is not None and Decimal(str(o["price"])) in grid_prices:
                errors.append(
                    f"На уровне {o['price']} уже стоит ордер того же направления {o.get('clientOrderId') or o.get('id')} - сетка, вероятно, уже выставлена"
                )

    if specs:
        total = sum((s.notional for s in specs), Decimal(0))
        need_margin = total / Decimal(leverage)

        try:
            free = Decimal(str(api_call(ex.fetch_balance)["USDT"]["free"]))
            if need_margin > free:
                warnings.append(
                    f"Нужная маржа ~{fmt(need_margin.quantize(Decimal('0.01')))} USDT "
                    f"(сумма {fmt(total)} / плечо {leverage}x), доступно {fmt(free)} USDT"
                )
        except (ccxt.BaseError, KeyError, TypeError) as e:
            warnings.append(f"Не удалось сверить маржу с балансом: {_err_text(e) if isinstance(e, ccxt.BaseError) else e!r}")

        try:
            tiers = api_call(ex.fetch_market_leverage_tiers, sym)
            caps = [Decimal(str(t["maxNotional"])) for t in tiers if t.get("maxLeverage") and t["maxLeverage"] >= leverage and t.get("maxNotional")]
            if caps and total > max(caps):
                warnings.append(
                    f"Сумма сетки {fmt(total)} USDT выше лимита позиции {fmt(max(caps))} USDT при плече {leverage}x "
                    f"(без учета уже открытой позиции)"
                )
        except (ccxt.BaseError, KeyError, TypeError, ValueError) as e:
            warnings.append(f"Не удалось проверить лимит позиции по плечу: {e!r}")

    for w in warnings:
        log.warning("ПРЕДУПРЕЖДЕНИЕ: %s", w)

    if errors:
        raise GridError("Проверки не пройдены, ордера НЕ отправлялись:\n- " + "\n- ".join(errors))

    return ex, market, specs, hedged


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Сетка лимитных ордеров на USDT-M фьючерсах: short = SELL выше рынка, long = BUY ниже рынка (batch + проверка). "
        "Позиционно: SYMBOL SIDE FILE [EXCHANGE] [MODE]; ключи необязательны.",
        epilog="По умолчанию (без -foc/--first-order-control) первый уровень из CSV отправляется как MARKET и должен исполниться. "
        "Флаг -foc/--first-order-control включает прежний контроль цены первого ордера.",
    )
    p.add_argument("pos", nargs="*", metavar="SYMBOL SIDE FILE [EXCHANGE] [MODE]", help="например: ESPORTSUSDT short Bot_levels.csv binance real")
    p.add_argument("--symbol", "-s", help="фьючерсный символ, например BTCUSDT")
    p.add_argument("--side", help="long (BUY ниже рынка) или short (SELL выше рынка); обязателен, без умолчания")
    p.add_argument("--file", "-f", help="путь к CSV (колонки Level и Size)")
    p.add_argument("--exchange", "-e", help="биржа (по умолчанию binance)")
    p.add_argument("--mode", "-m", choices=MODES, help="real | demo | testnet (по умолчанию demo)")
    p.add_argument("--dry-run", action="store_true", help="выполнить все проверки и показать план, не отправляя ордера")
    p.add_argument(
        "--allow-marketable",
        action="store_true",
        help="разрешить уровни, исполняющиеся сразу по рынку (short: не выше bid, long: не ниже ask); по умолчанию это ошибка",
    )
    p.add_argument(
        "-foc",
        "--first-order-control",
        action="store_true",
        help="включить контроль цены первого ордера (прежняя логика). "
        "Если флаг не указан, первый уровень из CSV принудительно отправляется как MARKET и должен исполниться; "
        "первая строка CSV обязана быть ближайшей к текущей цене",
    )

    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    load_dotenv(Path(__file__).resolve().parent / ".env")
    load_dotenv()

    try:
        resolve_args(args)
    except GridError as e:
        log.error("%s", e)
        return 1

    run_id = time.strftime("%y%m%d%H%M%S")

    try:
        ex, market, specs, hedged = prepare(args, run_id)
    except GridError as e:
        log.error("%s", e)
        return 1
    except ccxt.BaseError as e:
        log.error("Ошибка биржи при подготовке: %s", _err_text(e))
        return 1

    market_spec = next((s for s in specs if s.is_market), None)
    if market_spec is not None:
        log.info(
            "FOC off: первый уровень #%02d (Level=%s) будет отправлен как MARKET",
            market_spec.idx,
            fmt(market_spec.price),
        )

    print_plan(specs)

    if args.dry_run:
        log.info("--dry-run: проверки пройдены, ордера не отправлялись")
        return 0

    try:
        status: dict[str, OrderState] = {}
        limit_specs = specs

        if market_spec is not None:
            m_state = place_market_first(ex, market, market_spec, hedged, args.side)
            status[market_spec.cid] = m_state

            if m_state.state != "filled":
                print(
                    f"\n{market_spec.idx:>3} {fmt(market_spec.price):>14} {fmt(market_spec.amount):>14}  "
                    f"ПРОБЛЕМА: {m_state.state} {m_state.detail}".rstrip()
                )
                log.error(
                    "FOC off: первый рыночный ордер #%02d не исполнен; остальные ордера не отправлялись. "
                    "Проверьте биржу/маржу/связь",
                    market_spec.idx,
                )
                return 2

            if "частично" in m_state.detail:
                log.warning(
                    "FOC off: первый рыночный ордер #%02d исполнен частично (%s); повторная отправка не выполняется",
                    market_spec.idx,
                    m_state.detail,
                )
            else:
                log.info("FOC off: первый рыночный ордер #%02d исполнен (%s)", market_spec.idx, m_state.detail)

            limit_specs = [s for s in specs if not s.is_market]

        if limit_specs:
            status.update(run_grid(ex, market, limit_specs, hedged, args.side))

    except ccxt.AuthenticationError as e:
        log.error("Ошибка авторизации/прав API-ключа: %s", _err_text(e))
        return 2
    except ccxt.BaseError as e:
        log.error(
            "Не удалось выполнить проверку после размещения: %s. Проверьте открытые ордера вручную (префикс %s%s)",
            _err_text(e),
            SIDE_CFG[args.side]["prefix"],
            run_id,
        )
        return 2

    try:
        out_path = write_placed_orders(market["id"], args.side, args.mode, specs, status)
    except OSError as e:
        log.error("Не удалось записать файл placed_orders: %s", e)

    def _spec_ok(s: Spec) -> bool:
        st = status[s.cid]
        if s.is_market:
            return st.state == "filled"
        return st.ok

    bad = [s for s in specs if not _spec_ok(s)]

    print(f"\n{'#':>3} {'Level':>14} {'Qty':>14}  Статус")
    for s in specs:
        st = status[s.cid]
        if s.is_market:
            if st.state == "filled":
                txt = f"OK: MARKET {st.detail}".strip()
            else:
                txt = f"ПРОБЛЕМА: {st.state} {st.detail}".strip()
        else:
            txt = "OK: стоит на бирже" if st.ok else f"ПРОБЛЕМА: {st.state} {st.detail}".strip()

        print(f"{s.idx:>3} {fmt(s.price):>14} {fmt(s.amount):>14}  {txt}")

    if bad:
        log.error(
            "ИТОГ: на бирже подтверждено %d из %d ордеров; проблемные: %s",
            len(specs) - len(bad),
            len(specs),
            ", ".join(f"#{s.idx:02d}" for s in bad),
        )
        return 2

    if market_spec is not None:
        if limit_specs:
            log.info(
                "ИТОГ: первый рыночный ордер исполнен, %d лимитных подтверждены на бирже (%s, %s)",
                len(limit_specs),
                market["id"],
                args.mode,
            )
        else:
            log.info("ИТОГ: первый рыночный ордер исполнен (%s, %s)", market["id"], args.mode)
    else:
        log.info("ИТОГ: все %d ордеров подтверждены на бирже (%s, %s)", len(specs), market["id"], args.mode)

    if out_path:
        log.info("Установленные ордера (реальные значения с биржи) записаны:")
        log.info(out_path)
    else:
        has_placed_limits = any((not s.is_market and status[s.cid].ok) for s in specs)
        if market_spec is not None and not has_placed_limits:
            log.info("Файл placed_orders не создавался: рыночный первый ордер не остается на бирже, лимитных подтвержденных ордеров нет")
        else:
            log.warning("Нет подтвержденных ордеров - файл placed_orders не создан")

    return 0


if __name__ == "__main__":
    sys.exit(main())