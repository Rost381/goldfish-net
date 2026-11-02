#!/usr/bin/env python3
"""
place_sell_grid.py - сетка лимитных SELL-ордеров на USDT-M фьючерсах (batch + верификация).

Примеры (позиционно: SYMBOL FILE [EXCHANGE] [MODE]; ключи --symbol/--file/--exchange/--mode тоже работают):
    python place_sell_grid.py BTCUSDT Bot_levels.csv                  # binance, demo (по умолчанию)
    python place_sell_grid.py BTCUSDT Bot_levels.csv real             # режим вместо биржи - допустимо
    python place_sell_grid.py BTCUSDT Bot_levels.csv binance real
    python place_sell_grid.py BTCUSDT Bot_levels.csv --dry-run        # только проверки
    python place_sell_grid.py --symbol BTCUSDT --file Bot_levels.csv --mode testnet

Ключи берутся из .env (см. .env.example). Для каждого режима - свои ключи:
    real    -> BINANCE_API_KEY / BINANCE_API_SECRET
    demo    -> BINANCE_DEMO_API_KEY / BINANCE_DEMO_API_SECRET      (Demo Trading реального аккаунта, demo-fapi.binance.com)
    testnet -> BINANCE_TESTNET_API_KEY / BINANCE_TESTNET_API_SECRET (отдельный testnet, testnet.binancefuture.com)

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

if sys.version_info[:3] < (3, 12):
    sys.exit(f"Нужен Python >=3.12.3, сейчас {sys.version.split()[0]}")

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
MODES = ("real", "demo", "testnet")
OUTPUT_DIR = Path(__file__).resolve().parent / "placed_orders"  # рядом со скриптом
CCXT_CLASS = {"binance": "binanceusdm"}  # сюда позже добавятся другие биржи ccxt

log = logging.getLogger("sell_grid")
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

    @property
    def notional(self) -> Decimal:
        return self.price * self.amount


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
) -> tuple[list[Spec], list[str], list[str]]:
    sym = market["symbol"]
    last = ticker.get("last") or ticker.get("close")
    if not last:
        raise GridError(f"Биржа не вернула текущую цену для {market['id']}")
    cur = Decimal(str(last))
    bid = Decimal(str(ticker["bid"])) if ticker.get("bid") else cur

    p_min, p_max = _limit(market, "price", "min"), _limit(market, "price", "max")
    a_min, a_max = _limit(market, "amount", "min"), _limit(market, "amount", "max")
    c_min = _limit(market, "cost", "min")

    errors: list[str] = []
    warnings: list[str] = []
    specs: list[Spec] = []
    seen_prices: dict[Decimal, int] = {}

    for n, row in enumerate(rows, 1):
        where = f"уровень #{n} (строка {row.line}, Level={fmt(row.level)})"
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
        if price <= bid:
            msg = f"{where}: SELL-лимит не выше текущего bid ({fmt(bid)}) - исполнится сразу по рынку"
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

        specs.append(Spec(n, row.line, f"sg{run_id}-{n:02d}", row.size, price, amount))
    return specs, errors, warnings


def order_request(market: dict[str, Any], s: Spec, hedged: bool) -> dict[str, Any]:
    params: dict[str, Any] = {"clientOrderId": s.cid, "timeInForce": "GTC"}  # TP/SL не ставятся
    if hedged:
        params["hedged"] = True  # ccxt для sell выставит positionSide=SHORT
    return {
        "symbol": market["symbol"],
        "type": "limit",
        "side": "sell",
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


def place_batches(ex: ccxt.Exchange, market: dict[str, Any], specs: list[Spec], hedged: bool) -> None:
    """Размещение пачками по BATCH_SIZE. Итог НЕ интерпретируется как истина - его проверяет verify()."""
    # Почему не ex.create_orders(): внутри он вызывает parse_orders(), который СОРТИРУЕТ результат по timestamp,
    # а у отклоненных записей ({"code":..,"msg":..}) timestamp нет - они уезжают в начало, и соответствие
    # "i-й элемент ответа = i-й ордер запроса" (гарантия Binance) теряется. Поэтому тело запроса собирает ccxt
    # (create_order_request: точность, positionSide, clientOrderId), а ответ читаем сырым и в исходном порядке.
    for chunk in _chunks(specs, BATCH_SIZE):
        batch = []
        for s in chunk:
            r = order_request(market, s, hedged)
            batch.append(ex.create_order_request(
                r["symbol"], r["type"], r["side"], r["amount"], r["price"], {**r["params"], "isAlgoOrder": False}))
        try:
            response = ex.fapiPrivatePostBatchOrders({"batchOrders": batch})
        except ccxt.AuthenticationError:
            raise
        except ccxt.BaseError as e:
            log.warning("Batch #%d-#%d: запрос завершился ошибкой (%s) - статус ордеров выяснит проверка",
                        chunk[0].idx, chunk[-1].idx, _err_text(e))
            continue
        if not isinstance(response, list) or len(response) != len(chunk):
            log.warning("Batch #%d-#%d: неожиданный ответ %.200s - полагаемся на проверку",
                        chunk[0].idx, chunk[-1].idx, response)
            continue
        for s, item in zip(chunk, response):
            if isinstance(item, dict) and "orderId" in item:
                log.info("  #%02d %s @ %s принят (orderId=%s)", s.idx, fmt(s.amount), fmt(s.price), item["orderId"])
            else:
                log.warning("  #%02d %s @ %s отклонен: %s", s.idx, fmt(s.amount), fmt(s.price), item)


def place_single(ex: ccxt.Exchange, market: dict[str, Any], specs: list[Spec], hedged: bool) -> None:
    for s in specs:
        req = order_request(market, s, hedged)
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


def verify(ex: ccxt.Exchange, market: dict[str, Any], specs: list[Spec]) -> dict[str, OrderState]:
    """Источник истины - биржа: открытые ордера по символу + статус каждого отсутствующего по clientOrderId."""
    sym = market["symbol"]
    open_orders = api_call(ex.fetch_open_orders, sym)
    by_cid = {o.get("clientOrderId"): o for o in open_orders if o.get("clientOrderId")}

    result: dict[str, OrderState] = {}
    for s in specs:
        o = by_cid.get(s.cid)
        if o is not None:
            same = (
                o.get("side") == "sell"
                and Decimal(str(o.get("price"))) == s.price
                and Decimal(str(o.get("amount"))) == s.amount
            )
            result[s.cid] = OrderState("open", "", *_real(o)) if same else OrderState(
                "mismatch", f"на бирже side={o.get('side')} price={o.get('price')} amount={o.get('amount')}")
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


def run_grid(ex: ccxt.Exchange, market: dict[str, Any], specs: list[Spec], hedged: bool) -> dict[str, OrderState]:
    log.info("Размещение %d ордеров batch-запросами по %d ...", len(specs), BATCH_SIZE)
    place_batches(ex, market, specs, hedged)

    status: dict[str, OrderState] = {}
    for attempt in range(MAX_RETRY_ROUNDS + 1):
        time.sleep(0.5)  # дать бирже обработать конкурентно исполняемый batch
        status = verify(ex, market, specs)
        n_ok = sum(1 for v in status.values() if v.ok)
        log.info("Проверка на бирже: подтверждено %d из %d", n_ok, len(specs))
        # Повторяем ТОЛЬКО те, о которых биржа достоверно сказала "такого ордера нет":
        # исполненные/отмененные/неопределенные заново не ставим, чтобы не задвоить позицию.
        missing = [s for s in specs if status[s.cid].state == "absent"]
        if not missing or attempt == MAX_RETRY_ROUNDS:
            break
        log.info("Повтор %d/%d: выставляю по одному %d ордеров: %s",
                 attempt + 1, MAX_RETRY_ROUNDS, len(missing), ", ".join(f"#{s.idx:02d}" for s in missing))
        place_single(ex, market, missing, hedged)
    return status


def write_placed_orders(symbol_id: str, specs: list[Spec], status: dict[str, OrderState], out_dir: Path = OUTPUT_DIR) -> Optional[Path]:
    """Пишет в placed_orders/SYMBOL_Short_dd_mm_hh-mm.csv ордера, РЕАЛЬНО стоящие на бирже (цена/количество с биржи)."""
    placed = [(s, status[s.cid]) for s in specs if status[s.cid].ok and status[s.cid].price is not None]
    if not placed:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"{symbol_id}_Short_{datetime.now():%d_%m_%H-%M}"  # локальное время машины
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
    """Разбор позиционных аргументов: SYMBOL FILE [EXCHANGE] [MODE]. Ключи имеют приоритет и не обязательны."""
    pos = list(args.pos)
    modes = ", ".join(MODES)
    if args.symbol is None and pos:
        args.symbol = pos.pop(0)
    if args.file is None and pos:
        args.file = pos.pop(0)
        if args.file.lower() in MODES or args.file.lower() in CCXT_CLASS:
            raise GridError(f"Вторым аргументом должен быть путь к CSV, а не {args.file!r}. Порядок: SYMBOL FILE [EXCHANGE] [MODE]")
    if args.symbol is None or args.file is None:
        raise GridError("Нужны SYMBOL и FILE: python place_sell_grid.py SYMBOL FILE [EXCHANGE] [MODE]")

    exchange, mode = None, None
    if len(pos) == 1:
        if pos[0].lower() in MODES:
            mode = pos[0]
        else:
            exchange = pos[0]
    elif len(pos) == 2:
        exchange, mode = pos
    elif len(pos) > 2:
        raise GridError(f"Лишние аргументы: {' '.join(pos[2:])}. Порядок: SYMBOL FILE [EXCHANGE] [MODE]")
    if exchange is not None and exchange.lower() in MODES:
        raise GridError(f"Порядок: SYMBOL FILE [EXCHANGE] [MODE]; режим {exchange!r} должен стоять после биржи")
    if mode is not None and mode.lower() not in MODES:
        raise GridError(f"Неизвестный режим {mode!r} (допустимо: {modes}). Порядок: SYMBOL FILE [EXCHANGE] [MODE]")
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
        print(f"{s.idx:>3} {fmt(s.price):>14} {fmt(s.amount):>14} {fmt(s.notional):>14}  {s.cid}")
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
        leverage = lev_info.get("shortLeverage") or lev_info.get("longLeverage")
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

    log.info("%s: last=%s bid=%s | плечо=%sx (%s) | режим позиций=%s",
             market["id"], ticker.get("last"), ticker.get("bid"), leverage, margin_mode, "hedge" if hedged else "one-way")

    specs, errors, warnings = build_specs(ex, market, rows, ticker, run_id, args.allow_marketable)

    # Уже стоящие ордера на тех же уровнях - вероятно, сетка уже выставлена (защита от дублей при перезапуске)
    if specs:
        existing = api_call(ex.fetch_open_orders, sym)
        grid_prices = {s.price: s for s in specs}
        for o in existing:
            if o.get("side") == "sell" and o.get("price") is not None and Decimal(str(o["price"])) in grid_prices:
                errors.append(f"На уровне {o['price']} уже стоит SELL-ордер {o.get('clientOrderId') or o.get('id')} - сетка, вероятно, уже выставлена")

    if specs:
        total = sum((s.notional for s in specs), Decimal(0))
        need_margin = total / Decimal(leverage)
        try:
            free = Decimal(str(api_call(ex.fetch_balance)["USDT"]["free"]))
            if need_margin > free:
                warnings.append(f"Нужная маржа ~{fmt(need_margin.quantize(Decimal('0.01')))} USDT (сумма {fmt(total)} / плечо {leverage}x), доступно {fmt(free)} USDT")
        except (ccxt.BaseError, KeyError, TypeError) as e:
            warnings.append(f"Не удалось сверить маржу с балансом: {_err_text(e) if isinstance(e, ccxt.BaseError) else e!r}")
        try:
            tiers = api_call(ex.fetch_market_leverage_tiers, sym)
            caps = [Decimal(str(t["maxNotional"])) for t in tiers if t.get("maxLeverage") and t["maxLeverage"] >= leverage and t.get("maxNotional")]
            if caps and total > max(caps):
                warnings.append(f"Сумма сетки {fmt(total)} USDT выше лимита позиции {fmt(max(caps))} USDT при плече {leverage}x (без учета уже открытой позиции)")
        except (ccxt.BaseError, KeyError, TypeError, ValueError) as e:
            warnings.append(f"Не удалось проверить лимит позиции по плечу: {e!r}")

    for w in warnings:
        log.warning("ПРЕДУПРЕЖДЕНИЕ: %s", w)
    if errors:
        raise GridError("Проверки не пройдены, ордера НЕ отправлялись:\n  - " + "\n  - ".join(errors))
    return ex, market, specs, hedged


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Сетка лимитных SELL-ордеров на USDT-M фьючерсах (batch + проверка). "
                    "Позиционно: SYMBOL FILE [EXCHANGE] [MODE]; ключи --symbol/--file/--exchange/--mode необязательны.")
    p.add_argument("pos", nargs="*", metavar="SYMBOL FILE [EXCHANGE] [MODE]", help="например: ESPORTSUSDT Bot_levels.csv binance real")
    p.add_argument("--symbol", "-s", help="фьючерсный символ, например BTCUSDT")
    p.add_argument("--file", "-f", help="путь к CSV (колонки Level и Size)")
    p.add_argument("--exchange", "-e", help="биржа (по умолчанию binance)")
    p.add_argument("--mode", "-m", choices=MODES, help="real | demo | testnet (по умолчанию demo)")
    p.add_argument("--dry-run", action="store_true", help="выполнить все проверки и показать план, не отправляя ордера")
    p.add_argument("--allow-marketable", action="store_true",
                   help="разрешить уровни не выше bid (такие SELL исполнятся сразу по рынку); по умолчанию это ошибка")
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

    print_plan(specs)
    if args.dry_run:
        log.info("--dry-run: проверки пройдены, ордера не отправлялись")
        return 0

    try:
        status = run_grid(ex, market, specs, hedged)
    except ccxt.AuthenticationError as e:
        log.error("Ошибка авторизации/прав API-ключа: %s", _err_text(e))
        return 2
    except ccxt.BaseError as e:
        log.error("Не удалось выполнить проверку после размещения: %s. Проверьте открытые ордера вручную (префикс sg%s)", _err_text(e), run_id)
        return 2

    try:
        out_path = write_placed_orders(market["id"], specs, status)
        if out_path:
            log.info("Установленные ордера (реальные значения с биржи) записаны: %s", out_path)
        else:
            log.warning("Нет подтвержденных ордеров - файл placed_orders не создан")
    except OSError as e:
        log.error("Не удалось записать файл placed_orders: %s", e)

    bad = [s for s in specs if not status[s.cid].ok]
    print(f"\n{'#':>3} {'Level':>14} {'Qty':>14}  Статус")
    for s in specs:
        st = status[s.cid]
        print(f"{s.idx:>3} {fmt(s.price):>14} {fmt(s.amount):>14}  {'OK: стоит на бирже' if st.ok else f'ПРОБЛЕМА: {st.state} {st.detail}'.strip()}")
    if bad:
        log.error("ИТОГ: на бирже подтверждено %d из %d ордеров; проблемные: %s",
                  len(specs) - len(bad), len(specs), ", ".join(f"#{s.idx:02d}" for s in bad))
        return 2
    log.info("ИТОГ: все %d ордеров подтверждены на бирже (%s, %s)", len(specs), market["id"], args.mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
