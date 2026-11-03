#!/usr/bin/env python3
"""
place_market_grid.py - сетка УСЛОВНЫХ РЫНОЧНЫХ ордеров (STOP_MARKET / TAKE_PROFIT_MARKET) на USDT-M фьючерсах, hedge-режим.

    long : BUY  - вход при достижении уровня         short: SELL - вход при достижении уровня
    Тип ордера выбирается автоматически по положению уровня относительно текущей цены (CONTRACT_PRICE):
        BUY : уровень выше цены -> STOP_MARKET,        ниже -> TAKE_PROFIT_MARKET
        SELL: уровень ниже цены -> STOP_MARKET,        выше -> TAKE_PROFIT_MARKET
    Контроля положения сетки относительно цены нет: сетка может быть выше, ниже и вокруг текущей цены.

Позиционно: SYMBOL SIDE FILE [EXCHANGE] [MODE] [-p] [-b]
    python place_market_grid.py BTCUSDT long Bot_levels.csv
    python place_market_grid.py BTCUSDT Short Bot_levels.csv binance real -p -b
    python place_market_grid.py BTCUSDT long Bot_levels.csv demo --profit --break-even --dry-run

    -p / --profit      Take-Profit каждого ордера (TAKE_PROFIT_MARKET) = уровень следующего ордера -/+ спред;
                       для последнего - +/- средний шаг сетки в %
    -b / --break-even  Stop-Loss (STOP_MARKET) ордера номер BREAK_EVEN_SL_ORDER = уровень ордера BREAK_EVEN_LEVEL_ORDER +/- спред

Все условные ордера Binance идут через /fapi/v1/algoOrder (по одному ордеру на запрос, batch невозможен).
Требует place_limit_grid.py в той же папке (общий разбор CSV, подключение, .env, проверка).
Результат: {placed|demo_placed|testnet_placed}_orders/SYMBOL_{Short|Long}_Market_dd.mm-hh.mm.csv

Коды выхода: 0 - все ордера подтверждены на бирже; 1 - ошибка конфигурации/валидации (ничего не отправлено);
             2 - часть ордеров не удалось выставить/подтвердить.
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Optional

import ccxt
from dotenv import load_dotenv

import place_limit_grid as g

PROG = Path(__file__).name

# --------------------------------------------------------------------------------------
# Константы, которые можно менять
# --------------------------------------------------------------------------------------
BREAK_EVEN_SL_ORDER = 4  # -b: номер ордера (по направлению прибыли), которому ставится Stop-Loss
BREAK_EVEN_LEVEL_ORDER = 2  # -b: номер ордера, на чей уровень (+/- спред) ставится этот Stop-Loss
# TP/SL ставятся заранее только если цена по пути к их триггеру гарантированно сначала пройдет вход.
# Иначе TP/SL мог бы сработать РАНЬШЕ входа (когда закрывать еще нечего) и тихо пропасть. False - ставить и такие.
STRICT_ENTRY_FIRST = True
WORKING_TYPE = "CONTRACT_PRICE"  # по какой цене срабатывают триггеры (последняя цена); проверки цены используют ее же

log = logging.getLogger("market_grid")


@dataclass
class Algo:
    role: str  # ENTRY | TP | SL
    n: int  # номер входного ордера сетки (1..N по направлению прибыли); для TP/SL - номер ордера, который они закрывают
    cid: str  # clientAlgoId
    order_type: str  # STOP_MARKET | TAKE_PROFIT_MARKET
    side: str  # buy | sell
    position_side: str  # LONG | SHORT
    trigger: Decimal
    amount: Decimal
    line: int = 0  # строка CSV (для входов)

    @property
    def label(self) -> str:
        return f"{self.role}#{self.n}"


@dataclass
class Skipped:
    role: str
    n: int
    trigger: Optional[Decimal]
    reason: str


# --------------------------------------------------------------------------------------
# План
# --------------------------------------------------------------------------------------
def _market_lot(market: dict[str, Any]) -> tuple[Optional[Decimal], Optional[Decimal]]:
    for f in (market.get("info") or {}).get("filters", []):
        if f.get("filterType") == "MARKET_LOT_SIZE":
            return Decimal(str(f["minQty"])), Decimal(str(f["maxQty"]))
    return None, None


def entry_type(long: bool, trigger: Decimal, cur: Decimal) -> str:
    """BUY выше цены / SELL ниже цены -> STOP_MARKET; BUY ниже / SELL выше -> TAKE_PROFIT_MARKET."""
    above = trigger > cur
    return "STOP_MARKET" if above == long else "TAKE_PROFIT_MARKET"


def make_closing(role: str, entry: Algo, raw_trigger: Decimal, cur: Decimal, long: bool, ex: ccxt.Exchange,
                 sym: str, run_id: str, prefix: str) -> tuple[Optional[Algo], Optional[Skipped]]:
    """TP/SL закрывает ровно количество своего входа. Hedge: закрывающая сторона + positionSide позиции, без reduceOnly."""
    trigger = Decimal(ex.price_to_precision(sym, str(raw_trigger)))
    tp = role == "TP"

    def skip(reason: str) -> tuple[None, Skipped]:
        return None, Skipped(role, entry.n, trigger, reason)

    if tp and (trigger <= entry.trigger if long else trigger >= entry.trigger):
        return skip(f"TP не на стороне прибыли от входа {g.fmt(entry.trigger)} (шаг сетки не больше спреда)")
    if not tp and (trigger >= entry.trigger if long else trigger <= entry.trigger):
        return skip(f"SL не на стороне убытка от входа {g.fmt(entry.trigger)}")
    # Binance: условный ордер, который уже "сработал бы" по текущей цене, отклоняется (-2021)
    valid = (trigger > cur) if (tp == long) else (trigger < cur)
    if not valid:
        return skip(f"триггер уже за текущей ценой {g.fmt(cur)} - Binance отклонит (-2021)")
    if STRICT_ENTRY_FIRST and not (min(cur, trigger) < entry.trigger < max(cur, trigger)):
        return skip(f"цена дойдет до триггера раньше, чем сработает вход #{entry.n} ({g.fmt(entry.trigger)}): закрывать будет нечего")
    suffix = f"{'T' if tp else 'S'}{entry.n:02d}"
    return Algo(role, entry.n, f"{prefix}{run_id}-{suffix}", "TAKE_PROFIT_MARKET" if tp else "STOP_MARKET",
                "sell" if long else "buy", "LONG" if long else "SHORT", trigger, entry.amount), None


def build_plan(ex: ccxt.Exchange, market: dict[str, Any], rows: list[g.CsvRow], ticker: dict[str, Any], side: str,
               run_id: str, profit: bool, break_even: bool
               ) -> tuple[list[Algo], list[Algo], list[Skipped], list[str], list[str]]:
    sym, long = market["symbol"], side == "long"
    last = ticker.get("last") or ticker.get("close")
    if not last:
        raise g.GridError(f"Биржа не вернула текущую цену для {market['id']}")
    cur = Decimal(str(last))
    bid, ask = ticker.get("bid"), ticker.get("ask")
    spread = Decimal(str(ask)) - Decimal(str(bid)) if bid and ask else Decimal(0)
    errors: list[str] = []
    warnings: list[str] = []
    if spread <= 0:
        warnings.append("Спред по символу не получен или равен 0 - TP/SL считаются без поправки на спред")
    if not 1 <= BREAK_EVEN_LEVEL_ORDER < BREAK_EVEN_SL_ORDER:
        raise g.GridError("Константы BREAK_EVEN_*: нужно 1 <= BREAK_EVEN_LEVEL_ORDER < BREAK_EVEN_SL_ORDER")

    p_min, p_max = g._limit(market, "price", "min"), g._limit(market, "price", "max")
    a_min, a_max = _market_lot(market)
    a_min = a_min or g._limit(market, "amount", "min")
    a_max = a_max or g._limit(market, "amount", "max")
    c_min = g._limit(market, "cost", "min")
    prefix = "ml" if long else "ms"

    # Нумерация - по направлению прибыли: long по возрастанию цены, short по убыванию
    entries: list[Algo] = []
    seen: dict[Decimal, int] = {}
    for n, row in enumerate(sorted(rows, key=lambda r: r.level, reverse=not long), 1):
        where = f"ордер #{n} (строка {row.line}, Level={g.fmt(row.level)})"
        try:
            trigger = Decimal(ex.price_to_precision(sym, str(row.level)))
        except ccxt.BaseError as e:
            errors.append(f"{where}: не удалось округлить цену до tickSize: {e}")
            continue
        if trigger <= 0:
            errors.append(f"{where}: после округления до tickSize цена равна 0")
            continue
        if trigger > cur * g.MAX_PRICE_RATIO or trigger < cur / g.MAX_PRICE_RATIO:
            errors.append(f"{where}: отличается от текущей цены {g.fmt(cur)} более чем в {g.MAX_PRICE_RATIO} раз")
            continue
        if trigger == cur:
            errors.append(f"{where}: уровень равен текущей цене {g.fmt(cur)} - сработал бы сразу (-2021)")
            continue
        if (p_min is not None and trigger < p_min) or (p_max is not None and trigger > p_max):
            errors.append(f"{where}: цена {g.fmt(trigger)} вне допустимого диапазона [{p_min}, {p_max}]")
            continue
        if trigger in seen:
            errors.append(f"{where}: после округления совпадает с ордером #{seen[trigger]} (цена {g.fmt(trigger)})")
            continue
        seen[trigger] = n
        try:
            amount = Decimal(ex.amount_to_precision(sym, str(row.size / trigger)))  # вниз до stepSize
        except ccxt.BaseError as e:
            errors.append(f"{where}: размер {g.fmt(row.size)} USDT слишком мал для шага лота: {e}")
            continue
        notional = amount * trigger
        if amount <= 0 or (a_min is not None and amount < a_min):
            errors.append(f"{where}: количество {g.fmt(amount)} меньше минимального {a_min}")
            continue
        if a_max is not None and amount > a_max:
            errors.append(f"{where}: количество {g.fmt(amount)} больше максимального для рыночных ордеров {a_max}")
            continue
        if c_min is not None and notional < c_min:
            errors.append(f"{where}: стоимость {g.fmt(notional)} USDT меньше минимальной {g.fmt(c_min)} USDT")
            continue
        if notional < row.size * Decimal("0.9"):
            warnings.append(f"{where}: из-за шага лота стоимость {g.fmt(notional)} USDT заметно меньше заданных {g.fmt(row.size)}")
        entries.append(Algo("ENTRY", n, f"{prefix}{run_id}-E{n:02d}", entry_type(long, trigger, cur),
                            "buy" if long else "sell", "LONG" if long else "SHORT", trigger, amount, row.line))

    closings: list[Algo] = []
    skipped: list[Skipped] = []
    if errors:
        return entries, closings, skipped, errors, warnings
    n_total = len(entries)

    def add(res: tuple[Optional[Algo], Optional[Skipped]]) -> None:
        if res[0]:
            closings.append(res[0])
        if res[1]:
            skipped.append(res[1])

    if profit:
        step = Decimal(0)
        if n_total >= 2:  # средний шаг в % от уровня первого ордера: размах / (N-1 интервалов)
            step = abs(entries[-1].trigger - entries[0].trigger) / entries[0].trigger / (n_total - 1)
        for i, e in enumerate(entries):
            if i < n_total - 1:
                nxt = entries[i + 1].trigger
                raw = nxt - spread if long else nxt + spread
            elif n_total >= 2:
                raw = e.trigger * (1 + step if long else 1 - step)
            else:
                warnings.append("-p: в сетке один ордер - TP последнего ордера (средний шаг) рассчитать нельзя")
                continue
            add(make_closing("TP", e, raw, cur, long, ex, sym, run_id, prefix))
    if break_even:
        if n_total < BREAK_EVEN_SL_ORDER:
            warnings.append(f"-b: в сетке {n_total} ордеров, меньше {BREAK_EVEN_SL_ORDER} - Stop-Loss не ставится, сетка без SL")
        else:
            base = entries[BREAK_EVEN_LEVEL_ORDER - 1].trigger
            add(make_closing("SL", entries[BREAK_EVEN_SL_ORDER - 1], base + spread if long else base - spread,
                             cur, long, ex, sym, run_id, prefix))
    return entries, closings, skipped, errors, warnings


# --------------------------------------------------------------------------------------
# Размещение и проверка
# --------------------------------------------------------------------------------------
def place_one(ex: ccxt.Exchange, sym: str, s: Algo) -> None:
    params = {"triggerPrice": g.fmt(s.trigger), "positionSide": s.position_side,
              "clientOrderId": s.cid, "workingType": WORKING_TYPE}
    try:
        ex.create_order(sym, s.order_type, s.side, float(s.amount), None, params)
        log.info("  %-8s %-18s %s %s @ %s принят", s.label, s.order_type, s.side.upper(), g.fmt(s.amount), g.fmt(s.trigger))
    except ccxt.AuthenticationError:
        raise
    except ccxt.BaseError as e:
        log.warning("  %-8s %-18s @ %s не принят: %s", s.label, s.order_type, g.fmt(s.trigger), g._err_text(e))


def verify(ex: ccxt.Exchange, sym: str, specs: list[Algo]) -> dict[str, g.OrderState]:
    """Источник истины - биржа: открытые algo-ордера символа + статус каждого отсутствующего по clientAlgoId."""
    open_orders = g.api_call(ex.fetch_open_orders, sym, params={"trigger": True})
    by_cid = {o.get("clientOrderId"): o for o in open_orders if o.get("clientOrderId")}
    result: dict[str, g.OrderState] = {}
    for s in specs:
        o = by_cid.get(s.cid)
        if o is None:
            try:
                o = g.api_call(ex.fetch_order, s.cid, sym, {"trigger": True, "clientOrderId": s.cid})
            except ccxt.OrderNotFound:
                result[s.cid] = g.OrderState("absent")
                continue
            except ccxt.BaseError as e:
                result[s.cid] = g.OrderState("unknown", g._err_text(e))
                continue
            if o.get("status") != "open":
                status = (o.get("info") or {}).get("algoStatus") or o.get("status")
                result[s.cid] = g.OrderState("closed", f"algoStatus={status}")
                continue
        info = o.get("info") or {}
        trig, qty = Decimal(str(o.get("triggerPrice"))), Decimal(str(o.get("amount")))
        same = (o.get("side") == s.side and trig == s.trigger and qty == s.amount
                and info.get("positionSide", s.position_side) == s.position_side)
        oid = o.get("id") or info.get("algoId")
        result[s.cid] = g.OrderState("open", "", trig, qty, str(oid) if oid is not None else None) if same else g.OrderState(
            "mismatch", f"на бирже side={o.get('side')} trigger={o.get('triggerPrice')} amount={o.get('amount')}")
    return result


def run_grid(ex: ccxt.Exchange, sym: str, specs: list[Algo]) -> dict[str, g.OrderState]:
    log.info("Размещение %d условных ордеров по одному (algoOrder) ...", len(specs))
    for s in specs:
        place_one(ex, sym, s)
    status: dict[str, g.OrderState] = {}
    for attempt in range(g.MAX_RETRY_ROUNDS + 1):
        time.sleep(0.5)
        status = verify(ex, sym, specs)
        log.info("Проверка на бирже: подтверждено %d из %d", sum(1 for v in status.values() if v.ok), len(specs))
        missing = [s for s in specs if status[s.cid].state == "absent"]  # только "биржа достоверно не знает такого ордера"
        if not missing or attempt == g.MAX_RETRY_ROUNDS:
            break
        log.info("Повтор %d/%d: %s", attempt + 1, g.MAX_RETRY_ROUNDS, ", ".join(s.label for s in missing))
        for s in missing:
            place_one(ex, sym, s)
    return status


def write_placed_orders(symbol_id: str, side: str, mode: str, specs: list[Algo], status: dict[str, g.OrderState]) -> Optional[Path]:
    placed = [(s, status[s.cid]) for s in specs if status[s.cid].ok and status[s.cid].price is not None]
    if not placed:
        return None
    out_dir = g.BASE_DIR / g.OUTPUT_DIRS[mode]
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"{symbol_id}_{g.SIDE_CFG[side]['title']}_Market_{datetime.now():%d.%m-%H.%M}"
    path, n = out_dir / f"{base}.csv", 1
    while path.exists():
        n += 1
        path = out_dir / f"{base}_{n}.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["Role", "Order", "Type", "Side", "PositionSide", "Trigger", "Coins", "USDT", "AlgoId", "ClientAlgoId"])
        for s, st in placed:
            assert st.price is not None and st.amount is not None
            w.writerow([s.role, s.n, s.order_type, s.side.upper(), s.position_side, g.fmt(st.price), g.fmt(st.amount),
                        g.fmt(st.price * st.amount), st.order_id or "", s.cid])
    return path


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def print_plan(entries: list[Algo], closings: list[Algo], skipped: list[Skipped]) -> None:
    print(f"\n{'Роль':<8} {'Тип':<19} {'Сторона':<5} {'Trigger':>14} {'Qty':>14} {'USDT':>12}  clientAlgoId")
    for s in entries + closings:
        print(f"{s.label:<8} {s.order_type:<19} {s.side.upper():<5} {g.fmt(s.trigger):>14} {g.fmt(s.amount):>14} "
              f"{g.fmt(s.trigger * s.amount):>12}  {s.cid}")
    print(f"Итого: входов {len(entries)}, TP/SL {len(closings)}, на {g.fmt(sum((s.trigger * s.amount for s in entries), Decimal(0)))} USDT")
    if skipped:
        print("\nTP/SL, которые НЕ ставятся (причина):")
        for k in skipped:
            print(f"  {k.role}#{k.n} trigger {g.fmt(k.trigger) if k.trigger is not None else '-'}: {k.reason}")
    print()


def prepare(args: argparse.Namespace, run_id: str) -> tuple[ccxt.Exchange, dict[str, Any], list[Algo], list[Algo], list[Skipped]]:
    rows = g.parse_csv(Path(args.file))
    ex = g.build_exchange(args.exchange, args.mode)
    log.info("Режим=%s, загружаю рынки %s ...", args.mode, args.exchange)
    try:
        g.api_call(ex.load_markets)
    except ccxt.AuthenticationError as e:
        raise g.GridError(f"Биржа отклонила ключи для mode={args.mode}: {g._err_text(e)}") from e
    except ccxt.BaseError as e:
        raise g.GridError(f"Не удалось загрузить рынки в режиме {args.mode}: {g._err_text(e)}") from e

    market = g.resolve_market(ex, args.symbol, args.exchange, args.mode)
    sym = market["symbol"]
    try:
        ticker = g.api_call(ex.fetch_ticker, sym)
    except ccxt.BaseError as e:
        raise g.GridError(f"Символ {market['id']} есть в списке, но данные по нему недоступны в режиме {args.mode}: {g._err_text(e)}") from e
    try:
        lev_info = g.api_call(ex.fetch_leverage, sym)
        keys = ("shortLeverage", "longLeverage") if args.side == "short" else ("longLeverage", "shortLeverage")
        leverage = lev_info.get(keys[0]) or lev_info.get(keys[1])
        hedged = bool(g.api_call(ex.fetch_position_mode, sym).get("hedged"))
    except ccxt.AuthenticationError as e:
        raise g.GridError(f"Биржа отклонила ключи для mode={args.mode}: {g._err_text(e)}") from e
    except ccxt.BaseError as e:
        raise g.GridError(f"Не удалось получить плечо/режим позиций для {market['id']}: {g._err_text(e)}") from e
    if not leverage:
        raise g.GridError(f"Биржа не вернула плечо для {market['id']}")
    if not hedged:
        raise g.GridError("Аккаунт не в hedge-режиме: скрипт рассчитан только на hedge (positionSide LONG/SHORT)")
    log.info("%s %s: last=%s bid=%s ask=%s | плечо=%sx | hedge", market["id"], args.side, ticker.get("last"),
             ticker.get("bid"), ticker.get("ask"), leverage)

    entries, closings, skipped, errors, warnings = build_plan(ex, market, rows, ticker, args.side, run_id, args.profit, args.break_even)

    if entries and not errors:  # уже стоящие условные ордера на тех же уровнях - вероятно, сетка уже выставлена
        planned = {(e.side, e.position_side, e.trigger) for e in entries}
        for o in g.api_call(ex.fetch_open_orders, sym, params={"trigger": True}):
            info = o.get("info") or {}
            if o.get("triggerPrice") is not None and (o.get("side"), info.get("positionSide"), Decimal(str(o["triggerPrice"]))) in planned:
                errors.append(f"На уровне {o['triggerPrice']} уже стоит условный ордер {o.get('clientOrderId') or o.get('id')} - сетка, вероятно, уже выставлена")
        total = sum((e.trigger * e.amount for e in entries), Decimal(0))
        try:
            free = Decimal(str(g.api_call(ex.fetch_balance)["USDT"]["free"]))
            if total / Decimal(leverage) > free:
                warnings.append(f"Если сработают все входы: нужная маржа ~{g.fmt((total / Decimal(leverage)).quantize(Decimal('0.01')))} USDT "
                                f"(сумма {g.fmt(total)} / плечо {leverage}x), доступно {g.fmt(free)} USDT")
        except (ccxt.BaseError, KeyError, TypeError) as e:
            warnings.append(f"Не удалось сверить маржу с балансом: {e!r}")
        try:
            tiers = g.api_call(ex.fetch_market_leverage_tiers, sym)
            caps = [Decimal(str(t["maxNotional"])) for t in tiers if t.get("maxLeverage") and t["maxLeverage"] >= leverage and t.get("maxNotional")]
            if caps and total > max(caps):
                warnings.append(f"Сумма сетки {g.fmt(total)} USDT выше лимита позиции {g.fmt(max(caps))} USDT при плече {leverage}x")
        except (ccxt.BaseError, KeyError, TypeError, ValueError) as e:
            warnings.append(f"Не удалось проверить лимит позиции по плечу: {e!r}")

    for w in warnings:
        log.warning("ПРЕДУПРЕЖДЕНИЕ: %s", w)
    if errors:
        raise g.GridError("Проверки не пройдены, ордера НЕ отправлялись:\n  - " + "\n  - ".join(errors))
    return ex, market, entries, closings, skipped


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Сетка условных рыночных ордеров (STOP_MARKET / TAKE_PROFIT_MARKET) на USDT-M, hedge-режим. "
                    "Позиционно: SYMBOL SIDE FILE [EXCHANGE] [MODE]; ключи необязательны.")
    p.add_argument("pos", nargs="*", metavar="SYMBOL SIDE FILE [EXCHANGE] [MODE]", help="например: ESPORTSUSDT short Bot_levels.csv binance real")
    p.add_argument("--symbol", "-s")
    p.add_argument("--side", help="long (BUY) или short (SELL); обязателен, без умолчания")
    p.add_argument("--file", "-f", help="путь к CSV (колонки Level и Size)")
    p.add_argument("--exchange", "-e", help="биржа (по умолчанию binance)")
    p.add_argument("--mode", "-m", choices=g.MODES, help="real | demo | testnet (по умолчанию demo)")
    p.add_argument("-p", "--profit", action="store_true", help="выставить Take-Profit для каждого ордера")
    p.add_argument("-b", "--break-even", action="store_true", help="выставить Stop-Loss (безубыток) по константам BREAK_EVEN_*")
    p.add_argument("--dry-run", action="store_true", help="выполнить все проверки и показать план, не отправляя ордера")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    load_dotenv(Path(__file__).resolve().parent / ".env")
    load_dotenv()
    try:
        try:
            g.resolve_args(args)  # те же правила: SYMBOL SIDE FILE [EXCHANGE] [MODE]
        except g.GridError as e:
            raise g.GridError(str(e).replace("place_limit_grid.py", PROG)) from e
    except g.GridError as e:
        log.error("%s", e)
        return 1

    run_id = time.strftime("%y%m%d%H%M%S")
    try:
        ex, market, entries, closings, skipped = prepare(args, run_id)
    except g.GridError as e:
        log.error("%s", e)
        return 1
    except ccxt.BaseError as e:
        log.error("Ошибка биржи при подготовке: %s", g._err_text(e))
        return 1

    print_plan(entries, closings, skipped)
    if args.dry_run:
        log.info("--dry-run: проверки пройдены, ордера не отправлялись")
        return 0

    specs = entries + closings
    sym = market["symbol"]
    try:
        status = run_grid(ex, sym, specs)
    except ccxt.AuthenticationError as e:
        log.error("Ошибка авторизации/прав API-ключа: %s", g._err_text(e))
        return 2
    except ccxt.BaseError as e:
        log.error("Не удалось выполнить проверку после размещения: %s. Проверьте открытые условные ордера вручную (префикс %s%s)",
                  g._err_text(e), specs[0].cid[:2], run_id)
        return 2

    try:
        out_path = write_placed_orders(market["id"], args.side, args.mode, specs, status)
        if out_path:
            log.info("Установленные ордера (реальные значения с биржи) записаны: %s", out_path)
        else:
            log.warning("Нет подтвержденных ордеров - файл не создан")
    except OSError as e:
        log.error("Не удалось записать файл с ордерами: %s", e)

    bad = [s for s in specs if not status[s.cid].ok]
    print(f"\n{'Роль':<8} {'Trigger':>14} {'Qty':>14}  Статус")
    for s in specs:
        st = status[s.cid]
        print(f"{s.label:<8} {g.fmt(s.trigger):>14} {g.fmt(s.amount):>14}  "
              f"{'OK: стоит на бирже' if st.ok else f'ПРОБЛЕМА: {st.state} {st.detail}'.strip()}")
    if skipped:
        log.warning("Не ставились (см. причины выше): %s", ", ".join(f"{k.role}#{k.n}" for k in skipped))
    if bad:
        log.error("ИТОГ: на бирже подтверждено %d из %d ордеров; проблемные: %s", len(specs) - len(bad), len(specs),
                  ", ".join(s.label for s in bad))
        return 2
    log.info("ИТОГ: все %d ордеров подтверждены на бирже (%s %s, %s)", len(specs), market["id"], args.side, args.mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
