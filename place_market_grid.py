#!/usr/bin/env python3
"""
place_market_grid.py - сетка УСЛОВНЫХ РЫНОЧНЫХ ордеров на USDT-M фьючерсах Binance и Bybit, hedge-режим.

    long : BUY  - вход при достижении уровня         short: SELL - вход при достижении уровня
    Тип ордера выбирается автоматически по положению уровня относительно текущей цены (CONTRACT_PRICE / LastPrice):
        BUY : уровень выше цены -> STOP_MARKET,        ниже -> TAKE_PROFIT_MARKET
        SELL: уровень ниже цены -> STOP_MARKET,        выше -> TAKE_PROFIT_MARKET
    Контроля положения сетки относительно цены нет: сетка может быть выше, ниже и вокруг текущей цены.

Позиционно: SYMBOL SIDE FILE [EXCHANGE] [MODE] [-p] [-b]
    python place_market_grid.py BTCUSDT long Bot_levels.csv
    python place_market_grid.py BTCUSDT Short Bot_levels.csv binance real -p -b
    python place_market_grid.py BTCUSDT long Bot_levels.csv bybit demo -p -b
    python place_market_grid.py BTCUSDT long Bot_levels.csv demo --profit --break-even --dry-run
    python place_market_grid.py BTCUSDT long Bot_levels.csv demo -p=0,7      # TP на 70% пути до уровня профита

    -p / --profit [ДОЛЯ]  Take-Profit каждого ордера. Без значения: TP = уровень следующего ордера
                       -/+ спред; для последнего - +/- средний шаг сетки в %.
                       Со значением (0 < ДОЛЯ <= 1, разделитель "." или ","): уровень профита считается так же, но TP
                       ставится на ДОЛЮ пути от входа до него: -p=0,7 - на 70% пути, то есть на 30% ближе к входу, чем
                       уровень следующего ордера с учетом спреда (то же для последнего ордера). Округление - до tickSize.
    -b / --break-even  Stop-Loss ордера номер BREAK_EVEN_SL_ORDER = уровень ордера BREAK_EVEN_LEVEL_ORDER +/- спред

Binance: все условные ордера идут через /fapi/v1/algoOrder (по одному на запрос, batch невозможен); входы, TP и SL -
    отдельные ордера (TP = TAKE_PROFIT_MARKET, SL = STOP_MARKET с positionSide, без reduceOnly).
Bybit:   вход - обычный рыночный ордер с triggerPrice + triggerDirection (логические типы STOP_MARKET/TAKE_PROFIT_MARKET
    сохраняются только в логах и CSV); TP/SL привязываются к самому входному ордеру (takeProfit/stopLoss, tpslMode=Partial,
    tpOrderType/slOrderType=Market, триггер по LastPrice) и создаются биржей при его срабатывании. Отдельных закрывающих
    ордеров нет, поэтому отмена сетки не затрагивает TP/SL уже открытых позиций. Лимит биржи - не более 10 активных
    условных ордеров на символ: входы + уже стоящие условные ордера символа должны укладываться в него.
Требует place_limit_grid.py и grid_exchange.py в той же папке (общий разбор CSV, подключение, .env, проверка).
Результат: папка результатов биржи/режима (см. place_limit_grid.py), SYMBOL_{Short|Long}_Market_dd.mm-hh.mm.csv

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
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Optional

import ccxt
from dotenv import load_dotenv

import grid_exchange as ge
import place_limit_grid as g

PROG = Path(__file__).name

# --------------------------------------------------------------------------------------
# Константы, которые можно менять
# --------------------------------------------------------------------------------------
BREAK_EVEN_SL_ORDER = 4  # -b: номер ордера (по направлению прибыли), которому ставится Stop-Loss
BREAK_EVEN_LEVEL_ORDER = 2  # -b: номер ордера, на чей уровень (+/- спред) ставится этот Stop-Loss
# Только Binance (отдельные TP/SL-ордера): TP/SL ставятся заранее, если цена по пути к их триггеру гарантированно сначала
# пройдет вход. Иначе TP/SL мог бы сработать РАНЬШЕ входа (когда закрывать еще нечего) и тихо пропасть. False - ставить и такие.
STRICT_ENTRY_FIRST = True
WORKING_TYPE = "CONTRACT_PRICE"  # Binance: по какой цене срабатывают триггеры (последняя цена); проверки цены используют ее же

log = logging.getLogger("market_grid")


@dataclass
class Algo:
    role: str  # ENTRY | TP | SL
    n: int  # номер входного ордера сетки (1..N по направлению прибыли); для TP/SL - номер ордера, который они закрывают
    cid: str  # clientAlgoId (Binance) / orderLinkId (Bybit)
    order_type: str  # STOP_MARKET | TAKE_PROFIT_MARKET (логический тип)
    side: str  # buy | sell
    position_side: str  # LONG | SHORT
    trigger: Decimal
    amount: Decimal
    line: int = 0  # строка CSV (для входов)
    tp: Optional[Decimal] = None  # Bybit: TP/SL, привязанные ко входу
    sl: Optional[Decimal] = None

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
def parse_profit_fraction(text: str) -> Decimal:
    """Значение -p/--profit: доля пути (0, 1]; разделитель "." или ","."""
    try:
        value = Decimal(text.strip().replace(",", "."))
    except InvalidOperation:
        value = None
    if value is None or not value.is_finite() or not 0 < value <= 1:
        raise g.GridError(f"-p/--profit: значение {text!r} не подходит - нужна доля пути от входа до уровня профита, "
                          f"число больше 0 и не больше 1 (например -p=0,7 или --profit=0.7 для 70%). "
                          f"Если вы хотели указать SYMBOL/SIDE/FILE/MODE - ставьте -p/--profit после них")
    return value


def _pct(value: Decimal) -> str:
    return format((value * 100).normalize(), "f")


def entry_type(long: bool, trigger: Decimal, cur: Decimal) -> str:
    """BUY выше цены / SELL ниже цены -> STOP_MARKET; BUY ниже / SELL выше -> TAKE_PROFIT_MARKET."""
    above = trigger > cur
    return "STOP_MARKET" if above == long else "TAKE_PROFIT_MARKET"


def make_closing(role: str, entry: Algo, raw_trigger: Decimal, cur: Decimal, long: bool, ex: ccxt.Exchange,
                 sym: str, run_id: str, prefix: str, attach: bool = False) -> tuple[Optional[Algo], Optional[Skipped]]:
    """TP/SL закрывает ровно количество своего входа.
    Binance (attach=False): отдельный ордер, hedge: закрывающая сторона + positionSide позиции, без reduceOnly.
    Bybit (attach=True): значение привязывается ко входу, проверяется только сторона относительно триггера входа."""
    trigger = Decimal(ex.price_to_precision(sym, str(raw_trigger)))
    tp = role == "TP"

    def skip(reason: str) -> tuple[None, Skipped]:
        return None, Skipped(role, entry.n, trigger, reason)

    if tp and (trigger <= entry.trigger if long else trigger >= entry.trigger):
        return skip(f"TP не на стороне прибыли от входа {g.fmt(entry.trigger)} (шаг сетки не больше спреда "
                    f"или доля -p слишком мала для tickSize)")
    if not tp and (trigger >= entry.trigger if long else trigger <= entry.trigger):
        return skip(f"SL не на стороне убытка от входа {g.fmt(entry.trigger)}")
    if not attach:
        # Binance: условный ордер, который уже "сработал бы" по текущей цене, отклоняется (-2021)
        valid = (trigger > cur) if (tp == long) else (trigger < cur)
        if not valid:
            return skip(f"триггер уже за текущей ценой {g.fmt(cur)} - Binance отклонит (-2021)")
        if STRICT_ENTRY_FIRST and not (min(cur, trigger) < entry.trigger < max(cur, trigger)):
            return skip(f"цена дойдет до триггера раньше, чем сработает вход #{entry.n} ({g.fmt(entry.trigger)}): закрывать будет нечего")
    suffix = f"{'T' if tp else 'S'}{entry.n:02d}"
    return Algo(role, entry.n, f"{prefix}{run_id}-{suffix}", "TAKE_PROFIT_MARKET" if tp else "STOP_MARKET",
                "sell" if long else "buy", "LONG" if long else "SHORT", trigger, entry.amount), None


def build_plan(adapter: ge.ExchangeAdapter, market: dict[str, Any], rows: list[g.CsvRow], ticker: dict[str, Any], side: str,
               run_id: str, profit: Optional[Decimal], break_even: bool
               ) -> tuple[list[Algo], list[Algo], list[Skipped], list[str], list[str]]:
    ex, attach = adapter.ex, not adapter.separate_closing_orders  # attach: TP/SL привязаны ко входу (Bybit)
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
    a_min, a_max = adapter.market_qty_limits(market)
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
            errors.append(f"{where}: уровень равен текущей цене {g.fmt(cur)} - сработал бы сразу")
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
        c, k = res
        if c and attach:  # Bybit: TP/SL - поля входного ордера, отдельных ордеров нет
            entry = entries[c.n - 1]
            if c.role == "TP":
                entry.tp = c.trigger
            else:
                entry.sl = c.trigger
        elif c:
            closings.append(c)
        if k:
            skipped.append(k)

    if profit is not None:  # profit = доля пути от входа до уровня профита; 1 = сам уровень (поведение -p без значения)
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
            if profit != 1:  # уровень профита -> точка на доле пути от входа к нему (дальше make_closing округлит до tickSize)
                raw = e.trigger + (raw - e.trigger) * profit
            add(make_closing("TP", e, raw, cur, long, ex, sym, run_id, prefix, attach))
    if break_even:
        if n_total < BREAK_EVEN_SL_ORDER:
            warnings.append(f"-b: в сетке {n_total} ордеров, меньше {BREAK_EVEN_SL_ORDER} - Stop-Loss не ставится, сетка без SL")
        else:
            base = entries[BREAK_EVEN_LEVEL_ORDER - 1].trigger
            add(make_closing("SL", entries[BREAK_EVEN_SL_ORDER - 1], base + spread if long else base - spread,
                             cur, long, ex, sym, run_id, prefix, attach))
    return entries, closings, skipped, errors, warnings


# --------------------------------------------------------------------------------------
# Размещение и проверка
# --------------------------------------------------------------------------------------
def place_one(adapter: ge.ExchangeAdapter, market: dict[str, Any], s: Algo, last: Decimal) -> None:
    try:
        adapter.create_conditional(market, cid=s.cid, order_type=s.order_type, side=s.side, position_side=s.position_side,
                                   trigger=s.trigger, amount=s.amount, last=last, tp=s.tp, sl=s.sl)
        extra = f" TP={g.fmt(s.tp)}" if s.tp is not None else ""
        extra += f" SL={g.fmt(s.sl)}" if s.sl is not None else ""
        log.info("  %-8s %-18s %s %s @ %s%s принят", s.label, s.order_type, s.side.upper(), g.fmt(s.amount), g.fmt(s.trigger), extra)
    except ccxt.AuthenticationError:
        raise
    except ccxt.BaseError as e:
        log.warning("  %-8s %-18s @ %s не принят: %s", s.label, s.order_type, g.fmt(s.trigger), g._err_text(e))


def verify(adapter: ge.ExchangeAdapter, market: dict[str, Any], specs: list[Algo]) -> dict[str, g.OrderState]:
    """Источник истины - биржа: открытые условные ордера символа + статус каждого отсутствующего по clientId."""
    by_cid = {o.cid: o for o in adapter.open_orders(market, conditional=True) if o.cid}
    result: dict[str, g.OrderState] = {}
    for s in specs:
        o = by_cid.get(s.cid)
        if o is None:
            try:
                o = adapter.order_by_cid(market, s.cid, conditional=True)
            except ccxt.OrderNotFound:
                result[s.cid] = g.OrderState("absent")
                continue
            except ccxt.BaseError as e:
                result[s.cid] = g.OrderState("unknown", g._err_text(e))
                continue
            if o.status != "open":
                result[s.cid] = g.OrderState("closed", f"status={o.raw_status}")
                continue
        same = (o.side == s.side and o.trigger == s.trigger and o.amount == s.amount
                and (not o.position_side or o.position_side == s.position_side) and o.tp == s.tp and o.sl == s.sl)
        result[s.cid] = g.OrderState("open", "", o.trigger, o.amount, o.id or None) if same else g.OrderState(
            "mismatch", f"на бирже side={o.side} trigger={o.trigger} amount={o.amount} tp={o.tp} sl={o.sl}")
    return result


def run_grid(adapter: ge.ExchangeAdapter, market: dict[str, Any], specs: list[Algo], last: Decimal) -> dict[str, g.OrderState]:
    log.info("Размещение %d условных ордеров по одному ...", len(specs))
    for s in specs:
        place_one(adapter, market, s, last)
    status: dict[str, g.OrderState] = {}
    for attempt in range(g.MAX_RETRY_ROUNDS + 1):
        time.sleep(0.5)
        status = verify(adapter, market, specs)
        log.info("Проверка на бирже: подтверждено %d из %d", sum(1 for v in status.values() if v.ok), len(specs))
        missing = [s for s in specs if status[s.cid].state == "absent"]  # только "биржа достоверно не знает такого ордера"
        if not missing or attempt == g.MAX_RETRY_ROUNDS:
            break
        log.info("Повтор %d/%d: %s", attempt + 1, g.MAX_RETRY_ROUNDS, ", ".join(s.label for s in missing))
        for s in missing:
            place_one(adapter, market, s, last)
    return status


def write_placed_orders(symbol_id: str, side: str, exchange: str, mode: str, specs: list[Algo], status: dict[str, g.OrderState]) -> Optional[Path]:
    placed = [(s, status[s.cid]) for s in specs if status[s.cid].ok and status[s.cid].price is not None]
    if not placed:
        return None
    attached = not ge.ADAPTERS[exchange].separate_closing_orders
    base = f"{symbol_id}_{g.SIDE_CFG[side]['title']}_Market_{datetime.now():%d.%m-%H.%M}"
    path = g.unique_csv_path(g.BASE_DIR / ge.output_dir_name(exchange, mode), base)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=";")
        if attached:  # Bybit: только входы, TP/SL - триггеры, привязанные ко входу
            w.writerow(["Role", "Order", "Type", "Side", "PositionSide", "Trigger", "Coins", "USDT", "TP", "SL", "OrderId", "ClientOrderId"])
        else:
            w.writerow(["Role", "Order", "Type", "Side", "PositionSide", "Trigger", "Coins", "USDT", "AlgoId", "ClientAlgoId"])
        for s, st in placed:
            assert st.price is not None and st.amount is not None
            row = [s.role, s.n, s.order_type, s.side.upper(), s.position_side, g.fmt(st.price), g.fmt(st.amount), g.fmt(st.price * st.amount)]
            if attached:
                row += [g.fmt(s.tp) if s.tp is not None else "", g.fmt(s.sl) if s.sl is not None else ""]
            w.writerow(row + [st.order_id or "", s.cid])
    return path


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def print_plan(entries: list[Algo], closings: list[Algo], skipped: list[Skipped], attached: bool) -> None:
    tpsl = f" {'TP':>14} {'SL':>14}" if attached else ""
    print(f"\n{'Роль':<8} {'Тип':<19} {'Сторона':<5} {'Trigger':>14} {'Qty':>14} {'USDT':>12}{tpsl}  clientOrderId")
    for s in entries + closings:
        extra = f" {g.fmt(s.tp) if s.tp is not None else '-':>14} {g.fmt(s.sl) if s.sl is not None else '-':>14}" if attached else ""
        print(f"{s.label:<8} {s.order_type:<19} {s.side.upper():<5} {g.fmt(s.trigger):>14} {g.fmt(s.amount):>14} "
              f"{g.fmt(s.trigger * s.amount):>12}{extra}  {s.cid}")
    n_tp, n_sl = (sum(e.tp is not None for e in entries), sum(e.sl is not None for e in entries)) if attached else (0, 0)
    closing_txt = f"TP {n_tp} и SL {n_sl} привязаны ко входам" if attached else f"TP/SL {len(closings)}"
    print(f"Итого: входов {len(entries)}, {closing_txt}, на {g.fmt(sum((s.trigger * s.amount for s in entries), Decimal(0)))} USDT")
    if skipped:
        print("\nTP/SL, которые НЕ ставятся (причина):")
        for k in skipped:
            print(f"  {k.role}#{k.n} trigger {g.fmt(k.trigger) if k.trigger is not None else '-'}: {k.reason}")
    print()


def prepare(args: argparse.Namespace, run_id: str) -> tuple[g.Context, list[Algo], list[Algo], list[Skipped]]:
    rows = g.parse_csv(Path(args.file))
    ctx = g.connect(args.exchange, args.mode, args.symbol, args.side)
    if not ctx.hedged:
        raise g.GridError("Аккаунт не в hedge-режиме: скрипт рассчитан только на hedge (positionSide LONG/SHORT)")
    ad, market = ctx.adapter, ctx.market

    entries, closings, skipped, errors, warnings = build_plan(ad, market, rows, ctx.ticker, args.side, run_id, args.profit, args.break_even)

    if entries and not errors:  # уже стоящие условные ордера на тех же уровнях - вероятно, сетка уже выставлена
        existing = ad.open_orders(market, conditional=True)
        planned = {(e.side, e.position_side, e.trigger) for e in entries}
        for o in existing:
            if o.trigger is not None and (o.side, o.position_side, o.trigger) in planned:
                errors.append(f"На уровне {g.fmt(o.trigger)} уже стоит условный ордер {o.cid or o.id} - сетка, вероятно, уже выставлена")
        if ad.max_conditional is not None and len(existing) + len(entries) > ad.max_conditional:
            errors.append(f"Лимит {ad.id}: не более {ad.max_conditional} активных условных ордеров на символ; "
                          f"сейчас стоит {len(existing)}, в сетке {len(entries)}. Уменьшите число уровней или отмените лишние ордера")
        g.margin_warnings(ctx, sum((e.trigger * e.amount for e in entries), Decimal(0)), warnings)

    for w in warnings:
        log.warning("ПРЕДУПРЕЖДЕНИЕ: %s", w)
    if errors:
        raise g.GridError("Проверки не пройдены, ордера НЕ отправлялись:\n  - " + "\n  - ".join(errors))
    return ctx, entries, closings, skipped


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Сетка условных рыночных ордеров (STOP_MARKET / TAKE_PROFIT_MARKET) на USDT-M Binance/Bybit, hedge-режим. "
                    "Позиционно: SYMBOL SIDE FILE [EXCHANGE] [MODE]; ключи необязательны.")
    p.add_argument("pos", nargs="*", metavar="SYMBOL SIDE FILE [EXCHANGE] [MODE]", help="например: ESPORTSUSDT short Bot_levels.csv bybit real")
    p.add_argument("--symbol", "-s")
    p.add_argument("--side", help="long (BUY) или short (SELL); обязателен, без умолчания")
    p.add_argument("--file", "-f", help="путь к CSV (колонки Level и Size)")
    p.add_argument("--exchange", "-e", help=f"биржа: {' | '.join(ge.ADAPTERS)} (по умолчанию binance)")
    p.add_argument("--mode", "-m", choices=g.MODES, help="real | demo | testnet (по умолчанию demo; testnet только Binance)")
    p.add_argument("-p", "--profit", nargs="?", const="1", default=None, metavar="ДОЛЯ",
                   help="выставить Take-Profit для каждого ордера. Без значения: на уровне открытия следующего ордера "
                        "с учетом спреда (для последнего - средний шаг сетки). Со значением (0 < ДОЛЯ <= 1, разделитель . или ,): "
                        "на эту долю пути от входа до этого уровня, например -p=0,7 или --profit=0.7 - на 70%% пути, то есть "
                        "на 30%% ближе к входу, чем уровень следующего ордера с учетом спреда (то же для последнего). "
                        "Округляется до tickSize. Ставьте после SYMBOL SIDE FILE [EXCHANGE] [MODE]")
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
        if args.profit is not None:
            args.profit = parse_profit_fraction(args.profit)
    except g.GridError as e:
        log.error("%s", e)
        return 1

    run_id = time.strftime("%y%m%d%H%M%S")
    try:
        ctx, entries, closings, skipped = prepare(args, run_id)
    except g.GridError as e:
        log.error("%s", e)
        return 1
    except ccxt.BaseError as e:
        log.error("Ошибка биржи при подготовке: %s", g._err_text(e))
        return 1

    adapter, market = ctx.adapter, ctx.market
    if args.profit is not None and args.profit != 1:
        log.info("-p=%s: Take-Profit ставится на %s%% пути от входа до уровня профита (на %s%% ближе к входу, чем уровень "
                 "следующего ордера с учетом спреда), округление до tickSize", g.fmt(args.profit), _pct(args.profit), _pct(1 - args.profit))
    print_plan(entries, closings, skipped, not adapter.separate_closing_orders)
    if args.dry_run:
        log.info("--dry-run: проверки пройдены, ордера не отправлялись")
        return 0

    specs = entries + closings
    last = Decimal(str(ctx.ticker.get("last") or ctx.ticker.get("close")))
    try:
        status = run_grid(adapter, market, specs, last)
    except ccxt.AuthenticationError as e:
        log.error("Ошибка авторизации/прав API-ключа: %s", g._err_text(e))
        return 2
    except ccxt.BaseError as e:
        log.error("Не удалось выполнить проверку после размещения: %s. Проверьте открытые условные ордера вручную (префикс %s%s)",
                  g._err_text(e), specs[0].cid[:2], run_id)
        return 2

    try:
        out_path = write_placed_orders(market["id"], args.side, args.exchange, args.mode, specs, status)
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
    log.info("ИТОГ: все %d ордеров подтверждены на бирже (%s %s, %s %s)", len(specs), market["id"], args.side, args.exchange, args.mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
