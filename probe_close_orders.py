#!/usr/bin/env python3
"""
probe_close_orders.py - ПРОБА на demo/testnet: какие условные (algo) ордера Binance USDT-M Futures принимает
в hedge-режиме, когда позиции по символу еще нет. Нужна, чтобы принять решение по TP/SL в place_market_grid.py.

Ничего не торгует (кроме опционального --with-position, только demo/testnet): все принятые ордера сразу отменяются.
real запрещен намеренно.

    python probe_close_orders.py XRPUSDT                    # demo, только ордера без позиции
    python probe_close_orders.py XRPUSDT --with-position    # + открывает минимальный LONG на demo и проверяет TP/SL "больше позиции"
    python probe_close_orders.py XRPUSDT testnet

Результат печатается и сохраняется в probe_<SYMBOL>_<mode>.txt рядом со скриптом - пришлите его целиком.
Требует place_limit_grid.py в той же папке (переиспользуется подключение, .env и разбор символа).
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Optional

import ccxt
from dotenv import load_dotenv

import place_limit_grid as g

DIST = Decimal("0.04")  # триггеры на +-4% от цены: заведомо "с правильной / неправильной" стороны
LINES: list[str] = []


def out(text: str = "") -> None:
    print(text)
    LINES.append(text)


@dataclass
class Result:
    ok: bool
    code: Optional[int] = None
    msg: str = ""


def code_of(e: BaseException) -> tuple[Optional[int], str]:
    text = str(e).replace("\n", " ")
    m = re.search(r'"code"\s*:\s*(-?\d+)', text)
    msg = re.search(r'"msg"\s*:\s*"([^"]*)"', text)
    return (int(m.group(1)) if m else None), (msg.group(1) if msg else text[:160])


class Probe:
    def __init__(self, ex: ccxt.Exchange, market: dict[str, Any], price: Decimal, qty: Decimal, tag: str) -> None:
        self.ex, self.market, self.sym, self.price, self.qty, self.tag = ex, market, market["symbol"], price, qty, tag
        self.n = 0
        self.live: list[str] = []  # clientOrderId принятых и еще не отмененных ордеров

    def trig(self, pct: Decimal) -> str:
        return self.ex.price_to_precision(self.sym, str(self.price * (1 + pct)))

    def place(self, title: str, order_type: str, side: str, pos_side: str, pct: Decimal,
              qty: Optional[Decimal] = None, close_position: bool = False, keep: bool = False) -> Result:
        self.n += 1
        cid = f"{self.tag}{self.n:02d}"
        params: dict[str, Any] = {"triggerPrice": self.trig(pct), "positionSide": pos_side,
                                  "clientOrderId": cid, "workingType": "CONTRACT_PRICE"}
        amount: Optional[float] = None
        if close_position:
            params["closePosition"] = True
        else:
            amount = float(qty if qty is not None else self.qty)
        try:
            self.ex.create_order(self.sym, order_type, side, amount, None, params)
            res = Result(True)
            self.live.append(cid)
            if not keep:
                self.cancel(cid)
        except ccxt.AuthenticationError:
            raise
        except ccxt.BaseError as e:
            res = Result(False, *code_of(e))
        tail = "ПРИНЯТ" if res.ok else f"ОТКЛОНЕН code={res.code} {res.msg}"
        out(f"  {title:<62} trigger {params['triggerPrice']:>10}  -> {tail}")
        return res

    def cancel(self, cid: str) -> None:
        try:
            self.ex.cancel_order(cid, self.sym, {"trigger": True, "clientOrderId": cid})
            self.live.remove(cid)
        except ccxt.BaseError as e:
            out(f"  ! не удалось отменить {cid}: {code_of(e)[1]} (отмените вручную)")

    def cleanup(self) -> None:
        for cid in list(self.live):
            self.cancel(cid)


def main() -> int:
    p = argparse.ArgumentParser(description="Проба: что Binance принимает без позиции (hedge). Только demo/testnet.")
    p.add_argument("symbol")
    p.add_argument("mode", nargs="?", default="demo", choices=("demo", "testnet"))
    p.add_argument("--with-position", action="store_true", help="открыть минимальный LONG (demo/testnet) и проверить TP/SL больше позиции")
    args = p.parse_args()
    symbol = args.symbol.strip().upper()
    if not symbol.endswith("USDT"):
        sys.exit("Символ должен оканчиваться на USDT")

    load_dotenv(g.Path(__file__).resolve().parent / ".env")
    load_dotenv()
    try:
        ex = g.build_exchange("binance", args.mode)
        g.api_call(ex.load_markets)
        market = g.resolve_market(ex, symbol, "binance", args.mode)
        sym = market["symbol"]
        price = Decimal(str(g.api_call(ex.fetch_ticker, sym)["last"]))
        hedged = bool(g.api_call(ex.fetch_position_mode, sym).get("hedged"))
        positions = [x for x in g.api_call(ex.fetch_positions, [sym]) if abs(float(x.get("contracts") or 0)) > 0]
    except (g.GridError, ccxt.BaseError) as e:
        sys.exit(f"Ошибка подготовки: {e}")
    if not hedged:
        sys.exit("Аккаунт не в hedge-режиме - проба для hedge, остановлено")
    if positions:
        sys.exit(f"По {market['id']} уже есть открытая позиция - для чистой пробы закройте ее")

    cost_min = g._limit(market, "cost", "min") or Decimal(5)
    qty = Decimal(ex.amount_to_precision(sym, str(max(cost_min * Decimal("2"), Decimal(10)) / price)))
    out(f"Проба {market['id']} | режим={args.mode} | hedge=да | цена={g.fmt(price)} | qty={g.fmt(qty)} (~{g.fmt(qty * price)} USDT)")
    out(f"ccxt {ccxt.__version__} | {time.strftime('%Y-%m-%d %H:%M:%S')}")

    pr = Probe(ex, market, price, qty, "pb" + time.strftime("%H%M%S"))
    up, dn = DIST, -DIST
    try:
        out("\nA. ВХОДЫ (условные ордера на открытие), позиции нет")
        a1 = pr.place("BUY  STOP_MARKET        LONG  (выше цены)", "STOP_MARKET", "buy", "LONG", up)
        a2 = pr.place("BUY  TAKE_PROFIT_MARKET LONG  (ниже цены)", "TAKE_PROFIT_MARKET", "buy", "LONG", dn)
        a3 = pr.place("SELL STOP_MARKET        SHORT (ниже цены)", "STOP_MARKET", "sell", "SHORT", dn)
        a4 = pr.place("SELL TAKE_PROFIT_MARKET SHORT (выше цены)", "TAKE_PROFIT_MARKET", "sell", "SHORT", up)
        a5 = pr.place("BUY  STOP_MARKET        LONG  (НЕ с той стороны, ждем -2021)", "STOP_MARKET", "buy", "LONG", dn)
        a6 = pr.place("SELL STOP_MARKET        SHORT (НЕ с той стороны, ждем -2021)", "STOP_MARKET", "sell", "SHORT", up)

        out("\nB. ЗАКРЫВАЮЩИЕ (TP/SL) БЕЗ позиции, с quantity")
        b1 = pr.place("LONG  TP: SELL TAKE_PROFIT_MARKET qty", "TAKE_PROFIT_MARKET", "sell", "LONG", up)
        b2 = pr.place("LONG  SL: SELL STOP_MARKET        qty", "STOP_MARKET", "sell", "LONG", dn)
        b3 = pr.place("SHORT TP: BUY  TAKE_PROFIT_MARKET qty", "TAKE_PROFIT_MARKET", "buy", "SHORT", dn)
        b4 = pr.place("SHORT SL: BUY  STOP_MARKET        qty", "STOP_MARKET", "buy", "SHORT", up)
        b5 = pr.place("LONG  SL: SELL STOP_MARKET closePosition=true", "STOP_MARKET", "sell", "LONG", dn, close_position=True)
        b6 = pr.place("LONG  SL выше цены (НЕ с той стороны, ждем -2021)", "STOP_MARKET", "sell", "LONG", up)

        c: list[Result] = []
        if args.with_position:
            out("\nC. С ОТКРЫТОЙ позицией (LONG, минимальный объем), TP/SL БОЛЬШЕ позиции")
            try:
                ex.create_order(sym, "market", "buy", float(qty), None, {"positionSide": "LONG"})
                out(f"  открыт LONG {g.fmt(qty)} по рынку")
                c.append(pr.place("LONG TP qty=3x позиции", "TAKE_PROFIT_MARKET", "sell", "LONG", up, qty * 3, keep=True))
                c.append(pr.place("LONG TP qty=3x еще один (сумма 6x позиции)", "TAKE_PROFIT_MARKET", "sell", "LONG", up * Decimal("1.5"), qty * 3, keep=True))
                c.append(pr.place("LONG SL qty=3x позиции", "STOP_MARKET", "sell", "LONG", dn, qty * 3, keep=True))
            except ccxt.BaseError as e:
                out(f"  не удалось открыть позицию: {code_of(e)[1]}")
            finally:
                pr.cleanup()
                try:
                    ex.create_order(sym, "market", "sell", float(qty), None, {"positionSide": "LONG"})
                    out("  позиция закрыта")
                except ccxt.BaseError as e:
                    out(f"  ! ЗАКРОЙТЕ ПОЗИЦИЮ ВРУЧНУЮ: {code_of(e)[1]}")
    finally:
        pr.cleanup()

    def verdict(rs: list[Result]) -> str:
        return "ПРИНИМАЮТСЯ" if all(r.ok for r in rs) else ("ОТКЛОНЯЮТСЯ" if not any(r.ok for r in rs) else "СМЕШАННО")

    out("\nИТОГ")
    out(f"  Входы STOP_MARKET/TAKE_PROFIT_MARKET с нужной стороны цены: {verdict([a1, a2, a3, a4])}")
    out(f"  Вход с неправильной стороны цены:                           {verdict([a5, a6])} (ожидаем ОТКЛОНЯЮТСЯ, -2021)")
    out(f"  TP/SL с quantity без позиции (hedge):                       {verdict([b1, b2, b3, b4])}")
    out(f"  SL closePosition=true без позиции:                          {verdict([b5])}")
    out(f"  SL с неправильной стороны цены:                             {verdict([b6])} (ожидаем ОТКЛОНЯЮТСЯ, -2021)")
    if c:
        out(f"  TP/SL больше позиции (с позицией):                          {verdict(c)}")

    path = g.Path(__file__).resolve().parent / f"probe_{market['id']}_{args.mode}.txt"
    path.write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print(f"\nСохранено: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
