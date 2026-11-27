#!/usr/bin/env python3
"""
grid_exchange.py - адаптеры бирж для сеточных скриптов: Binance USDT-M Futures и Bybit V5 (linear, USDT perpetual).

Все различия бирж собраны здесь; place_limit_grid.py / place_market_grid.py / cancel_grid.py работают через
единый интерфейс ExchangeAdapter и нормализованные структуры OrderView / OpenRow.

Binance: real | demo | testnet.   Bybit: real | demo (testnet не поддерживается).
Ключи (.env): real -> {EXCHANGE}_API_KEY/SECRET, demo -> {EXCHANGE}_DEMO_API_KEY/SECRET, testnet -> {EXCHANGE}_TESTNET_API_KEY/SECRET.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Optional, TypeVar

import ccxt

log = logging.getLogger("grid_exchange")
T = TypeVar("T")

API_RETRIES = 3  # повторы для сетевых ошибок на чтении (не на размещении)

ALL_MODES = ("real", "demo", "testnet")
EXCHANGE_MODES: dict[str, tuple[str, ...]] = {"binance": ("real", "demo", "testnet"), "bybit": ("real", "demo")}
# Папки результатов (рядом со скриптами). Папки Binance не менялись, для Bybit - отдельные.
OUTPUT_DIRS: dict[str, dict[str, str]] = {
    "binance": {"real": "placed_orders", "demo": "demo_placed_orders", "testnet": "testnet_placed_orders"},
    "bybit": {"real": "bybit_placed_orders", "demo": "bybit_demo_placed_orders"},
}


class GridError(Exception):
    """Ошибка конфигурации или валидации: ничего не отправлено на биржу."""


def api_call(fn: Callable[..., T], *args: Any, tries: int = API_RETRIES, **kwargs: Any) -> T:
    """Повторяет только сетевые ошибки. Использовать для чтения/отмены, не для размещения ордеров."""
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


def env_names(exchange: str, mode: str) -> tuple[str, str]:
    infix = "" if mode == "real" else f"_{mode.upper()}"
    prefix = exchange.upper()
    return f"{prefix}{infix}_API_KEY", f"{prefix}{infix}_API_SECRET"


def output_dir_name(exchange: str, mode: str) -> str:
    try:
        return OUTPUT_DIRS[exchange][mode]
    except KeyError:
        raise GridError(f"Режим {mode!r} не поддерживается для биржи {exchange!r}") from None


def exchange_mode_from_folder(folder: str) -> Optional[tuple[str, str]]:
    """Имя папки результатов -> (биржа, режим) или None."""
    for exch, modes in OUTPUT_DIRS.items():
        for mode, name in modes.items():
            if name == folder:
                return exch, mode
    return None


def _dec(value: Any) -> Optional[Decimal]:
    if value is None or value == "":
        return None
    try:
        d = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None
    return d if d.is_finite() else None


def _pos(value: Any) -> Optional[Decimal]:
    d = _dec(value)
    return d if d is not None and d > 0 else None


def _fmt(d: Decimal) -> str:
    return format(d.normalize(), "f")


def _err(e: BaseException) -> str:
    return str(e).replace("\n", " ")[:300]


# --------------------------------------------------------------------------------------
# Нормализованные структуры
# --------------------------------------------------------------------------------------
@dataclass
class OrderView:
    """Ордер (обычный или условный) в одинаковом виде для обеих бирж; значения - как на бирже."""

    cid: str
    id: str
    side: str  # buy | sell
    amount: Decimal
    status: str  # open | filled | closed
    raw_status: str = ""  # статус биржи как есть (NEW, Untriggered, TRIGGERED, ...)
    price: Optional[Decimal] = None
    trigger: Optional[Decimal] = None
    filled: Decimal = Decimal(0)
    avg: Optional[Decimal] = None
    position_side: str = ""  # LONG | SHORT | "" (one-way)
    tp: Optional[Decimal] = None  # привязанные TP/SL (Bybit)
    sl: Optional[Decimal] = None


@dataclass
class OpenRow:
    """Строка списка открытых ордеров для cancel_grid (без загрузки рынков)."""

    cid: str
    oid: str
    side: str  # buy | sell
    position_side: str  # LONG | SHORT | "" (one-way)
    order_type: str  # LIMIT | MARKET | ...
    reduce_only: bool = False

    @property
    def key(self) -> str:
        return self.cid or self.oid


@dataclass
class AccountState:
    hedged: bool
    leverage: Decimal
    margin_mode: str


@dataclass
class BatchResult:
    ok: bool
    order_id: str = ""
    error: str = ""


Ref = tuple[str, str]  # (clientOrderId, orderId): отмена по cid, если он есть, иначе по id


# --------------------------------------------------------------------------------------
# Базовый адаптер
# --------------------------------------------------------------------------------------
class ExchangeAdapter:
    id = ""
    ccxt_class = ""
    batch_size = 5  # ордеров в одном batch-запросе размещения
    cancel_batch_size = 10
    separate_closing_orders = True  # TP/SL - отдельные условные ордера (Binance) или привязаны ко входу (Bybit)
    max_conditional: Optional[int] = None  # лимит активных условных ордеров на символ
    can_cancel_all_conditional = False
    conditional_batch_cancel = False  # есть ли batch-отмена условных ордеров
    plain_order_params: dict[str, Any] = {}

    def __init__(self, mode: str) -> None:
        if mode not in EXCHANGE_MODES[self.id]:
            raise GridError(f"Режим {mode!r} не поддерживается для {self.id} (допустимо: {', '.join(EXCHANGE_MODES[self.id])})")
        self.mode = mode
        self.calls = 0  # счётчик запросов (для отчёта cancel_grid)
        self.ex = self._make_exchange()

    # ----- подключение -----
    def _credentials(self) -> tuple[str, str]:
        key_var, secret_var = env_names(self.id, self.mode)
        key, secret = os.getenv(key_var, "").strip(), os.getenv(secret_var, "").strip()
        if not key or not secret:
            raise GridError(f"Для mode={self.mode} в .env нужны {key_var} и {secret_var}")
        return key, secret

    def _make_exchange(self) -> ccxt.Exchange:
        raise NotImplementedError

    def call(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        self.calls += 1
        return api_call(fn, *args, **kwargs)

    def load_markets(self) -> None:
        self.call(self.ex.load_markets)

    def sync_time(self) -> None:
        self.call(self.ex.load_time_difference)

    def trading_ok(self, market: dict[str, Any]) -> bool:
        raise NotImplementedError

    def market_qty_limits(self, market: dict[str, Any]) -> tuple[Optional[Decimal], Optional[Decimal]]:
        """(min, max) количества для РЫНОЧНЫХ ордеров."""
        raise NotImplementedError

    # ----- аккаунт -----
    def account_state(self, market: dict[str, Any], side: str) -> AccountState:
        raise NotImplementedError

    # ----- ордера (с загруженными рынками) -----
    def open_orders(self, market: dict[str, Any], conditional: bool = False) -> list[OrderView]:
        raise NotImplementedError

    def order_by_cid(self, market: dict[str, Any], cid: str, conditional: bool = False) -> OrderView:
        """Бросает ccxt.OrderNotFound, если биржа такого ордера не знает."""
        raise NotImplementedError

    def send_limit_batch(self, market: dict[str, Any], requests: list[dict[str, Any]]) -> Optional[list[BatchResult]]:
        """requests - аргументы create_order_request. None - неожиданный ответ (полагаемся на проверку)."""
        raise NotImplementedError

    def create_order(self, market: dict[str, Any], order_type: str, side: str, amount: float, price: Optional[float],
                     params: dict[str, Any]) -> Optional[OrderView]:
        o = self.ex.create_order(market["symbol"], order_type, side, amount, price, {**self.plain_order_params, **params})
        return self._view_from_create(o)

    def _view_from_create(self, o: Any) -> Optional[OrderView]:
        return None

    def create_conditional(self, market: dict[str, Any], *, cid: str, order_type: str, side: str, position_side: str,
                           trigger: Decimal, amount: Decimal, last: Decimal, tp: Optional[Decimal] = None,
                           sl: Optional[Decimal] = None) -> None:
        raise NotImplementedError

    # ----- cancel_grid (сырые запросы по id символа биржи, без load_markets) -----
    def list_open(self, symbol_id: str, conditional: bool) -> list[OpenRow]:
        raise NotImplementedError

    def cancel_batch(self, symbol_id: str, refs: list[Ref], conditional: bool) -> list[str]:
        """Возвращает тексты ошибок из ответа (для лога)."""
        raise NotImplementedError

    def cancel_one(self, symbol_id: str, ref: Ref, conditional: bool) -> None:
        raise NotImplementedError

    def cancel_all_conditional(self, symbol_id: str) -> None:
        raise NotImplementedError

    def conditional_status(self, symbol_id: str, cid: str) -> tuple[str, str]:
        """(executed | not_executed | unknown, подробность) для условного ордера по clientId."""
        raise NotImplementedError


# --------------------------------------------------------------------------------------
# Binance USDT-M
# --------------------------------------------------------------------------------------
class BinanceAdapter(ExchangeAdapter):
    id = "binance"
    ccxt_class = "binanceusdm"
    batch_size = 5  # POST /fapi/v1/batchOrders
    cancel_batch_size = 10  # DELETE /fapi/v1/batchOrders
    separate_closing_orders = True
    can_cancel_all_conditional = True  # DELETE /fapi/v1/algoOpenOrders
    plain_order_params = {"isAlgoOrder": False}
    # algoStatus входа: сработал / не сработал; остальное - "неизвестно" (защитные ордера остаются)
    EXECUTED = {"TRIGGERING", "TRIGGERED", "FINISHED"}
    NOT_EXECUTED = {"CANCELED", "CANCELLED", "EXPIRED", "REJECTED"}

    def _make_exchange(self) -> ccxt.Exchange:
        key, secret = self._credentials()
        ex = ccxt.binanceusdm({"apiKey": key, "secret": secret, "enableRateLimit": True, "timeout": 20_000,
                               "options": {"defaultType": "swap", "adjustForTimeDifference": True}})
        if self.mode == "demo":
            ex.enable_demo_trading(True)  # demo-fapi.binance.com
        elif self.mode == "testnet":
            ex.set_sandbox_mode(True)  # testnet.binancefuture.com
        return ex

    def trading_ok(self, market: dict[str, Any]) -> bool:
        status = (market.get("info") or {}).get("status")
        return market.get("active") is not False and (not status or status == "TRADING")

    def market_qty_limits(self, market: dict[str, Any]) -> tuple[Optional[Decimal], Optional[Decimal]]:
        for f in (market.get("info") or {}).get("filters", []):
            if f.get("filterType") == "MARKET_LOT_SIZE":
                return _dec(f.get("minQty")), _dec(f.get("maxQty"))
        return None, None

    def account_state(self, market: dict[str, Any], side: str) -> AccountState:
        sym = market["symbol"]
        info = self.call(self.ex.fetch_leverage, sym)
        keys = ("shortLeverage", "longLeverage") if side == "short" else ("longLeverage", "shortLeverage")
        leverage = _pos(info.get(keys[0])) or _pos(info.get(keys[1]))
        if not leverage:
            raise GridError(f"Биржа не вернула плечо для {market['id']}")
        hedged = bool(self.call(self.ex.fetch_position_mode, sym).get("hedged"))
        return AccountState(hedged, leverage, str(info.get("marginMode") or ""))

    # ----- нормализация -----
    @staticmethod
    def _view(o: dict[str, Any]) -> OrderView:
        info = o.get("info") or {}
        cs = o.get("status")
        oid = info.get("orderId") or info.get("algoId") or o.get("id")
        return OrderView(
            cid=str(o.get("clientOrderId") or ""),
            id=str(oid) if oid is not None else "",
            side=str(o.get("side") or ""),
            amount=_dec(info.get("origQty") or info.get("quantity") or o.get("amount")) or Decimal(0),
            status="open" if cs == "open" else "filled" if cs == "closed" else "closed",
            raw_status=str(info.get("algoStatus") or info.get("status") or cs or ""),
            price=_pos(info.get("price") or o.get("price")),
            trigger=_pos(o.get("triggerPrice")),
            filled=_dec(o.get("filled")) or Decimal(0),
            avg=_pos(o.get("average") or info.get("avgPrice")),
            position_side=str(info.get("positionSide") or "").replace("BOTH", ""),
        )

    def open_orders(self, market: dict[str, Any], conditional: bool = False) -> list[OrderView]:
        params = {"trigger": True} if conditional else {}
        return [self._view(o) for o in self.call(self.ex.fetch_open_orders, market["symbol"], params=params)]

    def order_by_cid(self, market: dict[str, Any], cid: str, conditional: bool = False) -> OrderView:
        params = {"trigger": True, "clientOrderId": cid} if conditional else {"origClientOrderId": cid}
        return self._view(self.call(self.ex.fetch_order, cid, market["symbol"], params))

    def _view_from_create(self, o: Any) -> Optional[OrderView]:
        return self._view(o) if isinstance(o, dict) else None

    def send_limit_batch(self, market: dict[str, Any], requests: list[dict[str, Any]]) -> Optional[list[BatchResult]]:
        # Не ex.create_orders(): parse_orders() сортирует результат по timestamp, и у отклонённых элементов
        # ({"code","msg"}) его нет - соответствие "i-й ответ = i-й ордер" теряется. Тело строит ccxt, ответ читаем сырым.
        batch = [self.ex.create_order_request(r["symbol"], r["type"], r["side"], r["amount"], r["price"],
                                              {**r["params"], **self.plain_order_params}) for r in requests]
        resp = self.ex.fapiPrivatePostBatchOrders({"batchOrders": batch})
        if not isinstance(resp, list) or len(resp) != len(requests):
            log.warning("Неожиданный ответ batch: %.200s", resp)
            return None
        return [BatchResult(True, str(i["orderId"])) if isinstance(i, dict) and "orderId" in i else BatchResult(False, error=str(i)) for i in resp]

    def create_conditional(self, market: dict[str, Any], *, cid: str, order_type: str, side: str, position_side: str,
                           trigger: Decimal, amount: Decimal, last: Decimal, tp: Optional[Decimal] = None,
                           sl: Optional[Decimal] = None) -> None:
        # order_type: STOP_MARKET | TAKE_PROFIT_MARKET -> POST /fapi/v1/algoOrder (clientOrderId -> clientAlgoId)
        params = {"triggerPrice": _fmt(trigger), "positionSide": position_side, "clientOrderId": cid, "workingType": "CONTRACT_PRICE"}
        self.ex.create_order(market["symbol"], order_type, side, float(amount), None, params)

    # ----- cancel_grid -----
    def list_open(self, symbol_id: str, conditional: bool) -> list[OpenRow]:
        if conditional:
            rows = self.call(self.ex.fapiPrivateGetOpenAlgoOrders, {"symbol": symbol_id})
            cid_key, oid_key, type_key = "clientAlgoId", "algoId", "orderType"
        else:
            rows = self.call(self.ex.fapiPrivateGetOpenOrders, {"symbol": symbol_id})
            cid_key, oid_key, type_key = "clientOrderId", "orderId", "type"
        return [OpenRow(str(r.get(cid_key) or ""), str(r.get(oid_key) or ""), str(r.get("side") or "").lower(),
                        str(r.get("positionSide") or "").replace("BOTH", ""), str(r.get(type_key) or "").upper(),
                        str(r.get("reduceOnly")).lower() == "true" or str(r.get("closePosition")).lower() == "true")
                for r in rows]

    def cancel_batch(self, symbol_id: str, refs: list[Ref], conditional: bool) -> list[str]:
        if conditional:  # у algo-ордеров batch-отмены нет (conditional_batch_cancel = False)
            raise NotImplementedError("Binance: batch-отмены algo-ордеров не существует")
        by_cid = [c for c, _ in refs if c]
        resp = self.call(self.ex.fapiPrivateDeleteBatchOrders, {"symbol": symbol_id, "origClientOrderIdList": by_cid})
        return [f"code={r.get('code')} msg={r.get('msg')}" for r in resp if isinstance(r, dict) and "code" in r and "orderId" not in r] if isinstance(resp, list) else []

    def cancel_one(self, symbol_id: str, ref: Ref, conditional: bool) -> None:
        cid, oid = ref
        if conditional:
            self.call(self.ex.fapiPrivateDeleteAlgoOrder, {"symbol": symbol_id, "clientAlgoId": cid})
        elif cid:
            self.call(self.ex.fapiPrivateDeleteOrder, {"symbol": symbol_id, "origClientOrderId": cid})
        else:
            self.call(self.ex.fapiPrivateDeleteOrder, {"symbol": symbol_id, "orderId": oid})

    def cancel_all_conditional(self, symbol_id: str) -> None:
        self.call(self.ex.fapiPrivateDeleteAlgoOpenOrders, {"symbol": symbol_id})

    def conditional_status(self, symbol_id: str, cid: str) -> tuple[str, str]:
        try:
            row = self.call(self.ex.fapiPrivateGetAlgoOrder, {"clientAlgoId": cid})
        except ccxt.OrderNotFound:
            return "unknown", "биржа не нашла ордер"
        except ccxt.BaseError as e:
            return "unknown", _err(e)
        status = str(row.get("algoStatus") or "").upper()
        if status in self.EXECUTED:
            return "executed", f"algoStatus={status}"
        if status in self.NOT_EXECUTED:
            return "not_executed", f"algoStatus={status}"
        return "unknown", f"algoStatus={status or '?'}"


# --------------------------------------------------------------------------------------
# Bybit V5 (linear USDT perpetual, UTA)
# --------------------------------------------------------------------------------------
class BybitAdapter(ExchangeAdapter):
    id = "bybit"
    ccxt_class = "bybit"
    batch_size = 20  # POST /v5/order/create-batch, linear
    cancel_batch_size = 20  # POST /v5/order/cancel-batch, linear
    separate_closing_orders = False  # TP/SL привязываются ко входному ордеру (takeProfit/stopLoss, tpslMode=Partial)
    max_conditional = 10  # документация create-order: не более 10 активных условных ордеров на символ
    can_cancel_all_conditional = False
    conditional_batch_cancel = True  # cancel-batch принимает и условные ордера (по orderLinkId/orderId)
    CATEGORY = "linear"
    OPEN = {"New", "PartiallyFilled", "Untriggered", "Created", "Active"}
    EXECUTED = {"Triggered", "Filled", "PartiallyFilled", "PartiallyFilledCanceled"}
    NOT_EXECUTED = {"Cancelled", "Rejected", "Deactivated"}

    def _make_exchange(self) -> ccxt.Exchange:
        key, secret = self._credentials()
        ex = ccxt.bybit({"apiKey": key, "secret": secret, "enableRateLimit": True, "timeout": 20_000,
                         "options": {"defaultType": "swap", "defaultSubType": "linear", "adjustForTimeDifference": True,
                                     "fetchMarkets": {"types": ["linear"]}}})  # без spot/inverse/option - быстрее
        ex.has["fetchCurrencies"] = False  # приватный запрос монет не нужен и требует лишних прав ключа
        if self.mode == "demo":
            ex.enable_demo_trading(True)  # api-demo.bybit.com
        return ex

    def trading_ok(self, market: dict[str, Any]) -> bool:
        status = (market.get("info") or {}).get("status")
        return market.get("active") is not False and (not status or status == "Trading")

    def market_qty_limits(self, market: dict[str, Any]) -> tuple[Optional[Decimal], Optional[Decimal]]:
        lot = (market.get("info") or {}).get("lotSizeFilter") or {}
        return _dec(lot.get("minOrderQty")), _dec(lot.get("maxMktOrderQty"))

    def _result_list(self, resp: Any) -> tuple[list[dict[str, Any]], Optional[str]]:
        res = (resp or {}).get("result") or {}
        return list(res.get("list") or []), res.get("nextPageCursor") or None

    def _paged(self, fn: Callable[..., Any], params: dict[str, Any]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        cursor: Optional[str] = None
        for _ in range(20):  # 20 страниц x 50 - с запасом над лимитом 500 ордеров на символ
            page, cursor = self._result_list(self.call(fn, {**params, "limit": 50, **({"cursor": cursor} if cursor else {})}))
            rows += page
            if not cursor or not page:
                break
        return rows

    # режим позиций и плечо - одним запросом: GET /v5/position/list (fetch_position_mode в ccxt для bybit не реализован)
    def account_state(self, market: dict[str, Any], side: str) -> AccountState:
        rows, _ = self._result_list(self.call(self.ex.privateGetV5PositionList, {"category": self.CATEGORY, "symbol": market["id"]}))
        if not rows:
            raise GridError(f"Bybit не вернул данных о позиции/плече для {market['id']}")
        by_idx = {int(r.get("positionIdx") or 0): r for r in rows}
        hedged = bool(set(by_idx) & {1, 2})
        row = by_idx.get((1 if side == "long" else 2) if hedged else 0) or rows[0]
        leverage = _pos(row.get("leverage"))
        if not leverage:
            raise GridError(f"Биржа не вернула плечо для {market['id']}")
        return AccountState(hedged, leverage, "isolated" if str(row.get("tradeMode")) == "1" else "cross")

    # ----- нормализация -----
    def _view(self, r: dict[str, Any]) -> OrderView:
        raw = str(r.get("orderStatus") or "")
        return OrderView(
            cid=str(r.get("orderLinkId") or ""),
            id=str(r.get("orderId") or ""),
            side=str(r.get("side") or "").lower(),
            amount=_dec(r.get("qty")) or Decimal(0),
            status="open" if raw in self.OPEN else "filled" if raw == "Filled" else "closed",
            raw_status=raw,
            price=_pos(r.get("price")),
            trigger=_pos(r.get("triggerPrice")),
            filled=_dec(r.get("cumExecQty")) or Decimal(0),
            avg=_pos(r.get("avgPrice")),
            position_side={1: "LONG", 2: "SHORT"}.get(int(r.get("positionIdx") or 0), ""),
            tp=_pos(r.get("takeProfit")),
            sl=_pos(r.get("stopLoss")),
        )

    @staticmethod
    def _filter(conditional: bool) -> str:
        return "StopOrder" if conditional else "Order"

    def open_orders(self, market: dict[str, Any], conditional: bool = False) -> list[OrderView]:
        rows = self._paged(self.ex.privateGetV5OrderRealtime, {"category": self.CATEGORY, "symbol": market["id"],
                                                                "orderFilter": self._filter(conditional)})
        return [self._view(r) for r in rows]

    def _lookup(self, symbol_id: str, cid: str, conditional: bool) -> Optional[dict[str, Any]]:
        # ccxt fetch_order для bybit видит только последние 500 ордеров и требует acknowledged, поэтому сырые запросы:
        # активные (realtime) -> затем история по тому же orderLinkId
        base = {"category": self.CATEGORY, "symbol": symbol_id, "orderLinkId": cid}
        rows, _ = self._result_list(self.call(self.ex.privateGetV5OrderRealtime, {**base, "orderFilter": self._filter(conditional)}))
        if not rows:
            rows, _ = self._result_list(self.call(self.ex.privateGetV5OrderHistory, base))
        return rows[0] if rows else None

    def order_by_cid(self, market: dict[str, Any], cid: str, conditional: bool = False) -> OrderView:
        row = self._lookup(market["id"], cid, conditional)
        if row is None:
            raise ccxt.OrderNotFound(f"bybit: ордер {cid} не найден")
        return self._view(row)

    def send_limit_batch(self, market: dict[str, Any], requests: list[dict[str, Any]]) -> Optional[list[BatchResult]]:
        items = []
        for r in requests:
            item = self.ex.create_order_request(r["symbol"], r["type"], r["side"], r["amount"], r["price"], {**r["params"], **self.plain_order_params})
            item.pop("category", None)
            items.append(item)
        resp = self.ex.privatePostV5OrderCreateBatch({"category": self.CATEGORY, "request": items})
        rows, _ = self._result_list(resp)
        codes = (((resp or {}).get("retExtInfo") or {}).get("list")) or []
        if len(rows) != len(requests) or len(codes) != len(requests):  # порядок ответа = порядку запроса
            log.warning("Неожиданный ответ batch: %.200s", resp)
            return None
        out = []
        for row, code in zip(rows, codes):
            ok = str(code.get("code")) == "0" and bool(row.get("orderId"))
            out.append(BatchResult(ok, str(row.get("orderId") or ""), "" if ok else f"code={code.get('code')} msg={code.get('msg')}"))
        return out

    def create_conditional(self, market: dict[str, Any], *, cid: str, order_type: str, side: str, position_side: str,
                           trigger: Decimal, amount: Decimal, last: Decimal, tp: Optional[Decimal] = None,
                           sl: Optional[Decimal] = None) -> None:
        # Условный рыночный ордер: обычный POST /v5/order/create с triggerPrice + triggerDirection (1 - цена растёт до
        # триггера, 2 - падает). order_type (STOP_MARKET/TAKE_PROFIT_MARKET) - логическое имя, на бирже оно не нужно.
        params: dict[str, Any] = {"clientOrderId": cid, "triggerPrice": _fmt(trigger), "triggerBy": "LastPrice",
                                  "triggerDirection": "ascending" if trigger > last else "descending", "hedged": True}
        if tp is not None:
            params.update({"takeProfit": {"triggerPrice": _fmt(tp)}, "tpslMode": "Partial", "tpOrderType": "Market", "tpTriggerBy": "LastPrice"})
        if sl is not None:
            params.update({"stopLoss": {"triggerPrice": _fmt(sl)}, "tpslMode": "Partial", "slOrderType": "Market", "slTriggerBy": "LastPrice"})
        self.ex.create_order(market["symbol"], "market", side, float(amount), None, params)

    # ----- cancel_grid -----
    def list_open(self, symbol_id: str, conditional: bool) -> list[OpenRow]:
        rows = self._paged(self.ex.privateGetV5OrderRealtime, {"category": self.CATEGORY, "symbol": symbol_id,
                                                               "orderFilter": self._filter(conditional)})
        return [OpenRow(str(r.get("orderLinkId") or ""), str(r.get("orderId") or ""), str(r.get("side") or "").lower(),
                        {1: "LONG", 2: "SHORT"}.get(int(r.get("positionIdx") or 0), ""), str(r.get("orderType") or "").upper(),
                        bool(r.get("reduceOnly")) or bool(r.get("closeOnTrigger")) or str(r.get("stopOrderType") or "") not in ("", "UNKNOWN", "Stop"))
                for r in rows]

    @staticmethod
    def _ref_body(symbol_id: str, ref: Ref) -> dict[str, str]:
        cid, oid = ref
        return {"symbol": symbol_id, **({"orderLinkId": cid} if cid else {"orderId": oid})}

    def cancel_batch(self, symbol_id: str, refs: list[Ref], conditional: bool) -> list[str]:
        resp = self.call(self.ex.privatePostV5OrderCancelBatch, {"category": self.CATEGORY, "request": [self._ref_body(symbol_id, r) for r in refs]})
        codes = (((resp or {}).get("retExtInfo") or {}).get("list")) or []
        return [f"code={c.get('code')} msg={c.get('msg')}" for c in codes if str(c.get("code")) != "0"]

    def cancel_one(self, symbol_id: str, ref: Ref, conditional: bool) -> None:
        self.call(self.ex.privatePostV5OrderCancel, {"category": self.CATEGORY, **self._ref_body(symbol_id, ref)})

    def conditional_status(self, symbol_id: str, cid: str) -> tuple[str, str]:
        try:
            row = self._lookup(symbol_id, cid, True)
        except ccxt.BaseError as e:
            return "unknown", _err(e)
        if row is None:
            return "unknown", "биржа не нашла ордер"
        status = str(row.get("orderStatus") or "")
        if status in self.EXECUTED:
            return "executed", f"orderStatus={status}"
        if status in self.NOT_EXECUTED:
            return "not_executed", f"orderStatus={status}"
        return "unknown", f"orderStatus={status or '?'}"


ADAPTERS: dict[str, type[ExchangeAdapter]] = {"binance": BinanceAdapter, "bybit": BybitAdapter}


def build_adapter(exchange_id: str, mode: str) -> ExchangeAdapter:
    cls = ADAPTERS.get(exchange_id)
    if cls is None:
        raise GridError(f"Биржа {exchange_id!r} пока не поддерживается (доступно: {', '.join(ADAPTERS)})")
    return cls(mode)
