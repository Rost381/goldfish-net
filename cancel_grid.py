#!/usr/bin/env python3
"""
cancel_grid.py - удаляет ордера сетки по файлу (place_limit_grid.py / place_market_grid.py)
ИЛИ по символу и направлению (удаляет все открытые лимитные ордера этого направления).

Синтаксис:
  python cancel_grid.py file.csv [mode]
  python cancel_grid.py SYMBOL side [mode]

Примеры:
  python cancel_grid.py AIOTUSDT_Short_Limit_04.11-16.49.csv
  python cancel_grid.py placed_orders/AIOTUSDT_Long_Market_04.11-16.49.csv real
  python cancel_grid.py BLESSUSDT short
  python cancel_grid.py BTCUSDT LONG demo --dry-run
  python cancel_grid.py ETHUSDT Short testnet

mode: real | demo | testnet (по умолчанию demo)
side: long | short (регистр не важен)
SYMBOL: должен заканчиваться на USDT (регистр не важен)

Символ и тип сетки (Limit / Market) берутся из имени файла. Если указано только имя файла и его нет в текущей
папке, он ищется в папке режима рядом со скриптом (placed_orders / demo_placed_orders / testnet_placed_orders).
Экономия запросов: сначала ОДИН запрос списка открытых ордеров символа, удаляются только те, что еще стоят
(исполненные или отмененные вручную пропускаются без запросов).
Limit : DELETE /fapi/v1/batchOrders, до 10 ордеров за запрос (по clientOrderId).
Market: algo-ордера batch-отмены не имеют; если нужно отменить ВСЕ открытые algo-ордера символа - один запрос
DELETE /fapi/v1/algoOpenOrders, иначе по одному DELETE /fapi/v1/algoOrder.
Market: TP/SL ордера, чей вход уже сработал (позиция в работе), НЕ удаляются. TP/SL удаляются, только если вход
еще не сработал (или был отменен/истек - тогда TP/SL осиротели).
После удаления - проверка по бирже, не отмененные добиваются по одному.
Коды выхода: 0 - все, что нужно было удалить, удалено; 1 - ошибка аргументов/файла/подключения; 2 - часть ордеров не удалена.
"""
from __future__ import annotations
import argparse
import csv
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional
import ccxt
from dotenv import load_dotenv
import place_limit_grid as g

BATCH_CANCEL_SIZE = 10  # DELETE /fapi/v1/batchOrders: максимум 10 ордеров на запрос
RETRY_ROUNDS = 2  # повторные раунды отмены по одному после проверки

# Статусы algo-ордера входа (algoStatus): вход сработал / не сработал
EXECUTED = {"TRIGGERING", "TRIGGERED", "FINISHED"}
NOT_EXECUTED = {"CANCELED", "CANCELLED", "EXPIRED", "REJECTED"}

FILE_RE = re.compile(r"^(?P<symbol>[A-Z0-9]+USDT)_(?P<side>Short|Long)_(?P<kind>Limit|Market)_.+\.csv$", re.I)

log = logging.getLogger("cancel_grid")

@dataclass
class Item:
    role: str  # ENTRY | TP | SL
    n: int  # номер ордера сетки (для TP/SL - номер входа, который они закрывают)
    cid: str  # clientOrderId (Limit) / clientAlgoId (Market)
    oid: str  # orderId / algoId
    action: str = ""  # cancel | gone | keep
    reason: str = ""

    @property
    def label(self) -> str:
        return f"{self.role}#{self.n}"

class Api:
    """Тонкая обертка: считает запросы и повторяет только сетевые ошибки (g.api_call)."""
    def __init__(self) -> None:
        self.count = 0

    def __call__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        self.count += 1
        return g.api_call(fn, *args, **kwargs)

# --------------------------------------------------------------------------------------
# Файл
# --------------------------------------------------------------------------------------
def locate_file(arg: str, mode: str) -> Path:
    path = Path(arg)
    if not path.is_file() and path.parent == Path("."):  # только имя файла - ищем в папке режима
        alt = g.BASE_DIR / g.OUTPUT_DIRS[mode] / path.name
        if alt.is_file():
            path = alt
    if not path.is_file():
        raise g.GridError(f"Файл не найден: {arg}")
    return path

def read_items(path: Path) -> tuple[str, str, str, list[Item]]:
    m = FILE_RE.match(path.name)
    if not m:
        raise g.GridError(f"Имя файла {path.name!r} не похоже на SYMBOL_{{Short|Long}}_{{Limit|Market}}_дата-время.csv: "
                          f"символ и тип сетки берутся из имени, не переименовывайте файл")
    symbol, side, kind = m["symbol"].upper(), m["side"].title(), m["kind"].title()
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        fields = reader.fieldnames or []
        rows = list(reader)
        items: list[Item] = []
        if kind == "Limit":
            if "ClientOrderId" not in fields:
                raise g.GridError("В Limit-файле нет колонки ClientOrderId (ожидается файл place_limit_grid.py)")
            for i, r in enumerate(rows, 1):
                items.append(Item("ENTRY", i, (r["ClientOrderId"] or "").strip(), (r.get("OrderId") or "").strip()))
        else:
            if "ClientAlgoId" not in fields:
                raise g.GridError("В Market-файле нет колонки ClientAlgoId (ожидается файл place_market_grid.py)")
            for r in rows:
                items.append(Item((r["Role"] or "").strip().upper(), int(r["Order"]), (r["ClientAlgoId"] or "").strip(),
                                  (r.get("AlgoId") or "").strip()))
        items = [it for it in items if it.cid]
        if not items:
            raise g.GridError("В файле нет ни одного ордера")
    return symbol, side, kind, items

# --------------------------------------------------------------------------------------
# План
# --------------------------------------------------------------------------------------
def plan_limit(items: list[Item], open_cids: set[str]) -> None:
    for it in items:
        it.action, it.reason = ("cancel", "стоит на бирже") if it.cid in open_cids else ("gone", "уже нет на бирже (исполнен или отменен)")

def plan_market(items: list[Item], open_cids: set[str], entry_status: Callable[[Item], tuple[str, str]]) -> None:
    entries = {it.n: it for it in items if it.role == "ENTRY"}
    verdict: dict[int, tuple[str, str]] = {}  # номер входа -> (executed | not_executed | unknown, подробность)
    for it in items:
        is_open = it.cid in open_cids
        if it.role == "ENTRY":
            it.action, it.reason = ("cancel", "вход еще не сработал") if is_open else ("gone", "уже нет на бирже (сработал или отменен)")
            continue
        if not is_open:
            it.action, it.reason = "gone", "уже нет на бирже (сработал или отменен)"
            continue
        entry = entries.get(it.n)
        if entry is None:
            it.action, it.reason = "keep", f"входа #{it.n} нет в файле - не удаляю (защитный ордер), проверьте вручную"
        elif entry.cid in open_cids:
            it.action, it.reason = "cancel", f"вход #{it.n} еще не сработал"
        else:
            if it.n not in verdict:
                verdict[it.n] = entry_status(entry)  # один запрос на вход, только если у него остались открытые TP/SL
            state, detail = verdict[it.n]
            if state == "not_executed":
                it.action, it.reason = "cancel", f"вход #{it.n} не исполнялся ({detail}) - ордер осиротел"
            elif state == "executed":
                it.action, it.reason = "keep", f"вход #{it.n} исполнен ({detail}), позиция в работе - оставлен"
            else:
                it.action, it.reason = "keep", f"не удалось определить статус входа #{it.n} ({detail}) - оставлен"

# --------------------------------------------------------------------------------------
# Удаление и проверка
# --------------------------------------------------------------------------------------
def _chunks(seq: list[str], n: int) -> list[list[str]]:
    return [seq[i : i + n] for i in range(0, len(seq), n)]

def cancel_limit_batches(ex: ccxt.Exchange, api: Api, symbol: str, cids: list[str]) -> None:
    for chunk in _chunks(cids, BATCH_CANCEL_SIZE):
        try:
            resp = api(ex.fapiPrivateDeleteBatchOrders, {"symbol": symbol, "origClientOrderIdList": chunk})
        except ccxt.AuthenticationError:
            raise
        except ccxt.BaseError as e:
            log.warning("Batch-отмена %d ордеров не удалась (%s) - статус выяснит проверка", len(chunk), g._err_text(e))
            continue
        errors = [r for r in resp if isinstance(r, dict) and "code" in r and "orderId" not in r] if isinstance(resp, list) else []
        log.info("  batch-отмена: %d ордеров, ошибок в ответе: %d", len(chunk), len(errors))
        for r in errors:
            log.warning("    code=%s msg=%s", r.get("code"), r.get("msg"))

def cancel_one(ex: ccxt.Exchange, api: Api, symbol: str, kind: str, cid: str) -> None:
    try:
        if kind == "Limit":
            api(ex.fapiPrivateDeleteOrder, {"symbol": symbol, "origClientOrderId": cid})
        else:
            api(ex.fapiPrivateDeleteAlgoOrder, {"symbol": symbol, "clientAlgoId": cid})
    except ccxt.AuthenticationError:
        raise
    except ccxt.OrderNotFound:
        pass  # уже нет - цель достигнута
    except ccxt.BaseError as e:
        log.warning("  %s не отменен: %s", cid, g._err_text(e))

def fetch_open_cids(ex: ccxt.Exchange, api: Api, symbol: str, kind: str) -> set[str]:
    if kind == "Limit":
        rows = api(ex.fapiPrivateGetOpenOrders, {"symbol": symbol})
        return {r["clientOrderId"] for r in rows if r.get("clientOrderId")}
    rows = api(ex.fapiPrivateGetOpenAlgoOrders, {"symbol": symbol})
    return {r["clientAlgoId"] for r in rows if r.get("clientAlgoId")}

def execute(ex: ccxt.Exchange, api: Api, symbol: str, kind: str, to_cancel: list[Item], all_open: set[str]) -> set[str]:
    """Возвращает множество cid, которые все еще стоят после отмены и повторов."""
    cids = [it.cid for it in to_cancel]
    if kind == "Limit":
        log.info("Batch-отмена %d ордеров (по %d за запрос) ...", len(cids), BATCH_CANCEL_SIZE)
        cancel_limit_batches(ex, api, symbol, cids)
    elif len(cids) > 1 and set(cids) == all_open:
        log.info("Отменяются ВСЕ открытые algo-ордера символа - один запрос algoOpenOrders ...")
        try:
            api(ex.fapiPrivateDeleteAlgoOpenOrders, {"symbol": symbol})
        except ccxt.AuthenticationError:
            raise
        except ccxt.BaseError as e:
            log.warning("algoOpenOrders не удался (%s) - статус выяснит проверка", g._err_text(e))
    else:
        log.info("Отмена %d algo-ордеров по одному (batch для algo-ордеров не существует) ...", len(cids))
        for cid in cids:
            cancel_one(ex, api, symbol, kind, cid)

    left: set[str] = set()
    for attempt in range(RETRY_ROUNDS + 1):
        now_open = fetch_open_cids(ex, api, symbol, kind)
        left = {c for c in cids if c in now_open}
        log.info("Проверка на бирже: из %d отменено %d, осталось %d", len(cids), len(cids) - len(left), len(left))
        if not left or attempt == RETRY_ROUNDS:
            break
        log.info("Повтор %d/%d: отмена по одному (%d)", attempt + 1, RETRY_ROUNDS, len(left))
        for cid in sorted(left):
            cancel_one(ex, api, symbol, kind, cid)
    return left

# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Удаляет ордера сетки. Укажите либо CSV-файл, либо символ и направление.",
        epilog="""Примеры:
  python cancel_grid.py AIOTUSDT_Short_Limit_04.11-16.49.csv
  python cancel_grid.py placed_orders/AIOTUSDT_Long_Market_04.11-16.49.csv real
  python cancel_grid.py BLESSUSDT short
  python cancel_grid.py BTCUSDT LONG demo --dry-run
  python cancel_grid.py ETHUSDT Short testnet"""
    )
    p.add_argument("inputs", nargs="+", metavar="ARG",
                   help="file.csv [mode] ИЛИ SYMBOL side [mode]. "
                        "mode: real|demo|testnet (по умолчанию demo). "
                        "side: long|short (регистр не важен). "
                        "SYMBOL должен заканчиваться на USDT.")
    p.add_argument("--dry-run", action="store_true", help="показать, что будет удалено, ничего не удаляя")
    
    args = p.parse_args(argv)
    
    # Разбор позиционных аргументов
    inputs = args.inputs
    
    # Определяем mode
    mode = "demo"
    if inputs[-1].lower() in g.MODES:
        mode = inputs[-1].lower()
        inputs = inputs[:-1]
    
    # Теперь inputs содержит либо [file], либо [symbol, side]
    if len(inputs) == 1:
        # Это file
        file_arg = inputs[0]
        symbol_arg = None
        side_arg = None
    elif len(inputs) == 2:
        # Это symbol и side
        file_arg = None
        symbol_arg = inputs[0]
        side_arg = inputs[1]
    else:
        p.error(f"Ожидается 1 или 2 аргумента (file ИЛИ symbol side), получено {len(inputs)}: {' '.join(inputs)}")
    
    # Валидация
    if file_arg and (symbol_arg or side_arg):
        p.error("Нельзя указать одновременно file и symbol/side")
    if not file_arg and not (symbol_arg and side_arg):
        p.error("Укажите либо file, либо symbol и side")
    if (symbol_arg and not side_arg) or (side_arg and not symbol_arg):
        p.error("symbol и side должны использоваться вместе")
    
    # Валидация symbol
    if symbol_arg:
        symbol_arg = symbol_arg.upper()
        if not symbol_arg.endswith("USDT"):
            p.error(f"Символ {symbol_arg!r} должен заканчиваться на 'USDT'")
        side_arg = side_arg.lower()
        if side_arg not in ("long", "short"):
            p.error(f"side должен быть 'long' или 'short', получено {side_arg!r}")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    load_dotenv(Path(__file__).resolve().parent / ".env")
    load_dotenv()
    
    api = Api()
    
    try:
        if file_arg:
            path = locate_file(file_arg, mode)
            folder = path.resolve().parent.name
            if folder in g.OUTPUT_DIRS.values() and folder != g.OUTPUT_DIRS[mode]:
                wanted = next(m for m, d in g.OUTPUT_DIRS.items() if d == folder)
                raise g.GridError(f"Файл лежит в папке {folder!r} (режим {wanted}), а указан режим {mode}. Укажите режим {wanted}")
            symbol, side, kind, items = read_items(path)
            log.info("Файл %s: %s %s %s, ордеров в файле: %d, режим=%s", path.name, symbol, side, kind, len(items), mode)
        else:
            symbol = symbol_arg
            side = side_arg.title()
            kind = "Limit"
            items = []
            log.info("Символ: %s, направление: %s, тип: %s, режим=%s", symbol, side, kind, mode)

        ex = g.build_exchange("binance", mode)
        
        try:
            api(ex.load_time_difference)
            if file_arg:
                open_cids = fetch_open_cids(ex, api, symbol, kind)
            else:
                open_orders_raw = api(ex.fapiPrivateGetOpenOrders, {"symbol": symbol})
                binance_side = "BUY" if side == "Long" else "SELL"
                filtered_orders = [
                    o for o in open_orders_raw 
                    if o.get("side") == binance_side and o.get("type") == "LIMIT"
                ]
                items = []
                for i, o in enumerate(filtered_orders, 1):
                    items.append(Item("ENTRY", i, o["clientOrderId"], o["orderId"]))
                open_cids = {o["clientOrderId"] for o in filtered_orders}
                if not items:
                    raise g.GridError(f"Не найдено открытых лимитных ордеров {side} для символа {symbol}")
                log.info("Найдено открытых лимитных ордеров %s: %d", side, len(items))
                
        except ccxt.AuthenticationError as e:
            raise g.GridError(f"Биржа отклонила ключи для mode={mode}: {g._err_text(e)}") from e
        except ccxt.BaseError as e:
            raise g.GridError(f"Не удалось получить открытые ордера {symbol} в режиме {mode}: {g._err_text(e)}") from e
        except g.GridError as e:
            log.error("%s", e)
            return 1

        def entry_status(entry: Item) -> tuple[str, str]:
            try:
                row = api(ex.fapiPrivateGetAlgoOrder, {"clientAlgoId": entry.cid})
            except ccxt.OrderNotFound:
                return "unknown", "биржа не нашла ордер"
            except ccxt.BaseError as e:
                return "unknown", g._err_text(e)
            status = str(row.get("algoStatus") or "").upper()
            if status in EXECUTED:
                return "executed", f"algoStatus={status}"
            if status in NOT_EXECUTED:
                return "not_executed", f"algoStatus={status}"
            return "unknown", f"algoStatus={status or '?'}"

        try:
            if kind == "Limit":
                plan_limit(items, open_cids)
                all_open = open_cids
            else:
                plan_market(items, open_cids, entry_status)
                all_open = open_cids
        except ccxt.BaseError as e:
            log.error("Ошибка при определении статуса ордеров: %s", g._err_text(e))
            return 1

        print(f"\n{'Ордер':<9} {'Действие':<8} {'clientId':<24} Причина")
        for it in items:
            print(f"{it.label:<9} {it.action:<8} {it.cid:<24} {it.reason}")
            
        to_cancel = [it for it in items if it.action == "cancel"]
        kept = [it for it in items if it.action == "keep"]
        gone = [it for it in items if it.action == "gone"]
        print(f"\nК удалению: {len(to_cancel)}, оставлено: {len(kept)}, уже нет на бирже: {len(gone)}\n")

        if args.dry_run:
            log.info("--dry-run: ничего не удалялось. Запросов к бирже: %d", api.count)
            return 0

        if not to_cancel:
            log.info("Удалять нечего. Запросов к бирже: %d", api.count)
            return 0

        try:
            left = execute(ex, api, symbol, kind, to_cancel, all_open)
        except ccxt.AuthenticationError as e:
            log.error("Ошибка авторизации/прав API-ключа: %s", g._err_text(e))
            return 2
        except ccxt.BaseError as e:
            log.error("Не удалось выполнить отмену/проверку: %s. Проверьте открытые ордера %s вручную", g._err_text(e), symbol)
            return 2

        if kept:
            log.warning("Оставлены (позиция в работе или статус неизвестен): %s", ", ".join(it.label for it in kept))
            
        if left:
            log.error("ИТОГ: не удалось удалить %d из %d: %s. Запросов к бирже: %d", len(left), len(to_cancel),
                      ", ".join(it.label for it in to_cancel if it.cid in left), api.count)
            return 2
            
        log.info("ИТОГ: удалено %d, оставлено %d, уже не было %d. Запросов к бирже: %d", len(to_cancel), len(kept), len(gone), api.count)
        return 0

    except g.GridError as e:
        log.error("%s", e)
        return 1

if __name__ == "__main__":
    sys.exit(main())