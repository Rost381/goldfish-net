#!/usr/bin/env python3
"""
cancel_grid.py - удаляет ордера сетки по файлу (place_limit_grid.py / place_market_grid.py)
ИЛИ по символу и направлению (удаляет все открытые лимитные ордера этого направления). Binance и Bybit.

Синтаксис:
  python cancel_grid.py file.csv [EXCHANGE] [MODE]
  python cancel_grid.py SYMBOL side [EXCHANGE] [MODE]

Примеры:
  python cancel_grid.py AIOTUSDT_Short_Limit_04.11-16.49.csv
  python cancel_grid.py placed_orders/AIOTUSDT_Long_Market_04.11-16.49.csv real
  python cancel_grid.py bybit_demo_placed_orders/BTCUSDT_Long_Limit_04.11-16.49.csv bybit demo
  python cancel_grid.py BLESSUSDT short
  python cancel_grid.py BTCUSDT LONG bybit real --dry-run
  python cancel_grid.py ETHUSDT Short binance testnet

EXCHANGE: binance | bybit (если не указана - берется из папки файла, для SYMBOL side - binance)
MODE: real | demo | testnet (по умолчанию demo; testnet только Binance)
side: long | short (регистр не важен);  SYMBOL: должен заканчиваться на USDT (регистр не важен)

Символ и тип сетки (Limit / Market) берутся из имени файла. Если указано только имя файла и его нет в текущей
папке, он ищется в папке биржи/режима рядом со скриптом (placed_orders, bybit_demo_placed_orders, ...).
Экономия запросов: сначала ОДИН запрос списка открытых ордеров символа, удаляются только те, что еще стоят
(исполненные или отмененные вручную пропускаются без запросов).
Режим SYMBOL side: отменяются открытые LIMIT-ордера ВХОДА этого направления (Long: BUY, Short: SELL; в hedge - своей
стороны позиции). Закрывающие ордера (reduceOnly, противоположная сторона/positionSide) не затрагиваются.
Binance Limit : DELETE /fapi/v1/batchOrders, до 10 ордеров за запрос (по clientOrderId).
Binance Market: algo-ордера batch-отмены не имеют; если нужно отменить ВСЕ открытые algo-ордера символа - один запрос
                DELETE /fapi/v1/algoOpenOrders, иначе по одному DELETE /fapi/v1/algoOrder.
                TP/SL ордера, чей вход уже сработал (позиция в работе), НЕ удаляются. TP/SL удаляются, только если вход
                еще не сработал (или был отменен/истек - тогда TP/SL осиротели).
Bybit Limit/Market: POST /v5/order/cancel-batch, до 20 ордеров за запрос (по orderLinkId; без него - по orderId).
                В Bybit TP/SL привязаны ко входу и отдельных ордеров не имеют: отмена сработавшего входа невозможна,
                а TP/SL уже открытых позиций остаются у биржи.
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
from typing import Callable, Optional

import ccxt
from dotenv import load_dotenv

import grid_exchange as ge
import place_limit_grid as g

RETRY_ROUNDS = 2  # повторные раунды отмены по одному после проверки

FILE_RE = re.compile(r"^(?P<symbol>[A-Z0-9]+USDT)_(?P<side>Short|Long)_(?P<kind>Limit|Market)_.+\.csv$", re.I)

log = logging.getLogger("cancel_grid")


@dataclass
class Item:
    role: str  # ENTRY | TP | SL
    n: int  # номер ордера сетки (для TP/SL - номер входа, который они закрывают)
    cid: str  # clientOrderId (Binance Limit, Bybit) / clientAlgoId (Binance Market)
    oid: str  # orderId / algoId
    action: str = ""  # cancel | gone | keep
    reason: str = ""

    @property
    def key(self) -> str:
        return self.cid or self.oid

    @property
    def ref(self) -> ge.Ref:
        return self.cid, self.oid

    @property
    def label(self) -> str:
        return f"{self.role}#{self.n}"


# --------------------------------------------------------------------------------------
# Файл
# --------------------------------------------------------------------------------------
def locate_file(arg: str, exchange: str, mode: str) -> Path:
    path = Path(arg)
    if not path.is_file() and path.parent == Path("."):  # только имя файла - ищем в папке биржи/режима
        alt = g.BASE_DIR / ge.output_dir_name(exchange, mode) / path.name
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
    cid_col = next((c for c in ("ClientOrderId", "ClientAlgoId") if c in fields), None)
    if cid_col is None:
        raise g.GridError("В файле нет колонки ClientOrderId/ClientAlgoId (ожидается файл place_limit_grid.py / place_market_grid.py)")
    oid_col = "OrderId" if "OrderId" in fields else "AlgoId"
    items: list[Item] = []
    for i, r in enumerate(rows, 1):
        cid, oid = (r.get(cid_col) or "").strip(), (r.get(oid_col) or "").strip()
        if kind == "Limit":
            items.append(Item("ENTRY", i, cid, oid))
        else:
            items.append(Item((r["Role"] or "").strip().upper(), int(r["Order"]), cid, oid))
    items = [it for it in items if it.key]
    if not items:
        raise g.GridError("В файле нет ни одного ордера")
    return symbol, side, kind, items


# --------------------------------------------------------------------------------------
# План
# --------------------------------------------------------------------------------------
def plan_limit(items: list[Item], open_keys: set[str]) -> None:
    for it in items:
        it.action, it.reason = ("cancel", "стоит на бирже") if it.key in open_keys else ("gone", "уже нет на бирже (исполнен или отменен)")


def plan_market(items: list[Item], open_keys: set[str], entry_status: Callable[[Item], tuple[str, str]]) -> None:
    entries = {it.n: it for it in items if it.role == "ENTRY"}
    verdict: dict[int, tuple[str, str]] = {}  # номер входа -> (executed | not_executed | unknown, подробность)
    for it in items:
        is_open = it.key in open_keys
        if it.role == "ENTRY":
            it.action, it.reason = ("cancel", "вход еще не сработал") if is_open else ("gone", "уже нет на бирже (сработал или отменен)")
            continue
        if not is_open:
            it.action, it.reason = "gone", "уже нет на бирже (сработал или отменен)"
            continue
        entry = entries.get(it.n)
        if entry is None:
            it.action, it.reason = "keep", f"входа #{it.n} нет в файле - не удаляю (защитный ордер), проверьте вручную"
        elif entry.key in open_keys:
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


def is_branch_order(row: ge.OpenRow, side: str) -> bool:
    """Открытый LIMIT-ордер ВХОДА направления side (Long/Short): BUY/LONG либо SELL/SHORT, не закрывающий."""
    want_side, want_pos = ("buy", "LONG") if side == "Long" else ("sell", "SHORT")
    return row.order_type == "LIMIT" and row.side == want_side and row.position_side in ("", want_pos) and not row.reduce_only


# --------------------------------------------------------------------------------------
# Удаление и проверка
# --------------------------------------------------------------------------------------
def _chunks(seq: list[ge.Ref], n: int) -> list[list[ge.Ref]]:
    return [seq[i : i + n] for i in range(0, len(seq), n)]


def cancel_one(ad: ge.ExchangeAdapter, symbol: str, conditional: bool, ref: ge.Ref) -> None:
    try:
        ad.cancel_one(symbol, ref, conditional)
    except ccxt.AuthenticationError:
        raise
    except ccxt.OrderNotFound:
        pass  # уже нет - цель достигнута
    except ccxt.BaseError as e:
        log.warning("  %s не отменен: %s", ref[0] or ref[1], g._err_text(e))


def execute(ad: ge.ExchangeAdapter, symbol: str, conditional: bool, to_cancel: list[Item], all_open: set[str]) -> set[str]:
    """Возвращает множество ключей, которые все еще стоят после отмены и повторов."""
    refs = [it.ref for it in to_cancel]
    keys = [it.key for it in to_cancel]
    if conditional and ad.can_cancel_all_conditional and len(keys) > 1 and set(keys) == all_open:
        log.info("Отменяются ВСЕ открытые условные ордера символа - один запрос ...")
        try:
            ad.cancel_all_conditional(symbol)
        except ccxt.AuthenticationError:
            raise
        except ccxt.BaseError as e:
            log.warning("Массовая отмена условных ордеров не удалась (%s) - статус выяснит проверка", g._err_text(e))
    elif conditional and not ad.conditional_batch_cancel:
        log.info("Отмена %d условных ордеров по одному (batch для них не существует) ...", len(refs))
        for ref in refs:
            cancel_one(ad, symbol, True, ref)
    else:
        log.info("Batch-отмена %d ордеров (по %d за запрос) ...", len(refs), ad.cancel_batch_size)
        for chunk in _chunks(refs, ad.cancel_batch_size):
            try:
                errors = ad.cancel_batch(symbol, chunk, conditional)
            except ccxt.AuthenticationError:
                raise
            except ccxt.BaseError as e:
                log.warning("Batch-отмена %d ордеров не удалась (%s) - статус выяснит проверка", len(chunk), g._err_text(e))
                continue
            log.info("  batch-отмена: %d ордеров, ошибок в ответе: %d", len(chunk), len(errors))
            for e_text in errors:
                log.warning("    %s", e_text)

    left: set[str] = set()
    for attempt in range(RETRY_ROUNDS + 1):
        now_open = {r.key for r in ad.list_open(symbol, conditional)}
        left = {k for k in keys if k in now_open}
        log.info("Проверка на бирже: из %d отменено %d, осталось %d", len(keys), len(keys) - len(left), len(left))
        if not left or attempt == RETRY_ROUNDS:
            break
        log.info("Повтор %d/%d: отмена по одному (%d)", attempt + 1, RETRY_ROUNDS, len(left))
        for it in to_cancel:
            if it.key in left:
                cancel_one(ad, symbol, conditional, it.ref)
    return left


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def parse_inputs(inputs: list[str]) -> tuple[list[str], Optional[str], Optional[str]]:
    """Хвост [EXCHANGE] [MODE] (с конца: сначала режим, затем биржа) -> (остаток, exchange | None, mode | None)."""
    rest = list(inputs)
    mode = exchange = None
    if rest and rest[-1].lower() in g.MODES:
        mode = rest.pop().lower()
    if rest and rest[-1].lower() in ge.ADAPTERS:
        exchange = rest.pop().lower()
    return rest, exchange, mode


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Удаляет ордера сетки (Binance/Bybit). Укажите либо CSV-файл, либо символ и направление.",
        epilog="""Примеры:
  python cancel_grid.py AIOTUSDT_Short_Limit_04.11-16.49.csv
  python cancel_grid.py placed_orders/AIOTUSDT_Long_Market_04.11-16.49.csv real
  python cancel_grid.py bybit_demo_placed_orders/BTCUSDT_Long_Limit_04.11-16.49.csv bybit demo
  python cancel_grid.py BLESSUSDT short
  python cancel_grid.py BTCUSDT LONG bybit real --dry-run""",
    )
    p.add_argument("inputs", nargs="+", metavar="ARG",
                   help="file.csv [EXCHANGE] [MODE] ИЛИ SYMBOL side [EXCHANGE] [MODE]. EXCHANGE: binance|bybit. "
                        "MODE: real|demo|testnet (по умолчанию demo). side: long|short. SYMBOL должен заканчиваться на USDT.")
    p.add_argument("--dry-run", action="store_true", help="показать, что будет удалено, ничего не удаляя")
    args = p.parse_args(argv)

    rest, exchange_arg, mode_arg = parse_inputs(args.inputs)
    if len(rest) == 1:
        file_arg, symbol_arg, side_arg = rest[0], None, None
    elif len(rest) == 2:
        file_arg, symbol_arg, side_arg = None, rest[0].upper(), rest[1].lower()
    else:
        p.error(f"Ожидается 1 или 2 аргумента (file ИЛИ symbol side), получено {len(rest)}: {' '.join(rest)}")
    if symbol_arg:
        if not symbol_arg.endswith("USDT") or len(symbol_arg) <= len("USDT"):
            p.error(f"Символ {symbol_arg!r} должен заканчиваться на 'USDT'")
        if side_arg not in g.SIDES:
            p.error(f"side должен быть 'long' или 'short', получено {side_arg!r}")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    load_dotenv(Path(__file__).resolve().parent / ".env")
    load_dotenv()

    try:
        mode = mode_arg or "demo"
        exchange = exchange_arg
        path: Optional[Path] = None
        if file_arg:
            probe = Path(file_arg)
            folder_info = ge.exchange_mode_from_folder(probe.resolve().parent.name) if probe.is_file() else None
            exchange = exchange or (folder_info[0] if folder_info else "binance")
            g.check_exchange_mode(exchange, mode)
            path = locate_file(file_arg, exchange, mode)
            folder_info = ge.exchange_mode_from_folder(path.resolve().parent.name)
            if folder_info and folder_info != (exchange, mode):
                raise g.GridError(f"Файл лежит в папке {path.resolve().parent.name!r} (биржа {folder_info[0]}, режим {folder_info[1]}), "
                                  f"а указано: биржа {exchange}, режим {mode}. Укажите: {folder_info[0]} {folder_info[1]}")
            symbol, side, kind, items = read_items(path)
            log.info("Файл %s: %s %s %s, ордеров в файле: %d, биржа=%s, режим=%s", path.name, symbol, side, kind, len(items), exchange, mode)
        else:
            exchange = exchange or "binance"
            g.check_exchange_mode(exchange, mode)
            assert symbol_arg and side_arg
            symbol, side, kind, items = symbol_arg, side_arg.title(), "Limit", []
            log.info("Символ: %s, направление: %s, тип: %s, биржа=%s, режим=%s", symbol, side, kind, exchange, mode)

        ad = ge.build_adapter(exchange, mode)
        conditional = kind == "Market"

        try:
            ad.sync_time()
            open_rows = ad.list_open(symbol, conditional)
            if not file_arg:
                branch = [r for r in open_rows if is_branch_order(r, side)]
                items = [Item("ENTRY", i, r.cid, r.oid) for i, r in enumerate(branch, 1)]
                if not items:
                    raise g.GridError(f"Не найдено открытых лимитных ордеров {side} для символа {symbol}")
                log.info("Найдено открытых лимитных ордеров %s: %d (других открытых ордеров символа не затрагиваю: %d)",
                         side, len(items), len(open_rows) - len(items))
                open_keys = {r.key for r in branch}
            else:
                open_keys = {r.key for r in open_rows}
        except ccxt.AuthenticationError as e:
            raise g.GridError(f"Биржа отклонила ключи для mode={mode}: {g._err_text(e)}") from e
        except ccxt.BaseError as e:
            raise g.GridError(f"Не удалось получить открытые ордера {symbol} в режиме {mode}: {g._err_text(e)}") from e

        try:
            if conditional:
                plan_market(items, open_keys, lambda entry: ad.conditional_status(symbol, entry.cid))
            else:
                plan_limit(items, open_keys)
        except ccxt.BaseError as e:
            log.error("Ошибка при определении статуса ордеров: %s", g._err_text(e))
            return 1

        print(f"\n{'Ордер':<9} {'Действие':<8} {'clientId':<24} Причина")
        for it in items:
            print(f"{it.label:<9} {it.action:<8} {it.key:<24} {it.reason}")

        to_cancel = [it for it in items if it.action == "cancel"]
        kept = [it for it in items if it.action == "keep"]
        gone = [it for it in items if it.action == "gone"]
        print(f"\nК удалению: {len(to_cancel)}, оставлено: {len(kept)}, уже нет на бирже: {len(gone)}\n")

        if args.dry_run:
            log.info("--dry-run: ничего не удалялось. Запросов к бирже: %d", ad.calls)
            return 0
        if not to_cancel:
            log.info("Удалять нечего. Запросов к бирже: %d", ad.calls)
            return 0

        try:
            # "все открытые" для массовой отмены: в режиме файла - все открытые условные ордера символа
            left = execute(ad, symbol, conditional, to_cancel, open_keys)
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
                      ", ".join(it.label for it in to_cancel if it.key in left), ad.calls)
            return 2

        log.info("ИТОГ: удалено %d, оставлено %d, уже не было %d. Запросов к бирже: %d", len(to_cancel), len(kept), len(gone), ad.calls)
        return 0

    except g.GridError as e:
        log.error("%s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
