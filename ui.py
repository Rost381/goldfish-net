#!/usr/bin/env python3
"""
ui.py - TUI launcher for place_limit_grid.py / place_market_grid.py / cancel_grid.py (Binance, Bybit)
Features:
- browser-like tabs: placement / cancellation;
- manual symbol input + paste from clipboard;
- symbol validation: only A-Z/0-9 and must end with USDT (both scripts);
- CSV file selection from disk;
- CSV preview + optional trimming (placement);
- cancel by placed-orders CSV file OR by symbol+side (side disabled when file chosen);
- grid type switch: limit orders (-foc) or conditional market orders (-p take-profit, -b break-even);
- exchange selection (binance / bybit); the mode list depends on the exchange;
- all form settings and the active tab are saved to ui.cfg (JSON) and restored on start;
- launching scripts with selected options.
Requires:
pip install textual
"""
from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Optional

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    DataTable,
    DirectoryTree,
    Footer,
    Header,
    Input,
    Label,
    RichLog,
    Select,
    Static,
    Switch,
    TabbedContent,
    TabPane,
)

# Path to the main scripts.
# By default, expected next to this file; can be overridden via environment.
SCRIPT_PATH = Path(
    os.getenv(
        "PLACE_LIMIT_GRID_SCRIPT",
        str(Path(__file__).resolve().with_name("place_limit_grid.py")),
    )
).expanduser().resolve()

CANCEL_SCRIPT_PATH = Path(
    os.getenv(
        "CANCEL_GRID_SCRIPT",
        str(Path(__file__).resolve().with_name("cancel_grid.py")),
    )
).expanduser().resolve()

MARKET_SCRIPT_PATH = Path(
    os.getenv(
        "PLACE_MARKET_GRID_SCRIPT",
        str(Path(__file__).resolve().with_name("place_market_grid.py")),
    )
).expanduser().resolve()

# All form settings and the active tab are stored here between launches.
CFG_PATH = Path(
    os.getenv("GRID_UI_CFG", str(Path(__file__).resolve().with_name("ui.cfg")))
).expanduser().resolve()

# Base project directory and default CSV file.
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = BASE_DIR / "input" / "Long26.csv"

MAX_GRID_ORDERS = 50
BYBIT_MAX_CONDITIONAL = 10  # Bybit: active conditional orders per symbol
SYMBOL_RE = re.compile(r"^[A-Z0-9]+USDT$")

# Mirror of cancel_grid.py: placed-orders file name mask and exchange/mode folders.
CANCEL_FILE_RE = re.compile(
    r"^(?P<symbol>[A-Z0-9]+USDT)_(?P<side>Short|Long)_(?P<kind>Limit|Market)_.+\.csv$",
    re.I,
)
EXCHANGE_MODES: dict[str, tuple[str, ...]] = {
    "binance": ("demo", "real", "testnet"),
    "bybit": ("demo", "real"),  # Bybit testnet is not supported
}
OUTPUT_DIRS: dict[str, dict[str, str]] = {
    "binance": {"real": "placed_orders", "demo": "demo_placed_orders", "testnet": "testnet_placed_orders"},
    "bybit": {"real": "bybit_placed_orders", "demo": "bybit_demo_placed_orders"},
}


def folder_info(folder: str) -> Optional[tuple[str, str]]:
    """Results folder name -> (exchange, mode) or None."""
    for exchange, modes in OUTPUT_DIRS.items():
        for mode, name in modes.items():
            if name == folder:
                return exchange, mode
    return None


# --------------------------------------------------------------------------------------
# Settings file (ui.cfg, JSON)
# --------------------------------------------------------------------------------------
PLACE_DEFAULTS: dict[str, Any] = {
    "symbol": "", "file": "", "trim": "", "kind": "limit", "side": "short", "exchange": "binance", "mode": "demo",
    "dry_run": False, "allow_marketable": False, "foc": False, "profit": False, "profit_value": "", "break_even": False,
}
CANCEL_DEFAULTS: dict[str, Any] = {
    "source": "symbol", "symbol": "", "side": "short", "file": "", "exchange": "binance", "mode": "demo", "dry_run": False,
}
CHOICES = {"kind": ("limit", "market"), "side": ("long", "short"), "source": ("symbol", "file")}
TABS = ("tab_place", "tab_cancel")


def _clean_section(defaults: dict[str, Any], raw: Any) -> dict[str, Any]:
    """Takes only known keys of the right type; anything invalid falls back to the default."""
    out = dict(defaults)
    if isinstance(raw, dict):
        for key, default in defaults.items():
            value = raw.get(key)
            if isinstance(default, bool):
                out[key] = value if isinstance(value, bool) else default
            elif isinstance(value, str):
                out[key] = value
    for key, allowed in CHOICES.items():
        if key in out and out[key] not in allowed:
            out[key] = defaults[key]
    if out["exchange"] not in EXCHANGE_MODES:
        out["exchange"] = defaults["exchange"]
    if out["mode"] not in EXCHANGE_MODES[out["exchange"]]:
        out["mode"] = "demo"
    return out


def load_cfg(path: Path) -> tuple[dict[str, Any], str]:
    """Returns (settings, warning). A missing or broken file is not fatal: defaults are used."""
    raw: Any = None
    warning = ""
    if path.is_file():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            warning = f"ВНИМАНИЕ: не удалось прочитать {path.name} ({exc}) - использованы значения по умолчанию"
    if not isinstance(raw, dict):
        raw = {}
    tab = raw.get("tab")
    return {
        "tab": tab if tab in TABS else "tab_place",
        "place": _clean_section(PLACE_DEFAULTS, raw.get("place")),
        "cancel": _clean_section(CANCEL_DEFAULTS, raw.get("cancel")),
    }, warning


def save_cfg(path: Path, data: dict[str, Any]) -> None:
    """Atomic write: temp file + replace, so a crash never leaves a half-written ui.cfg."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"version": 1, **data}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


# --------------------------------------------------------------------------------------
# Symbol validation
# --------------------------------------------------------------------------------------
def normalize_symbol(raw: str) -> str:
    """Remove quotes/spaces and convert the symbol to upper case."""
    return raw.strip().strip("'\"").replace(" ", "").upper()


def validate_symbol(raw: str) -> tuple[str, str]:
    """Return (normalized_symbol, error_message)."""
    symbol = normalize_symbol(raw)
    if not symbol:
        return symbol, "Символ обязателен"
    if len(symbol) <= len("USDT") or not SYMBOL_RE.match(symbol):
        return (
            symbol,
            "Неверный символ: нужен формат вида BTCUSDT "
            "(заканчивается на USDT, только A-Z и 0-9)",
        )
    return symbol, ""


# --------------------------------------------------------------------------------------
# Clipboard helpers
# --------------------------------------------------------------------------------------
def _decode_bytes(data: bytes) -> str:
    """Decode output from system clipboard commands."""
    if data.startswith(b"\xff\xfe"):
        return data.decode("utf-16-le", "ignore")
    if data.startswith(b"\xfe\xff"):
        return data.decode("utf-16-be", "ignore")
    for enc in ("utf-8", "cp1251", "cp866"):
        try:
            return data.decode(enc)
        except Exception:
            pass
    return data.decode("utf-8", "ignore")


def _cmd_output(cmd: list[str]) -> str:
    """Run a command and return its text output."""
    try:
        data = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        return _decode_bytes(data)
    except Exception:
        return ""


def get_clipboard_text() -> str:
    """
    Return the first non-empty line from the clipboard.
    Uses pyperclip if installed; otherwise falls back to system commands.
    """
    try:
        import pyperclip

        text = pyperclip.paste()
        if text:
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            if lines:
                return lines[0]
    except Exception:
        pass

    text = ""
    if sys.platform == "win32":
        text = _cmd_output(
            [
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "Get-Clipboard",
            ]
        )
    elif sys.platform == "darwin":
        text = _cmd_output(["pbpaste"])
    else:
        candidates = (
            ["wl-paste", "--no-newline"],
            ["xclip", "-selection", "clipboard", "-o"],
            ["xsel", "--clipboard", "--output"],
        )
        for cmd in candidates:
            if shutil.which(cmd[0]):
                text = _cmd_output(cmd)
                if text:
                    break

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[0] if lines else ""


# --------------------------------------------------------------------------------------
# CSV preview and trimming
# --------------------------------------------------------------------------------------
@dataclass
class PreviewRow:
    line: int
    raw: str
    level: str
    size: str
    error: str = ""


@dataclass
class CsvPreview:
    path: Path
    header: str = ""
    delimiter: str = ""
    rows: list[PreviewRow] = field(default_factory=list)
    fatal: str = ""


def _norm_name(token: str) -> str:
    """Normalize a CSV header column name."""
    return token.strip().strip("\"'").strip().lower()


def _split_line(line: str, delim: str) -> list[str]:
    """Split one CSV line using the detected delimiter."""
    if delim.isspace() and delim != "\t":
        return line.split()
    return next(csv.reader([line], delimiter=delim, skipinitialspace=True), [])


def _detect_delimiter(header: str) -> str:
    """Detect CSV delimiter from the header between Level and Size."""
    m = re.search(r"\b(level|size)\b([^A-Za-z0-9]*?)\b(level|size)\b", header, re.I)
    if not m:
        raise ValueError("Заголовок CSV должен содержать колонки 'Level' и 'Size' рядом, например 'Level;Size'")
    sep = m.group(2).replace('"', "").replace("'", "")
    if "\t" in sep:
        return "\t"
    sep = sep.strip()
    if not sep:
        return " "
    if len(sep) != 1:
        raise ValueError(f"Не удалось определить разделитель полей в заголовке: {header!r}")
    return sep


def _parse_decimal_preview(token: str, what: str) -> tuple[str, str]:
    """Validate a decimal token for preview purposes."""
    t = token.strip().strip("\"'").replace("\u00a0", "").replace(" ", "")
    if "," in t and "." in t:
        return token, f"{what}: неоднозначный формат числа (и ',' и '.')"
    if t.count(",") > 1 or t.count(".") > 1:
        return token, f"{what}: неверный формат числа"
    t2 = t.replace(",", ".")
    try:
        value = Decimal(t2)
    except InvalidOperation:
        return token, f"{what}: не число"
    if not value.is_finite() or value <= 0:
        return token, f"{what}: должно быть положительным числом"
    return token, ""


def parse_csv_preview(path: Path) -> CsvPreview:
    """Parse CSV file for preview without failing on the first bad row."""
    preview = CsvPreview(path=path)
    if not path.is_file():
        preview.fatal = "Файл не найден"
        return preview
    try:
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = raw.decode("cp1251")
        text = text.replace("\x00", "")
    except Exception as e:
        preview.fatal = f"Не удалось прочитать файл: {e}"
        return preview

    numbered = [(i, ln) for i, ln in enumerate(text.splitlines(), 1) if ln.strip()]
    if not numbered:
        preview.fatal = "CSV пустой"
        return preview

    header_line, header = numbered[0]
    try:
        delim = _detect_delimiter(header)
    except ValueError as e:
        preview.fatal = str(e)
        return preview

    names = [_norm_name(t) for t in _split_line(header, delim)]
    if names.count("level") != 1 or names.count("size") != 1:
        preview.fatal = f"Строка {header_line}: в заголовке должны быть ровно по одной колонке Level и Size, найдено: {names}"
        return preview
    i_level, i_size = names.index("level"), names.index("size")

    preview.header = header.rstrip()
    preview.delimiter = delim

    for line_no, line in numbered[1:]:
        tokens = _split_line(line, delim)
        # Skip visually empty lines such as ";;".
        if all(not t.strip() for t in tokens):
            continue
        error = ""
        level_text = ""
        size_text = ""
        if any(t.strip() for t in tokens[len(names):]):
            error = "больше колонок, чем в заголовке"
        elif len(tokens) <= max(i_level, i_size):
            error = "не хватает колонок"
        else:
            level_text = tokens[i_level].strip()
            size_text = tokens[i_size].strip()
            if not level_text and not size_text:
                continue
            if not level_text or not size_text:
                error = "заполнено только одно из полей Level/Size"
            else:
                _, err_level = _parse_decimal_preview(level_text, "Level")
                _, err_size = _parse_decimal_preview(size_text, "Size")
                if err_level:
                    error = err_level
                elif err_size:
                    error = err_size
        preview.rows.append(
            PreviewRow(
                line=line_no,
                raw=line.rstrip(),
                level=level_text,
                size=size_text,
                error=error,
            )
        )
    return preview


# --------------------------------------------------------------------------------------
# File picker modal dialog
# --------------------------------------------------------------------------------------
class FilePicker(ModalScreen):
    BINDINGS = [
        ("escape", "cancel", "Отмена"),
    ]
    CSS = """
    FilePicker {
        align: center middle;
    }
    #picker_dialog {
        width: 100;
        height: 80%;
        border: round;
        padding: 1;
    }
    #picker_tree {
        height: 1fr;
        border: round;
    }
    #picker_buttons {
        height: auto;
        margin-top: 1;
    }
    """

    def __init__(self, initial: Path):
        self.initial_path = initial.expanduser()
        if self.initial_path.is_file():
            self.initial_dir = self.initial_path.parent
            self.initial_value = str(self.initial_path)
        elif self.initial_path.is_dir():
            self.initial_dir = self.initial_path
            self.initial_value = str(self.initial_path)
        else:
            self.initial_dir = Path.cwd()
            self.initial_value = str(Path.cwd())
        super().__init__()

    def _make_tree(self):
        """Create DirectoryTree; filter argument may be unsupported in some Textual versions."""
        try:
            return DirectoryTree(
                self.initial_dir,
                filter=["*.csv", "*.CSV", "*.txt", "*.TXT"],
                id="picker_tree",
            )
        except TypeError:
            return DirectoryTree(self.initial_dir, id="picker_tree")

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static("Выбор CSV файла"),
            Input(value=self.initial_value, id="picker_path"),
            self._make_tree(),
            Horizontal(
                Button("Выбрать", id="picker_ok", variant="primary"),
                Button("Отмена", id="picker_cancel"),
                id="picker_buttons",
            ),
            id="picker_dialog",
        )

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        self.query_one("#picker_path", Input).value = str(event.path)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "picker_path":
            p = Path(event.value.strip()).expanduser()
            try:
                tree = self.query_one("#picker_tree", DirectoryTree)
                if p.is_dir():
                    tree.path = p
                elif p.parent.is_dir():
                    tree.path = p.parent
            except Exception:
                pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "picker_ok":
            self.dismiss(self.query_one("#picker_path", Input).value.strip())
        elif event.button.id == "picker_cancel":
            self.dismiss(None)


# --------------------------------------------------------------------------------------
# Main TUI application
# --------------------------------------------------------------------------------------
class GridTUI(App):
    TITLE = "grid TUI"
    BINDINGS = [
        ("ctrl+q", "quit", "Выход"),
        ("ctrl+1", "tab_place", "Размещение"),
        ("ctrl+2", "tab_cancel", "Отмена"),
    ]
    CSS = """
    Screen {
        layout: vertical;
    }
    /* ---- browser-like tabs ---- */
    TabbedContent {
        height: 1fr;
    }
    Tabs {
        height: 3;
    }
    Tab {
        height: 3;
        width: auto;
        padding: 0 3;
        margin-right: 1;
        background: $panel;
        color: $text-muted;
        border-top: round $panel-lighten-2;
        content-align: center middle;
    }
    Tab:hover {
        background: $panel-lighten-1;
        color: $text;
    }
    Tab.-active {
        background: $surface;
        color: $text;
        text-style: bold;
        border-top: round $accent;
    }
    Underline {
        display: none;
    }
    TabPane {
        padding: 0;
    }
    /* ---- form ---- */
    VerticalScroll {
        height: 1fr;
        border: round;
        padding: 1 2;
    }
    #title, #c_title {
        text-style: bold;
        margin-bottom: 1;
    }
    #defaults, #c_defaults {
        color: gray;
        margin-bottom: 1;
    }
    Horizontal {
        height: auto;
        margin-bottom: 1;
    }
    Label {
        width: 28;
    }
    Input {
        width: 1fr;
    }
    Select {
        width: 1fr;
    }
    #symbol_status, #csv_status, #c_symbol_status, #c_file_status {
        margin-left: 28;
        height: 1;
        margin-bottom: 1;
    }
    #preview {
        height: 12;
        border: round;
        margin-bottom: 1;
    }
    #output {
        height: 10;
        border: round;
        margin-top: 1;
    }
    Button {
        margin-right: 1;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self.cfg, self.cfg_warning = load_cfg(CFG_PATH)
        self._cfg_ready = False  # True after mount: before that widget changes must not overwrite ui.cfg
        self._save_failed = False

    def compose(self) -> ComposeResult:
        p, c = self.cfg["place"], self.cfg["cancel"]
        file_value = p["file"] or (str(DEFAULT_CSV) if DEFAULT_CSV.is_file() else "")
        yield Header()
        with TabbedContent(id="tabs", initial=self.cfg["tab"]):
            with TabPane("Размещение ордеров", id="tab_place"):
                yield VerticalScroll(
                    Static("", id="title"),
                    Static("", id="defaults"),
                    Horizontal(
                        Label("Symbol*"),
                        Input(value=p["symbol"], placeholder="BTCUSDT", id="symbol"),
                        Button("Вставить из буфера", id="paste_symbol"),
                    ),
                    Label("", id="symbol_status"),
                    Horizontal(
                        Label("CSV file*"),
                        Input(
                            value=file_value,
                            placeholder=f"{DEFAULT_CSV} (не найден — выберите файл)",
                            id="file",
                        ),
                        Button("Выбрать", id="browse_file"),
                    ),
                    Horizontal(
                        Label("Обрезать до уровней"),
                        Input(value=p["trim"], placeholder="пусто = весь файл", id="trim"),
                        Button("Обновить предпросмотр", id="refresh_preview"),
                    ),
                    Label("", id="csv_status"),
                    DataTable(id="preview"),
                    Horizontal(
                        Label("Тип сетки*"),
                        Select(
                            options=[
                                ("Лимитные ордера (place_limit_grid.py)", "limit"),
                                ("Условные рыночные (place_market_grid.py)", "market"),
                            ],
                            value=p["kind"],
                            allow_blank=False,
                            id="kind",
                        ),
                    ),
                    Horizontal(
                        Label("Side*"),
                        Select(
                            options=[("short", "short"), ("long", "long")],
                            value=p["side"],
                            allow_blank=False,
                            id="side",
                        ),
                    ),
                    Horizontal(
                        Label("Exchange"),
                        Select(
                            options=[(name, name) for name in EXCHANGE_MODES],
                            value=p["exchange"],
                            allow_blank=False,
                            id="exchange",
                        ),
                    ),
                    Horizontal(
                        Label("Mode"),
                        Select(
                            options=[(m, m) for m in EXCHANGE_MODES[p["exchange"]]],
                            value=p["mode"],
                            allow_blank=False,
                            id="mode",
                        ),
                    ),
                    Horizontal(
                        Label("--dry-run: только проверки"),
                        Switch(value=p["dry_run"], id="dry_run"),
                    ),
                    Horizontal(
                        Label("--allow-marketable"),
                        Switch(value=p["allow_marketable"], id="allow_marketable"),
                        id="row_allow",
                    ),
                    Horizontal(
                        Label("-foc: первый ордер - Лимитный"),
                        Switch(value=p["foc"], id="foc"),
                        id="row_foc",
                    ),
                    Horizontal(
                        Label("-p: Take-Profit"),
                        Switch(value=p["profit"], id="profit"),
                        id="row_profit",
                    ),
                    Horizontal(
                        Label("   доля пути (0 < x <= 1)"),
                        Input(
                            value=p["profit_value"],
                            placeholder="пусто = уровень следующего ордера; 0,7 = 70% пути",
                            id="profit_value",
                        ),
                        id="row_profit_value",
                    ),
                    Horizontal(
                        Label("-b: Stop-Loss (безубыток)"),
                        Switch(value=p["break_even"], id="break_even"),
                        id="row_break_even",
                    ),
                    Horizontal(
                        Button("Запустить", id="run", variant="primary"),
                        Button("Выход", id="quit"),
                    ),
                )
            with TabPane("Отмена ордеров", id="tab_cancel"):
                yield VerticalScroll(
                    Static("Запуск cancel_grid.py", id="c_title"),
                    Static(
                        "По умолчанию: источник=символ+side, side=short, exchange=binance, mode=demo, "
                        "--dry-run выключен. При источнике File символ и side берутся из имени файла, "
                        "а биржа и режим должны совпадать с папкой файла.",
                        id="c_defaults",
                    ),
                    Horizontal(
                        Label("Источник*"),
                        Select(
                            options=[
                                ("Символ + Side", "symbol"),
                                ("CSV файл", "file"),
                            ],
                            value=c["source"],
                            allow_blank=False,
                            id="c_source",
                        ),
                    ),
                    Horizontal(
                        Label("Symbol"),
                        Input(value=c["symbol"], placeholder="BTCUSDT", id="c_symbol"),
                        Button("Вставить из буфера", id="c_paste_symbol"),
                    ),
                    Label("", id="c_symbol_status"),
                    Horizontal(
                        Label("Side"),
                        Select(
                            options=[("short", "short"), ("long", "long")],
                            value=c["side"],
                            allow_blank=False,
                            id="c_side",
                        ),
                    ),
                    Horizontal(
                        Label("CSV file"),
                        Input(
                            value=c["file"],
                            placeholder="SYMBOL_Short_Limit_дата-время.csv (placed_orders / bybit_demo_placed_orders / ...)",
                            id="c_file",
                        ),
                        Button("Выбрать", id="c_browse_file"),
                    ),
                    Label("", id="c_file_status"),
                    Horizontal(
                        Label("Exchange"),
                        Select(
                            options=[(name, name) for name in EXCHANGE_MODES],
                            value=c["exchange"],
                            allow_blank=False,
                            id="c_exchange",
                        ),
                    ),
                    Horizontal(
                        Label("Mode"),
                        Select(
                            options=[(m, m) for m in EXCHANGE_MODES[c["exchange"]]],
                            value=c["mode"],
                            allow_blank=False,
                            id="c_mode",
                        ),
                    ),
                    Horizontal(
                        Label("--dry-run: только план"),
                        Switch(value=c["dry_run"], id="c_dry_run"),
                    ),
                    Horizontal(
                        Button("Запустить", id="c_run", variant="primary"),
                        Button("Выход", id="c_quit"),
                    ),
                )
        yield RichLog(
            id="output",
            wrap=True,
            markup=False,
            auto_scroll=True,
        )
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#c_source" if self.cfg["tab"] == "tab_cancel" else "#symbol").focus()
        self.append_output(f"Скрипт размещения (лимитные): {SCRIPT_PATH}")
        self.append_output(f"Скрипт размещения (условные рыночные): {MARKET_SCRIPT_PATH}")
        self.append_output(f"Скрипт отмены: {CANCEL_SCRIPT_PATH}")
        self.append_output(f"Настройки интерфейса: {CFG_PATH}")
        for path, env_name in (
            (SCRIPT_PATH, "PLACE_LIMIT_GRID_SCRIPT"),
            (MARKET_SCRIPT_PATH, "PLACE_MARKET_GRID_SCRIPT"),
            (CANCEL_SCRIPT_PATH, "CANCEL_GRID_SCRIPT"),
        ):
            if not path.is_file():
                self.append_output(
                    f"ВНИМАНИЕ: {path.name} не найден. Положи его рядом с ui.py или задай переменную окружения {env_name}."
                )
        if self.cfg_warning:
            self.append_output(self.cfg_warning)
        self._sync_kind()
        self._sync_profit()
        self.load_preview()
        self._update_cancel_source()
        self._cfg_ready = True  # с этого момента изменения виджетов сохраняются в ui.cfg

    # ----------------------------------------------------------------------------------
    # Output helpers
    # ----------------------------------------------------------------------------------
    def append_output(self, text: str) -> None:
        self.query_one("#output", RichLog).write(text or " ")

    def show_error(self, message: str) -> None:
        self.append_output(f"ОШИБКА: {message}")

    def _set_status(self, selector: str, message: str, color: str) -> None:
        status = self.query_one(selector, Label)
        status.update(message)
        status.styles.color = color

    def set_symbol_status(self, message: str, ok: bool) -> None:
        self._set_status("#symbol_status", message, "green" if ok else "red")

    # ----------------------------------------------------------------------------------
    # Tab switching (browser-like hotkeys)
    # ----------------------------------------------------------------------------------
    def _show_tab(self, tab: str) -> None:
        # TabbedContent returns to the tab that holds the focused widget, so the focus must move into the target tab too
        self.query_one("#tabs", TabbedContent).active = tab
        self.query_one("#symbol" if tab == "tab_place" else "#c_source").focus()

    def action_tab_place(self) -> None:
        self._show_tab("tab_place")

    def action_tab_cancel(self) -> None:
        self._show_tab("tab_cancel")

    # ----------------------------------------------------------------------------------
    # Settings persistence (ui.cfg) and dynamic form state
    # ----------------------------------------------------------------------------------
    def collect_state(self) -> dict[str, Any]:
        def sel(wid: str, default: str) -> str:
            value = self.query_one(wid, Select).value
            return value if isinstance(value, str) else default

        def txt(wid: str) -> str:
            return self.query_one(wid, Input).value

        def sw(wid: str) -> bool:
            return bool(self.query_one(wid, Switch).value)

        return {
            "tab": self.query_one("#tabs", TabbedContent).active,
            "place": {
                "symbol": txt("#symbol"),
                "file": txt("#file"),
                "trim": txt("#trim"),
                "kind": sel("#kind", "limit"),
                "side": sel("#side", "short"),
                "exchange": sel("#exchange", "binance"),
                "mode": sel("#mode", "demo"),
                "dry_run": sw("#dry_run"),
                "allow_marketable": sw("#allow_marketable"),
                "foc": sw("#foc"),
                "profit": sw("#profit"),
                "profit_value": txt("#profit_value"),
                "break_even": sw("#break_even"),
            },
            "cancel": {
                "source": sel("#c_source", "symbol"),
                "symbol": txt("#c_symbol"),
                "side": sel("#c_side", "short"),
                "file": txt("#c_file"),
                "exchange": sel("#c_exchange", "binance"),
                "mode": sel("#c_mode", "demo"),
                "dry_run": sw("#c_dry_run"),
            },
        }

    def save_state(self) -> None:
        """Сохраняет все настройки формы в ui.cfg. До завершения запуска (_cfg_ready) ничего не пишет."""
        if not self._cfg_ready:
            return
        try:
            save_cfg(CFG_PATH, self.collect_state())
            self._save_failed = False
        except (OSError, ValueError) as exc:
            if not self._save_failed:  # сообщаем один раз, а не на каждое нажатие клавиши
                self.append_output(f"ВНИМАНИЕ: не удалось сохранить настройки в {CFG_PATH}: {exc}")
            self._save_failed = True

    async def action_quit(self) -> None:
        self.save_state()
        self.exit()

    def _sync_modes(self, exchange_id: str, mode_id: str) -> None:
        """Список режимов зависит от биржи (testnet есть только у Binance); выбранный режим сохраняется, если он допустим."""
        exchange = str(self.query_one(f"#{exchange_id}", Select).value or "binance")
        mode_select = self.query_one(f"#{mode_id}", Select)
        modes = EXCHANGE_MODES.get(exchange, ("demo", "real"))
        current = mode_select.value if isinstance(mode_select.value, str) else "demo"
        mode_select.set_options([(m, m) for m in modes])
        mode_select.value = current if current in modes else "demo"

    def _sync_kind(self) -> None:
        """Лимитная сетка: -foc/--allow-marketable. Условная рыночная: -p/-b. Лишние строки скрываются."""
        market = self.query_one("#kind", Select).value == "market"
        for wid in ("#row_allow", "#row_foc"):
            self.query_one(wid).display = not market
        for wid in ("#row_profit", "#row_profit_value", "#row_break_even"):
            self.query_one(wid).display = market
        self.query_one("#title", Static).update(
            "Запуск place_market_grid.py (условные рыночные ордера, hedge-режим)" if market else "Запуск place_limit_grid.py"
        )
        self.query_one("#defaults", Static).update(
            "По умолчанию: side=short, exchange=binance, mode=demo, -p и -b выключены. "
            "Bybit: TP/SL привязываются ко входу, не более 10 условных ордеров на символ."
            if market
            else f"По умолчанию: side=short, exchange=binance, mode=demo, CSV={DEFAULT_CSV} (если существует), -foc выключен"
        )

    def _sync_profit(self) -> None:
        self.query_one("#profit_value", Input).disabled = not self.query_one("#profit", Switch).value

    # ----------------------------------------------------------------------------------
    # Events
    # ----------------------------------------------------------------------------------
    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "symbol":
            _, msg = validate_symbol(event.value)
            self.set_symbol_status(msg or "OK", not msg)
        elif event.input.id == "c_symbol":
            _, msg = validate_symbol(event.value)
            self._set_status("#c_symbol_status", msg or "OK", "red" if msg else "green")
        elif event.input.id == "c_file":
            self.update_cancel_file_status()
        self.save_state()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id in ("file", "trim"):
            self.load_preview()
        elif event.input.id == "c_file":
            self.update_cancel_file_status()

    def on_select_changed(self, event: Select.Changed) -> None:
        sid = event.select.id
        if sid == "c_source":
            self._update_cancel_source()
        elif sid == "c_exchange":
            self._sync_modes("c_exchange", "c_mode")
            self.update_cancel_file_status()
        elif sid == "c_mode":
            self.update_cancel_file_status()
        elif sid == "exchange":
            self._sync_modes("exchange", "mode")
            self.load_preview()
        elif sid == "kind":
            self._sync_kind()
            self.load_preview()
        self.save_state()

    def on_switch_changed(self, event: Switch.Changed) -> None:
        if event.switch.id == "profit":
            self._sync_profit()
        self.save_state()

    def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        self.save_state()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id in ("quit", "c_quit"):
            self.save_state()
            self.exit()
        elif event.button.id == "paste_symbol":
            self.paste_symbol()
        elif event.button.id == "browse_file":
            self.open_file_dialog()
        elif event.button.id == "refresh_preview":
            self.load_preview()
        elif event.button.id == "run":
            self.start_run()
        elif event.button.id == "c_paste_symbol":
            self.paste_symbol("#c_symbol", "#c_symbol_status")
        elif event.button.id == "c_browse_file":
            self.open_cancel_file_dialog()
        elif event.button.id == "c_run":
            self.start_cancel_run()

    # ----------------------------------------------------------------------------------
    # Symbol clipboard paste (shared by both tabs)
    # ----------------------------------------------------------------------------------
    def paste_symbol(self, input_id: str = "#symbol", status_id: str = "#symbol_status") -> None:
        text = get_clipboard_text()
        if not text:
            self.append_output("Буфер обмена пуст или недоступен")
            return
        symbol = normalize_symbol(text)
        self.query_one(input_id, Input).value = symbol
        _, msg = validate_symbol(symbol)
        self._set_status(status_id, msg or "OK", "red" if msg else "green")
        self.query_one(input_id, Input).focus()

    # ----------------------------------------------------------------------------------
    # File picker (placement)
    # ----------------------------------------------------------------------------------
    def open_file_dialog(self) -> None:
        raw = self.query_one("#file", Input).value.strip().strip("'\"")
        if raw:
            initial = Path(os.path.expandvars(raw)).expanduser()
        elif DEFAULT_CSV.parent.is_dir():
            initial = DEFAULT_CSV.parent
        else:
            initial = BASE_DIR
        self.push_screen(FilePicker(initial), self.on_file_picked)

    def on_file_picked(self, result: Optional[str]) -> None:
        if result:
            self.query_one("#file", Input).value = result
            self.load_preview()

    # ----------------------------------------------------------------------------------
    # Cancel tab: source switch, file picker, validation
    # ----------------------------------------------------------------------------------
    def _update_cancel_source(self) -> None:
        """File source disables Symbol/Side; symbol source disables file widgets."""
        by_file = self.query_one("#c_source", Select).value == "file"
        for wid in ("#c_symbol", "#c_paste_symbol", "#c_side"):
            self.query_one(wid).disabled = by_file
        for wid in ("#c_file", "#c_browse_file"):
            self.query_one(wid).disabled = not by_file
        if by_file:
            self._set_status("#c_symbol_status", "Берётся из имени файла", "gray")
        else:
            _, msg = validate_symbol(self.query_one("#c_symbol", Input).value)
            self._set_status("#c_symbol_status", msg or "OK", "red" if msg else "green")
        self.update_cancel_file_status()

    def open_cancel_file_dialog(self) -> None:
        raw = self.query_one("#c_file", Input).value.strip().strip("'\"")
        if raw:
            initial = Path(os.path.expandvars(raw)).expanduser()
        else:
            exchange = str(self.query_one("#c_exchange", Select).value or "binance")
            mode = str(self.query_one("#c_mode", Select).value or "demo")
            folder = BASE_DIR / OUTPUT_DIRS.get(exchange, {}).get(mode, "demo_placed_orders")
            initial = folder if folder.is_dir() else BASE_DIR
        self.push_screen(FilePicker(initial), self.on_cancel_file_picked)

    def on_cancel_file_picked(self, result: Optional[str]) -> None:
        if result:
            self.query_one("#c_file", Input).value = result
            self.update_cancel_file_status()

    def _cancel_file_path(self) -> tuple[Optional[Path], str]:
        """Validate placed-orders CSV: exists, name mask, mode folder."""
        raw = self.query_one("#c_file", Input).value.strip().strip("'\"")
        if not raw:
            return None, "Укажите CSV-файл с ордерами"
        path = Path(os.path.expandvars(raw)).expanduser()
        if not path.is_file():
            return None, f"Файл не найден: {path}"
        if not CANCEL_FILE_RE.match(path.name):
            return None, (
                "Имя файла должно быть SYMBOL_{Short|Long}_{Limit|Market}_дата-время.csv "
                "(символ и тип сетки берутся из имени, не переименовывайте файл)"
            )
        exchange = str(self.query_one("#c_exchange", Select).value or "binance")
        mode = str(self.query_one("#c_mode", Select).value or "demo")
        info = folder_info(path.resolve().parent.name)
        if info and info != (exchange, mode):
            return None, (
                f"Файл лежит в папке биржи {info[0]}, режима {info[1]}, а выбрано {exchange} / {mode}. "
                f"Выберите {info[0]} / {info[1]}"
            )
        return path, ""

    def update_cancel_file_status(self) -> None:
        if self.query_one("#c_source", Select).value != "file":
            self._set_status("#c_file_status", "Не используется: источник — символ + side", "gray")
            return
        raw = self.query_one("#c_file", Input).value.strip().strip("'\"")
        path, err = self._cancel_file_path()
        if err:
            self._set_status("#c_file_status", err, "yellow" if not raw else "red")
            return
        m = CANCEL_FILE_RE.match(path.name)
        self._set_status(
            "#c_file_status",
            f"OK: {m['symbol'].upper()} {m['side'].title()} {m['kind'].title()}",
            "green",
        )

    # ----------------------------------------------------------------------------------
    # CSV preview (placement)
    # ----------------------------------------------------------------------------------
    def load_preview(self) -> None:
        table = self.query_one("#preview", DataTable)
        table.clear(columns=True)
        table.add_columns("#", "CSV строка", "Level", "Size", "Статус")
        status = self.query_one("#csv_status", Label)
        file_raw = self.query_one("#file", Input).value.strip().strip("'\"")
        if not file_raw:
            status.update(f"CSV не выбран. Файл по умолчанию не найден: {DEFAULT_CSV}")
            status.styles.color = "yellow"
            return
        path = Path(os.path.expandvars(file_raw)).expanduser()
        if not path.is_file():
            status.update(f"Файл не найден: {path}")
            status.styles.color = "red"
            return
        preview = parse_csv_preview(path)
        if preview.fatal:
            status.update(preview.fatal)
            status.styles.color = "red"
            return

        trim_raw = self.query_one("#trim", Input).value.strip()
        selected = preview.rows
        trim_error = ""
        levels = len(preview.rows)
        if trim_raw:
            try:
                levels = int(trim_raw)
                if levels <= 0:
                    trim_error = "Число уровней должно быть больше 0"
                else:
                    selected = preview.rows[:levels]
            except ValueError:
                trim_error = "Обрезать: укажите целое число уровней или оставьте пустым"
        if trim_error:
            selected = preview.rows

        for i, row in enumerate(selected, 1):
            table.add_row(
                str(i),
                str(row.line),
                row.level or "-",
                row.size or "-",
                row.error or "OK",
            )
        err_count = sum(1 for r in selected if r.error)
        if trim_error:
            msg = trim_error
            color = "red"
        elif preview.fatal:
            msg = preview.fatal
            color = "red"
        elif err_count:
            msg = f"CSV содержит ошибки в показанных строках: {err_count}. Всего уровней: {len(preview.rows)}"
            color = "red"
        elif len(selected) > MAX_GRID_ORDERS:
            msg = f"Слишком много уровней: {len(selected)}, максимум {MAX_GRID_ORDERS}"
            color = "red"
        else:
            msg = f"Разделитель={preview.delimiter!r} | всего уровней={len(preview.rows)} | показано={len(selected)}"
            if trim_raw and levels < len(preview.rows):
                msg += f" | будет обрезано до {len(selected)}"
            elif trim_raw:
                msg += " | обрезка не требуется (уровней меньше или равно)"
            color = "green"
            if (
                self.query_one("#kind", Select).value == "market"
                and self.query_one("#exchange", Select).value == "bybit"
                and len(selected) > BYBIT_MAX_CONDITIONAL
            ):
                msg += f" | ВНИМАНИЕ: Bybit допускает не более {BYBIT_MAX_CONDITIONAL} активных условных ордеров на символ"
                color = "yellow"
        status.update(msg)
        status.styles.color = color

    def prepare_csv_for_run(self) -> tuple[Optional[Path], str]:
        """Validate CSV and create a trimmed temporary file if trimming is requested."""
        file_raw = self.query_one("#file", Input).value.strip().strip("'\"")
        if not file_raw:
            return None, "Укажите путь к CSV-файлу"
        path = Path(os.path.expandvars(file_raw)).expanduser()
        if not path.is_file():
            return None, f"CSV-файл не найден: {path}"
        preview = parse_csv_preview(path)
        if preview.fatal:
            return None, preview.fatal
        if not preview.rows:
            return None, "В CSV нет строк данных"

        trim_raw = self.query_one("#trim", Input).value.strip()
        if trim_raw:
            try:
                levels = int(trim_raw)
            except ValueError:
                return None, "Обрезать: укажите целое число уровней или оставьте поле пустым"
            if levels <= 0:
                return None, "Обрезать: число уровней должно быть больше 0"
            selected = preview.rows[:levels]
            if not selected:
                return None, "Обрезать: в файле нет нужного количества уровней"
        else:
            levels = len(preview.rows)
            selected = preview.rows

        if len(selected) > MAX_GRID_ORDERS:
            return None, f"Слишком много уровней: {len(selected)}, максимум {MAX_GRID_ORDERS}"
        for idx, row in enumerate(selected, 1):
            if row.error:
                return None, f"CSV ошибка в уровне #{idx} (строка {row.line}): {row.error}"

        # If trimming is actually needed, create a separate trimmed file.
        # The original CSV file is not modified.
        if trim_raw and levels < len(preview.rows):
            out_path = path.with_name(path.stem + ".trimmed" + path.suffix)
            lines = [preview.header] + [r.raw for r in selected]
            try:
                out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            except OSError as e:
                return None, f"Не удалось записать обрезанный файл: {e}"
            self.append_output(f"Создан обрезанный CSV: {out_path} (уровней: {len(selected)})")
            return out_path, ""
        return path, ""

    # ----------------------------------------------------------------------------------
    # Run: placement
    # ----------------------------------------------------------------------------------
    def start_run(self) -> None:
        kind = str(self.query_one("#kind", Select).value or "limit")
        script = MARKET_SCRIPT_PATH if kind == "market" else SCRIPT_PATH
        if not script.is_file():
            self.show_error(f"Не найден скрипт: {script}")
            return
        symbol_input = self.query_one("#symbol", Input)
        symbol, msg = validate_symbol(symbol_input.value)
        symbol_input.value = symbol
        if msg:
            self.set_symbol_status(msg, False)
            self.show_error(msg)
            return
        self.set_symbol_status("OK", True)

        file_path, msg = self.prepare_csv_for_run()
        if file_path is None:
            self.show_error(msg)
            return

        side = self.query_one("#side", Select).value
        if side not in ("long", "short"):
            self.show_error("Выберите Side: long или short")
            return
        exchange = str(self.query_one("#exchange", Select).value or "binance")
        mode = str(self.query_one("#mode", Select).value or "demo")
        if mode not in EXCHANGE_MODES.get(exchange, ()):
            self.show_error(f"Режим {mode} не поддерживается для {exchange}")
            return

        cmd: list[str] = [sys.executable, str(script), symbol, str(side), str(file_path), exchange, mode]
        if bool(self.query_one("#dry_run", Switch).value):
            cmd.append("--dry-run")
        if kind == "limit":
            if bool(self.query_one("#allow_marketable", Switch).value):
                cmd.append("--allow-marketable")
            if bool(self.query_one("#foc", Switch).value):
                cmd.append("--first-order-control")
        else:
            if bool(self.query_one("#profit", Switch).value):
                fraction = self.query_one("#profit_value", Input).value.strip().replace(",", ".")
                if fraction:
                    try:
                        value = Decimal(fraction)
                    except InvalidOperation:
                        value = Decimal(-1)
                    if not value.is_finite() or not 0 < value <= 1:
                        self.show_error("-p: доля пути должна быть числом больше 0 и не больше 1 (например 0,7) или пустой")
                        return
                    cmd.append(f"--profit={fraction}")
                else:
                    cmd.append("--profit")
            if bool(self.query_one("#break_even", Switch).value):
                cmd.append("--break-even")

        self.save_state()
        self._launch(cmd)

    # ----------------------------------------------------------------------------------
    # Run: cancellation
    # ----------------------------------------------------------------------------------
    def start_cancel_run(self) -> None:
        if not CANCEL_SCRIPT_PATH.is_file():
            self.show_error(f"Не найден скрипт: {CANCEL_SCRIPT_PATH}")
            return

        source = self.query_one("#c_source", Select).value
        if source == "file":
            path, msg = self._cancel_file_path()
            if path is None:
                self.update_cancel_file_status()
                self.show_error(msg)
                return
            target = [str(path)]
        else:
            symbol_input = self.query_one("#c_symbol", Input)
            symbol, msg = validate_symbol(symbol_input.value)
            symbol_input.value = symbol
            if msg:
                self._set_status("#c_symbol_status", msg, "red")
                self.show_error(msg)
                return
            self._set_status("#c_symbol_status", "OK", "green")
            side = self.query_one("#c_side", Select).value
            if side not in ("long", "short"):
                self.show_error("Выберите Side: long или short")
                return
            target = [symbol, str(side)]

        exchange = str(self.query_one("#c_exchange", Select).value or "binance")
        mode = str(self.query_one("#c_mode", Select).value or "demo")
        if mode not in EXCHANGE_MODES.get(exchange, ()):
            self.show_error(f"Режим {mode} не поддерживается для {exchange}")
            return
        cmd: list[str] = [sys.executable, str(CANCEL_SCRIPT_PATH), *target, exchange, mode]
        if bool(self.query_one("#c_dry_run", Switch).value):
            cmd.append("--dry-run")

        self.save_state()
        self._launch(cmd)

    # ----------------------------------------------------------------------------------
    # Shared launch / process streaming
    # ----------------------------------------------------------------------------------
    def _launch(self, cmd: list[str]) -> None:
        pretty_cmd = " ".join(f'"{part}"' if " " in part else part for part in cmd)
        self.append_output("Команда: " + pretty_cmd)
        self._set_running(True)
        thread = threading.Thread(
            target=self._run_process,
            args=(cmd,),
            daemon=True,
        )
        thread.start()

    def _set_running(self, running: bool) -> None:
        self.query_one("#run", Button).disabled = running
        self.query_one("#c_run", Button).disabled = running

    def _run_process(self, cmd: list[str]) -> None:
        """Run a script in a background thread and stream output to the TUI."""
        try:
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            with subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                encoding="utf-8",
                errors="replace",
            ) as proc:
                if proc.stdout:
                    for line in proc.stdout:
                        self.call_from_thread(self.append_output, line.rstrip())
                proc.wait()
                self.call_from_thread(
                    self.append_output,
                    f"Код завершения: {proc.returncode}",
                )
        except FileNotFoundError:
            self.call_from_thread(
                self.append_output,
                f"Не найден исполняемый файл: {cmd[0]}",
            )
        except Exception as exc:
            self.call_from_thread(
                self.append_output,
                f"Ошибка запуска: {exc}",
            )
        finally:
            try:
                self.call_from_thread(self._set_running, False)
            except Exception:
                pass


if __name__ == "__main__":
    GridTUI().run()