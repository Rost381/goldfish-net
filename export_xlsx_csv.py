#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Универсальный экспорт листов книги Bot_levels.xlsx в CSV.

Что делает:
  * открывает книгу (по умолчанию — input/Bot_levels.xlsx рядом со скриптом);
  * пропускает служебные листы: имя начинается с «main» в любом регистре
    (Main, main, MAIN, Main2, Main_old — не важно);
  * каждый оставшийся лист (Level/Size или любая другая структура) сохраняет
    в CSV с тем же именем в ту же папку, где лежит книга — например:
    Long8.csv, Long26.csv, Short8.csv, Short26.csv, Long-Squeeze.csv.
  * разделитель колонок — «;» (открывается в Excel русской локали без настройки).

Числа записываются так, как они отображаются в Excel: учитывается числовой
формат ячейки (0.00000000 -> 8 знаков, 0.00 -> 2 знака), а «мусор» вычислений
(например 15.999999999999998) превращается в чистое значение (16).

Использование:
  python export_bot_levels_csv.py                  # книга input/Bot_levels.xlsx
  python export_bot_levels_csv.py путь/к/книге.xlsx

Зависимости: openpyxl (pip install openpyxl)
"""

import csv
import re
import sys
from pathlib import Path

try:
    from openpyxl import load_workbook
except ImportError:
    sys.exit("Не найден модуль openpyxl. Установите его командой: pip install openpyxl")

CSV_DELIMITER = ";"             # разделитель колонок в CSV
SKIP_PREFIX = "main"            # листы с именем на «main» (в любом регистре) не экспортируются
XLSX_NAME = "Bot_levels.xlsx"   # имя книги по умолчанию


def fmt_number(value: float, number_format: str) -> str:
    """Число -> строка «как в Excel», без хвостов плавающей точки."""
    nf = (number_format or "General").split(";")[0]   # первый раздел формата
    m = re.search(r"\.(0+)", nf)
    if m:  # формат задаёт число знаков после запятой: 0.00 -> 2, 0.00000000 -> 8
        s = f"{value:.{len(m.group(1))}f}"
    else:  # General: гасим шум вычислений 10 знаками после запятой
        if value != 0 and abs(value) < 1e-10:          # слишком малое число не теряем
            s = f"{value:.12g}"
        else:
            s = f"{value:.10f}"
    s = s.rstrip("0").rstrip(".")                      # 16.00 -> 16, 0.21710000 -> 0.2171
    return s if s not in ("", "-") else "0"


def fmt_cell(value, number_format: str) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return fmt_number(value, number_format)
    return str(value).strip()


def safe_name(name: str) -> str:
    """Имя листа -> безопасное имя файла (на случай служебных символов)."""
    return re.sub(r'[<>:"/\\|?*]', "_", name).strip() or "sheet"


def main() -> None:
    # Путь к книге: аргумент командной строки либо input/Bot_levels.xlsx рядом со скриптом
    base = Path(__file__).resolve().parent
    if len(sys.argv) > 1:
        xlsx_path = Path(sys.argv[1])
    else:
        candidates = [base / "input" / XLSX_NAME, base / XLSX_NAME]
        xlsx_path = next((p for p in candidates if p.is_file()), candidates[0])

    if not xlsx_path.is_file():
        sys.exit(f"Файл не найден: {xlsx_path}")

    out_dir = xlsx_path.parent
    wb = load_workbook(xlsx_path, data_only=True)  # значения формул, а не сами формулы

    print(f"Книга: {xlsx_path}")
    exported = 0
    for ws in wb.worksheets:
        if ws.title.strip().lower().startswith(SKIP_PREFIX):
            print(f"  лист '{ws.title}' — пропущен (служебный: main*)")
            continue

        rows_out = []
        for row in ws.iter_rows():
            cells = [fmt_cell(c.value, c.number_format) for c in row]
            while cells and cells[-1] == "":   # пустые ячейки в конце строки не пишем
                cells.pop()
            if cells:                          # полностью пустые строки пропускаем
                rows_out.append(cells)

        out_path = out_dir / (safe_name(ws.title) + ".csv")
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f, delimiter=CSV_DELIMITER).writerows(rows_out)

        data_rows = max(len(rows_out) - 1, 0)  # без строки заголовка
        tail = ""
        if not rows_out:
            tail = " (лист пустой)"
        elif data_rows == 0:
            tail = " (только заголовок)"
        print(f"  {out_path.name}: строк данных {data_rows}{tail}")
        exported += 1

    print(f"Готово: {exported} CSV в {out_dir}")


if __name__ == "__main__":
    main()
