#!/usr/bin/env python3
"""
scan_gotoscreen_only.py

Simple usage:
  python scan_gotoscreen_only.py "<project_root>" --screen DialogChangeBag --xlsx DialogChangeBag.xlsx

What it does:
  - Recursively scans C/C++ files for goToScreen/GotoScreen/GotoScreenImmediate
  - Extracts LAST token of the screen (e.g., DialogChangeBag)
  - Keeps ONLY the requested screen (--screen) and its variations
  - Writes Excel with 2 sheets:
      Raw: CodeLine | Screen | Variation | File | Line
      ByScreen: Screen | Variations | VariationsJSON | NumVariations | Count | FirstFile | FirstLine
"""

import argparse
import os
import re
import json
from typing import Dict, Iterable, List

try:
    import pandas as pd
except Exception:
    pd = None

CALL_RE = re.compile(
    r'''(?ix)
    \bgo?toScreen(?:Immediate)?          # goToScreen/GotoScreen/GotoScreenImmediate
    \s*\(\s*
    ScreenDriver::
    (?P<screen>[A-Za-z_]\w*(?:::[A-Za-z_]\w*)?)  # Dialog::Info | DialogChangeBag | Setup
    \s*,\s*
    (?P<variation>[A-Za-z_]\w*|\d+)      # 0 | 1 | dynamic | CONFIGURATION
    \s*\)
    '''
)

EXTS = {".cpp", ".cc", ".cxx", ".hpp", ".hh", ".h", ".hxx", ".inl"}

def iter_source_files(root: str):
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if os.path.splitext(fn)[1].lower() in EXTS:
                yield os.path.join(dp, fn)

def read_text(path: str) -> str:
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            with open(path, "r", encoding=enc, errors="ignore") as f:
                return f.read()
        except Exception:
            pass
    with open(path, "rb") as f:
        return f.read().decode("utf-8", "ignore")

def last_token(screen: str) -> str:
    parts = screen.split("::")
    return parts[-1] if parts else screen

def find_calls_in_file(path: str) -> List[dict]:
    text = read_text(path)
    lines = text.splitlines()
    out: List[dict] = []
    for m in CALL_RE.finditer(text):
        start = m.start()
        line_no = text.count("\n", 0, start) + 1
        code_line = lines[line_no - 1].strip() if 0 <= line_no - 1 < len(lines) else ""
        out.append({
            "File": os.path.relpath(path),
            "Line": line_no,
            "CodeLine": code_line,
            "ScreenFull": m.group("screen"),
            "Screen": last_token(m.group("screen")),
            "Variation": m.group("variation"),
        })
    return out

def natural_key(v: str):
    try:
        return (0, int(v))
    except Exception:
        return (1, str(v).lower())

def main():
    ap = argparse.ArgumentParser(description="Extract ONE screen and its variations into Excel.")
    ap.add_argument("root", help="Project root folder (where your .cpp files live)")
    ap.add_argument("--screen", required=True, help="Screen name to keep (LAST token), e.g., DialogChangeBag")
    ap.add_argument("--xlsx", default=None, help="Output Excel file name (default: <screen>.xlsx)")
    args = ap.parse_args()

    root = args.root
    if not os.path.isdir(root):
        print(f"ERROR: folder not found: {root}")
        print("Tip: drag the folder into the terminal to paste the correct path, and wrap it in quotes.")
        return

    rows: List[dict] = []
    for path in iter_source_files(root):
        try:
            rows.extend(find_calls_in_file(path))
        except Exception:
            pass

    # Filter to the requested screen (case-insensitive)
    want = args.screen.lower()
    rows = [r for r in rows if r["Screen"].lower() == want]

    if not rows:
        print(f"No matches found for screen: {args.screen}")
        return

    rows.sort(key=lambda r: (natural_key(str(r["Variation"])), r["File"].lower(), r["Line"]))

    # Build summary (single screen)
    variations = sorted({str(r["Variation"]) for r in rows}, key=natural_key)
    first_file = rows[0]["File"]
    first_line = rows[0]["Line"]

    summary = [{
        "Screen": args.screen,
        "Variations": ", ".join(variations),
        "VariationsJSON": json.dumps(variations),
        "NumVariations": len(variations),
        "Count": len(rows),
        "FirstFile": first_file,
        "FirstLine": first_line
    }]

    # Output file
    out_xlsx = args.xlsx or f"{args.screen}.xlsx"

    try:
        import pandas as pd
        import openpyxl  # ensure writer engine present
        df_raw = pd.DataFrame(rows, columns=["CodeLine", "Screen", "Variation", "File", "Line"])
        df_sum = pd.DataFrame(summary, columns=["Screen", "Variations", "VariationsJSON", "NumVariations", "Count", "FirstFile", "FirstLine"])
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
            df_raw.to_excel(xw, index=False, sheet_name="Raw")
            df_sum.to_excel(xw, index=False, sheet_name="ByScreen")
        print(f"Wrote Excel: {out_xlsx}")
    except Exception:
        # CSV fallback
        import csv
        raw_csv = os.path.splitext(out_xlsx)[0] + "_raw.csv"
        sum_csv = os.path.splitext(out_xlsx)[0] + "_byscreen.csv"
        with open(raw_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["CodeLine", "Screen", "Variation", "File", "Line"])
            w.writeheader()
            for r in rows:
                w.writerow({k: r[k] for k in ["CodeLine", "Screen", "Variation", "File", "Line"]})
        with open(sum_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["Screen", "Variations", "VariationsJSON", "NumVariations", "Count", "FirstFile", "FirstLine"])
            w.writeheader()
            w.writerow(summary[0])
        print(f"No pandas/openpyxl: wrote CSVs -> {raw_csv}, {sum_csv}")

if __name__ == "__main__":
    main()
