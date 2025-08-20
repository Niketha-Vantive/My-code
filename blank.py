# detect_blanks.py  (updated)
import os
import re
import json
import argparse
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient

# -----------------------
# Regex / helpers
# -----------------------
PLACEHOLDER_RE = re.compile(
    r"^\s*(?:_{2,}|\.{2,}|-+|<\s*insert[^>]*>|<[^>]+>|tbd|tba|n/?a|not\s+provided|see\s+(?:stamp|signature))\s*$",
    re.IGNORECASE,
)
LABEL_WORD_RE = re.compile(r"(name|date|title|author|email|id|amount|by|rev|revision|document)\b", re.IGNORECASE)

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def is_placeholder_or_empty(s: str) -> bool:
    s = (s or "").strip()
    return (not s) or bool(PLACEHOLDER_RE.match(s))

def poly_to_bbox(polygon) -> Tuple[float, float, float, float]:
    xs = [p.x for p in polygon]
    ys = [p.y for p in polygon]
    return (min(xs), min(ys), max(xs), max(ys))

def bbox_to_dict(b):
    x0, y0, x1, y1 = b
    return {"x0": x0, "y0": y0, "x1": x1, "y1": y1}

def rect_overlap(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    return iw * ih

def rect_area(b) -> float:
    x0, y0, x1, y1 = b
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)

def min_gap_threshold(avg_char_w: float, page_w: float) -> float:
    return max(1.5 * max(avg_char_w, 1.0), 0.05 * page_w)

def has_real_words(words, check_rect, margin=0.0) -> bool:
    """Return True if any non-placeholder word overlaps the rect (with optional padding)."""
    x0, y0, x1, y1 = check_rect
    padded = (x0 - margin, y0 - margin, x1 + margin, y1 + margin)
    for w in words:
        wb = poly_to_bbox(w.polygon)
        if rect_overlap(wb, padded) > 0:
            if not is_placeholder_or_empty(w.content):
                return True
    return False

# -----------------------
# Detection
# -----------------------
def detect_table_blanks(result, words_by_page, debug_counts) -> List[Dict[str, Any]]:
    blanks = []
    for t_idx, table in enumerate(result.tables or []):
        # collect column headers
        col_headers = {}
        for cell in table.cells:
            if getattr(cell, "kind", None) == "columnHeader" or cell.row_index == 0:
                col_headers[cell.column_index] = norm(cell.content)

        for cell in table.cells:
            # skip header cells
            if getattr(cell, "kind", None) in {"columnHeader", "rowHeader"} or cell.row_index == 0:
                continue

            txt = norm(cell.content)
            # Use cell bbox to double-check real text isn't there (OCR sometimes doesn't pipe content up)
            if cell.bounding_regions:
                reg = cell.bounding_regions[0]
                page_no = reg.page_number
                cell_bbox = poly_to_bbox(reg.polygon)
                # If any real words are inside, it's NOT blank → skip
                if has_real_words(words_by_page.get(page_no, []), cell_bbox, margin=0.0):
                    debug_counts["tiny_gap_noise"] += 1
                    continue

            if is_placeholder_or_empty(txt):
                col_lbl = col_headers.get(cell.column_index, f"Col {cell.column_index+1}")
                row_lbl = None
                # attempt a row header in same row
                for rc in table.cells:
                    if rc.row_index == cell.row_index and getattr(rc, "kind", None) == "rowHeader":
                        row_lbl = norm(rc.content)
                        break
                label = " | ".join([s for s in [row_lbl, col_lbl] if s]) or f"Table {t_idx+1} R{cell.row_index+1}C{cell.column_index+1}"

                if cell.bounding_regions:
                    reg = cell.bounding_regions[0]
                    blanks.append({
                        "page": reg.page_number,
                        "bbox": bbox_to_dict(poly_to_bbox(reg.polygon)),
                        "label": label,
                        "source": "table",
                        "reason": "table_cell_empty",
                        "confidence": 0.90
                    })
    return blanks

def detect_kv_blanks(result, pages_by_no, words_by_page, debug_counts) -> List[Dict[str, Any]]:
    blanks = []
    for kv in (result.key_value_pairs or []):
        key_txt = norm(getattr(kv.key, "content", "") if kv.key else "")
        val_obj  = getattr(kv, "value", None)
        val_txt  = norm(getattr(val_obj, "content", "") if val_obj else "")
        val_conf = getattr(val_obj, "confidence", 0.0) if val_obj else 0.0

        # choose a primary region to inspect
        region = None
        if val_obj and getattr(val_obj, "bounding_regions", None):
            region = val_obj.bounding_regions[0]
        elif kv.key and getattr(kv.key, "bounding_regions", None):
            region = kv.key.bounding_regions[0]
        if not region:
            continue

        page_no = region.page_number
        page = pages_by_no.get(page_no)
        page_w = page.width if page else 1000.0

        # If value is missing/low confidence → before flagging, check if words exist immediately to the right of the key
        # Build a small "value search" rect to the right of key region to catch cases like "Document #: F2000"
        key_reg = kv.key.bounding_regions[0] if (kv.key and kv.key.bounding_regions) else region
        kx0, ky0, kx1, ky1 = poly_to_bbox(key_reg.polygon)
        line_height = max(8.0, (ky1 - ky0))
        search_rect = (kx1, ky0, min(kx1 + 0.25 * page_w, kx1 + 4 * (kx1 - kx0)), ky0 + line_height)

        if (is_placeholder_or_empty(val_txt) or val_conf < 0.25):
            # If there are real words in the value search region → it's filled; skip
            if has_real_words(words_by_page.get(page_no, []), search_rect, margin=0.0):
                debug_counts["tiny_gap_noise"] += 1
                continue

            blanks.append({
                "page": page_no,
                "bbox": bbox_to_dict(poly_to_bbox(region.polygon)),
                "label": key_txt or "KeyValue",
                "source": "key_value",
                "reason": "kv_missing_or_low_conf",
                "confidence": 0.80
            })
    return blanks

def detect_selection_mark_blanks(result) -> List[Dict[str, Any]]:
    blanks = []
    for page in (result.pages or []):
        for mark in (page.selection_marks or []):
            if mark.state and str(mark.state).lower() == "unselected":
                if mark.polygon:
                    blanks.append({
                        "page": page.page_number,
                        "bbox": bbox_to_dict(poly_to_bbox(mark.polygon)),
                        "label": "Checkbox/Radio",
                        "source": "selection_mark",
                        "reason": "unselected_selection_mark",
                        "confidence": 0.85
                    })
    return blanks

def detect_label_gap_blanks(result, pages_by_no, words_by_page, debug_counts, enable=True) -> List[Dict[str, Any]]:
    if not enable:
        return []
    blanks = []
    for page in (result.pages or []):
        page_no = page.page_number
        page_w = page.width or 1000.0
        words = words_by_page.get(page_no, [])
        for line in (page.lines or []):
            text = norm(line.content)
            if not text:
                continue
            looks_label = (text.endswith(":") or LABEL_WORD_RE.search(text) is not None)
            if not looks_label:
                continue

            lx0, ly0, lx1, ly1 = poly_to_bbox(line.polygon)
            line_w = max(lx1 - lx0, 1.0)
            avg_char_w = line_w / max(len(text), 10)
            min_gap = min_gap_threshold(avg_char_w, page_w)

            # Use the right 35% of the line as the candidate gap region
            gap_start = lx0 + 0.65 * line_w
            gap_end   = lx1
            gap_rect  = (gap_start, ly0, gap_end, ly1)
            gap_w     = max(0.0, gap_end - gap_start)

            # If any real word sits in the gap → not a blank
            if has_real_words(words, gap_rect, margin=0.0):
                debug_counts["tiny_gap_noise"] += 1
                continue

            if gap_w >= min_gap:
                blanks.append({
                    "page": page_no,
                    "bbox": bbox_to_dict(gap_rect),
                    "label": text.rstrip(":"),
                    "source": "label_gap",
                    "reason": "label_right_gap",
                    "confidence": 0.65
                })
    return blanks

# -----------------------
# Runner
# -----------------------
def analyze_document(endpoint: str, key: str, file_path: str = None, url: str = None):
    client = DocumentAnalysisClient(endpoint, AzureKeyCredential(key))
    if url:
        poller = client.begin_analyze_document_from_url("prebuilt-document", url)
    else:
        with open(file_path, "rb") as f:
            poller = client.begin_analyze_document("prebuilt-document", document=f)
    return poller.result()

def build_page_indexes(result):
    pages_by_no = {p.page_number: p for p in (result.pages or [])}
    words_by_page = {p.page_number: (p.words or []) for p in (result.pages or [])}
    return pages_by_no, words_by_page

def main():
    load_dotenv()
    endpoint = os.getenv("AZURE_DI_ENDPOINT")
    key = os.getenv("AZURE_DI_KEY")
    assert endpoint and key, "Set AZURE_DI_ENDPOINT and AZURE_DI_KEY in your environment or .env"

    parser = argparse.ArgumentParser(description="Detect generic blanks with Azure Document Intelligence")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", type=str, help="Local file path (PDF, TIFF, JPG, PNG, etc.)")
    src.add_argument("--url", type=str, help="Public/SAS URL to the document")
    parser.add_argument("--include-label-gaps", action="store_true",
                        help="Also flag label→gap blanks (heuristic). Defaults to False.")
    parser.add_argument("--debug", action="store_true", help="Print skip counts and per-page summary.")
    args = parser.parse_args()

    result = analyze_document(endpoint, key, file_path=args.file, url=args.url)
    pages_by_no, words_by_page = build_page_indexes(result)

    debug_counts = {"tiny_gap_noise": 0}

    blanks: List[Dict[str, Any]] = []
    blanks += detect_table_blanks(result, words_by_page, debug_counts)
    blanks += detect_kv_blanks(result, pages_by_no, words_by_page, debug_counts)
    blanks += detect_selection_mark_blanks(result)
    blanks += detect_label_gap_blanks(result, pages_by_no, words_by_page, debug_counts, enable=args.include_label_gaps)

    # Sort by page, then source priority
    priority = {"table": 0, "selection_mark": 1, "key_value": 2, "label_gap": 3}
    blanks.sort(key=lambda b: (b["page"], priority.get(b["source"], 99), -b.get("confidence", 0.0)))

    print(json.dumps({"blanks": blanks}, indent=2))

    if args.debug:
        # per-page summary + tiny-gap noise avoided
        per_page = {}
        for b in blanks:
            per_page.setdefault(b["page"], 0)
            per_page[b["page"]] += 1
        print("\n--- Summary ---")
        for p in sorted(per_page.keys()):
            print(f"Page {p}: {per_page[p]} blanks")
        if not per_page:
            print("No blanks detected.")
        print(f"Skipped due to tiny_gap_noise (value tokens found in/near bbox): {debug_counts['tiny_gap_noise']}")

if __name__ == "__main__":
    main()
