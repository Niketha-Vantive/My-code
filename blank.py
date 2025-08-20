# detect_blanks.py — document-agnostic blank detector (Azure Document Intelligence)
# Python 3.9+
# pip install azure-ai-formrecognizer>=3.2 azure-core python-dotenv

import os
import re
import json
import argparse
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient

# ---------------------------------------------------------------------
# Regex / helpers
# ---------------------------------------------------------------------
PLACEHOLDER_RE = re.compile(
    r"^\s*(?:_{2,}|\.{2,}|-+|<\s*insert[^>]*>|<[^>]+>|tbd|tba|n/?a|not\s+provided|see\s+(?:stamp|signature))\s*$",
    re.IGNORECASE,
)
LABEL_WORD_RE = re.compile(
    r"(name|date|title|author|email|id|amount|by|rev|revision|document)\b",
    re.IGNORECASE,
)

def norm(s: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def is_placeholder_or_empty(s: Optional[str]) -> bool:
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

def min_gap_threshold(avg_char_w: float, page_w: float) -> float:
    # dynamic tiny-gap guard
    return max(1.5 * max(avg_char_w, 1.0), 0.05 * page_w)

def has_real_words(words, check_rect, margin=0.0) -> bool:
    """True if any non-placeholder word overlaps the rect (with optional padding)."""
    x0, y0, x1, y1 = check_rect
    padded = (x0 - margin, y0 - margin, x1 + margin, y1 + margin)
    for w in words:
        if not getattr(w, "polygon", None):
            continue
        wb = poly_to_bbox(w.polygon)
        if rect_overlap(wb, padded) > 0:
            if not is_placeholder_or_empty(w.content):
                return True
    return False

# ---------------------------------------------------------------------
# Debug visibility
# ---------------------------------------------------------------------
def debug_print_tables(result):
    for idx, t in enumerate(result.tables or []):
        pages = sorted({br.page_number for c in t.cells for br in (c.bounding_regions or [])})
        print(f"[DI] table#{idx+1}: rows≈{t.row_count} cols≈{t.column_count} pages={pages or ['?']} cells={len(t.cells)}")

def trace(msg: str, enabled: bool):
    if enabled:
        print(f"[trace] {msg}")

# ---------------------------------------------------------------------
# Primary detectors (use DI tables/kv/marks/lines)
# ---------------------------------------------------------------------
def detect_table_blanks(result, words_by_page, debug_counts) -> List[Dict[str, Any]]:
    blanks = []
    for t_idx, table in enumerate(result.tables or []):
        # headers either by 'kind' or fallback first non-empty row
        col_headers: Dict[int, str] = {}
        header_rows = set()
        for cell in table.cells:
            if getattr(cell, "kind", None) == "columnHeader":
                col_headers[cell.column_index] = norm(cell.content)
                header_rows.add(cell.row_index)
        if not col_headers:
            rows_text = {}
            for cell in table.cells:
                rows_text.setdefault(cell.row_index, 0)
                if norm(cell.content):
                    rows_text[cell.row_index] += 1
            fallback_row = min([r for r, cnt in rows_text.items() if cnt >= 2], default=None)
            if fallback_row is not None:
                for cell in table.cells:
                    if cell.row_index == fallback_row:
                        col_headers[cell.column_index] = norm(cell.content)
                header_rows.add(fallback_row)

        for cell in table.cells:
            if cell.row_index in header_rows:
                continue
            if not cell.bounding_regions:
                continue  # no geometry → let synthetic handle

            reg = cell.bounding_regions[0]
            page_no = reg.page_number
            cell_bbox = poly_to_bbox(reg.polygon)

            # word presence guard
            if has_real_words(words_by_page.get(page_no, []), cell_bbox, margin=0.0):
                debug_counts["tiny_gap_noise"] += 1
                continue

            txt = norm(cell.content)
            if is_placeholder_or_empty(txt):
                col_lbl = col_headers.get(cell.column_index, f"Col {cell.column_index+1}")
                row_lbl = None
                for rc in table.cells:
                    if rc.row_index == cell.row_index and getattr(rc, "kind", None) == "rowHeader":
                        row_lbl = norm(rc.content)
                        break
                label = " | ".join([s for s in [row_lbl, col_lbl] if s]) or f"Table {t_idx+1} R{cell.row_index+1}C{cell.column_index+1}"
                blanks.append({
                    "page": page_no,
                    "bbox": bbox_to_dict(cell_bbox),
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

        key_reg = kv.key.bounding_regions[0] if (kv.key and kv.key.bounding_regions) else region
        kx0, ky0, kx1, ky1 = poly_to_bbox(key_reg.polygon)
        line_h = max(8.0, (ky1 - ky0))
        search_rect = (kx1, ky0, min(kx1 + 0.25 * page_w, kx1 + 4 * (kx1 - kx0)), ky0 + line_h)

        if is_placeholder_or_empty(val_txt) or val_conf < 0.25:
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

            gap_start = lx0 + 0.65 * line_w
            gap_end   = lx1
            gap_rect  = (gap_start, ly0, gap_end, ly1)
            gap_w     = max(0.0, gap_end - gap_start)

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

# ---------------------------------------------------------------------
# Generic synthetic table fallback (works when DI emits no cells)
# ---------------------------------------------------------------------
def _cluster_rows_by_y(words, header_y, page_h, trace_on=False):
    row_lines = []
    for w in words:
        if not getattr(w, "polygon", None):
            continue
        wx0, wy0, wx1, wy1 = poly_to_bbox(w.polygon)
        if wy0 > header_y:  # below header
            row_lines.append((wy0, wy1))
    if not row_lines:
        return []

    row_lines.sort()
    row_bands = []
    cur = list(row_lines[0])
    for (y0, y1) in row_lines[1:]:
        if y0 - cur[1] < 0.02 * page_h:  # merge close lines
            cur[1] = max(cur[1], y1)
        else:
            row_bands.append(tuple(cur)); cur = [y0, y1]
    row_bands.append(tuple(cur))
    return row_bands

def detect_synthetic_tables_generic(result, words_by_page, debug_counts, trace_on=False) -> List[Dict[str, Any]]:
    """
    Generic synthetic table detector:
    - choose a header row (the row with the widest range of x)
    - build column bands from header word boxes
    - build row bands from words below header
    - flag empty intersections (no real words)
    """
    blanks = []
    for page in (result.pages or []):
        page_no = page.page_number
        page_w  = page.width or 1000.0
        page_h  = page.height or 1000.0
        words   = words_by_page.get(page_no, [])

        if not words:
            trace("no words on page", trace_on); 
            continue

        # 1) group words into rows by y-overlap
        #    (simple heuristic: bucket words whose centers fall into same y band)
        rows: Dict[int, List[Any]] = {}
        centers = []
        for w in words:
            if not getattr(w, "polygon", None):
                continue
            x0,y0,x1,y1 = poly_to_bbox(w.polygon)
            cy = (y0+y1)/2
            centers.append(cy)
        if not centers:
            continue
        centers.sort()
        bands = [centers[0]]
        for c in centers[1:]:
            if abs(c - bands[-1]) > 0.015 * page_h:
                bands.append(c)
        # map words to nearest band index
        for w in words:
            if not getattr(w, "polygon", None):
                continue
            x0,y0,x1,y1 = poly_to_bbox(w.polygon)
            cy = (y0+y1)/2
            # nearest index
            idx = min(range(len(bands)), key=lambda i: abs(bands[i]-cy))
            rows.setdefault(idx, []).append(w)

        # 2) pick header row = row with largest x-span and at least 3 tokens
        header_idx, header_span = None, 0.0
        for idx, ws in rows.items():
            xs = []
            for w in ws:
                bx = poly_to_bbox(w.polygon)
                xs += [bx[0], bx[2]]
            if not xs:
                continue
            span = max(xs) - min(xs)
            if span > header_span and len(ws) >= 3:
                header_idx, header_span = idx, span

        if header_idx is None:
            trace(f"page {page_no}: no header row candidate", trace_on)
            continue

        header_words = sorted(rows[header_idx], key=lambda w: poly_to_bbox(w.polygon)[0])
        header_boxes = [poly_to_bbox(w.polygon) for w in header_words]
        header_texts = [norm(w.content) for w in header_words]
        header_y = min(b[1] for b in header_boxes)
        trace(f"page {page_no}: header row with {len(header_boxes)} tokens: {header_texts}", trace_on)

        # 3) build column bands from header boxes
        #    columns = from each header left edge to next header left edge
        xbands = []
        for i, box in enumerate(header_boxes):
            name = header_texts[i]
            xstart = box[0]
            xend = header_boxes[i+1][0] if i+1 < len(header_boxes) else min(page_w, max(b[2] for b in header_boxes) + 0.25 * page_w)
            xbands.append((name, xstart, xend))

        if len(xbands) < 2:
            trace(f"page {page_no}: header had <2 columns", trace_on)
            continue

        # 4) build row bands from words below header
        row_bands = _cluster_rows_by_y(words, header_y, page_h, trace_on)
        if not row_bands:
            trace(f"page {page_no}: no row bands below header", trace_on)
            continue

        # 5) flag empty intersections (skip the first column only if it looks like a label column)
        first_col_labelish = True  # heuristic—adjustable
        for (ry0, ry1) in row_bands:
            for ci, (name, x0, x1) in enumerate(xbands):
                if first_col_labelish and ci == 0:
                    continue
                cell_rect = (x0, ry0, x1, ry1)
                if has_real_words(words, cell_rect, margin=0.0):
                    debug_counts["tiny_gap_noise"] += 1
                    continue
                blanks.append({
                    "page": page_no,
                    "bbox": bbox_to_dict(cell_rect),
                    "label": name,
                    "source": "synthetic_table",
                    "reason": "no_table_cells_from_DI",
                    "confidence": 0.72
                })

    return blanks

# ---------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------
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
                        help="Also flag label→gap blanks (heuristic).")
    parser.add_argument("--debug", action="store_true", help="Print DI tables and per-page summary.")
    parser.add_argument("--trace", action="store_true", help="Verbose reasons for synthetic fallback decisions.")
    args = parser.parse_args()

    result = analyze_document(endpoint, key, file_path=args.file, url=args.url)

    if args.debug:
        debug_print_tables(result)

    pages_by_no, words_by_page = build_page_indexes(result)
    debug_counts = {"tiny_gap_noise": 0}

    blanks: List[Dict[str, Any]] = []
    # primary detectors
    blanks += detect_table_blanks(result, words_by_page, debug_counts)
    blanks += detect_kv_blanks(result, pages_by_no, words_by_page, debug_counts)
    blanks += detect_selection_mark_blanks(result)
    blanks += detect_label_gap_blanks(result, pages_by_no, words_by_page, debug_counts, enable=args.include_label_gaps)
    # synthetic fallbacks (generic)
    blanks += detect_synthetic_tables_generic(result, words_by_page, debug_counts, trace_on=args.trace)

    # order by page, then by source type
    priority = {"table": 0, "selection_mark": 1, "key_value": 2, "synthetic_table": 3, "label_gap": 4}
    blanks.sort(key=lambda b: (b["page"], priority.get(b["source"], 99), -b.get("confidence", 0.0)))

    print(json.dumps({"blanks": blanks}, indent=2))

    if args.debug:
        per_page = {}
        for b in blanks:
            per_page.setdefault(b["page"], 0)
            per_page[b["page"]] += 1
        print("\n--- Summary ---")
        if per_page:
            for p in sorted(per_page.keys()):
                print(f"Page {p}: {per_page[p]} blanks")
        else:
            print("No blanks detected.")
        print(f"Skipped due to tiny_gap_noise (words present where blank was suspected): {debug_counts['tiny_gap_noise']}")

if __name__ == "__main__":
    main()
