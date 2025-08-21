# detect_blanks.py — strict blank detector + red-circle overlay
# Python 3.9+
# pip install azure-ai-formrecognizer>=3.2 azure-core python-dotenv pymupdf

import os
import re
import json
import argparse
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient

import fitz  # PyMuPDF

# -----------------------
# Regex / helpers
# -----------------------
PLACEHOLDER_RE = re.compile(
    r"^\s*(?:_{2,}|\.{2,}|-+|<\s*insert[^>]*>|<[^>]+>|tbd|tba|n/?a|not\s+provided|see\s+(?:stamp|signature))\s*$",
    re.IGNORECASE,
)
LABEL_WORD_RE = re.compile(r"(name|date|title|author|email|id|amount|by|rev|revision|document)\b", re.IGNORECASE)

def norm(s: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def is_placeholder_or_empty(s: Optional[str]) -> bool:
    s = (s or "").strip()
    return (not s) or bool(PLACEHOLDER_RE.match(s))

def poly_to_bbox(poly) -> Tuple[float, float, float, float]:
    xs = [p.x for p in poly] if poly else []
    ys = [p.y for p in poly] if poly else []
    if not xs or not ys:
        return (0, 0, 0, 0)
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

def has_real_words(words, rect, margin=0.0) -> bool:
    x0, y0, x1, y1 = rect
    padded = (x0 - margin, y0 - margin, x1 + margin, y1 + margin)
    for w in words or []:
        if not getattr(w, "polygon", None):
            continue
        wb = poly_to_bbox(w.polygon)
        if rect_overlap(wb, padded) > 0:
            if not is_placeholder_or_empty(w.content):
                return True
    return False

def grow(rect, left=0, up=0, right=0, down=0):
    x0, y0, x1, y1 = rect
    return (x0 - left, y0 - up, x1 + right, y1 + down)

def min_gap_threshold(avg_char_w: float, page_w: float) -> float:
    return max(1.5 * max(avg_char_w, 1.0), 0.05 * page_w)

# -----------------------
# Debug (optional)
# -----------------------
def debug_overview(result):
    print("\n=== DI Overview ===")
    print(f"pages={len(result.pages or [])} tables={len(result.tables or [])} kv_pairs={len(result.key_value_pairs or [])}")
    for p in (result.pages or []):
        print(f"  Page {p.page_number}: words={len(p.words or [])}, lines={len(p.lines or [])}, marks={len(p.selection_marks or [])}, size=({p.width}x{p.height})")

# -----------------------
# Detectors (strict)
# -----------------------
def detect_table_blanks(result, words_by_page, skipped) -> List[Dict[str, Any]]:
    out = []
    for t_idx, t in enumerate(result.tables or []):
        col_headers: Dict[int, str] = {}
        header_rows = set()
        for c in t.cells:
            if getattr(c, "kind", None) == "columnHeader":
                col_headers[c.column_index] = norm(c.content)
                header_rows.add(c.row_index)
        if not col_headers:
            row_counts: Dict[int, int] = {}
            for c in t.cells:
                if norm(c.content):
                    row_counts[c.row_index] = row_counts.get(c.row_index, 0) + 1
            if row_counts:
                first_row = min((r for r, n in row_counts.items() if n >= 2), default=None)
                if first_row is not None:
                    for c in t.cells:
                        if c.row_index == first_row:
                            col_headers[c.column_index] = norm(c.content)
                    header_rows.add(first_row)

        for c in t.cells:
            if c.row_index in header_rows:
                continue
            if not c.bounding_regions:
                continue
            reg = c.bounding_regions[0]
            page_no = reg.page_number
            cell_rect = poly_to_bbox(reg.polygon)
            if has_real_words(words_by_page.get(page_no), cell_rect):
                skipped["table_words_present"] += 1
                continue
            if is_placeholder_or_empty(norm(c.content)):
                col_lbl = col_headers.get(c.column_index, f"Col {c.column_index+1}")
                row_lbl = None
                for rc in t.cells:
                    if rc.row_index == c.row_index and getattr(rc, "kind", None) == "rowHeader":
                        row_lbl = norm(rc.content); break
                label = " | ".join([x for x in [row_lbl, col_lbl] if x]) or f"Table {t_idx+1} R{c.row_index+1}C{c.column_index+1}"
                out.append({
                    "page": page_no,
                    "bbox": bbox_to_dict(cell_rect),
                    "label": label,
                    "source": "table",
                    "reason": "table_cell_empty",
                    "confidence": 0.90
                })
    return out

def detect_kv_blanks(result, pages_by_no, words_by_page, skipped) -> List[Dict[str, Any]]:
    out = []
    for kv in (result.key_value_pairs or []):
        key_txt = norm(getattr(kv.key, "content", "") if kv.key else "")
        val = getattr(kv, "value", None)
        val_txt  = norm(getattr(val, "content", "") if val else "")
        val_conf = getattr(val, "confidence", 0.0) if val else 0.0

        region = None
        if val and getattr(val, "bounding_regions", None):
            region = val.bounding_regions[0]
        elif kv.key and getattr(kv.key, "bounding_regions", None):
            region = kv.key.bounding_regions[0]
        if not region:
            continue

        page_no = region.page_number
        page = pages_by_no.get(page_no)
        page_w = page.width if page else 1000.0

        value_has_words = False
        if val and getattr(val, "bounding_regions", None):
            vrect = poly_to_bbox(val.bounding_regions[0].polygon)
            value_has_words = has_real_words(words_by_page.get(page_no), vrect)

        if kv.key and getattr(kv.key, "bounding_regions", None):
            kx0, ky0, kx1, ky1 = poly_to_bbox(kv.key.bounding_regions[0].polygon)
            line_h = max(8.0, (ky1 - ky0))
            search_rect = (kx1, ky0, min(kx1 + 0.25 * page_w, kx1 + 4 * (kx1 - kx0)), ky0 + line_h)
            right_has_words = has_real_words(words_by_page.get(page_no), search_rect)
        else:
            right_has_words = False

        if (is_placeholder_or_empty(val_txt) or val_conf < 0.25):
            if value_has_words or right_has_words:
                skipped["kv_value_present"] += 1
                continue
            out.append({
                "page": page_no,
                "bbox": bbox_to_dict(poly_to_bbox(region.polygon)),
                "label": key_txt or "KeyValue",
                "source": "key_value",
                "reason": "kv_missing_or_low_conf",
                "confidence": 0.80
            })
    return out

def detect_selection_mark_blanks(result) -> List[Dict[str, Any]]:
    out = []
    for p in (result.pages or []):
        for m in (p.selection_marks or []):
            if getattr(m, "polygon", None) and str(m.state).lower() == "unselected":
                out.append({
                    "page": p.page_number,
                    "bbox": bbox_to_dict(poly_to_bbox(m.polygon)),
                    "label": "Checkbox/Radio",
                    "source": "selection_mark",
                    "reason": "unselected_selection_mark",
                    "confidence": 0.85
                })
    return out

def detect_label_gap_blanks(result, words_by_page, enable=False) -> List[Dict[str, Any]]:
    if not enable:
        return []
    out = []
    for p in (result.pages or []):
        page_no = p.page_number
        page_w  = p.width or 1000.0
        words   = words_by_page.get(page_no) or []
        for line in (p.lines or []):
            text = norm(line.content)
            if not text or not text.endswith(":"):
                continue
            lx0, ly0, lx1, ly1 = poly_to_bbox(line.polygon)
            line_w = max(lx1 - lx0, 1.0)
            avg_char_w = line_w / max(len(text), 10)
            min_gap = min_gap_threshold(avg_char_w, page_w)
            gap = (lx0 + 0.65*line_w, ly0, lx1, ly1)
            gap_w = max(0.0, gap[2]-gap[0])
            if has_real_words(words, gap):
                continue
            below = grow(gap, up=0, down=(ly1-ly0)*0.9)
            if has_real_words(words, below):
                continue
            if gap_w >= min_gap:
                out.append({
                    "page": page_no,
                    "bbox": bbox_to_dict(gap),
                    "label": text[:-1],
                    "source": "label_gap",
                    "reason": "label_right_gap",
                    "confidence": 0.65
                })
    return out

# -----------------------
# Overlay (red circles)
# -----------------------
def overlay_red_circles(input_pdf_path: str, output_pdf_path: str, blanks: List[Dict[str, Any]], di_pages: Dict[int, Any]):
    """
    Draw a red circle at each blank's center.
    Coordinate mapping:
      - Azure DI gives (x,y) in DI page units; DI provides page.width/page.height.
      - PyMuPDF page.rect is in points; we scale per page: sx = page.rect.width / di_page.width, sy = page.rect.height / di_page.height.
    """
    os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
    doc = fitz.open(input_pdf_path)

    # index DI pages by number for width/height
    di_page_map = {p.page_number: p for p in di_pages}

    for b in blanks:
        pno = b["page"]  # 1-based
        if pno < 1 or pno > len(doc):
            continue
        page = doc[pno - 1]
        di_p = di_page_map.get(pno)
        if not di_p:
            continue

        # scales
        sx = page.rect.width / (di_p.width or 1.0)
        sy = page.rect.height / (di_p.height or 1.0)

        # bbox center in DI units → PDF points
        bb = b["bbox"]
        cx_di = (bb["x0"] + bb["x1"]) / 2.0
        cy_di = (bb["y0"] + bb["y1"]) / 2.0
        w_di  = max(1e-3, (bb["x1"] - bb["x0"]))
        h_di  = max(1e-3, (bb["y1"] - bb["y0"]))
        # circle radius = half of larger side, scaled, with a minimum visual size
        r_pts = max((w_di * sx), (h_di * sy)) * 0.55
        r_pts = max(r_pts, 6)  # min radius in points so it’s visible

        cx_pts = cx_di * sx
        cy_pts = cy_di * sy

        # draw a red circle (stroke only)
        shape = page.new_shape()
        shape.draw_circle(fitz.Point(cx_pts, cy_pts), r_pts)
        shape.finish(color=(1, 0, 0), fill=None, width=1.5)  # red outline
        shape.commit()

    doc.save(output_pdf_path, deflate=True)
    doc.close()

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

    parser = argparse.ArgumentParser(description="Strict blank detector + overlay (Azure Document Intelligence)")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", type=str, help="Local file path (PDF, TIFF, JPG, PNG, etc.)")
    src.add_argument("--url", type=str, help="Public/SAS URL to the document")
    parser.add_argument("--include-label-gaps", action="store_true",
                        help="Also flag label→gap blanks (very conservative). OFF by default.")
    parser.add_argument("--debug", action="store_true", help="Print DI overview and per-page summary.")
    parser.add_argument("--overlay", type=str, help="Output PDF path to save circles overlay, e.g., out/annotated.pdf")
    args = parser.parse_args()

    result = analyze_document(endpoint, key, file_path=args.file, url=args.url)
    if args.debug:
        debug_overview(result)

    pages_by_no, words_by_page = build_page_indexes(result)
    skipped = {"table_words_present": 0, "kv_value_present": 0}

    blanks: List[Dict[str, Any]] = []
    blanks += detect_table_blanks(result, words_by_page, skipped)
    blanks += detect_kv_blanks(result, pages_by_no, words_by_page, skipped)
    blanks += detect_selection_mark_blanks(result)
    blanks += detect_label_gap_blanks(result, words_by_page, enable=args.include_label_gaps)

    # sort results
    priority = {"table": 0, "selection_mark": 1, "key_value": 2, "label_gap": 3}
    blanks.sort(key=lambda b: (b["page"], priority.get(b["source"], 99), -b.get("confidence", 0.0)))

    # print JSON
    print(json.dumps({"blanks": blanks}, indent=2))

    # overlay, if requested
    if args.overlay:
        if not args.file:
            # overlay needs the local PDF path
            raise ValueError("Overlay requires --file to draw on a local PDF.")
        overlay_red_circles(args.file, args.overlay, blanks, result.pages or [])
        print(f"\nSaved overlay PDF → {args.overlay}")

    # debug summary
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
        print(f"Skipped table cells with words present: {skipped['table_words_present']}")
        print(f"Skipped KV because a value word was present: {skipped['kv_value_present']}")

if __name__ == "__main__":
    main()
