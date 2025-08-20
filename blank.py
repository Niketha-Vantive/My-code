# detect_blanks.py
import os
import re
import json
import argparse
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient

# -----------------------
# Helpers (generic)
# -----------------------
PLACEHOLDER_RE = re.compile(
    r"^\s*(?:_{2,}|\.{2,}|-+|<\s*insert[^>]*>|<[^>]+>|tbd|tba|n/?a|not\s+provided|see\s+(?:stamp|signature))\s*$",
    re.IGNORECASE,
)

LABEL_WORD_RE = re.compile(r"(name|date|title|author|email|id|amount|by|rev)\b", re.IGNORECASE)

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def is_placeholder_or_empty(s: str) -> bool:
    s = (s or "").strip()
    return (not s) or bool(PLACEHOLDER_RE.match(s))

def poly_to_bbox(polygon) -> Tuple[float, float, float, float]:
    # polygon is [Point(x,y), ...] length 4 or 8 depending on SDK
    xs = [p.x for p in polygon]
    ys = [p.y for p in polygon]
    return (min(xs), min(ys), max(xs), max(ys))

def bbox_to_dict(b):
    x0, y0, x1, y1 = b
    return {"x0": x0, "y0": y0, "x1": x1, "y1": y1}

def min_gap_threshold(avg_char_w: float, page_w: float) -> float:
    # dynamic tiny-gap guard
    return max(1.5 * max(avg_char_w, 1.0), 0.05 * page_w)

def pct(value: float, total: float) -> float:
    return (value / total) if total else 0.0

# -----------------------
# Detection functions
# -----------------------
def detect_table_blanks(result) -> List[Dict[str, Any]]:
    blanks = []
    for t_idx, table in enumerate(result.tables or []):
        # try to collect column headers from row 0 or by "kind"
        col_headers = {}
        for cell in table.cells:
            if getattr(cell, "kind", None) == "columnHeader" or cell.row_index == 0:
                col_headers[cell.column_index] = norm(cell.content)

        for cell in table.cells:
            # skip header cells
            if getattr(cell, "kind", None) in {"columnHeader", "rowHeader"} or cell.row_index == 0:
                continue
            txt = norm(cell.content)
            if is_placeholder_or_empty(txt):
                # Build a human label from row+column context where possible
                col_lbl = col_headers.get(cell.column_index, f"Col {cell.column_index+1}")
                # try to find a row header in same row
                row_lbl = None
                for rc in table.cells:
                    if rc.row_index == cell.row_index and (getattr(rc, "kind", None) == "rowHeader"):
                        row_lbl = norm(rc.content)
                        break
                label = " | ".join([s for s in [row_lbl, col_lbl] if s])
                if not label:
                    label = f"Table {t_idx+1} R{cell.row_index+1}C{cell.column_index+1}"
                # bounding region
                if cell.bounding_regions:
                    reg = cell.bounding_regions[0]
                    bbox = poly_to_bbox(reg.polygon)
                    blanks.append({
                        "page": reg.page_number,
                        "bbox": bbox_to_dict(bbox),
                        "label": label,
                        "source": "table",
                        "confidence": 0.90
                    })
    return blanks

def detect_kv_blanks(result) -> List[Dict[str, Any]]:
    blanks = []
    for kv in (result.key_value_pairs or []):
        key_txt = norm(getattr(kv.key, "content", "") if kv.key else "")
        val_obj = getattr(kv, "value", None)
        val_txt = norm(getattr(val_obj, "content", "") if val_obj else "")
        val_conf = getattr(val_obj, "confidence", 0.0) if val_obj else 0.0

        if is_placeholder_or_empty(val_txt) or val_conf < 0.25:
            # use value bbox if present, else key bbox
            region = None
            if val_obj and getattr(val_obj, "bounding_regions", None):
                region = val_obj.bounding_regions[0]
            elif kv.key and getattr(kv.key, "bounding_regions", None):
                region = kv.key.bounding_regions[0]
            if region:
                bbox = poly_to_bbox(region.polygon)
                blanks.append({
                    "page": region.page_number,
                    "bbox": bbox_to_dict(bbox),
                    "label": key_txt or "KeyValue",
                    "source": "key_value",
                    "confidence": 0.80
                })
    return blanks

def detect_selection_mark_blanks(result) -> List[Dict[str, Any]]:
    blanks = []
    for page in (result.pages or []):
        for mark in (page.selection_marks or []):
            if mark.state and str(mark.state).lower() == "unselected":
                if mark.polygon:
                    bbox = poly_to_bbox(mark.polygon)
                    blanks.append({
                        "page": page.page_number,
                        "bbox": bbox_to_dict(bbox),
                        "label": "Checkbox/Radio",
                        "source": "selection_mark",
                        "confidence": 0.85
                    })
    return blanks

def detect_label_gap_blanks(result) -> List[Dict[str, Any]]:
    """
    Simple, safe heuristic:
    - If a line ends with ':' OR looks like a label word line,
    - and there are no words to the right inside the line bbox,
    - and right-side slack >= dynamic min_gap, then flag the right side as blank.
    (We don't hard-map words to the colon position; this keeps it robust across SDK versions.)
    """
    blanks = []
    for page in (result.pages or []):
        page_w = page.width or 1000.0
        # Precompute page's words to estimate avg char width per line
        words = page.words or []
        for line in (page.lines or []):
            text = norm(line.content)
            if not text:
                continue

            looks_label = (text.endswith(":") or LABEL_WORD_RE.search(text) is not None)
            if not looks_label:
                continue

            # line bbox and rough metrics
            lx0, ly0, lx1, ly1 = poly_to_bbox(line.polygon)
            line_w = max(lx1 - lx0, 1.0)
            avg_char_w = line_w / max(len(text), 10)

            # check if any word centroid is significantly to the right half of the line
            right_words = []
            for w in words:
                wx0, wy0, wx1, wy1 = poly_to_bbox(w.polygon)
                # consider horizontal right-of-center and vertical overlap with line
                if wx0 > (lx0 + 0.65 * line_w) and not (wy1 < ly0 or wy0 > ly1):
                    right_words.append(w)

            # right slack = 35% of line width by default; refine with words if present
            gap_start = lx0 + 0.65 * line_w
            gap_end = lx1
            gap_w = max(gap_end - gap_start, 0.0)

            min_gap = min_gap_threshold(avg_char_w, page_w)
            if not right_words and gap_w >= min_gap:
                blanks.append({
                    "page": page.page_number,
                    "bbox": bbox_to_dict((gap_start, ly0, gap_end, ly1)),
                    "label": text.rstrip(":"),
                    "source": "label_gap",
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
    args = parser.parse_args()

    result = analyze_document(endpoint, key, file_path=args.file, url=args.url)

    blanks: List[Dict[str, Any]] = []
    blanks += detect_table_blanks(result)
    blanks += detect_kv_blanks(result)
    blanks += detect_selection_mark_blanks(result)
    if args.include_label_gaps:
        blanks += detect_label_gap_blanks(result)

    # Sort by page, then source priority
    priority = {"table": 0, "selection_mark": 1, "key_value": 2, "label_gap": 3}
    blanks.sort(key=lambda b: (b["page"], priority.get(b["source"], 99), -b.get("confidence", 0.0)))

    print(json.dumps({"blanks": blanks}, indent=2))

if __name__ == "__main__":
    main()
