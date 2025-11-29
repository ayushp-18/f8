from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os, re, tempfile, requests
from pdf2image import convert_from_path
import pytesseract
import base64
import io
from typing import Optional
from PIL import Image

# ==========================
# CONFIG
# Use environment variables so deployment can provide platform-specific paths
# POPPLER_BIN: path to poppler's bin directory (optional)
# TESSERACT_CMD: full path to tesseract executable (optional)
# ==========================
POPPLER_BIN = os.environ.get("POPPLER_BIN")
TESSERACT_EXE = os.environ.get("TESSERACT_CMD")
GEMINI_API_URL = os.environ.get("GEMIhttps://aistudio.google.com/api-keysNI_API_URL")
GEMINI_API_KEY = os.environ.get("AIzaSyAmmYdPLsOpwf-W3on2Kw7P080MsoLqZf8GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")
GEMINI_OCR_PROMPT = os.environ.get("GEMINI_OCR_PROMPT", "Please extract all textual content from the provided invoice page image. Return only the plain extracted text preserving line breaks. Do not add commentary or labels.")

if TESSERACT_EXE and os.path.exists(TESSERACT_EXE):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE

# ==========================
# HELPERS
# ==========================
def download_pdf(url: str) -> str:
    """Downloads a PDF from a URL and returns the local file path."""
    try:
        response = requests.get(url, timeout=20)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Failed to download PDF: HTTP {response.status_code}")

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        temp_file.write(response.content)
        temp_file.close()
        return temp_file.name

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF download failed: {e}")


def clean_num(s):
    if not s:
        return None
    s = str(s).replace(",", "").replace("₹", "").replace("$", "")
    m = re.findall(r"[-+]?\d*\.\d+|\d+", s)
    return float(m[0]) if m else None


numeric_suffix_re = re.compile(r"""
    (?P<prefix>.*\S)\s+
    (?P<qty>\d+(?:\.\d+)?)\s+
    (?P<rate>[-\d,\.]+)\s+
    (?P<discount>[-\d,\.]+)\s+
    (?P<net>[-\d,\.]+)\s*$
""", re.VERBOSE)

amount_only_re = re.compile(r'(?P<prefix>.*\S)\s+(?P<net>[-\d,\.]+)\s*$')


def parse_page_text(text):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    items = []
    buf_name = None

    for ln in lines:
        low = ln.lower()
        if any(x in low for x in ["description", "qty", "rate", "discount", "net amt", "total"]):
            buf_name = None
            continue

        m = numeric_suffix_re.match(ln)
        if m:
            items.append({
                "item_name": m.group("prefix").strip(),
                "item_quantity": clean_num(m.group("qty")),
                "item_rate": clean_num(m.group("rate")),
                "item_amount": clean_num(m.group("net"))
            })
            buf_name = None
            continue

        m2 = amount_only_re.match(ln)
        if buf_name and m2:
            items.append({
                "item_name": (buf_name + " " + m2.group("prefix")).strip(),
                "item_quantity": 1.0,
                "item_rate": None,
                "item_amount": clean_num(m2.group("net"))
            })
            buf_name = None
            continue

        buf_name = ln

    return items


JUNK_PATTERNS = [
    "pagewise line items",
    "response format",
    "item name",
    "tem_amount",
    "tem quantity",
]

def is_junk_page(txt):
    low = txt.lower()
    return any(p in low for p in JUNK_PATTERNS)


def normalize_name(s):
    s = re.sub(r"[^a-zA-Z0-9 ]+", " ", s).lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()


# ==========================
# OCR: pluggable wrapper
# If `GEMINI_API_URL` + `GEMINI_API_KEY` are provided, POST the page image
# as base64 JSON to that endpoint with `model` (default `gemini-2.5-pro`).
# Expected response should include extracted text in a top-level `text`
# field or in one of the common nested fields; otherwise we fallback to
# `pytesseract`.
# ==========================
def _image_to_base64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def ocr_page(image: Image.Image, prompt: Optional[str] = None) -> str:
    # If user configured a Gemini endpoint, try that first
    if GEMINI_API_URL and GEMINI_API_KEY:
        try:
            if prompt is None:
                prompt = GEMINI_OCR_PROMPT
            payload = {
                "model": GEMINI_MODEL,
                "image_base64": _image_to_base64_png(image),
                "prompt": prompt,
                # include messages for endpoints that accept chat-like inputs
                "messages": [{"role": "user", "content": prompt}]
            }
            headers = {
                "Authorization": f"Bearer {GEMINI_API_KEY}",
                "Content-Type": "application/json"
            }
            resp = requests.post(GEMINI_API_URL, json=payload, headers=headers, timeout=30)
            if resp.status_code == 200:
                j = resp.json()
                # Common possible fields
                if isinstance(j, dict):
                    if "text" in j and isinstance(j["text"], str):
                        return j["text"]
                    # Google-like `candidates` or `output` shapes
                    if "outputs" in j and isinstance(j["outputs"], list):
                        parts = []
                        for out in j["outputs"]:
                            if isinstance(out, dict):
                                for fk in ("text", "content", "output", "payload"):
                                    if fk in out and isinstance(out[fk], str):
                                        parts.append(out[fk])
                        if parts:
                            return "\n".join(parts)
                    # some proxy endpoints return {"predictions": [{"text": "..."}]}
                    if "predictions" in j and isinstance(j["predictions"], list):
                        for p in j["predictions"]:
                            if isinstance(p, dict) and "text" in p:
                                return p["text"]
                # fallback: try raw string body
                body = resp.text.strip()
                if body:
                    return body
            else:
                # Non-200: fall through to pytesseract
                print(f"Gemini OCR endpoint returned {resp.status_code}; falling back to Tesseract")
        except Exception as e:
            print(f"Gemini OCR request failed: {e}; falling back to Tesseract")

    # Default fallback to tesseract
    try:
        return pytesseract.image_to_string(image)
    except Exception as e:
        print(f"Tesseract OCR failed: {e}")
        return ""


# ==========================
# FASTAPI APP
# ==========================
app = FastAPI(title="Bill OCR Extractor API")


class ExtractRequest(BaseModel):
    document: str   # URL or local path


@app.post("/extract-bill-data")
async def extract_bill_data(req: ExtractRequest):
    doc = req.document.strip()

    # Case 1: URL download
    if doc.startswith("http://") or doc.startswith("https://"):
        try:
            r = requests.get(doc)
            r.raise_for_status()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to download PDF URL: {e}")

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.write(r.content)
        tmp.close()
        pdf_path = tmp.name

    # Case 2: Local file
    elif os.path.exists(doc):
        pdf_path = doc

    # Invalid input
    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid document path. Provide a URL or a valid local file path."
        )

    # Convert PDF
    try:
        pages = convert_from_path(pdf_path, dpi=300, poppler_path=POPPLER_BIN)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF conversion failed: {e}")

    pagewise = []
    collected = []

    for i, page in enumerate(pages, 1):
        txt = ocr_page(page, prompt=GEMINI_OCR_PROMPT)

        if is_junk_page(txt):
            continue

        items = parse_page_text(txt)

        for it in items:
            it["_page_no"] = str(i)
            collected.append(it)

        pagewise.append({
            "page_no": str(i),
            "page_type": "Bill Detail",
            "bill_items": items
        })

    # Dedupe
    seen = set()
    final_items = []
    for it in collected:
        key = (normalize_name(it["item_name"]), it.get("item_amount"))
        if key not in seen:
            seen.add(key)
            final_items.append(it)

    total_amt = round(sum((it["item_amount"] or 0) for it in final_items), 2)

    return {
        "is_success": True,
        "data": {
            "pagewise_line_items": pagewise,
            "unique_line_items": final_items,
            "total_items_count": len(final_items),
            "sum_total": total_amt
        }
    }
