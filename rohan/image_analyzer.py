"""
image_analyzer.py
Improved OCR medicine detection.

Features
• EasyOCR text extraction
• rapidfuzz / fuzzywuzzy fuzzy matching
• 3-char prefix index for fast 50k+ medicine lookup
• phrase matching + strength detection
"""

import io
import re
import os
import csv
import time
import warnings

import numpy as np
from PIL import Image, ImageEnhance

warnings.filterwarnings("ignore", message=".*pin_memory.*")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ─────────────────────────────────────────────
# OCR
# ─────────────────────────────────────────────

OCR_OK = False
_reader = None

try:
    import easyocr

    _reader = easyocr.Reader(['en'], gpu=False, verbose=False)

    OCR_OK = True
    print("  [ocr] EasyOCR ✓")

except Exception as e:
    print(f"  [ocr] EasyOCR unavailable: {e}")


# ─────────────────────────────────────────────
# Fuzzy matching
# ─────────────────────────────────────────────

_fuzz = None
_fwp = None

try:
    from rapidfuzz import fuzz as _fuzz
    from rapidfuzz import process as _fwp

    print("  [ocr] rapidfuzz ✓")

except ImportError:

    try:
        from fuzzywuzzy import fuzz as _fuzz
        from fuzzywuzzy import process as _fwp

        print("  [ocr] fuzzywuzzy ✓")

    except ImportError:
        print("  [ocr] WARNING: install rapidfuzz")


# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────

_BASE = os.path.dirname(os.path.abspath(__file__))

CSV_PATH = os.path.join(_BASE, "data", "medicines.csv")


# ─────────────────────────────────────────────
# Noise words
# ─────────────────────────────────────────────

NOISE = {
    "tablet","tablets","capsule","capsules","syrup","injection",
    "cream","gel","ointment","solution","suspension","drops",
    "mg","ml","strip","box","pack","batch","mfg","exp","date",
    "manufactured","expiry","store","below","children",
    "contains","ip","bp","usp","ltd","pharma","india",
    "dosage","doctor","prescription","schedule","drug",
    "use","before","keep","out","reach","read","leaflet",
    "carefully","only","not","for","the","and","with",
    "each","per","fast","relief","advance","forte"
}


# ─────────────────────────────────────────────
# Medicine index
# ─────────────────────────────────────────────

_INDEX = {}
_ALL = []
_NAMES = []


def _build_index(medicine_list):

    global _INDEX, _ALL, _NAMES

    _ALL = medicine_list
    _NAMES = [m["Name"] for m in medicine_list]

    _INDEX = {}

    for m in medicine_list:

        name = m["Name"].strip().lower()

        key = name[:3] if len(name) >= 3 else name

        _INDEX.setdefault(key, []).append(m)

    print(f"  [ocr] index built ({len(_ALL)} medicines, {len(_INDEX)} buckets)")


# ─────────────────────────────────────────────
# CSV loader
# ─────────────────────────────────────────────

def _load_csv():

    if not os.path.exists(CSV_PATH):
        print("  [ocr] medicines.csv not found")
        return []

    rows = []

    with open(CSV_PATH, encoding="utf-8-sig") as f:

        reader = csv.DictReader(f)

        for row in reader:

            rows.append({

                "Name": row.get("Name","").strip(),

                "Category": row.get("Category","").strip(),

                "Dosage Form": row.get("Dosage Form","").strip(),

                "Strength": row.get("Strength","").strip(),

                "Manufacturer": row.get("Manufacturer","").strip(),

                "Indication": row.get("Indication","").strip(),

                "Classification": row.get("Classification","").strip()
            })

    print(f"  [ocr] loaded {len(rows)} medicines")

    return rows


_build_index(_load_csv())


# ─────────────────────────────────────────────
# Image preprocessing
# ─────────────────────────────────────────────

def _preprocess(img):

    img = img.convert("RGB")

    img = ImageEnhance.Contrast(img).enhance(1.5)

    img = ImageEnhance.Sharpness(img).enhance(2.0)

    return img


# ─────────────────────────────────────────────
# OCR
# ─────────────────────────────────────────────

def run_ocr(image_bytes):

    if not OCR_OK:
        return ""

    img = Image.open(io.BytesIO(image_bytes))

    img = _preprocess(img)

    results = _reader.readtext(np.array(img), detail=1)

    texts = [text for (_, text, conf) in results if conf > 0.35]

    return " ".join(texts)


# ─────────────────────────────────────────────
# Token extraction
# ─────────────────────────────────────────────

def extract_tokens(text):

    clean = re.sub(r"[^a-zA-Z0-9\s\-]", " ", text)

    words = clean.split()

    tokens = []

    for w in words:

        lw = w.lower()

        if lw in NOISE:
            continue

        if len(w) < 3:
            continue

        tokens.append(w)

    bigrams = []

    for i in range(len(words)-1):

        bigrams.append(f"{words[i]} {words[i+1]}")

    phrases = []

    for i in range(len(words)-2):

        phrases.append(f"{words[i]} {words[i+1]} {words[i+2]}")

    return tokens + bigrams + phrases


# ─────────────────────────────────────────────
# Strength detection
# ─────────────────────────────────────────────

def _extract_strength(token):

    m = re.search(r"\b(\d{2,4})\b", token)

    if m:
        return m.group(1)

    return None


# ─────────────────────────────────────────────
# Candidate lookup
# ─────────────────────────────────────────────

def _candidates_for(token):

    t = token.lower()

    for l in (3,2,1):

        key = t[:l]

        bucket = _INDEX.get(key)

        if bucket:

            sub = [m for m in bucket if t in m["Name"].lower()]

            return sub if sub else bucket

    return _ALL


# ─────────────────────────────────────────────
# Fuzzy search
# ─────────────────────────────────────────────

def _fuzzy_search(token, threshold=70):

    if not _fwp:
        return []

    candidates = _candidates_for(token)

    names = [m["Name"] for m in candidates]

    results = _fwp.extract(
        token,
        names,
        scorer=_fuzz.token_set_ratio,
        limit=5
    )

    strength = _extract_strength(token)

    out = []

    for name, score, *_ in results:

        if score < threshold:
            continue

        row = next((m for m in candidates if m["Name"] == name), None)

        if not row:
            continue

        if strength and strength in row.get("Strength",""):
            score += 10

        out.append({

            "medicine": row,

            "score": round(min(score,100)/100,2),

            "matched_token": token
        })

    return out


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def detect_medicine(image_bytes, medicine_list):

    if medicine_list and len(medicine_list) != len(_ALL):
        _build_index(medicine_list)

    text = run_ocr(image_bytes)

    tokens = extract_tokens(text)

    matches = []

    seen = set()

    for token in tokens[:15]:

        for m in _fuzzy_search(token):

            name = m["medicine"]["Name"]

            if name not in seen:

                seen.add(name)

                matches.append(m)

        if len(matches) >= 5:
            break

    matches.sort(key=lambda x: x["score"], reverse=True)

    top_name = matches[0]["medicine"]["Name"] if matches else "Unknown"

    confidence = matches[0]["score"] if matches else 0.0

    return {

        "ocr_text": text.strip(),

        "tokens": tokens[:8],

        "matches": matches[:5],

        "top_name": top_name,

        "confidence": confidence
    }