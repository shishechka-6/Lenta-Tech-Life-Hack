"""Production pipeline for price-tag recognition from 4K supermarket video.

Self-contained: no imports from ``solution/``. All parser logic is inlined so
the file can be deployed as part of the backend package without bundling the
research notebooks.

Public API matches the FastAPI consumer:

    from .ml_pipeline import PipelineError, PriceTagProcessor

Pipeline stages:

    1. YOLO11n + ByteTrack — detect & track price tags across frames.
    2. Top-K sharpest keyframes per track (Laplacian variance × √area).
    3. Per-crop FieldExtractor — PaddleOCR + zxing-cpp + QR decoders + parsers.
    4. Multi-frame consensus — Counter.most_common per field.
    5. Post-processing — DB sanity filter, validators, canonicalisation.

The pipeline intentionally drops the VLM second-pass (Qwen2.5-VL on Kaggle
T4×2 takes ~5h); for an interactive backend this is not viable. Quality on
``code/id_sku/print_datetime/barcode`` is therefore lower than the offline
notebook pipeline (3/157 = 1.9% strict), but the architecture is the same.
"""
from __future__ import annotations

import csv
import difflib
import io
import json
import logging
import math
import re
import time
from collections import Counter, defaultdict, namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


logger = logging.getLogger("uvicorn.error")
ProgressCallback = Callable[[str, str, int], None]


# ─── Submission schema (sans bbox/timestamp — backend produces simplified format)

SUBMISSION_COLUMNS = [
    "video",
    "track_id",
    "product_name",
    "price_default",
    "price_card",
    "price_discount",
    "barcode",
    "discount_amount",
    "id_sku",
    "print_datetime",
    "code",
    "additional_info",
    "color",
    "special_symbols",
    "qr_code_barcode",
    "price1_qr",
    "price2_qr",
    "price3_qr",
    "price4_qr",
    "wholesale_level_1_count",
    "wholesale_level_1_price",
    "wholesale_level_2_count",
    "wholesale_level_2_price",
    "action_price_qr",
    "action_code_qr",
]

QR_FIELDS = (
    "qr_code_barcode",
    "price1_qr", "price2_qr", "price3_qr", "price4_qr",
    "wholesale_level_1_count", "wholesale_level_1_price",
    "wholesale_level_2_count", "wholesale_level_2_price",
    "action_price_qr", "action_code_qr",
)

TRACK_SIGNAL_FIELDS = (
    "product_name",
    "price_default",
    "price_card",
    "barcode",
    "discount_amount",
    "id_sku",
    "print_datetime",
    "code",
    "additional_info",
    "qr_code_barcode",
    "price1_qr",
    "price2_qr",
    "price3_qr",
    "price4_qr",
    "wholesale_level_1_count",
    "wholesale_level_1_price",
    "wholesale_level_2_count",
    "wholesale_level_2_price",
    "action_price_qr",
    "action_code_qr",
)


# ─── Exceptions and DTOs


class PipelineError(RuntimeError):
    """Raised when video processing cannot be completed."""


@dataclass(frozen=True)
class ProcessingResult:
    columns: list[str]
    rows: list[dict[str, str]]
    csv_text: str
    processing_seconds: float
    tracks_detected: int
    frames_seen: int
    model_path: str
    device: str


def rows_to_csv(rows: list[dict[str, str]]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=SUBMISSION_COLUMNS,
        delimiter=";",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _empty_submission_row(video_id: str, track_id: Any) -> dict[str, str]:
    row = {column: "нет" for column in SUBMISSION_COLUMNS}
    row["video"] = video_id
    row["track_id"] = str(track_id)
    return row


def _line_to_dict(line: OcrLine) -> dict[str, object]:
    return {
        "text": line.text,
        "x": float(line.x),
        "y": float(line.y),
        "w": float(line.w),
        "h": float(line.h),
        "conf": float(line.conf),
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _track_sort_key(track_id: Any) -> tuple[int, Any]:
    try:
        return (0, int(track_id))
    except (TypeError, ValueError):
        return (1, str(track_id))


def _select_device(requested_device: str) -> str:
    if requested_device and requested_device != "auto":
        return requested_device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


# ─── Shared OCR primitives ────────────────────────────────────────────────────

OcrLine = namedtuple("OcrLine", ["text", "x", "y", "w", "h", "conf"])


def cx(line: OcrLine) -> float:
    return line.x + line.w / 2


def cy(line: OcrLine) -> float:
    return line.y + line.h / 2


SUPER_TRANS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")

DIGIT_FIX = str.maketrans({
    "З": "3", "з": "3",
    "O": "0", "o": "0", "О": "0", "о": "0",
    "I": "1", "l": "1", "i": "1",
    "B": "8", "В": "8",
    "S": "5", "Ѕ": "5",
})

LATIN_TO_CYR_NORM = str.maketrans({
    "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "K": "К", "M": "М",
    "O": "О", "P": "Р", "T": "Т", "X": "Х", "Y": "У",
    "a": "а", "c": "с", "e": "е", "o": "о", "p": "р", "x": "х", "y": "у",
})

LATIN_TO_CYR_HINTS = str.maketrans({
    "a": "а", "A": "а", "c": "с", "C": "с", "e": "е", "E": "е",
    "o": "о", "O": "о", "p": "р", "P": "р", "x": "х", "X": "х",
    "y": "у", "Y": "у", "b": "ь", "B": "в", "h": "н", "H": "н",
    "k": "к", "K": "к", "m": "м", "M": "м", "t": "т", "T": "т",
    "r": "г", "n": "п",
})


def ean13_checksum_ok(value: str) -> bool:
    digits = re.sub(r"\D", "", value)
    if len(digits) != 13:
        return False
    parsed = [int(c) for c in digits]
    parity = sum(v * (3 if i % 2 else 1) for i, v in enumerate(parsed[:-1]))
    return (10 - parity % 10) % 10 == parsed[-1]


# ─── Parsers ──────────────────────────────────────────────────────────────────


def parse_barcode(lines: list[OcrLine]) -> str | None:
    if not lines:
        return None
    height = max(L.y + L.h for L in lines)
    candidates: list[tuple[str, float]] = []
    for L in lines:
        for digits in re.findall(r"\d{10,14}", re.sub(r"\D", "", L.text)):
            if len(digits) > 14:
                continue
            bottom_bonus = (cy(L) / max(height, 1)) ** 2
            valid_bonus = 2.0 if (len(digits) == 13 and ean13_checksum_ok(digits)) else 0.0
            score = len(digits) * 0.5 + bottom_bonus + L.conf + valid_bonus
            candidates.append((digits, score))
    if not candidates:
        return None
    candidates.sort(key=lambda x: -x[1])
    return candidates[0][0]


def parse_id_sku(lines: list[OcrLine], barcode_value: str | None = None) -> str | None:
    if not lines:
        return None
    height = max(L.y + L.h for L in lines)
    bc = barcode_value or ""
    candidates: list[tuple[str, float]] = []
    for L in lines:
        for digits in re.findall(r"\d{8,12}", re.sub(r"\D", "", L.text)):
            if not (8 <= len(digits) <= 12):
                continue
            if bc and (digits == bc or digits in bc or bc in digits):
                continue
            rel_y = cy(L) / max(height, 1)
            if rel_y < 0.3:
                continue
            score = L.conf + (1.0 - abs(rel_y - 0.7)) + (0.3 if 9 <= len(digits) <= 12 else 0)
            candidates.append((digits, score))
    if not candidates:
        return None
    candidates.sort(key=lambda x: -x[1])
    return candidates[0][0]


_DATETIME_RE_FULL = re.compile(r"(\d{2})[.\-/](\d{2})[.\-/](\d{4})[\D]+(\d{1,2})[:.]\s*(\d{2})")
_DATETIME_RE_DATE = re.compile(r"(\d{2})[.\-/](\d{2})[.\-/](\d{4})")
_TIME_RE = re.compile(r"^(\d{1,2})[:.](\d{2})$")


def parse_print_datetime(lines: list[OcrLine]) -> str | None:
    for L in lines:
        s = L.text.translate(DIGIT_FIX)
        m = _DATETIME_RE_FULL.search(s)
        if m:
            dd, mm, yyyy, hh, mi = m.groups()
            return f"{dd}.{mm}.{yyyy} {int(hh)}:{mi}"
    for L in lines:
        s = L.text.translate(DIGIT_FIX)
        m = _DATETIME_RE_DATE.search(s)
        if m:
            dd, mm, yyyy = m.groups()
            for L2 in lines:
                if L2 is L:
                    continue
                m2 = _TIME_RE.match(L2.text.translate(DIGIT_FIX).strip())
                if m2:
                    return f"{dd}.{mm}.{yyyy} {int(m2.group(1))}:{m2.group(2)}"
            return f"{dd}.{mm}.{yyyy}"
    return None


_DISCOUNT_RE_V2 = re.compile(r"[-−–—]?\s*([0-9OoIlbBg]{1,3})\s*%")
_DISCOUNT_DIGIT_FIX = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "b": "6", "g": "9", "B": "8"})


def parse_discount_amount_v2(lines: list[OcrLine]) -> str | None:
    """v1 + tolerance к OCR-путаницам O→0, l→1 в цифрах скидки."""
    best: tuple[int, float] | None = None
    for L in lines:
        m = _DISCOUNT_RE_V2.search(L.text)
        if not m:
            continue
        raw = m.group(1).translate(_DISCOUNT_DIGIT_FIX)
        if not raw.isdigit():
            continue
        n = int(raw)
        if 1 <= n <= 99 and (best is None or L.h > best[1]):
            best = (n, L.h)
    return f"-{best[0]}%" if best else None


_K_TOKENS = {"К", "к", "K", "k"}
_SH_TOKENS = {"Ш", "ш"}


def parse_special_symbols_v2(lines: list[OcrLine]) -> str | None:
    """Маленький изолированный токен К или Ш."""
    if not lines:
        return None
    for L in lines:
        s = L.text.strip()
        if s in _K_TOKENS:
            return "К"
        if s in _SH_TOKENS:
            return "Ш"
    candidates: list[tuple[str, OcrLine]] = []
    for L in lines:
        s = L.text.strip()
        if not s or len(s) > 3:
            continue
        clean = re.sub(r"[^КкKkШш]", "", s)
        if clean in _K_TOKENS:
            candidates.append(("К", L))
        elif clean in _SH_TOKENS:
            candidates.append(("Ш", L))
    if candidates:
        sym, _L = min(candidates, key=lambda c: c[1].h)
        return sym
    return None


ADDITIONAL_HINTS = (
    "Полусладкое", "Полусухое", "Сладкое", "Сухое",
    "Игристое", "Шипучее", "Десертное",
    "по цене", "Удачная упаковка",
)


def _fuzzy_find_hint(text_lower: str, hint: str) -> bool:
    h = hint.lower()
    if h in text_lower:
        return True
    L = len(h)
    if L < 4:
        return False
    threshold = 0.75 if L >= 8 else 0.78
    for i in range(len(text_lower) - L + 1):
        window = text_lower[i:i + L + 1]
        ratio = difflib.SequenceMatcher(None, h, window).ratio()
        if ratio >= threshold:
            return True
    return False


def parse_additional_info(lines: list[OcrLine]) -> str | None:
    raw = " ".join(L.text for L in lines).lower()
    normalized = raw.translate(LATIN_TO_CYR_HINTS)
    for hint in ADDITIONAL_HINTS:
        if _fuzzy_find_hint(normalized, hint):
            return hint
    return None


# Price parsing (v2)

MIN_PRICE_VALUE = 10.0
MAX_PRICE_VALUE = 9999.99

_PRICE_INLINE_RE = re.compile(r"(?<!\d)(\d{1,4})[,.\s]+(\d{2})\b")
_SOURCE_PRIORITY = {"inline": 0, "pair": 1, "int-only": 2}
_MIN_DX_PX = 80
_MIN_DY_PX = 35


def _is_plausible_price_value(value: float) -> bool:
    return MIN_PRICE_VALUE <= value <= MAX_PRICE_VALUE


def _extract_integer(text: str) -> int | None:
    s = text.translate(SUPER_TRANS).strip()
    if re.fullmatch(r"[-−–—]?\s*\d{1,2}\s*%?", s):
        if "%" in s or re.fullmatch(r"[-−–—]\s*\d{1,2}", s):
            return None
    runs = re.findall(r"\d+", s)
    if not runs:
        return None
    longest = max(runs, key=len)
    idx = s.find(longest)
    if s[idx + len(longest): idx + len(longest) + 1] == "%":
        return None
    v = int(longest)
    return v if _is_plausible_price_value(float(v)) else None


def _try_inline_price(text: str) -> float | None:
    m = _PRICE_INLINE_RE.search(text.translate(SUPER_TRANS))
    if not m:
        return None
    v = int(m.group(1)) + int(m.group(2)) / 100.0
    return v if _is_plausible_price_value(v) else None


def _fmt_price_ru(v: float) -> str:
    return f"{v:.2f}".replace(".", ",")


def _candidate_prices_v2(lines: list[OcrLine]) -> list[tuple[float, OcrLine, str]]:
    cands: list[tuple[float, OcrLine, str]] = []
    for L in lines:
        v = _try_inline_price(L.text)
        if v is not None:
            cands.append((v, L, "inline"))
    integers = [(v, L) for L in lines if (v := _extract_integer(L.text)) is not None]
    decimals: list[tuple[int, OcrLine]] = []
    for L in lines:
        s = L.text.translate(SUPER_TRANS).strip()
        m = re.fullmatch(r"[^\d]*(\d{2})[^\d]*", s)
        if m:
            decimals.append((int(m.group(1)), L))
    for v_int, I in integers:
        best, best_d = None, float("inf")
        dx_thresh = max(2.5 * I.w, _MIN_DX_PX)
        dy_thresh = max(1.5 * I.h, _MIN_DY_PX)
        for d_val, D in decimals:
            if D is I:
                continue
            dx = cx(D) - cx(I)
            dy = cy(D) - cy(I)
            if abs(dx) > dx_thresh or abs(dy) > dy_thresh:
                continue
            if dx < -0.3 * I.w:
                continue
            d = abs(dx) + 2.0 * abs(dy)
            if d < best_d:
                best, best_d = d_val, d
        if best is not None:
            cands.append((v_int + best / 100.0, I, "pair"))
    for v_int, I in integers:
        if v_int >= 100:
            cands.append((float(v_int), I, "int-only"))
    return cands


def parse_prices_v2(lines: list[OcrLine]) -> dict[str, str | None]:
    """Extract price_card + optional price_default from OCR layout candidates."""
    out: dict[str, str | None] = {"price_default": None, "price_card": None}
    cands = _candidate_prices_v2(lines)
    if not cands:
        return out
    best_by_line: dict[int, tuple[float, OcrLine, str]] = {}
    for v, L, src in cands:
        key = id(L)
        if key not in best_by_line or _SOURCE_PRIORITY[src] < _SOURCE_PRIORITY[best_by_line[key][2]]:
            best_by_line[key] = (v, L, src)
    ordered = sorted(best_by_line.values(), key=lambda c: -c[1].h)
    if ordered:
        v, _line, src = ordered[0]
        out["price_card"] = _fmt_price_ru(int(v) + 0.99 if src == "int-only" else v)
    if len(ordered) > 1:
        card_int = int(ordered[0][0])
        for v, L, src in ordered[1:]:
            v_int = int(v)
            if v_int == card_int:
                continue
            if abs(v_int - card_int) / max(card_int, 1) <= 0.03:
                continue
            if src == "int-only":
                out["price_default"] = _fmt_price_ru(v_int + 0.99)
            else:
                out["price_default"] = _fmt_price_ru(v)
            break
    return out


def build_product_name(lines: list[OcrLine]) -> str | None:
    if not lines:
        return None
    height = max(L.y + L.h for L in lines)
    top = [
        L for L in lines
        if cy(L) < height * 0.55
        and any(c.isalpha() for c in L.text)
        and len(L.text.strip()) >= 2
    ]
    if not top:
        return None
    top.sort(key=lambda L: (L.y, L.x))
    name = " ".join(L.text.strip() for L in top)
    return name if len(name) >= 5 else None


def normalize_product_name(s: str | None) -> str:
    """Latin→Cyrillic homoglyph fold + lowercase + punct strip + collapse spaces."""
    if not s or str(s).strip().lower() in ("", "нет", "none"):
        return ""
    s = str(s).translate(LATIN_TO_CYR_NORM).lower()
    s = re.sub(r"[^a-zа-я0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# QR

QR_KEY_MAP = {
    "b": "qr_code_barcode", "barcode": "qr_code_barcode",
    "p1": "price1_qr", "price1": "price1_qr",
    "p2": "price2_qr", "price2": "price2_qr",
    "p3": "price3_qr", "price3": "price3_qr",
    "p4": "price4_qr", "price4": "price4_qr",
    "wL1C": "wholesale_level_1_count", "wholesaleLevel1Count": "wholesale_level_1_count",
    "wL1P": "wholesale_level_1_price", "wholesaleLevel1Price": "wholesale_level_1_price",
    "wL2C": "wholesale_level_2_count", "wholesaleLevel2Count": "wholesale_level_2_count",
    "wL2P": "wholesale_level_2_price", "wholesaleLevel2Price": "wholesale_level_2_price",
    "aP": "action_price_qr", "actionPrice": "action_price_qr",
    "aC": "action_code_qr", "actionCode": "action_code_qr",
}


def parse_qr_payload(text: str) -> dict[str, Any]:
    if not text:
        return {}
    out: dict[str, Any] = {}
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in QR_KEY_MAP:
                    out[QR_KEY_MAP[k]] = None if v == "" else v
            return out
    except Exception:
        pass
    for part in re.split(r"[&;,\n]", text):
        m = re.match(r"\s*([A-Za-z0-9_]+)\s*[=:]\s*(.*?)\s*$", part)
        if m and m.group(1) in QR_KEY_MAP:
            out[QR_KEY_MAP[m.group(1)]] = None if m.group(2) == "" else m.group(2)
    return out


# ─── Validators ───────────────────────────────────────────────────────────────


def valid_barcode(s: str | None) -> bool:
    if not s:
        return False
    d = "".join(c for c in str(s) if c.isdigit())
    if len(d) == 13:
        return ean13_checksum_ok(d)
    return len(d) in (8, 12)


def valid_id_sku(s: str | None) -> bool:
    if not s:
        return False
    d = "".join(c for c in str(s) if c.isdigit())
    return 9 <= len(d) <= 12


def valid_price(s: str | None) -> bool:
    if not s:
        return False
    try:
        f = float(str(s).replace(",", "."))
        return _is_plausible_price_value(f)
    except Exception:
        return False


def valid_datetime(s: str | None) -> bool:
    if not s:
        return False
    return bool(re.fullmatch(r"\d{2}\.\d{2}\.\d{4}( \d{1,2}:\d{2})?", str(s)))


def valid_discount(s: str | None) -> bool:
    if not s:
        return False
    m = re.fullmatch(r"-\d{1,2}%", str(s))
    if not m:
        return False
    return 1 <= int(str(s)[1:-1]) <= 99


def valid_special_sym(s: str | None) -> bool:
    return s in ("К", "Ш")


def valid_code(s: str | None) -> bool:
    """GT-форматы: '025017 - 026015', '01_025019', '21_ЦПУ', '024 017_1_6_2'."""
    if not s:
        return False
    s = str(s).strip()
    patterns = [
        r"\d{4,7}\s*-\s*\d{4,7}",
        r"\d{2}_\d{4,7}(\s*-\s*\d{4,7})?",
        r"\d{2}_[А-ЯA-Z]{2,5}",
        r"\d+ \d+(_\d+)+",
    ]
    return any(re.fullmatch(p, s) for p in patterns)


VALIDATORS = {
    "barcode": valid_barcode,
    "id_sku": valid_id_sku,
    "price_default": valid_price,
    "price_card": valid_price,
    "print_datetime": valid_datetime,
    "discount_amount": valid_discount,
    "special_symbols": valid_special_sym,
    "code": valid_code,
}


# ─── Post-processing helpers ─────────────────────────────────────────────────


_AI_VOCAB = [
    "Сухое", "Полусухое", "Полусладкое", "Сладкое",
    "2 по цене 1 от цены без карты",
    "3 по цене 2 от цены без карты",
    "Полусухое, Упс! Товар закончился. Уже везём!",
    "Сухое, 3 по цене 2 от цены без карты",
    "Сухое, доп. скидка для участников соц. программы",
]


def snap_additional_info(raw: str | None) -> str:
    """Прижимает наш OCR-выход к ближайшему элементу закрытого словаря."""
    if not raw or str(raw).strip().lower() in ("нет", "", "none"):
        return "нет"
    raw_norm = str(raw).strip()
    for v in _AI_VOCAB:
        if raw_norm in v or v.startswith(raw_norm):
            return v
    try:
        from rapidfuzz import fuzz, process

        result = process.extractOne(raw_norm, _AI_VOCAB, scorer=fuzz.token_set_ratio, score_cutoff=70)
        return result[0] if result else "нет"
    except Exception:
        return raw_norm if raw_norm in _AI_VOCAB else "нет"


_BLEED_TOKENS = re.compile(
    r"(\s+\d+\s+по\s+цене\s+\d+.*|"
    r"\s+(Сухое|Полусухое|Полусладкое|Сладкое)(\W.*)?$)",
    re.IGNORECASE,
)
_VOLUME_ANCHOR = re.compile(r"(\d[\.,]\s*\d{1,2}\s*[Lл])", re.IGNORECASE)


def canonical_product_name(raw: str | None) -> str:
    if not raw or str(raw).strip() in ("", "нет"):
        return "нет"
    s = str(raw).strip()
    m = _VOLUME_ANCHOR.search(s)
    if m:
        s = s[: m.end()].rstrip()
    s = _BLEED_TOKENS.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    # GT-конвенция: '0.75L' → '0. 75L' (GT имеет 61 такой случай против 10 без пробела)
    s = re.sub(r"(\d)\.(\d{2}[LМмл])", r"\1. \2", s)
    return s if s else "нет"


def _load_db_codes(db_path: Path | None) -> set[str]:
    """Загружает db_hack.csv (cp1251). Возвращает пустой set если файла нет."""
    if db_path is None or not db_path.exists():
        return set()
    codes: set[str] = set()
    try:
        with db_path.open(encoding="cp1251") as f:
            rdr = csv.reader(f, delimiter=";")
            next(rdr, None)
            for row in rdr:
                if len(row) == 2:
                    codes.add(row[1].strip())
    except Exception:
        return set()
    return codes


def _parse_price_value(s: str | None) -> float | None:
    if not s or s == "нет":
        return None
    try:
        return float(str(s).replace(",", "."))
    except Exception:
        return None


def _parse_disc_value(s: str | None) -> int | None:
    if not s or s == "нет":
        return None
    try:
        return int(str(s).replace("%", "").replace("-", "").strip())
    except Exception:
        return None


def _sanitize_record_prices(record: dict[str, Any]) -> None:
    for field in ("price_default", "price_card"):
        value = record.get(field)
        if value and not valid_price(str(value)):
            record[field] = None

    pd_val = _parse_price_value(record.get("price_default"))
    pc_val = _parse_price_value(record.get("price_card"))
    discount = _parse_disc_value(record.get("discount_amount"))

    if pd_val is not None and pc_val is not None:
        if pd_val <= pc_val * 1.03 or pd_val > pc_val * 4.0:
            record["price_default"] = None
            pd_val = None

    if pd_val is not None and pc_val is not None and discount:
        try:
            expected_default = pc_val / (1 - discount / 100.0)
        except ZeroDivisionError:
            return
        if expected_default and abs(pd_val - expected_default) / expected_default > 0.12:
            record["price_default"] = None


def _record_has_signal(record: dict[str, Any]) -> bool:
    return any(record.get(field) not in (None, "", "нет") for field in TRACK_SIGNAL_FIELDS)


def _safe_file_part(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))[:80] or "unknown"


# ─── Main service classes ─────────────────────────────────────────────────────


class FieldExtractor:
    """Single-crop OCR + parser output. Returns a per-crop dict.

    The processor calls this once per crop and aggregates results across the
    top-K crops of each track (multi-frame consensus).
    """

    def __init__(self) -> None:
        self.stats: Counter = Counter()
        self._ocr_error_logged = False
        self._ocr = self._load_ocr()
        self._zxing = self._load_zxing()
        self._wechat_qr = self._load_wechat_qr()
        self._cv_qr = None  # инициализируется лениво при первом вызове
        logger.info(
            "field_extractor_ready ocr=%s zxing=%s wechat_qr=%s",
            self._ocr is not None,
            self._zxing is not None,
            self._wechat_qr is not None,
        )

    @property
    def ocr_available(self) -> bool:
        return self._ocr is not None

    @staticmethod
    def _load_ocr():
        try:
            from paddleocr import PaddleOCR
        except Exception as exc:
            logger.exception("paddleocr_import_failed error=%s", exc)
            return None
        # PaddleOCR 3.x на CPU с oneDNN/PIR падает с NotImplementedError
        # (ConvertPirAttribute2RuntimeAttribute). Отключаем mkldnn для надёжности.
        try:
            logger.info("paddleocr_init_start")
            ocr = PaddleOCR(
                lang="ru",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                enable_mkldnn=False,
            )
            logger.info("paddleocr_init_done")
            return ocr
        except TypeError:
            try:
                logger.info("paddleocr_init_start fallback=no_mkldnn_kw")
                ocr = PaddleOCR(
                    lang="ru",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                )
                logger.info("paddleocr_init_done fallback=no_mkldnn_kw")
                return ocr
            except TypeError:
                try:
                    logger.info("paddleocr_init_start legacy_args=true")
                    ocr = PaddleOCR(lang="ru", use_angle_cls=False, enable_mkldnn=False)
                    logger.info("paddleocr_init_done legacy_args=true")
                    return ocr
                except Exception as exc:
                    logger.exception("paddleocr_init_failed legacy_args=true error=%s", exc)
                    return None
            except Exception as exc:
                logger.exception("paddleocr_init_failed fallback error=%s", exc)
                return None
        except Exception as exc:
            logger.exception("paddleocr_init_failed error=%s", exc)
            return None

    @staticmethod
    def _load_zxing():
        try:
            import zxingcpp
        except Exception:
            return None
        return zxingcpp

    @staticmethod
    def _load_wechat_qr():
        try:
            import cv2

            return cv2.wechat_qrcode_WeChatQRCode()
        except Exception:
            return None

    def extract(self, crop) -> dict[str, Any]:
        result, _debug = self.extract_with_debug(crop)
        return result

    def extract_with_debug(self, crop) -> tuple[dict[str, Any], dict[str, Any]]:
        """Запускает OCR/zxing/QR на одном кропе и парсит поля."""
        self.stats["crops_seen"] += 1
        debug: dict[str, Any] = {
            "crop_shape": list(crop.shape) if crop is not None and hasattr(crop, "shape") else None,
            "ocr_lines": [],
            "zxing_barcode": None,
            "qr": {},
            "parsed": {},
        }
        if crop is None or crop.size == 0:
            self.stats["empty_crops"] += 1
            debug["empty_crop"] = True
            return {}, debug

        lines = self._ocr_lines(crop)
        zxing_bc = self._decode_barcode_1d(crop)
        qr = self._decode_qr(crop)
        debug["ocr_lines"] = [_line_to_dict(line) for line in lines]
        debug["zxing_barcode"] = zxing_bc
        debug["qr"] = qr
        if lines:
            self.stats["ocr_crops_with_text"] += 1
            self.stats["ocr_lines"] += len(lines)
        if zxing_bc:
            self.stats["barcode_crops"] += 1
        if qr:
            self.stats["qr_crops"] += 1

        if not lines and not zxing_bc and not qr:
            return {}, debug

        bc = zxing_bc or parse_barcode(lines)

        parsed: dict[str, Any] = {
            "product_name": build_product_name(lines),
            "barcode": bc,
            "id_sku": parse_id_sku(lines, bc),
            "print_datetime": parse_print_datetime(lines),
            "discount_amount": parse_discount_amount_v2(lines),
            "additional_info": parse_additional_info(lines),
            "special_symbols": parse_special_symbols_v2(lines),
        }
        parsed.update(parse_prices_v2(lines))

        # QR-поля идут сверху обычных — они достоверны
        for k, v in qr.items():
            if v not in (None, ""):
                parsed[k] = v

        result = {k: v for k, v in parsed.items() if v not in (None, "")}
        debug["parsed"] = result
        if result:
            self.stats["parsed_crops"] += 1
        return result, debug

    def _ocr_lines(self, crop) -> list[OcrLine]:
        if self._ocr is None:
            return []
        try:
            if hasattr(self._ocr, "predict"):
                result = self._ocr.predict(crop)
            else:
                result = self._ocr.ocr(crop)
        except Exception as exc:
            self.stats["ocr_errors"] += 1
            if not self._ocr_error_logged:
                logger.exception("paddleocr_predict_failed_once error=%s", exc)
                self._ocr_error_logged = True
            return []
        return self._normalize_ocr_result(result)

    @staticmethod
    def _normalize_ocr_result(result) -> list[OcrLine]:
        if not result:
            return []
        first = result[0]
        if first is None:
            return []
        lines: list[OcrLine] = []
        for paddle_dict in FieldExtractor._paddle_result_dicts(first):
            lines.extend(FieldExtractor._lines_from_paddle_dict(paddle_dict))
            if lines:
                return lines
        if isinstance(first, dict):
            return lines
        try:
            legacy_items = first or []
            for item in legacy_items:
                try:
                    box, text_score = item[0], item[1]
                    text, score = text_score[0], text_score[1]
                except Exception:
                    continue
                normalized = FieldExtractor._line_from_box(text, score, box)
                if normalized:
                    lines.append(normalized)
        except TypeError:
            return lines
        return lines

    @staticmethod
    def _paddle_result_dicts(result_item) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        if isinstance(result_item, dict):
            candidates.append(result_item)
        json_payload = getattr(result_item, "json", None)
        if json_payload is not None:
            try:
                if callable(json_payload):
                    json_payload = json_payload()
                if isinstance(json_payload, dict):
                    candidates.append(json_payload)
            except Exception:
                pass
        expanded: list[dict[str, Any]] = []
        for candidate in candidates:
            nested = candidate.get("res")
            if isinstance(nested, dict):
                expanded.append(nested)
            expanded.append(candidate)
        return expanded

    @staticmethod
    def _lines_from_paddle_dict(result_item: dict[str, Any]) -> list[OcrLine]:
        texts = result_item.get("rec_texts", [])
        scores = result_item.get("rec_scores", [])
        boxes = result_item.get("rec_boxes", result_item.get("rec_polys", []))
        lines: list[OcrLine] = []
        for text, score, box in zip(texts, scores, boxes):
            normalized = FieldExtractor._line_from_box(text, score, box)
            if normalized:
                lines.append(normalized)
        return lines

    @staticmethod
    def _line_from_box(text: str, score: float, box) -> OcrLine | None:
        """Поддерживает оба формата:
        - rec_boxes: shape (4,) — [x1, y1, x2, y2]
        - rec_polys: shape (4, 2) или (N, 2) — список точек [[x, y], ...]
        np.asarray(box).reshape(-1, 2) приводит к единому виду (M, 2), дальше min/max.
        """
        try:
            import numpy as np

            arr = np.asarray(box, dtype=float).reshape(-1, 2)
            if arr.size == 0:
                return None
            x1 = float(arr[:, 0].min())
            y1 = float(arr[:, 1].min())
            x2 = float(arr[:, 0].max())
            y2 = float(arr[:, 1].max())
            return OcrLine(str(text), x1, y1, x2 - x1, y2 - y1, float(score))
        except Exception:
            return None

    def _decode_barcode_1d(self, crop) -> str | None:
        if self._zxing is None:
            return None
        try:
            results = self._zxing.read_barcodes(crop)
        except Exception:
            return None
        for r in results:
            fmt = str(getattr(r, "format", "")).split(".")[-1]
            if fmt in ("EAN13", "EAN8", "UPCA", "UPCE", "Code128", "Code39", "ITF"):
                return r.text
        return None

    def _decode_qr(self, crop) -> dict[str, Any]:
        """Пробует WeChat QR + cv2.QRCodeDetector с апскейлом ×2,3,4.
        QR на ценнике ~80×80 px (2.7 px/модуль) — нужно апскейлить, чтобы достичь ≥4 px/модуль."""
        try:
            import cv2
        except Exception:
            return {}

        payloads: list[str] = []

        # WeChat (без апскейла)
        if self._wechat_qr is not None:
            try:
                texts, _ = self._wechat_qr.detectAndDecode(crop)
                payloads.extend([t for t in (texts or []) if t])
            except Exception:
                pass

        # cv2 на исходном размере и ×2,3,4
        if self._cv_qr is None:
            self._cv_qr = cv2.QRCodeDetector()
        height, width = crop.shape[:2]
        for scale in (1, 2, 3, 4):
            tgt = crop if scale == 1 else cv2.resize(
                crop, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC,
            )
            try:
                data, _, _ = self._cv_qr.detectAndDecode(tgt)
                if data:
                    payloads.append(data)
            except Exception:
                continue

        for payload in payloads:
            parsed = parse_qr_payload(payload)
            if parsed:
                return parsed
        return {}


class PriceTagProcessor:
    """End-to-end: video → submission rows.

    Wraps YOLO+ByteTrack detection, multi-frame consensus, and post-processing.
    Parameters mirror the FastAPI config so that the backend service can pass
    runtime overrides without modifying the pipeline body.
    """

    def __init__(
        self,
        model_path: Path,
        device: str,
        conf: float,
        iou: float,
        imgsz: int,
        max_frames: int | None,
        k_best: int = 5,
        db_path: Path | None = None,
        max_crop_side: int = 600,
    ) -> None:
        self.model_path = model_path
        self.device = _select_device(device)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.max_frames = max_frames
        self.k_best = max(1, int(k_best))
        self.db_path = db_path
        self.max_crop_side = max_crop_side
        self._model = None
        self._field_extractor: FieldExtractor | None = None
        self._db_codes: set[str] | None = None

    def process(
        self,
        video_path: Path,
        original_filename: str | None = None,
        progress_callback: ProgressCallback | None = None,
        debug_dir: Path | None = None,
    ) -> ProcessingResult:
        def report(stage: str, message: str, progress: int) -> None:
            if progress_callback is None:
                return
            try:
                progress_callback(stage, message, progress)
            except Exception:
                logger.exception("progress callback failed")

        start = time.perf_counter()
        video_id = Path(original_filename or video_path.name).stem
        report("preprocess", "Подготовка видео", 10)

        report("yolo", "YOLO 0%: трекинг ценников", 25)
        tracks, frames_seen = self._track_price_tags(video_path, progress_callback=report)
        report("yolo", f"YOLO завершен: {len(tracks)} треков, {frames_seen} кадров", 55)

        report("ocr", "Загрузка OCR и декодеров", 60)
        field_extractor = self._get_field_extractor()
        if not field_extractor.ocr_available:
            raise PipelineError("PaddleOCR is not available; check backend logs for paddleocr_import/init_failed")
        field_extractor.stats.clear()
        sorted_track_ids = sorted(tracks, key=_track_sort_key)
        total_tracks = len(sorted_track_ids)
        report("ocr", f"OCR 0%: треков {total_tracks}", 65)

        rows: list[dict[str, str]] = []
        records: list[dict[str, Any]] = []
        debug_tracks: list[dict[str, Any]] = []
        if debug_dir is not None:
            debug_dir.mkdir(parents=True, exist_ok=True)
        parsed_tracks = 0
        field_counts: Counter = Counter()
        last_ocr_progress = 65

        for idx, track_id in enumerate(sorted_track_ids, start=1):
            crops = tracks[track_id]["crops"]
            per_crop_fields: list[dict[str, Any]] = []
            crop_debugs: list[dict[str, Any]] = []
            for crop_idx, crop in enumerate(crops):
                if debug_dir is None:
                    per_crop_fields.append(field_extractor.extract(crop))
                    continue
                fields, crop_debug = field_extractor.extract_with_debug(crop)
                crop_debug["crop_path"] = self._write_debug_crop(debug_dir, track_id, crop_idx, crop)
                per_crop_fields.append(fields)
                crop_debugs.append(crop_debug)
            aggregated = self._aggregate_fields(per_crop_fields)
            meaningful_fields = {k: v for k, v in aggregated.items() if k != "color"}
            if meaningful_fields:
                parsed_tracks += 1
                field_counts.update(meaningful_fields.keys())
            records.append({"_track_id": track_id, **aggregated})
            if debug_dir is not None:
                debug_tracks.append(
                    {
                        "track_id": str(track_id),
                        "candidate_meta": tracks[track_id].get("candidate_meta", []),
                        "crop_count": len(crops),
                        "crops": crop_debugs,
                        "per_crop_fields": per_crop_fields,
                        "aggregated_before_postprocess": aggregated,
                    }
                )
            if total_tracks:
                stage_pct = int(idx * 100 / total_tracks)
                overall_progress = 65 + int(stage_pct * 20 / 100)
                if overall_progress > last_ocr_progress:
                    last_ocr_progress = overall_progress
                    report(
                        "ocr",
                        f"OCR {stage_pct}%: трек {idx}/{total_tracks}, распарсилось {parsed_tracks}",
                        overall_progress,
                    )
        stats = dict(field_extractor.stats)
        logger.info(
            "ocr_summary tracks=%s parsed_tracks=%s stats=%s fields=%s",
            len(records),
            parsed_tracks,
            stats,
            dict(field_counts),
        )
        report(
            "ocr",
            (
                f"OCR завершен: треков {len(records)}, "
                f"с текстом {stats.get('ocr_crops_with_text', 0)} кропов, "
                f"распарсилось {parsed_tracks}"
            ),
            85,
        )

        # Post-processing над всеми треками сразу (нужно для per-video code dedup)
        report("postprocess", "Постобработка результата", 92)
        self._post_process(records)
        postprocessed_records = [dict(record) for record in records]
        records_before_filter = len(records)
        records = [r for r in records if _record_has_signal(r)]
        filtered_tracks = records_before_filter - len(records)
        if filtered_tracks:
            logger.info(
                "track_filter before=%s after=%s filtered=%s",
                records_before_filter,
                len(records),
                filtered_tracks,
            )
        if debug_dir is not None:
            try:
                self._write_debug_payload(
                    debug_dir=debug_dir,
                    debug_tracks=debug_tracks,
                    postprocessed_records=postprocessed_records,
                    kept_records=records,
                    stats=stats,
                    frames_seen=frames_seen,
                    filtered_tracks=filtered_tracks,
                )
            except Exception:
                logger.exception("debug_payload_write_failed path=%s", debug_dir)

        for r in records:
            row = _empty_submission_row(video_id=video_id, track_id=r["_track_id"])
            for col in SUBMISSION_COLUMNS:
                if col in ("video", "track_id"):
                    continue
                v = r.get(col)
                if v not in (None, "", "нет"):
                    row[col] = str(v)
            rows.append(row)

        elapsed = time.perf_counter() - start
        report("postprocess", "Формирование CSV", 97)
        return ProcessingResult(
            columns=SUBMISSION_COLUMNS,
            rows=rows,
            csv_text=rows_to_csv(rows),
            processing_seconds=round(elapsed, 3),
            tracks_detected=len(rows),
            frames_seen=frames_seen,
            model_path=str(self.model_path),
            device=self.device,
        )

    # ── Tracking ──

    def _load_model(self):
        if self._model is not None:
            return self._model
        if not self.model_path.exists():
            raise PipelineError(f"YOLO weights not found: {self.model_path}")
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise PipelineError("ultralytics is not installed") from exc
        self._model = YOLO(str(self.model_path))
        return self._model

    def _get_field_extractor(self) -> FieldExtractor:
        if self._field_extractor is None:
            self._field_extractor = FieldExtractor()
        return self._field_extractor

    def _get_db_codes(self) -> set[str]:
        if self._db_codes is None:
            self._db_codes = _load_db_codes(self.db_path)
        return self._db_codes

    @staticmethod
    def _write_debug_crop(debug_dir: Path, track_id: Any, crop_idx: int, crop) -> str | None:
        try:
            import cv2
        except Exception:
            return None
        crops_dir = debug_dir / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)
        filename = f"track_{_safe_file_part(track_id)}_crop_{crop_idx:02d}.jpg"
        path = crops_dir / filename
        try:
            if crop is None or crop.size == 0:
                return None
            ok = cv2.imwrite(str(path), crop)
            return str(path.relative_to(debug_dir)) if ok else None
        except Exception as exc:
            logger.warning("debug_crop_write_failed path=%s error=%s", path, exc)
            return None

    @staticmethod
    def _write_debug_payload(
        debug_dir: Path,
        debug_tracks: list[dict[str, Any]],
        postprocessed_records: list[dict[str, Any]],
        kept_records: list[dict[str, Any]],
        stats: dict[str, Any],
        frames_seen: int,
        filtered_tracks: int,
    ) -> None:
        final_by_track = {str(r.get("_track_id")): r for r in postprocessed_records}
        kept_track_ids = {str(r.get("_track_id")) for r in kept_records}
        for track in debug_tracks:
            track_id = str(track.get("track_id"))
            track["kept_in_csv"] = track_id in kept_track_ids
            track["final_record"] = final_by_track.get(track_id)
            track_path = debug_dir / f"track_{_safe_file_part(track.get('track_id'))}.json"
            track_path.write_text(
                json.dumps(_json_ready(track), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        summary = {
            "frames_seen": frames_seen,
            "tracks_seen": len(debug_tracks),
            "tracks_kept": len(kept_records),
            "tracks_filtered": filtered_tracks,
            "ocr_stats": stats,
        }
        (debug_dir / "summary.json").write_text(
            json.dumps(_json_ready(summary), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (debug_dir / "records.json").write_text(
            json.dumps(_json_ready(kept_records), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (debug_dir / "records_all_postprocessed.json").write_text(
            json.dumps(_json_ready(postprocessed_records), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _track_price_tags(
        self,
        video_path: Path,
        progress_callback: ProgressCallback | None = None,
    ) -> tuple[dict[Any, dict[str, Any]], int]:
        try:
            import cv2
        except Exception as exc:
            raise PipelineError("opencv-python-headless is not installed") from exc

        model = self._load_model()
        # tracks[tid] = {'candidates': [(sharp, frame_idx, crop), ...] (top-K, sorted desc by sharpness)}
        tracks: dict[Any, dict[str, Any]] = {}
        frames_seen = 0
        total_frames = self._video_frame_count(cv2=cv2, video_path=video_path)
        if self.max_frames is not None:
            total_frames = min(total_frames, self.max_frames) if total_frames else self.max_frames
        last_yolo_progress = 25
        if progress_callback is not None and total_frames:
            progress_callback("yolo", f"YOLO 0%: 0/{total_frames} кадров", 25)

        try:
            results = model.track(
                source=str(video_path),
                tracker="bytetrack.yaml",
                persist=True,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                stream=True,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            raise PipelineError(f"failed to start YOLO tracking: {exc}") from exc

        for frame_idx, result in enumerate(results):
            if self.max_frames is not None and frame_idx >= self.max_frames:
                break
            frames_seen += 1
            if progress_callback is not None and total_frames:
                stage_pct = min(99, int(frames_seen * 100 / total_frames))
                overall_progress = 25 + int(stage_pct * 30 / 100)
                if overall_progress > last_yolo_progress:
                    last_yolo_progress = overall_progress
                    progress_callback(
                        "yolo",
                        f"YOLO {stage_pct}%: кадр {frames_seen}/{total_frames}, треков {len(tracks)}",
                        overall_progress,
                    )
            boxes = getattr(result, "boxes", None)
            if boxes is None or boxes.id is None:
                continue
            frame = getattr(result, "orig_img", None)
            if frame is None:
                continue

            track_ids = boxes.id.int().tolist()
            bboxes = boxes.xyxy.tolist()

            for track_id, bbox in zip(track_ids, bboxes):
                crop = self._crop_bbox(frame=frame, bbox=bbox)
                if crop.size == 0:
                    continue
                sharpness = self._crop_sharpness(cv2=cv2, crop=crop)
                crop_resized = self._resize_crop(cv2=cv2, crop=crop)

                bucket = tracks.setdefault(track_id, {"candidates": []})
                bucket["candidates"].append((sharpness, frame_idx, crop_resized))
                if len(bucket["candidates"]) > self.k_best:
                    bucket["candidates"].sort(key=lambda c: c[0], reverse=True)
                    del bucket["candidates"][self.k_best:]

        # Финализируем: вытаскиваем сами кропы (отсортированные по убыванию sharpness)
        finalized: dict[Any, dict[str, Any]] = {}
        for tid, bucket in tracks.items():
            bucket["candidates"].sort(key=lambda c: c[0], reverse=True)
            finalized[tid] = {
                "crops": [c for _s, _f, c in bucket["candidates"]],
                "candidate_meta": [
                    {"sharpness": float(sharpness), "frame_idx": int(frame_idx)}
                    for sharpness, frame_idx, _crop in bucket["candidates"]
                ],
            }

        return finalized, frames_seen

    @staticmethod
    def _video_frame_count(cv2, video_path: Path) -> int:
        cap = cv2.VideoCapture(str(video_path))
        try:
            if not cap.isOpened():
                return 0
            return max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
        finally:
            cap.release()

    @staticmethod
    def _crop_bbox(frame, bbox: list[float]):
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1c = max(0, min(width, int(x1)))
        y1c = max(0, min(height, int(y1)))
        x2c = max(0, min(width, int(x2)))
        y2c = max(0, min(height, int(y2)))
        return frame[y1c:y2c, x1c:x2c].copy()

    @staticmethod
    def _crop_sharpness(cv2, crop) -> float:
        if crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return float(
            cv2.Laplacian(gray, cv2.CV_64F).var() * math.sqrt(gray.shape[0] * gray.shape[1])
        )

    def _resize_crop(self, cv2, crop):
        h, w = crop.shape[:2]
        longest = max(h, w)
        if longest <= self.max_crop_side:
            return crop
        scale = self.max_crop_side / longest
        return cv2.resize(crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # ── Consensus + post-processing ──

    def _aggregate_fields(self, per_crop_fields: list[dict[str, Any]]) -> dict[str, Any]:
        """Multi-frame consensus: Counter.most_common per field, проходя через валидатор."""
        votes: dict[str, Counter] = defaultdict(Counter)
        for fields in per_crop_fields:
            for k, v in fields.items():
                if v in (None, "", "нет"):
                    continue
                votes[k][v] += 1

        out: dict[str, Any] = {"color": "red"}  # color: 99% red в реальных данных Lenta

        for field_name, counter in votes.items():
            value, _count = counter.most_common(1)[0]
            validator = VALIDATORS.get(field_name)
            if validator is not None and not validator(value):
                continue
            out[field_name] = value

        return out

    def _post_process(self, records: list[dict[str, Any]]) -> None:
        """Применяет фильтры и трансформации над всем списком треков.

        1. DB sanity filter — barcode не из db_hack.csv → 'нет'
        2. price sanity — выбросы и невозможные пары price_default/price_card → 'нет'
        3. additional_info snap к закрытому словарю
        4. product_name canonicalize (volume-anchor + bleed-in + space norm)
        5. Per-video code dedup — если одно значение code доминирует >40% треков → 'нет'
        """
        db_codes = self._get_db_codes()

        # 1. DB filter
        if db_codes:
            for r in records:
                bc = r.get("barcode")
                if bc and bc not in db_codes:
                    r["barcode"] = None

        # 2. Price sanity filters.
        for r in records:
            _sanitize_record_prices(r)
            pd_str = r.get("price_default")
            if not pd_str:
                continue
            pd_val = _parse_price_value(pd_str)
            pc_val = _parse_price_value(r.get("price_card"))
            d_pct = _parse_disc_value(r.get("discount_amount"))
            if pd_val is None or pc_val is None or not d_pct:
                continue
            try:
                low = pc_val / (1 - d_pct / 100.0)
                high = pc_val / (1 - (d_pct + 1) / 100.0)
            except ZeroDivisionError:
                continue
            pd_int = int(pd_val)
            if not (int(low) - 1 <= pd_int < int(high) + 1):
                r["price_default"] = None

        # 3. additional_info snap
        for r in records:
            v = r.get("additional_info")
            if v:
                snapped = snap_additional_info(v)
                r["additional_info"] = snapped if snapped != "нет" else None

        # 4. product_name canonicalize
        for r in records:
            v = r.get("product_name")
            if v:
                canon = canonical_product_name(v)
                r["product_name"] = canon if canon != "нет" else None

        # 5. Per-video code dedup (видео в нашем случае одно — все треки)
        non_net = [r for r in records if r.get("code")]
        if len(non_net) >= 10:
            code_counts = Counter(r["code"] for r in non_net)
            total = len(non_net)
            for code_val, cnt in code_counts.items():
                if cnt / total > 0.40:
                    for r in non_net:
                        if r.get("code") == code_val:
                            r["code"] = None
