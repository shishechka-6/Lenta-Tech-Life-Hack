from __future__ import annotations

import csv
import io
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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


class PriceTagProcessor:
    """Thin service wrapper around the current detection stage.

    The notebook pipeline already contains OCR and field parsers, but it is not yet
    packaged as importable production code. This class gives the backend a stable
    service contract today: one uploaded video in, submission-shaped rows out.
    """

    def __init__(
        self,
        model_path: Path,
        device: str,
        conf: float,
        iou: float,
        imgsz: int,
        max_frames: int | None,
    ) -> None:
        self.model_path = model_path
        self.device = _select_device(device)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.max_frames = max_frames
        self._model = None
        self._field_extractor: FieldExtractor | None = None

    def process(self, video_path: Path, original_filename: str | None = None) -> ProcessingResult:
        start = time.perf_counter()
        video_id = Path(original_filename or video_path.name).stem
        tracks, frames_seen = self._track_price_tags(video_path)

        rows = []
        field_extractor = self._get_field_extractor()
        for track_id in sorted(tracks, key=_track_sort_key):
            row = _empty_submission_row(video_id=video_id, track_id=track_id)
            crop = tracks[track_id].get("crop")
            if crop is not None:
                row.update(field_extractor.extract(crop))
            rows.append(row)

        elapsed = time.perf_counter() - start
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

    def _load_model(self):
        if self._model is not None:
            return self._model
        if not self.model_path.exists():
            raise PipelineError(f"YOLO weights not found: {self.model_path}")

        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise PipelineError(
                "ultralytics is not installed; install services/backend/requirements.txt"
            ) from exc

        self._model = YOLO(str(self.model_path))
        return self._model

    def _get_field_extractor(self) -> "FieldExtractor":
        if self._field_extractor is None:
            self._field_extractor = FieldExtractor()
        return self._field_extractor

    def _track_price_tags(self, video_path: Path) -> tuple[dict[Any, dict[str, Any]], int]:
        try:
            import cv2
        except Exception as exc:
            raise PipelineError(
                "opencv-python-headless is not installed; install backend requirements"
            ) from exc

        model = self._load_model()
        tracks: dict[Any, dict[str, Any]] = {}
        frames_seen = 0

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
                sharpness = self._crop_sharpness(cv2=cv2, crop=crop)
                current = tracks.get(track_id)
                if current is None or sharpness > current["sharpness"]:
                    tracks[track_id] = {
                        "sharpness": sharpness,
                        "frame_idx": frame_idx,
                        "bbox": bbox,
                        "crop": crop,
                    }

        return tracks, frames_seen

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
        return float(cv2.Laplacian(gray, cv2.CV_64F).var() * math.sqrt(gray.shape[0] * gray.shape[1]))


class OcrLine(tuple):
    __slots__ = ()

    def __new__(cls, text: str, x: float, y: float, w: float, h: float, conf: float):
        return tuple.__new__(cls, (text, x, y, w, h, conf))

    @property
    def text(self) -> str:
        return self[0]

    @property
    def x(self) -> float:
        return self[1]

    @property
    def y(self) -> float:
        return self[2]

    @property
    def w(self) -> float:
        return self[3]

    @property
    def h(self) -> float:
        return self[4]

    @property
    def conf(self) -> float:
        return self[5]


def _cx(line: OcrLine) -> float:
    return line.x + line.w / 2


def _cy(line: OcrLine) -> float:
    return line.y + line.h / 2


def _ean13_checksum_ok(value: str) -> bool:
    digits = re.sub(r"\D", "", value)
    if len(digits) != 13:
        return False
    parsed = [int(char) for char in digits]
    parity = sum(v * (3 if i % 2 else 1) for i, v in enumerate(parsed[:-1]))
    return (10 - parity % 10) % 10 == parsed[-1]


def _normalize_missing(value: Any) -> str:
    if value is None:
        return "нет"
    value = str(value).strip()
    return value if value else "нет"


class FieldExtractor:
    def __init__(self) -> None:
        self._ocr = self._load_ocr()
        self._zxing = self._load_zxing()
        self._qr_detector = None

    def extract(self, crop) -> dict[str, str]:
        fields: dict[str, str] = {}
        fields["color"] = self._detect_color(crop)

        qr_fields = self._decode_qr(crop)
        fields.update(qr_fields)

        barcode = self._decode_barcode_1d(crop)
        lines = self._ocr_lines(crop)
        if not barcode:
            barcode = self._parse_barcode(lines)

        parsed = {
            "product_name": self._build_product_name(lines),
            "barcode": barcode,
            "id_sku": self._parse_id_sku(lines, barcode),
            "print_datetime": self._parse_print_datetime(lines),
            "discount_amount": self._parse_discount_amount(lines),
            "additional_info": self._parse_additional_info(lines),
            "special_symbols": self._parse_special_symbols(lines),
        }
        parsed.update(self._parse_prices(lines))

        for key, value in parsed.items():
            if key in SUBMISSION_COLUMNS:
                fields[key] = _normalize_missing(value)

        return {key: value for key, value in fields.items() if value != "нет"}

    @staticmethod
    def _load_ocr():
        try:
            from paddleocr import PaddleOCR
        except Exception:
            return None

        try:
            return PaddleOCR(lang="ru", use_textline_orientation=True)
        except TypeError:
            return PaddleOCR(lang="ru", use_angle_cls=True)
        except Exception:
            return None

    @staticmethod
    def _load_zxing():
        try:
            import zxingcpp
        except Exception:
            return None
        return zxingcpp

    def _ocr_lines(self, crop) -> list[OcrLine]:
        if self._ocr is None or crop is None or crop.size == 0:
            return []
        try:
            if hasattr(self._ocr, "predict"):
                result = self._ocr.predict(crop)
            else:
                result = self._ocr.ocr(crop)
        except Exception:
            return []
        return self._normalize_ocr_result(result)

    @staticmethod
    def _normalize_ocr_result(result) -> list[OcrLine]:
        if not result:
            return []

        first = result[0]
        lines: list[OcrLine] = []
        if isinstance(first, dict):
            texts = first.get("rec_texts", [])
            scores = first.get("rec_scores", [])
            boxes = first.get("rec_boxes", first.get("rec_polys", []))
            for text, score, box in zip(texts, scores, boxes):
                normalized = FieldExtractor._line_from_box(text, score, box)
                if normalized:
                    lines.append(normalized)
            return lines

        for item in first or []:
            try:
                box, text_score = item[0], item[1]
                text, score = text_score[0], text_score[1]
            except Exception:
                continue
            normalized = FieldExtractor._line_from_box(text, score, box)
            if normalized:
                lines.append(normalized)
        return lines

    @staticmethod
    def _line_from_box(text: str, score: float, box) -> OcrLine | None:
        try:
            points = list(box)
            if len(points) == 4 and not isinstance(points[0], (list, tuple)):
                x1, y1, x2, y2 = [float(value) for value in points]
                return OcrLine(str(text), x1, y1, x2 - x1, y2 - y1, float(score))
            xs = [float(point[0]) for point in points]
            ys = [float(point[1]) for point in points]
            x1, y1 = min(xs), min(ys)
            return OcrLine(str(text), x1, y1, max(xs) - x1, max(ys) - y1, float(score))
        except Exception:
            return None

    def _decode_barcode_1d(self, crop) -> str | None:
        if self._zxing is None or crop is None or crop.size == 0:
            return None
        try:
            results = self._zxing.read_barcodes(crop)
        except Exception:
            return None
        for item in results:
            text = getattr(item, "text", "")
            digits = re.sub(r"\D", "", text)
            if len(digits) == 13 and _ean13_checksum_ok(digits):
                return digits
            if len(digits) in (8, 12):
                return digits
        return None

    def _decode_qr(self, crop) -> dict[str, str]:
        try:
            import cv2
        except Exception:
            return {}
        if crop is None or crop.size == 0:
            return {}
        if self._qr_detector is None:
            self._qr_detector = cv2.QRCodeDetector()

        payloads: list[str] = []
        try:
            data, _, _ = self._qr_detector.detectAndDecode(crop)
            if data:
                payloads.append(data)
        except Exception:
            pass

        height, width = crop.shape[:2]
        for scale in (2, 3, 4):
            try:
                resized = cv2.resize(crop, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC)
                data, _, _ = self._qr_detector.detectAndDecode(resized)
                if data:
                    payloads.append(data)
            except Exception:
                continue

        for payload in payloads:
            parsed = self._parse_qr_payload(payload)
            if parsed:
                return parsed
        return {}

    @staticmethod
    def _parse_qr_payload(payload: str) -> dict[str, str]:
        key_map = {
            "b": "qr_code_barcode",
            "barcode": "qr_code_barcode",
            "p1": "price1_qr",
            "price1": "price1_qr",
            "p2": "price2_qr",
            "price2": "price2_qr",
            "p3": "price3_qr",
            "price3": "price3_qr",
            "p4": "price4_qr",
            "price4": "price4_qr",
            "wL1C": "wholesale_level_1_count",
            "wholesaleLevel1Count": "wholesale_level_1_count",
            "wL1P": "wholesale_level_1_price",
            "wholesaleLevel1Price": "wholesale_level_1_price",
            "wL2C": "wholesale_level_2_count",
            "wholesaleLevel2Count": "wholesale_level_2_count",
            "wL2P": "wholesale_level_2_price",
            "wholesaleLevel2Price": "wholesale_level_2_price",
            "aP": "action_price_qr",
            "actionPrice": "action_price_qr",
            "aC": "action_code_qr",
            "actionCode": "action_code_qr",
        }
        out: dict[str, str] = {}
        for part in re.split(r"[&;,\n]", payload):
            match = re.match(r"\s*([A-Za-z0-9_]+)\s*[=:]\s*(.*?)\s*$", part)
            if not match:
                continue
            field = key_map.get(match.group(1))
            if field:
                out[field] = _normalize_missing(match.group(2))
        return out

    @staticmethod
    def _detect_color(crop) -> str:
        try:
            import cv2
        except Exception:
            return "нет"
        if crop is None or crop.size == 0:
            return "нет"
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        red1 = cv2.inRange(hsv, (0, 70, 50), (12, 255, 255))
        red2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
        yellow = cv2.inRange(hsv, (18, 55, 80), (42, 255, 255))
        red_score = int((red1 > 0).sum() + (red2 > 0).sum())
        yellow_score = int((yellow > 0).sum())
        if red_score < 50 and yellow_score < 50:
            return "нет"
        return "yellow" if yellow_score > red_score else "red"

    @staticmethod
    def _parse_barcode(lines: list[OcrLine]) -> str | None:
        candidates: list[tuple[str, float]] = []
        for line in lines:
            for digits in re.findall(r"\d{8,14}", re.sub(r"\D", "", line.text)):
                valid_bonus = 2.0 if len(digits) == 13 and _ean13_checksum_ok(digits) else 0.0
                candidates.append((digits, len(digits) + line.conf + valid_bonus))
        if not candidates:
            return None
        candidates.sort(key=lambda item: -item[1])
        return candidates[0][0]

    @staticmethod
    def _parse_id_sku(lines: list[OcrLine], barcode: str | None) -> str | None:
        if not lines:
            return None
        height = max(line.y + line.h for line in lines)
        candidates: list[tuple[str, float]] = []
        for line in lines:
            if _cy(line) / max(height, 1) < 0.3:
                continue
            for digits in re.findall(r"\d{8,12}", re.sub(r"\D", "", line.text)):
                if barcode and (digits == barcode or digits in barcode or barcode in digits):
                    continue
                candidates.append((digits, line.conf + _cy(line) / max(height, 1)))
        if not candidates:
            return None
        candidates.sort(key=lambda item: -item[1])
        return candidates[0][0]

    @staticmethod
    def _parse_print_datetime(lines: list[OcrLine]) -> str | None:
        digit_fix = str.maketrans({"З": "3", "з": "3", "O": "0", "o": "0", "О": "0", "о": "0", "I": "1", "l": "1", "B": "8", "В": "8"})
        full_re = re.compile(r"(\d{2})[.\-/](\d{2})[.\-/](\d{4})[\D]+(\d{1,2})[:.]\s*(\d{2})")
        date_re = re.compile(r"(\d{2})[.\-/](\d{2})[.\-/](\d{4})")
        for line in lines:
            text = line.text.translate(digit_fix)
            match = full_re.search(text)
            if match:
                day, month, year, hour, minute = match.groups()
                return f"{day}.{month}.{year} {int(hour)}:{minute}"
        for line in lines:
            text = line.text.translate(digit_fix)
            match = date_re.search(text)
            if match:
                day, month, year = match.groups()
                return f"{day}.{month}.{year}"
        return None

    @staticmethod
    def _parse_discount_amount(lines: list[OcrLine]) -> str | None:
        best: tuple[int, float] | None = None
        for line in lines:
            match = re.search(r"[-−–—]?\s*(\d{1,2})\s*%", line.text)
            if match and 1 <= int(match.group(1)) <= 99:
                if best is None or line.h > best[1]:
                    best = (int(match.group(1)), line.h)
        return f"-{best[0]}%" if best else None

    @staticmethod
    def _parse_prices(lines: list[OcrLine]) -> dict[str, str | None]:
        price_re = re.compile(r"(\d{1,5})[,.\s]+(\d{2})\b")
        candidates: list[tuple[float, OcrLine]] = []
        for line in lines:
            text = line.text.translate(str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789"))
            if re.search(r"\d{1,2}\s*%", text):
                continue
            match = price_re.search(text)
            if match:
                value = int(match.group(1)) + int(match.group(2)) / 100
                if value >= 10:
                    candidates.append((value, line))
                    continue
            runs = re.findall(r"\d+", text)
            if runs:
                value = int(max(runs, key=len))
                if value >= 100:
                    candidates.append((float(value), line))

        if not candidates:
            return {"price_default": None, "price_card": None}

        ordered = sorted(candidates, key=lambda item: -item[1].h)
        card = ordered[0][0]
        out = {"price_default": None, "price_card": f"{int(card) + 0.99:.2f}".replace(".", ",")}
        card_int = int(card)
        for value, _line in ordered[1:]:
            if int(value) != card_int:
                out["price_default"] = f"{value:.2f}".replace(".", ",")
                break
        return out

    @staticmethod
    def _build_product_name(lines: list[OcrLine]) -> str | None:
        if not lines:
            return None
        height = max(line.y + line.h for line in lines)
        top = [
            line
            for line in lines
            if _cy(line) < height * 0.55
            and any(char.isalpha() for char in line.text)
            and not re.search(r"\d{1,5}[,.\s]\d{2}", line.text)
            and "%" not in line.text
        ]
        top.sort(key=lambda line: (line.y, line.x))
        name = " ".join(line.text.strip() for line in top if len(line.text.strip()) >= 2)
        return name if len(name) >= 5 else None

    @staticmethod
    def _parse_additional_info(lines: list[OcrLine]) -> str | None:
        hints = ("Полусладкое", "Полусухое", "Сладкое", "Сухое", "Игристое", "Шипучее", "Десертное", "по цене", "Удачная упаковка")
        text = " ".join(line.text for line in lines).lower()
        for hint in hints:
            if hint.lower() in text:
                return hint
        return None

    @staticmethod
    def _parse_special_symbols(lines: list[OcrLine]) -> str | None:
        for line in lines:
            token = line.text.strip()
            if token in {"К", "к", "K", "k"}:
                return "К"
            if token in {"Ш", "ш"}:
                return "Ш"
        return None
