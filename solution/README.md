# Lenta Tech Life Hack — Решение

Автоматическое распознавание ценников супермаркета «Лента» по 4K-видео полок.

## Задача

**Input:** MP4 (3840×2160, ~20 fps) — робот проезжает вдоль стеллажей.
**Output:** CSV — 1 строка на уникальный ценник × 25 полей (название товара, цены, штрих-код, артикул, дата печати и др.).

**Главная метрика:** % ценников, где ≥80% полей распознаны правильно.

## Архитектура

```
Видео 4K (.mp4)
    │
    ▼
[1] YOLO11n              → bbox ценников на каждом кадре         notebooks/01_prepare_dataset.ipynb
    │                      (обучение на Kaggle GPU T4×2)         notebooks/02_train_yolo.ipynb
    ▼
[2] ByteTrack            → треки: один ценник = один track_id    notebooks/03_track_and_extract.ipynb
    │
    ▼
[3] Top-K keyframes      → K=10 самых резких кадров на трек      data/best_crops/<vid>/<tid>_<rank>.jpg
    │                      (Laplacian.var × √area)
    ▼
[4] Multi-frame OCR      → PaddleOCR + zxing + WeChat QR         notebooks/04_ocr.ipynb
    │                      на каждом из K кропов трека            → data/ocr_cache.json
    │
    ▼
[5] VLM (Qwen2.5-VL-7B)  → 4 поля мелкого шрифта (date/sku/      notebooks/05_vlm_kaggle.ipynb
    │                      barcode/code) через AI                  → data/vlm_cache_v3.json
    │                      (на Kaggle GPU T4×2)
    ▼
[6] Парсеры (v1 + v2)    → 25 полей: цены, дата, barcode, ...   notebooks/06_parsers.ipynb (v1)
    │                      Latin→Cyrillic нормализация            src/parsers_v2.py (v2 fixes)
    ▼
[7] Hybrid merge         → per-field source routing:              notebooks/07_main_pipeline.ipynb
    │                      VLM v3 → code/id_sku/barcode           ← entry point
    │                      VLM v2 → print_datetime
    │                      Classical → prices, product_name, ...
    │
    ▼
[8] Plausibility-фильтр  → отсев VLM-галлюцинаций:                (cell 5 в 07_main_pipeline)
    │                      - explicit blocklist (~80 значений)
    │                      - valid_*() гейты (EAN-13 checksum, ...)
    │
    ▼
[9] Post-processing      → QR-mirror, code dedup, addinfo snap     (cells 7-9 в 07_main_pipeline)
    │                      product_name canonicalization
    ▼
submission_v8.csv        → 1 строка на трек × 29 колонок          data/submission_v8.csv
```

## Структура папки

```
solution/
├── README.md
├── requirements.txt
├── sample.csv                       # пример submission от организаторов
├── notebooks/
│   ├── 01_prepare_dataset.ipynb     # подготовка YOLO-датасета
│   ├── 02_train_yolo.ipynb          # обучение YOLO (Kaggle)
│   ├── 03_track_and_extract.ipynb   # YOLO + ByteTrack → crops
│   ├── 04_ocr.ipynb                 # PaddleOCR + zxing + WeChat
│   ├── 05_vlm_kaggle.ipynb          # Qwen2.5-VL inference (Kaggle)
│   ├── 06_parsers.ipynb             # парсеры (импортируется из 07)
│   └── 07_main_pipeline.ipynb       # entry point — финальная сборка
├── src/
│   ├── parsers_v2.py                # v2 парсеры (импортируется из 07)
│   └── bytetrack_v2.yaml            # конфиг трекера
├── data/                            # кэши, GT, итоговый submission
│   ├── ocr_cache.json               # кэш PaddleOCR + zxing + QR (2.6 MB)
│   ├── vlm_cache_v2.json            # старый прогон Qwen2.5-VL
│   ├── vlm_cache_v3.json            # новый прогон Qwen2.5-VL
│   ├── best_crops_meta.json         # метаданные кропов (1.5 MB)
│   ├── per_frame_tracks.json        # треки по кадрам (6.5 MB)
│   ├── crops_meta.json              # GT-разметка
│   ├── submission_v8.csv            # ← итоговая submission
│   ├── eval_report_v8.json          # метрики
│   └── yolo/dataset.yaml            # конфиг для обучения YOLO
└── runs/tags_v1/weights/best.pt     # обученные веса YOLO11n (11 MB)
```

**Скачать отдельно с Google Drive** (тяжёлые файлы, ~1 GB):

| Папка | Размер | Куда положить | Ссылка |
|---|---|---|---|
| `Данные/` | 388 MB | `solution/Данные/` | `TODO: вставить ссылку на GDrive` |
| `Khasan_Dataset/` | 495 MB | `solution/Khasan_Dataset/` | `TODO: вставить ссылку на GDrive` |
| `best_crops/` | 138 MB | `solution/data/best_crops/` | `TODO: вставить ссылку на GDrive` |

> `Данные/` — исходные видео + GT от организаторов.
> `Khasan_Dataset/` — наша ручная разметка для обучения YOLO (301 кадр, Roboflow).
> `best_crops/` — вырезанные ценники с top-K keyframes (можно пересоздать запуском `03_track_and_extract.ipynb`).

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Все зависимости — чистый pip, без brew / системных библиотек.

## Воспроизведение результата

### Быстрый путь (только финальный шаг, ~15 секунд)

Все кэши уже лежат в `data/`. Достаточно прогнать только финальный ноутбук:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/07_main_pipeline.ipynb
```

→ `data/submission_v8.csv` + `data/eval_report_v8.json`.

### Полный путь (с нуля)

Перед запуском нужно скачать тяжёлые папки с Google Drive (см. секцию выше) и положить в `solution/`.

| # | Ноутбук | Где запускать | Время | Что генерит |
|---|---|---|---|---|
| 1 | `01_prepare_dataset.ipynb` | локально | 2 мин | `data/yolo/` |
| 2 | `02_train_yolo.ipynb` | **Kaggle GPU T4×2** | ~10 мин | `runs/tags_v1/weights/best.pt` |
| 3 | `03_track_and_extract.ipynb` | локально | ~5 мин | `data/best_crops/*`, `best_crops_meta.json` |
| 4 | `04_ocr.ipynb` | локально | ~65 мин (первый), ~30 сек (с кэшем) | `data/ocr_cache.json` |
| 5 | `05_vlm_kaggle.ipynb` | **Kaggle GPU T4×2** | ~5 часов | `data/vlm_cache_v3.json` |
| 7 | `07_main_pipeline.ipynb` | локально | <1 мин | `submission_v8.csv` |

(Ноутбук `06_parsers.ipynb` не запускается отдельно — его содержимое загружается ноутбуком `07` через `nbformat`.)

### Запуск VLM на Kaggle

1. Импортируй `notebooks/05_vlm_kaggle.ipynb` на kaggle.com
2. Прикрепи как Kaggle Dataset: папку `data/best_crops/` + файл `data/best_crops_meta.json`
3. Settings → Accelerator → **GPU T4 × 2** (32 GB VRAM суммарно — нужно для Qwen2.5-VL-7B FP16)
4. Run All → дождаться завершения (~5 часов)
5. Скачать `/kaggle/working/vlm_cache_v3.json` → положить в `data/`

Ноутбук поддерживает **resume**: если `vlm_cache_v3.json` уже частично заполнен, при перезапуске пропускает обработанные треки.

## Результаты

### Главная метрика

| Режим | Прошло | Из | % |
|---|---|---|---|
| **Strict (символ-в-символ)** | 3 | 157 | 1.9% |
| **Fuzzy (как у организаторов, для product_name)** | 3 | 157 | 1.9% |
| Bucket 70-80% strict (почти прошли) | 23 | 157 | 15% |
| Bucket 70-80% fuzzy (почти прошли) | 35 | 157 | 22% |

### Per-field accuracy (на 124 matched ценниках)

| Поле | Strict | Замечание |
|---|---|---|
| price_discount, action_*, price3_qr | 100% | бесплатные match-ы (GT='нет') |
| color | 99% | хардкод `red` |
| wholesale_level_*_count/price | 99% | бесплатные match-ы |
| discount_amount | 68% | крупный шрифт, multi-frame consensus |
| **price4_qr** | **66%** | QR-mirror: copies `price_card` (GT-аналитика: эти поля равны в 95%) |
| price_card | 64% | Lenta-конвенция `,99` |
| additional_info | 62% | snap к закрытому словарю (Сухое/Полусухое/…) |
| special_symbols (К/Ш) | 39% | детект одиночного токена в OCR |
| **print_datetime** | **25%** | VLM v2 + blocklist (было 6% без VLM) |
| **code** | **25%** | VLM v3 + расширенный validator + per-video dedup (было 13%) |
| id_sku | 7% | мелкий шрифт |
| price_default | 6% | копейки в superscript |
| price2_qr | 6% | partial QR-mirror |
| price1_qr | 5% | partial QR-mirror |
| barcode | 4% | мелкий шрифт + zxing на 4K-кропе |
| qr_code_barcode | 2% | mirror от barcode |
| product_name | 0% (strict) / 31% (fuzzy) | canonicalize+strip bleed-in; strict невозможен при OCR-ошибках |

### Capture rate

- **124/157 = 79%** — наш пайплайн вообще «нашёл» ценник
- Остальные 21% потеряны на стадии детекции/трекинга

## Ключевые решения

### Post-processing fixes (дали 2× прирост main metric)

Анализом GT обнаружили: **QR-поля = дубликаты обычных полей** (`price1_qr == price_default` 88%, `price4_qr == price_card` 95%, `qr_code_barcode == barcode` 99%). Раньше мы писали в них `'нет'`, потому что QR ~80×80 px физически не декодируется. Теперь после агрегации мирорим значения из обычных полей в QR. **price4_qr: 4.8% → 66.1% (+61pp)**.

Также:
- **Per-video code dedup**: если одно значение `code` доминирует >40% треков в видео (≥10 треков), это VLM-галлюцинация → `'нет'`
- **additional_info snap**: OCR-выход маппится в ближайший элемент закрытого словаря (`Сухое`, `Полусухое`, …) через `rapidfuzz.token_set_ratio`
- **product_name canonicalize**: обрезаем bleed-in после volume-anchor (`0.25L`), убираем хвосты типа `3 по цене 2` (которые принадлежат `additional_info`)

### Multi-frame consensus
На каждый трек собираем 10 самых резких кадров. PaddleOCR/zxing/WeChat прогоняются на всех 10, затем `Counter.most_common(1)` по каждому полю.

### Plausibility-валидаторы + blocklist
Перед записью в submission каждое значение проходит **двойную проверку**:
1. **Format validator** (`valid_barcode` с EAN-13 checksum, `valid_id_sku` 9-12 цифр, `valid_datetime` regex, `valid_code` для форматов `01_NNNNNN`, и др.)
2. **VLM blocklist** (~80 явных placeholder-значений типа `4000000000000`, `1234567890`, `01_025019`, которые Qwen2.5-VL стабильно галлюцинирует)

Невалидные значения → `'нет'`, что часто матчит GT `'нет'` (значительная доля полей в GT пустая).

### Hybrid per-field source routing
Разные источники лучше на разных полях:
- **VLM v3** (свежий промпт): `code`, `id_sku`, `barcode`
- **VLM v2** (старый промпт): `print_datetime` (v2 имела memorized 2026-даты, v3 их потерял)
- **Classical OCR** (PaddleOCR + regex): `price_default`, `price_card`, `discount_amount`, `product_name`, `additional_info`, `special_symbols`
- **zxing-cpp**: `barcode` (физическое декодирование штрих-кода, не OCR)
- **WeChat QR + cv2**: `qr_code_barcode`, `price*_qr`
- **Хардкод**: `color = 'red'` (99% точность — все ценники в датасете красные)

### Latin→Cyrillic нормализация
PaddleOCR часто путает кириллицу с латиницей-лукалайками (`Cyxoe` вместо `Сухое`). Перед fuzzy-matchем нормализуем `c→с`, `y→у`, `x→х`, `o→о` и т.д. Дала +10pp на `additional_info`.

## Стек

| Компонент | Технология |
|---|---|
| Детектор | YOLO11n (Ultralytics) |
| Трекер | ByteTrack |
| OCR | PaddleOCR PP-OCRv5 (`lang='ru'`) |
| Штрихкоды | zxing-cpp |
| QR | OpenCV WeChat QR + cv2.QRCodeDetector |
| VLM | Qwen2.5-VL-7B-Instruct (FP16, на Kaggle T4×2) |
| Видео | OpenCV |
| Среда | Python 3.12, macOS arm64 (M4) / Linux (CUDA) |

## Ограничения

- **OCR ceiling.** Источник 4K → ценник ~270×270 px → шрифт даты/SKU ~10-15 px. PaddleOCR требует ≥16-20 px. На этих полях strict ~5-30% физически.
- **QR-коды не декодируются.** QR на ценнике ~80×80 px = ~2.7 px/модуль (нужно ≥4). Ни WeChat, ни cv2.QRCodeDetector даже с апскейлом ×4 не справляются.
- **Capture rate 79%**, не 95%. На видео `26_12-20` — 59% (плотные полки, окклюзии).
- **product_name strict = 0%** не из-за алгоритма — точное совпадение длинной строки невозможно при OCR-ошибках (1 буква мимо = miss). Fuzzy на этом поле = 31%.
- **VLM-галлюцинации.** Qwen2.5-VL стабильно выдаёт placeholder-значения для нечитаемых полей. Blocklist+validator отсеивают, но мы теряем и единичные правильные значения, совпадающие с placeholder-ами.
