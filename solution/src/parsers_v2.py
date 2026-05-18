"""Улучшенные парсеры для solution_v2.

Импортируется в notebook'ах solution_v2. Содержит fix'ы которые НЕ возвращаем в
eda/05_parsers.ipynb чтобы baseline остался воспроизводимым.

Зависимости (нужны в globals при импорте):
- OcrLine, SUPER_TRANS, _extract_integer, _try_inline_price, cx, cy, _SOURCE_PRIORITY, _fmt_price_ru
  (загружаются из eda/05_parsers.ipynb)
"""
import re

# ── Tier 1.2: product_name fuzzy normalization ────────────────────────────────
# product_name strict (raw): 0%, после Latin→Cyrillic fold + lowercase + punct-strip:
#   strict 1.3%, fuzz_token_set ≥80: 14.7%, ≥85: 10.9%.
# Organizer FAQ: "не обязательно символ-в-символ" — почти точно fuzzy match.

LATIN_TO_CYR = str.maketrans({
    # Заглавные
    'A': 'А', 'B': 'В', 'C': 'С', 'E': 'Е', 'H': 'Н', 'K': 'К', 'M': 'М',
    'O': 'О', 'P': 'Р', 'T': 'Т', 'X': 'Х', 'Y': 'У',
    # Строчные
    'a': 'а', 'c': 'с', 'e': 'е', 'o': 'о', 'p': 'р', 'x': 'х', 'y': 'у',
})


def normalize_product_name(s):
    """Latin→Cyrillic homoglyph fold + lowercase + punct strip + collapse spaces.
    Подходит для сравнения с GT через равенство/fuzzy match."""
    if not s or str(s).strip().lower() in ('', 'нет', 'none'):
        return ''
    s = str(s).translate(LATIN_TO_CYR).lower()
    s = re.sub(r'[^a-zа-я0-9 ]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


# ── Tier 1.3a: special_symbols better detection ──────────────────────────────
# Baseline: 34% strict (matches только если OCR-line = ровно 'К'/'Ш'). GT: 75 К, 52 Ш, 30 нет.
# Идея: разрешаем 1-2 char токены содержащие К/Ш в верхнем регистре, ИЛИ Latin K
# с привязкой к мелкому шрифту маркера + не в составе слова.

_K_SET = {'К', 'к', 'K', 'k'}
_SH_SET = {'Ш', 'ш'}

def parse_special_symbols_v2(lines):
    """Ищет маленький изолированный токен К или Ш."""
    if not lines:
        return None
    # Шаг 1: строгий — целая строка ровно K/Ш
    for L in lines:
        s = L.text.strip()
        if s in _K_SET: return 'К'
        if s in _SH_SET: return 'Ш'
    # Шаг 2: мягкий — короткие 1-3 char токены, мелкие, содержат только К/Ш + шум
    cands = []
    for L in lines:
        s = L.text.strip()
        if not s or len(s) > 3: continue
        # Только если уверенно К или Ш
        clean = re.sub(r'[^КкKkШш]', '', s)
        if clean in _K_SET: cands.append(('К', L))
        elif clean in _SH_SET: cands.append(('Ш', L))
    if cands:
        # Выбираем самый мелкий (markers — маленькие)
        sym, _ = min(cands, key=lambda c: c[1].h)
        return sym
    return None


# ── Tier 1.3b: discount tolerance for OCR digit confusion ────────────────────
# Baseline: 68% strict. Добавляем tolerance для O→0, l→1, b→6.

_DISCOUNT_RE_V2 = re.compile(r'[-−–—]?\s*([0-9OoIlbBg]{1,3})\s*%')
_DIGIT_FIX = str.maketrans({'O':'0', 'o':'0', 'I':'1', 'l':'1', 'b':'6', 'g':'9', 'B':'8'})

def parse_discount_amount_v2(lines):
    """v1 + tolerance для путаниц O→0, l→1 в цифрах скидки."""
    best, best_h = None, 0
    for L in lines:
        m = _DISCOUNT_RE_V2.search(L.text)
        if not m: continue
        raw = m.group(1).translate(_DIGIT_FIX)
        if not raw.isdigit(): continue
        n = int(raw)
        if 1 <= n <= 99 and L.h > best_h:
            best, best_h = n, L.h
    return f'-{best}%' if best else None


# ── Tier 1.3c: additional_info compound detection ────────────────────────────
# Baseline: 61% strict. GT имеет составные значения "Сухое, 3 по цене 2 ...".
# Сейчас парсер возвращает первый матч — мы можем терять составные значения.

_PRIMARY_HINTS = ('Полусладкое', 'Полусухое', 'Сладкое', 'Сухое',
                  'Игристое', 'Шипучее', 'Десертное')
_PROMO_HINTS = ('по цене', 'Удачная упаковка', 'доп. скидка')


def product_name_match(pred, gt_value, threshold=80):
    """Возвращает True если pred ≈ gt_value (fuzz_token_set_ratio ≥ threshold
    на нормализованных строках) ИЛИ оба пусты ('нет')."""
    from rapidfuzz import fuzz
    p_norm = normalize_product_name(pred)
    g_norm = normalize_product_name(gt_value)
    if not p_norm and not g_norm:
        return True
    if not p_norm or not g_norm:
        return False
    return fuzz.token_set_ratio(p_norm, g_norm) >= threshold

# ── Tier 1.1: price_default fix ────────────────────────────────────────────────
# Базовый baseline parse_prices: 1 strict price_default из 124 матчей (0.8%)
# Этот v2: 7/124 = 5.6% (×7 improvement). Логика:
#   1. Расширенное окно integer+decimal пары: max(REL, ABS_MIN_PX)
#   2. Если для price_default только int-only fallback → форсим ,99 (а не ,00)
#      (в GT 13% price_default имеют ,99 копейки, остальные разнообразны;
#       ,00 практически никогда — было ~0% strict; форсинг ,99 даёт ~13% на тех ценниках)

_MIN_DX_PX = 80     # абсолютный минимум окна по X для pair (для мелкого шрифта price_default)
_MIN_DY_PX = 35

def _candidate_prices_v2(lines):
    """v2 кандидаты: расширенное spatial окно + int-only только для крупных шрифтов."""
    cands = []

    # 1) Inline (целая цена на одной строке)
    for L in lines:
        v = _try_inline_price(L.text)
        if v is not None:
            cands.append((v, L, 'inline'))

    # 2) Integers + 2-digit decimals
    integers = [(v, L) for L in lines if (v := _extract_integer(L.text)) is not None]
    decimals = []
    for L in lines:
        s = L.text.translate(SUPER_TRANS).strip()
        m = re.fullmatch(r'[^\d]*(\d{2})[^\d]*', s)
        if m:
            decimals.append((int(m.group(1)), L))

    # 3) Pair integer + decimal с расширенным окном
    for v_int, I in integers:
        best, best_d = None, float('inf')
        dx_thresh = max(2.5 * I.w, _MIN_DX_PX)
        dy_thresh = max(1.5 * I.h, _MIN_DY_PX)
        for d_val, D in decimals:
            if D is I: continue
            dx = cx(D) - cx(I)
            dy = cy(D) - cy(I)
            if abs(dx) > dx_thresh or abs(dy) > dy_thresh: continue
            if dx < -0.3 * I.w: continue
            d = abs(dx) + 2.0 * abs(dy)
            if d < best_d:
                best, best_d = d_val, d
        if best is not None:
            cands.append((v_int + best / 100.0, I, 'pair'))

    # 4) Int-only fallback для всех значений ≥100
    for v_int, I in integers:
        if v_int >= 100:
            cands.append((float(v_int), I, 'int-only'))
    return cands


def parse_prices_v2(lines):
    """price_card (всегда ,99) + price_default (агрессивный ,99 fallback для int-only)."""
    out = {'price_default': None, 'price_card': None}
    cs = _candidate_prices_v2(lines)
    if not cs:
        return out

    best_by_line = {}
    for v, L, src in cs:
        key = id(L)
        if key not in best_by_line or _SOURCE_PRIORITY[src] < _SOURCE_PRIORITY[best_by_line[key][2]]:
            best_by_line[key] = (v, L, src)
    ordered = sorted(best_by_line.values(), key=lambda c: -c[1].h)

    # price_card = крупная, ,99 (97% GT именно такие)
    if ordered:
        out['price_card'] = _fmt_price_ru(int(ordered[0][0]) + 0.99)

    # price_default = следующая отличающаяся
    if len(ordered) > 1:
        card_int = int(ordered[0][0])
        for v, L, src in ordered[1:]:
            v_int = int(v)
            if v_int == card_int: continue
            if abs(v_int - card_int) / max(card_int, 1) <= 0.03: continue
            if src == 'int-only':
                # Force ,99 — в GT 13% price_default такие, против ~0% с ,00
                out['price_default'] = _fmt_price_ru(v_int + 0.99)
            else:
                out['price_default'] = _fmt_price_ru(v)
            break
    return out
