"""
image_processor.py — עיבוד קובץ הקלט לתמונות מוכנות לתמלול

האחריות של הקובץ הזה:
1. לקבל נתיב לקובץ (PDF / JPG / PNG / WEBP)
2. להמיר PDF לעמודים נפרדים
3. לנקות ולשפר כל תמונה לפני שליחה למודל
4. להחזיר רשימה מסודרת של אובייקטי ProcessedImage

סדר העיבוד וההיגיון שמאחוריו:
  0. תיקון EXIF                — טלפונים שומרים סיבוב בתגית ולא בפיקסלים
  1. זיהוי כיוון לפי הטקסט      — למקרים שבהם אין EXIF
  2. איתור הדף ותיקון פרספקטיבה — מסיר את השולחן ומיישר עיוות טרפזי
  3. deskew                     — תיקון סיבוב קטן במישור
  4. נרמול תאורה                — הצעד המשמעותי ביותר בצילומי טלפון
  5. הגברת ניגודיות מקומית       — CLAHE
  6. החלקה משמרת קצוות + חידוד   — מדגיש קווי כתיבה דקים
  7. חיתוך לאזור הכתב           — חוסך טוקנים ומרכז את תשומת הלב
  8. הגבלת גודל

הערה חשובה על מה שהוסר:
בגרסה הקודמת היה בסוף העיבוד טשטוש גאוסי. הוא הוסר —
בכתב יד עם קווים דקים הוא מוחק בדיוק את המידע שצריך לשמר.
במקומו נכנסה החלקה משמרת קצוות, שמנקה רעש בלי לפגוע בקווים.
"""

from __future__ import annotations

import math
from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps
from pdf2image import convert_from_path
from loguru import logger


# ============================================================
# מבנה הנתונים של תמונה מעובדת
# ============================================================

@dataclass
class ProcessedImage:
    """
    מייצג עמוד אחד שעבר עיבוד ומוכן לשליחה למודל.

    page_number   — מספר העמוד (מתחיל מ-1)
    image         — אובייקט PIL.Image של התמונה המוכנה
    source_path   — הנתיב המקורי של הקובץ
    was_deskewed  — האם תוקן סיבוב קטן
    was_rotated   — האם תוקן סיבוב של 90 מעלות לפי הטקסט
    page_detected — האם אותר מרובע הדף ותוקנה פרספקטיבה
    crop_box      — לאיזה אזור נחתכה התמונה, אם נחתכה
    """
    page_number: int
    image: Image.Image
    source_path: str
    was_deskewed: bool = False
    was_rotated: bool = False
    page_detected: bool = False
    crop_box: tuple | None = None


# ============================================================
# קבועים
# ============================================================

# מעל 4096 פיקסל בציר אחד המודל עלול לדחות את התמונה
GEMINI_MAX_PX = 4096

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


# ============================================================
# פונקציה ראשית
# ============================================================

def prepare_images(
    input_path: str,
    dpi: int = 300,
    skip_preprocess: bool = False,
    auto_orient: bool = False,
) -> list[ProcessedImage]:
    """
    מקבלת נתיב קובץ ומחזירה רשימת ProcessedImage מוכנות לתמלול.

    input_path       — נתיב לקובץ PDF, JPG, PNG או WEBP
    dpi              — רזולוציה להמרת PDF
    skip_preprocess  — אם True, שולחים את התמונה כמעט כפי שהיא
    auto_orient      — זיהוי סיבוב 90 מעלות. כבוי כברירת מחדל!
                       בתמונות מטלפון תגית EXIF מטפלת בכיוון,
                       והזיהוי האוטומטי רק מוסיף סיכון לסיבוב שגוי.
                       להפעלה רק אם יש תמונות בלי EXIF: --auto-orient
    """
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"הקובץ לא נמצא: {input_path}")

    ext = path.suffix.lower()

    if ext == ".pdf":
        logger.info(f"קובץ PDF — ממיר עמודים ב-{dpi} DPI")
        pil_images = _pdf_to_images(path, dpi)
    elif ext in SUPPORTED_IMAGE_EXTS:
        logger.info(f"קובץ תמונה ({ext})")
        pil_images = [Image.open(path).convert("RGB")]
    else:
        raise ValueError(
            f"פורמט לא נתמך: {ext}\n"
            f"פורמטים מותרים: pdf, jpg, png, webp"
        )

    results = []
    for i, img in enumerate(pil_images, start=1):
        logger.info(f"מעבד עמוד {i}/{len(pil_images)}")

        if skip_preprocess:
            img = _fix_exif_rotation(img)
            img = _limit_size(img)
            results.append(ProcessedImage(
                page_number=i, image=img, source_path=str(path),
            ))
            logger.debug(f"עמוד {i}: דולג על עיבוד מקדים")
        else:
            results.append(_preprocess(img, i, str(path), auto_orient))

    return results


# ============================================================
# המרת PDF לתמונות
# ============================================================

def _pdf_to_images(pdf_path: Path, dpi: int) -> list[Image.Image]:
    """
    ממיר קובץ PDF לרשימת תמונות PIL, עמוד לפי עמוד.
    pdf2image עוטפת את poppler — כלי מהיר ומדויק להמרת PDF.
    """
    try:
        pages = convert_from_path(str(pdf_path), dpi=dpi)
        logger.info(f"PDF פוצל — {len(pages)} עמודים")
        return [p.convert("RGB") for p in pages]
    except Exception as e:
        raise RuntimeError(
            f"שגיאה בהמרת PDF. ודאי ש-poppler מותקן.\n"
            f"פרטים: {e}"
        )


# ============================================================
# צינור העיבוד
# ============================================================

def _preprocess(
    img: Image.Image,
    page_number: int,
    source: str,
    auto_orient: bool = False,
) -> ProcessedImage:
    """מריצה את כל שלבי העיבוד על תמונה אחת."""

    # --- 0. EXIF ---
    img = _fix_exif_rotation(img)
    rgb = np.array(img)

    # --- 1. כיוון (רק אם התבקש במפורש) ---
    was_rotated = False
    if auto_orient:
        rgb, was_rotated = _auto_orient(rgb)

    # --- 2. איתור הדף ותיקון פרספקטיבה ---
    rgb, page_detected = _correct_perspective(rgb)

    gray = cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)

    # --- 3. deskew ---
    gray, was_deskewed = _deskew(gray)

    # --- 4. נרמול תאורה ---
    gray = _flatten_illumination(gray)

    # --- 5. ניגודיות מקומית ---
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    # --- 6. החלקה משמרת קצוות + חידוד ---
    gray = cv2.bilateralFilter(gray, 5, 40, 40)
    gray = _unsharp(gray, amount=0.8, radius=2)

    # --- 7. חיתוך לאזור הכתב ---
    gray, crop_box = _crop_to_ink(gray)

    # --- 8. חזרה ל-PIL והגבלת גודל ---
    result = Image.fromarray(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))
    result = _limit_size(result)

    if was_rotated:
        logger.info(f"עמוד {page_number}: תוקן כיוון הדף")
    if page_detected:
        logger.debug(f"עמוד {page_number}: אותר הדף ותוקנה פרספקטיבה")
    if was_deskewed:
        logger.debug(f"עמוד {page_number}: תוקן סיבוב קטן")
    if crop_box:
        logger.debug(f"עמוד {page_number}: נחתך לאזור הכתב {crop_box}")

    return ProcessedImage(
        page_number=page_number,
        image=result,
        source_path=source,
        was_deskewed=was_deskewed,
        was_rotated=was_rotated,
        page_detected=page_detected,
        crop_box=crop_box,
    )


# ============================================================
# שלב 0 — EXIF
# ============================================================

def _fix_exif_rotation(img: Image.Image) -> Image.Image:
    """
    מתקנת סיבוב שמקורו בנתוני EXIF.
    מצלמות טלפון מצלמות לרוחב ושומרות תגית Orientation.
    בלי היישום הזה התמונה מגיעה מסובבת למודל.
    """
    try:
        return ImageOps.exif_transpose(img)
    except Exception:
        return img


# ============================================================
# שלב 1 — כיוון לפי הטקסט
# ============================================================

def _ink_mask(gray: np.ndarray) -> np.ndarray:
    """
    מפת דיו: פיקסלים כהים משמעותית מהרקע המקומי.
    השוואה לרקע מקומי ולא לסף גלובלי — כך תאורה לא אחידה לא מפריעה.
    """
    k = 61 if min(gray.shape) > 200 else 11
    bg = cv2.medianBlur(gray, k)
    diff = cv2.subtract(bg, gray)
    return (diff > 28).astype(np.uint8)


def _line_orientation_votes(gray: np.ndarray) -> tuple[int, int]:
    """
    סופרת קווים ישרים ארוכים ומסווגת אותם לאופקיים ואנכיים.
    מחזירה: (אופקיים, אנכיים)

    למה קווים ולא היטל דיו:
    הגרסה הראשונה של הפונקציה הזו מדדה את שונות היטל הדיו,
    והיא טעתה בפועל וסובבה דף תקין. הסיבה: בדף שכתוב רק בחלקו
    ההיטל מתנהג בצורה לא צפויה, והמדד אינו יציב.

    קווי המחברת המודפסים הם אות חד־משמעי בהרבה: הם ארוכים,
    ישרים, ומקבילים לשורות הכתיבה. אם הם אופקיים — הדף ישר.
    """
    small = cv2.resize(gray, None, fx=0.35, fy=0.35)
    edges = cv2.Canny(small, 50, 150, apertureSize=3)
    min_len = min(small.shape) // 3
    lines = cv2.HoughLinesP(edges, 1, math.pi / 180, threshold=90,
                            minLineLength=min_len, maxLineGap=12)
    if lines is None:
        return 0, 0

    horiz = vert = 0
    for ln in lines:
        x1, y1, x2, y2 = ln.flatten()
        ang = abs(math.degrees(math.atan2(y2 - y1, x2 - x1))) % 180
        if ang < 20 or ang > 160:
            horiz += 1
        elif 70 < ang < 110:
            vert += 1
    return horiz, vert


def _auto_orient(rgb: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    מתקנת סיבוב של 90 מעלות, כשאין תגית EXIF שתטפל בזה.

    הפונקציה שמרנית בכוונה: היא מסובבת רק כשהעדות חד־משמעית —
    הרבה קווים אנכיים ארוכים וכמעט אף אחד אופקי.

    למה שמרנית: סיבוב שגוי הרסני. הוא הופך דף תקין לדף שאי אפשר
    לקרוא, וגורר גם חיתוך שגוי אחריו. עדיף לא לסובב מאשר לסובב לחינם.

    שימי לב: השיטה מבחינה בין אופקי לאנכי, לא בין 0 ל-180.
    """
    gray = cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
    horiz, vert = _line_orientation_votes(gray)

    # דורשים מסה קריטית של עדויות, ורוב מוחץ לאנכי
    if vert >= 8 and vert > horiz * 4:
        logger.debug(f"כיוון: {vert} קווים אנכיים מול {horiz} אופקיים — מסובב")
        return cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE), True

    logger.debug(f"כיוון: {horiz} אופקיים מול {vert} אנכיים — לא מסובב")
    return rgb, False


# ============================================================
# שלב 2 — איתור הדף ותיקון פרספקטיבה
# ============================================================

def _correct_perspective(rgb: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    מאתרת את מרובע הדף ומיישרת אותו למלבן.

    למה זה לא אותו דבר כמו deskew:
    deskew מסובב את התמונה במישור — הוא מתקן הטיה בלבד.
    צילום בזווית יוצר עיוות טרפזי, שבו צד אחד של הדף רחב מהשני
    ורווחי השורות משתנים לאורך העמוד. סיבוב אינו מתקן את זה.
    כאן מחשבים הומוגרפיה — התמרה שמחזירה את הדף לצורת מלבן.

    בונוס: הרקע מחוץ לדף נחתך, כך שהשולחן לא נשלח למודל.
    """
    small = cv2.resize(rgb, None, fx=0.25, fy=0.25)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)

    # נייר: רוויה נמוכה ובהירות גבוהה. משטחים צבעוניים נופלים מהמסכה.
    mask = ((hsv[:, :, 1] < 70) & (hsv[:, :, 2] > 90)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return rgb, False

    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.25 * mask.size:
        return rgb, False        # לא נמצא משטח גדול שנראה כמו דף

    peri = cv2.arcLength(c, True)
    quad = None
    for eps in (0.02, 0.03, 0.05):
        ap = cv2.approxPolyDP(c, eps * peri, True)
        if len(ap) == 4:
            quad = (ap.reshape(4, 2) * 4).astype(np.float32)
            break
    if quad is None:
        return rgb, False

    q = _order_quad(quad)
    wA = np.linalg.norm(q[2] - q[3]); wB = np.linalg.norm(q[1] - q[0])
    hA = np.linalg.norm(q[1] - q[2]); hB = np.linalg.norm(q[0] - q[3])
    W, H = int(max(wA, wB)), int(max(hA, hB))
    if W < 50 or H < 50:
        return rgb, False

    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
    warped = cv2.warpPerspective(
        rgb, cv2.getPerspectiveTransform(q, dst), (W, H), flags=cv2.INTER_CUBIC
    )
    return warped, True


def _order_quad(p: np.ndarray) -> np.ndarray:
    """מסדרת ארבע נקודות: שמאל-עליון, ימין-עליון, ימין-תחתון, שמאל-תחתון."""
    s = p.sum(axis=1)
    d = np.diff(p, axis=1).ravel()
    return np.array([p[np.argmin(s)], p[np.argmin(d)],
                     p[np.argmax(s)], p[np.argmax(d)]], dtype=np.float32)


# ============================================================
# שלב 3 — deskew
# ============================================================

def _deskew(gray: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    מזהה זווית הטיה ומתקנת אם הסיבוב מעל 0.3 מעלות.

    Canny מוצא קצוות, Hough מוצא קווים ישרים (שורות טקסט או שורות הדף),
    ומחשבים את הזווית החציונית שלהם.
    """
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, math.pi / 180, threshold=80,
                            minLineLength=gray.shape[1] // 4, maxLineGap=20)
    if lines is None:
        return gray, False

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line.flatten()
        if x2 == x1:
            continue
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if -15 < angle < 15:
            angles.append(angle)

    if not angles:
        return gray, False

    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.3:
        return gray, False

    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    logger.debug(f"deskew: תוקן סיבוב של {median_angle:.2f} מעלות")
    return rotated, True


# ============================================================
# שלב 4 — נרמול תאורה
# ============================================================

def _flatten_illumination(gray: np.ndarray) -> np.ndarray:
    """
    מנרמלת תאורה לא אחידה — הצעד המשמעותי ביותר בצילומי טלפון.

    איך זה עובד:
    מחשבים אומדן של הרקע (הנייר עצמו) על ידי טשטוש חזק, ואז
    מחלקים את התמונה ברקע הזה. התוצאה: כל אזור נמדד ביחס
    לבהירות המקומית שלו, ולא ביחס לתמונה כולה.

    למה זה חשוב כאן:
    בצילום עם מקור אור מהצד, צד אחד של הדף כהה משמעותית מהשני.
    בכתב עם קווים דקים, האזור הכהה מאבד את הקווים לגמרי.
    אחרי הנרמול הנייר אחיד וכל הכתב נמצא באותה נקודת פתיחה.
    """
    k = max(31, (min(gray.shape) // 12) | 1)   # חייב להיות אי-זוגי
    bg = cv2.medianBlur(gray, 31)
    bg = cv2.GaussianBlur(bg, (k, k), 0)
    bg = np.maximum(bg, 1)
    flat = cv2.divide(gray.astype(np.float32), bg.astype(np.float32))
    return np.clip(flat * 200.0, 0, 255).astype(np.uint8)


# ============================================================
# שלב 6 — חידוד
# ============================================================

def _unsharp(gray: np.ndarray, amount: float = 0.8, radius: int = 2) -> np.ndarray:
    """
    חידוד קווים בשיטת unsharp mask: מחסירים גרסה מטושטשת מהמקור.
    מדגיש קווי כתיבה דקים בלי להוסיף רעש כמו שסף בינארי עושה.
    """
    blur = cv2.GaussianBlur(gray, (0, 0), radius)
    return cv2.addWeighted(gray, 1 + amount, blur, -amount, 0)


# ============================================================
# שלב 7 — חיתוך לאזור הכתב
# ============================================================

def _crop_to_ink(gray: np.ndarray, margin: int | None = None) -> tuple[np.ndarray, tuple | None]:
    """
    חותכת את התמונה לאזור שבו יש כתב בפועל.

    למה זה חוסך כסף ומשפר איכות:
    עמוד שכתוב בו רק בחלקו העליון נשלח כולו למודל — כולל שוליים
    ריקים שצורכים טוקנים ומדללים את תשומת הלב. חיתוך מגדיל
    את הרזולוציה האפקטיבית של הכתב בתוך אותו תקציב.

    הערה: קווי השורות המודפסים בדף מזוהים גם הם כדיו. לכן מסננים
    אותם בפתיחה אנכית — לקו מודפס אין גובה, לאותיות יש.
    """
    # שוליים פרופורציוניים ולא מספר קבוע — 40 פיקסלים על תמונה
    # של 2400 פיקסל הם כלום, ואותיות בקצה נחתכות למחצה.
    if margin is None:
        margin = max(60, int(0.02 * max(gray.shape)))

    ink = _ink_mask(gray)

    strokes = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((9, 1), np.uint8))
    if strokes.sum() < 200:
        return gray, None

    rows = strokes.sum(axis=1)
    cols = strokes.sum(axis=0)
    r = np.where(rows > rows.max() * 0.08)[0]
    c = np.where(cols > cols.max() * 0.08)[0]
    if len(r) == 0 or len(c) == 0:
        return gray, None

    h, w = gray.shape
    y0, y1 = max(0, int(r[0]) - margin), min(h, int(r[-1]) + margin)
    x0, x1 = max(0, int(c[0]) - margin), min(w, int(c[-1]) + margin)

    # חותכים רק אם החיסכון משמעותי — אחרת לא נוגעים
    if (y1 - y0) * (x1 - x0) > 0.92 * h * w:
        return gray, None

    return gray[y0:y1, x0:x1], (x0, y0, x1, y1)


# ============================================================
# שלב 8 — הגבלת גודל
# ============================================================

def _limit_size(img: Image.Image) -> Image.Image:
    """
    מקטינה אם אחד מהצירים עולה על המותר, תוך שמירת יחס הצדדים.
    תמונה גדולה מדי תידחה על ידי ה-API.
    """
    w, h = img.size
    max_dim = max(w, h)
    if max_dim <= GEMINI_MAX_PX:
        return img

    scale = GEMINI_MAX_PX / max_dim
    new_w, new_h = int(w * scale), int(h * scale)
    logger.debug(f"תמונה הוקטנה: {w}x{h} → {new_w}x{new_h}")
    return img.resize((new_w, new_h), Image.LANCZOS)
