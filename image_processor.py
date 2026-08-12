"""
image_processor.py — עיבוד קובץ הקלט לתמונות מוכנות לתמלול

האחריות של הקובץ הזה:
1. לקבל נתיב לקובץ (PDF / JPG / PNG / WEBP)
2. להמיר PDF לעמודים נפרדים
3. לנקות ולשפר כל תמונה לפני שליחה ל-Gemini
4. להחזיר רשימה מסודרת של אובייקטי ProcessedImage
"""

from __future__ import annotations

import math
from pathlib import Path
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
from loguru import logger


# ============================================================
# מבנה הנתונים של תמונה מעובדת
# ============================================================

@dataclass
class ProcessedImage:
    """
    מייצג עמוד אחד שעבר עיבוד ומוכן לשליחה ל-Gemini.

    page_number  — מספר העמוד (מתחיל מ-1)
    image        — אובייקט PIL.Image של התמונה המוכנה
    source_path  — הנתיב המקורי של הקובץ שהכנסנו
    was_deskewed — האם המערכת תיקנה סיבוב בתמונה הזו
    """
    page_number: int
    image: Image.Image
    source_path: str
    was_deskewed: bool = False


# ============================================================
# קבועים
# ============================================================

# gemini מגביל גודל תמונה — מעל 4096 פיקסל בציר אחד הוא יכול לדחות
GEMINI_MAX_PX = 4096

# סיומות קבצי תמונה שאנחנו תומכים בהן
SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


# ============================================================
# פונקציה ראשית
# ============================================================

def prepare_images(input_path: str, dpi: int = 300) -> list[ProcessedImage]:
    """
    הפונקציה המרכזית של המודול.
    מקבלת נתיב קובץ ומחזירה רשימת ProcessedImage מוכנות לתמלול.

    input_path — נתיב לקובץ PDF, JPG, PNG או WEBP
    dpi        — רזולוציה להמרת PDF (300 = באיכות טובה לכתב יד)
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
        processed, was_deskewed = _preprocess(img)
        results.append(ProcessedImage(
            page_number=i,
            image=processed,
            source_path=str(path),
            was_deskewed=was_deskewed,
        ))
        if was_deskewed:
            logger.warning(f"עמוד {i}: זוהה סיבוב — בוצע תיקון")

    return results


# ============================================================
# המרת PDF לתמונות
# ============================================================

def _pdf_to_images(pdf_path: Path, dpi: int) -> list[Image.Image]:
    """
    ממיר קובץ PDF לרשימת תמונות PIL, עמוד לפי עמוד.

    למה pdf2image ולא ספרייה אחרת?
    כי pdf2image עוטפת את poppler — כלי C++ מהיר ומדויק להמרת PDF.
    Gemini לא מקבל PDF ישירות כקלט, אז אין ברירה אלא לפרק אותו לתמונות.
    """
    try:
        pages = convert_from_path(str(pdf_path), dpi=dpi)
        logger.info(f"PDF ממוקם — {len(pages)} עמודים")
        return [p.convert("RGB") for p in pages]
    except Exception as e:
        raise RuntimeError(
            f"שגיאה בהמרת PDF. ודאי ש-poppler מותקן.\n"
            f"פרטים: {e}"
        )


# ============================================================
# עיבוד מקדים לכל תמונה
# ============================================================

def _preprocess(img: Image.Image) -> tuple[Image.Image, bool]:
    """
    מקבלת תמונת PIL גולמית ומחזירה תמונה נקייה יותר + האם בוצע deskew.

    שלבי העיבוד:
    1. המרה ל-numpy (cv2 עובד עם numpy, לא PIL)
    2. המרה לגווני אפור — מפשטת את הניתוח, cv2 לא צריך צבע לזיהוי זוויות
    3. תיקון סיבוב (deskew) — דף שנסרק בזווית יגרום לשגיאות תמלול
    4. שיפור ניגודיות (CLAHE) — מוציאה פרטים מדיו דהוי
    5. הפחתת רעש — Gaussian blur קל מסיר פיקסלים בודדים רועשים
    6. חזרה ל-PIL + הגבלת גודל ל-Gemini
    """

    # --- שלב 1: PIL → numpy BGR (פורמט cv2) ---
    arr = np.array(img)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    # --- שלב 2: גווני אפור ---
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # --- שלב 3: תיקון סיבוב ---
    straightened, was_deskewed = _deskew(gray)

    # --- שלב 4: שיפור ניגודיות עם CLAHE ---
    # CLAHE = Contrast Limited Adaptive Histogram Equalization
    # עובדת על אזורים קטנים בנפרד, כך שאזור בהיר ואזור כהה בדף אחד
    # משתפרים כל אחד לפי הקונטקסט שלו — עדיף על שיפור גלובלי
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(straightened)

    # --- שלב 5: הפחתת רעש ---
    # Gaussian blur קל (3x3) מרכך פיקסלים רועשים בודדים מבלי לטשטש אותיות
    denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # --- שלב 6: חזרה ל-PIL RGB ---
    # Gemini מקבל PIL/RGB, לא numpy
    result_rgb = cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)
    result_pil = Image.fromarray(result_rgb)

    # --- הגבלת גודל ---
    result_pil = _limit_size(result_pil)

    return result_pil, was_deskewed


def _deskew(gray: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    מזהה זווית הטיה בתמונת גווני אפור ומתקנת אם הסיבוב מעל 0.5°.

    כיצד עובד:
    - מפעיל Canny edge detection לזיהוי קצוות
    - Hough Line Transform מוצא קווים ישרים בתמונה (שורות טקסט)
    - מחשב את הזווית הממוצעת של הקווים
    - מסובב את כל התמונה לתיקון
    """
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, math.pi / 180, threshold=80,
                             minLineLength=gray.shape[1] // 4, maxLineGap=20)

    if lines is None:
        return gray, False  # לא נמצאו קווים — אין מה לתקן

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            continue
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        # מסנן זוויות קיצוניות (מעל ±15° — כנראה לא הטיית דף)
        if -15 < angle < 15:
            angles.append(angle)

    if not angles:
        return gray, False

    median_angle = float(np.median(angles))

    if abs(median_angle) < 0.5:
        return gray, False  # הסיבוב קטן מדי לתיקון

    # סיבוב סביב מרכז התמונה
    h, w = gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    logger.debug(f"deskew: תוקן סיבוב של {median_angle:.2f}°")
    return rotated, True


def _limit_size(img: Image.Image) -> Image.Image:
    """
    אם אחד מצירי התמונה עולה על GEMINI_MAX_PX (4096),
    מקטינה תוך שמירת יחס גובה-רוחב.

    למה? Gemini מגביל גודל קלט — תמונה גדולה מדי תגרום לשגיאת API.
    """
    w, h = img.size
    max_dim = max(w, h)
    if max_dim <= GEMINI_MAX_PX:
        return img

    scale = GEMINI_MAX_PX / max_dim
    new_w = int(w * scale)
    new_h = int(h * scale)
    logger.debug(f"תמונה הוקטנה: {w}x{h} → {new_w}x{new_h}")
    return img.resize((new_w, new_h), Image.LANCZOS)
