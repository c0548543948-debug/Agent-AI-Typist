"""
line_segmenter.py — חיתוך עמוד לשורות בודדות

למה זה הרכיב המרכזי בשלב הבא:

1. דיוק — שורה בודדת נשלחת למודל בגודל מלא, במקום עמוד שלם מוקטן.
   גרש הוא כמה פיקסלים; בעמוד מוקטן הוא נעלם, בשורה בודדת הוא ברור.
   בנוסף, בשורה אחת אין מרחב להמציא סיפור שלם — כפי שקרה לנו בעמוד המסובב.

2. הגהה — ממשק ההגהה מציג תמונת שורה מעל הטקסט שלה. בלי חיתוך
   אין מה להציג.

3. אימון — היחידה שמודלים של זיהוי כתב יד מאומנים עליה היא שורה,
   לא אות ולא עמוד. כל זוג (תמונת שורה, טקסט נכון) הוא נתון אימון.

השיטה:
מסכת דיו → סינון קווי המחברת המודפסים → מיזוג מילים לגושי שורה
→ רכיבי קשירות → תיבות שורה ממוינות מלמעלה למטה.

למה לא היטל אופקי פשוט: היטל רגיש להטיה קלה ולשוליים, ומתמזג
כשאותיות של שורה אחת יורדות לתחום השורה הבאה. מיזוג מורפולוגי
עמיד יותר בשני המקרים.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from loguru import logger


# ============================================================
# מבנה הנתונים של שורה
# ============================================================

@dataclass
class TextLine:
    """
    שורת טקסט אחת שחולצה מעמוד.

    index    — מספר השורה בעמוד, מ-1 ומלמעלה למטה
    box      — (x0, y0, x1, y1) בקואורדינטות העמוד המעובד
    image    — תמונת השורה עצמה
    """
    index: int
    box: tuple[int, int, int, int]
    image: Image.Image

    @property
    def height(self) -> int:
        return self.box[3] - self.box[1]

    @property
    def width(self) -> int:
        return self.box[2] - self.box[0]


# ============================================================
# מסכת דיו
# ============================================================

def _ink_mask(gray: np.ndarray) -> np.ndarray:
    """
    פיקסלים כהים משמעותית מהרקע המקומי.
    השוואה מקומית ולא סף גלובלי — כך תאורה לא אחידה לא מפריעה.
    """
    k = 61 if min(gray.shape) > 200 else 11
    bg = cv2.medianBlur(gray, k)
    return (cv2.subtract(bg, gray) > 28).astype(np.uint8)


def _strokes_only(ink: np.ndarray) -> np.ndarray:
    """
    מסירה את קווי המחברת המודפסים ומשאירה רק כתב יד.

    ההבחנה: לקו מודפס אין גובה — הוא שני-שלושה פיקסלים.
    לאותיות יש. פתיחה מורפולוגית אנכית משאירה רק מה שגבוה מספיק.
    """
    return cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((9, 1), np.uint8))


# ============================================================
# חיתוך לשורות
# ============================================================

def _estimate_pitch(profile: np.ndarray, page_height: int) -> int:
    """
    מעריכה את המרווח בין שורות בעזרת אוטוקורלציה של ההיטל.

    למה: כתיבה ביד היא מחזורית — שורה, רווח, שורה, רווח.
    האוטוקורלציה מוצאת את אורך המחזור הזה בלי שנצטרך לנחש אותו,
    וכך אותו קוד עובד גם על עמוד צפוף וגם על עמוד מרווח.
    """
    p = profile - profile.mean()
    ac = np.correlate(p, p, mode="full")[len(p) - 1:]
    lo = max(5, int(page_height * 0.015))
    hi = max(lo + 5, int(page_height * 0.20))
    return int(lo + int(np.argmax(ac[lo:hi])))


def segment_lines(
    image: Image.Image,
    ink_threshold: int = 35,
    min_height_ratio: float = 0.35,
    pad_ratio: float = 0.18,
) -> list[TextLine]:
    """
    מקבלת תמונת עמוד ומחזירה רשימת שורות, ממוינות מלמעלה למטה.

    השיטה: היטל אופקי של הדיו, אומדן מרווח השורות, איתור שיאים,
    וחיתוך בנקודת המינימום שבין כל שני שיאים סמוכים.

    למה שיאים ולא סף קבוע: בכתב יד צפיפות הדיו משתנה מאוד בין
    שורה לשורה. סף גלובלי מאחד שורות חלשות או מפצל שורות חזקות.
    חיתוך במינימום שבין שיאים עמיד בפני השוני הזה.

    ink_threshold    — כמה כהה מהרקע המקומי נחשב דיו
    min_height_ratio — גובה מזערי של שורה, כשבר מהמרווח המשוער
    pad_ratio        — הרחבת התיבה מעל ומתחת, כשבר מהמרווח.
                       דרושה כדי לא לקצוץ אותיות עולות ויורדות.
    """
    gray = np.array(image.convert("L"))
    h, w = gray.shape

    diff = cv2.subtract(cv2.medianBlur(gray, 61), gray)
    ink = (diff > ink_threshold).astype(np.uint8)
    if ink.sum() < 100:
        logger.warning("לא נמצא כתב בעמוד — לא בוצע חיתוך לשורות")
        return []

    profile = ink.sum(axis=1).astype(float)
    pitch = _estimate_pitch(profile, h)

    k = max(3, (pitch // 6) | 1)                 # גרעין החלקה אי-זוגי
    smooth = cv2.GaussianBlur(profile.reshape(-1, 1), (1, k), 0).ravel()

    # שיאים — כל אחד מייצג שורה. מרחק מזערי מונע פיצול שורה לשתיים.
    peaks: list[int] = []
    min_dist = max(3, int(pitch * 0.6))
    cutoff = smooth.max() * 0.20
    for i in np.argsort(-smooth):
        if smooth[i] < cutoff:
            break
        if all(abs(int(i) - j) >= min_dist for j in peaks):
            peaks.append(int(i))
    peaks.sort()

    if not peaks:
        logger.warning("לא זוהו שורות")
        return []

    # גבול בין שתי שורות = הנקודה הדלילה ביותר שביניהן
    bounds = [0]
    for a, b in zip(peaks, peaks[1:]):
        bounds.append(int(a + int(np.argmin(smooth[a:b]))))
    bounds.append(h)

    lines = []
    idx = 1
    for y0, y1 in zip(bounds, bounds[1:]):
        if y1 - y0 < pitch * min_height_ratio:
            continue
        pad = int(pitch * pad_ratio)
        yy0, yy1 = max(0, y0 - pad), min(h, y1 + pad)

        # צמצום אופקי לאזור שבו יש דיו בשורה הזו
        band = ink[yy0:yy1]
        cols = band.sum(axis=0)
        nz = np.where(cols > 0)[0]
        if len(nz) == 0:
            continue
        xx0, xx1 = max(0, int(nz[0]) - 15), min(w, int(nz[-1]) + 15)

        lines.append(TextLine(
            index=idx, box=(xx0, yy0, xx1, yy1),
            image=image.crop((xx0, yy0, xx1, yy1)),
        ))
        idx += 1

    logger.info(f"זוהו {len(lines)} שורות (מרווח משוער {pitch} פיקסלים)")
    return lines


def _merge_overlapping(boxes: list[tuple], overlap: float = 0.5) -> list[tuple]:
    """
    מאחדת תיבות שחופפות אנכית באופן משמעותי.

    למה צריך: מילה שנכתבה מעט גבוה מהשאר עלולה להתפצל לרכיב נפרד.
    אם היא חופפת אנכית לשורה קיימת — היא שייכת לה.
    """
    boxes = sorted(boxes, key=lambda b: b[1])
    out: list[list] = []

    for b in boxes:
        placed = False
        for o in out:
            top, bot = max(b[1], o[1]), min(b[3], o[3])
            inter = max(0, bot - top)
            small = min(b[3] - b[1], o[3] - o[1])
            if small > 0 and inter / small > overlap:
                o[0] = min(o[0], b[0]); o[1] = min(o[1], b[1])
                o[2] = max(o[2], b[2]); o[3] = max(o[3], b[3])
                placed = True
                break
        if not placed:
            out.append(list(b))

    return [tuple(o) for o in out]


# ============================================================
# שמירה ותצוגה
# ============================================================

def save_lines(lines: list[TextLine], out_dir: str | Path, stem: str) -> list[Path]:
    """
    שומרת כל שורה כקובץ תמונה נפרד.
    אלה הקבצים שנשמרים במסד הנתונים ומשמשים גם להגהה וגם לאימון.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for ln in lines:
        p = out / f"{stem}_line{ln.index:03d}.png"
        ln.image.save(p)
        paths.append(p)
    return paths


def draw_overlay(image: Image.Image, lines: list[TextLine]) -> Image.Image:
    """
    מציירת את תיבות השורות על העמוד — לבדיקה ויזואלית שהחיתוך נכון.
    """
    arr = cv2.cvtColor(np.array(image.convert("L")), cv2.COLOR_GRAY2BGR)
    for ln in lines:
        x0, y0, x1, y1 = ln.box
        cv2.rectangle(arr, (x0, y0), (x1, y1), (0, 0, 255), 3)
        cv2.putText(arr, str(ln.index), (x1 + 6, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


# ============================================================
# בדיקה מהירה משורת הפקודה
# ============================================================

if __name__ == "__main__":
    import argparse
    from image_processor import prepare_images

    ap = argparse.ArgumentParser(description="חיתוך עמוד לשורות")
    ap.add_argument("-i", "--input", required=True, help="נתיב תמונה או PDF")
    ap.add_argument("-o", "--out-dir", default="lines", help="לאן לשמור")
    ap.add_argument("--overlay", default=None, help="נתיב לשמירת תצוגת בדיקה")
    args = ap.parse_args()

    for page in prepare_images(args.input):
        lines = segment_lines(page.image)
        stem = Path(args.input).stem + f"_p{page.page_number}"
        paths = save_lines(lines, args.out_dir, stem)
        print(f"עמוד {page.page_number}: {len(lines)} שורות → {args.out_dir}")
        if args.overlay:
            draw_overlay(page.image, lines).save(args.overlay)
            print(f"תצוגת בדיקה: {args.overlay}")
