"""
ocr_engine.py — שליחת תמונה ל-Gemini וקבלת תמלול

האחריות של הקובץ הזה:
1. לטעון דוגמאות Few-Shot מהתיקייה (אם קיימות)
2. לבנות את הפרומפט המלא
3. לשלוח תמונה + פרומפט ל-Gemini API
4. לפרסר את התשובה לאובייקט מסודר
5. לטפל בשגיאות API מבלי לקרוס
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image
from loguru import logger

from config import AppConfig
from image_processor import ProcessedImage


# ============================================================
# מבנה הנתונים של תוצאת תמלול
# ============================================================

@dataclass
class TranscriptionResult:
    """
    תוצאת תמלול של עמוד אחד.

    page_number         — מספר העמוד (1-based)
    text                — הטקסט המתומלל המלא
    uncertain_words     — רשימת מילים שסומנו כ-[מילה?]
    detected_language   — "he" | "en" | "mixed" | "unknown"
    api_call_ms         — כמה זמן לקחה הבקשה (מילישניות)
    model_used          — שם המודל שהשיב
    success             — False אם הייתה שגיאה ונכנס placeholder
    error_message       — הודעת שגיאה אם success=False
    """
    page_number: int
    text: str
    uncertain_words: list[str] = field(default_factory=list)
    detected_language: str = "unknown"
    api_call_ms: int = 0
    model_used: str = ""
    success: bool = True
    error_message: str = ""


# ============================================================
# טעינת דוגמאות Few-Shot
# ============================================================

@dataclass
class FewShotSample:
    """זוג אחד: תמונה + תמלול נכון שלה"""
    image: Image.Image
    transcript: str


def load_few_shot_samples(few_shot_dir: str, max_samples: int) -> list[FewShotSample]:
    """
    סורקת תיקיית few_shot/ ומחזירה זוגות תמונה+תמלול.

    כללי ההתאמה:
    - קובץ תמונה (jpg/png/webp) + קובץ טקסט עם אותו שם בדיוק.
    - דוגמה: sample_01_image.jpg ←→ sample_01_image.txt
    - אם קובץ .txt חסר לתמונה — הזוג נדחה עם אזהרה.
    - נטענים לכל היותר max_samples זוגות.
    """
    dir_path = Path(few_shot_dir)
    if not dir_path.exists():
        logger.debug("תיקיית few_shot לא קיימת — ממשיך ללא כיול")
        return []

    image_exts = {".jpg", ".jpeg", ".png", ".webp"}
    samples = []

    # מיון לפי שם — כדי שהסדר יהיה עקבי בכל ריצה
    image_files = sorted(
        f for f in dir_path.iterdir()
        if f.suffix.lower() in image_exts and f.stem != ".gitkeep"
    )

    for img_path in image_files:
        if len(samples) >= max_samples:
            break

        txt_path = img_path.with_suffix(".txt")
        if not txt_path.exists():
            logger.warning(f"Few-Shot: לא נמצא קובץ טקסט עבור {img_path.name} — דילוג")
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            transcript = txt_path.read_text(encoding="utf-8").strip()
            samples.append(FewShotSample(image=img, transcript=transcript))
            logger.debug(f"Few-Shot טעון: {img_path.name}")
        except Exception as e:
            logger.warning(f"שגיאה בטעינת {img_path.name}: {e}")

    logger.info(f"נטענו {len(samples)} דוגמאות Few-Shot")
    return samples


# ============================================================
# בניית הפרומפט
# ============================================================

# הפרומפט המערכתי — נשלח בכל בקשה
SYSTEM_PROMPT = """You are a handwriting transcription specialist. Your ONLY job is to read and transcribe exactly what is physically written in the image — nothing more, nothing less.

STRICT RULES:
1. Transcribe ONLY what you can actually see written in the image. Do NOT add, invent, complete, or guess any content that is not clearly visible.
2. Go line by line, in the exact order the text appears in the image.
3. Preserve the original language of every word. The text may be Hebrew, English, or mixed. Do NOT translate.
4. Preserve paragraph breaks. Use a blank line between paragraphs.
5. If a word is unclear or you are not confident, write it as [word?] — replace "word" with your best guess. Example: [שלום?] or [hello?].
6. If an entire section is completely unreadable, write [unreadable].
7. Do NOT add punctuation that is not clearly written in the image.
8. Do NOT fix spelling or grammar.
9. Output ONLY the transcribed text. No explanations, no commentary, no headers."""


def _build_contents(
    image: Image.Image,
    few_shot_samples: list[FewShotSample],
    language_mode: str,
) -> list:
    """
    בונה את רשימת ה-contents לשליחה ל-Gemini.

    המבנה:
    [הוראה, דוגמה1_תמונה, דוגמה1_תמלול, דוגמה2_תמונה, דוגמה2_תמלול, ..., תמונה_חדשה]

    למה הסדר הזה?
    Gemini לומד מההקשר — כשרואה זוגות "תמונה→טקסט" לפני התמונה החדשה,
    הוא מבין את הסגנון הספציפי של כתב היד ומסתגל אליו.
    """
    contents = []

    # הוספת הנחיית שפה לפרומפט
    system = SYSTEM_PROMPT
    if language_mode == "he":
        system += "\n\nNote: The document is in Hebrew only."
    elif language_mode == "en":
        system += "\n\nNote: The document is in English only."

    contents.append(system)

    # הוספת דוגמאות Few-Shot
    for sample in few_shot_samples:
        contents.append(sample.image)          # תמונת דוגמה
        contents.append(f"תמלול:\n{sample.transcript}")   # התמלול הנכון שלה

    # הוספת ההנחיה לתמלול + התמונה החדשה
    contents.append("Transcribe ONLY what is physically written in the following image, line by line:")
    contents.append(image)

    return contents


# ============================================================
# פירסור תשובת Gemini
# ============================================================

def _parse_response(text: str, page_number: int, model: str, elapsed_ms: int) -> TranscriptionResult:
    """
    מקבלת טקסט חופשי מ-Gemini ומחזירה TranscriptionResult מסודר.

    מה אנחנו מחפשים:
    - כל המילים בפורמט [מילה?] — אלה המילים הלא-ודאיות
    - זיהוי שפה: אם יש תווים עבריים → he, אם רק לטינית → en, אחרת → mixed
    """
    # איסוף מילים לא-ודאיות — חיפוש תבנית [xxx?]
    uncertain = re.findall(r'\[([^\]]+)\?]', text)

    # זיהוי שפה לפי תווים בטקסט
    has_hebrew = bool(re.search(r'[֐-׿]', text))
    has_latin = bool(re.search(r'[a-zA-Z]', text))

    if has_hebrew and has_latin:
        lang = "mixed"
    elif has_hebrew:
        lang = "he"
    elif has_latin:
        lang = "en"
    else:
        lang = "unknown"

    return TranscriptionResult(
        page_number=page_number,
        text=text.strip(),
        uncertain_words=uncertain,
        detected_language=lang,
        api_call_ms=elapsed_ms,
        model_used=model,
        success=True,
    )


# ============================================================
# פונקציה ראשית
# ============================================================

def transcribe_image(
    processed: ProcessedImage,
    config: AppConfig,
    few_shot_samples: list[FewShotSample] | None = None,
) -> TranscriptionResult:
    """
    מקבלת ProcessedImage ומחזירה TranscriptionResult.

    זו הפונקציה שמחברת הכל: פרומפט + few-shot + API + פירסור.
    אם הבקשה נכשלת — מחזירה תוצאה עם success=False ו-placeholder,
    כך שהשאר ממשיך לפעול.
    """
    samples = few_shot_samples or []

    # בניית תוכן הבקשה
    contents = _build_contents(processed.image, samples, config.language_mode)

    # אתחול לקוח Gemini
    # הלקוח מנהל את החיבור ל-API, טיפול ב-auth, retry אוטומטי
    client = genai.Client(api_key=config.api_key)

    logger.info(f"שולח עמוד {processed.page_number} ל-Gemini ({config.model})...")
    start = time.monotonic()

    try:
        response = client.models.generate_content(
            model=config.model,
            contents=contents,
            config=types.GenerateContentConfig(
                # temperature=0 = פחות יצירתיות, יותר עקביות — חשוב לתמלול
                temperature=0.0,
                # max_output_tokens — כתב יד ארוך יכול להיות הרבה מילים
                max_output_tokens=8192,
            ),
        )
        elapsed_ms = int((time.monotonic() - start) * 1000)

        raw_text = response.text or ""
        logger.info(
            f"עמוד {processed.page_number}: התקבל תמלול "
            f"({len(raw_text)} תווים, {elapsed_ms}ms)"
        )

        return _parse_response(raw_text, processed.page_number, config.model, elapsed_ms)

    except Exception as e:
        elapsed_ms = int((time.monotonic() - start) * 1000)
        logger.error(f"עמוד {processed.page_number}: שגיאת API — {e}")

        # במקום לקרוס — מחזירים placeholder
        # document_writer יציג את זה בצורה ברורה במסמך
        return TranscriptionResult(
            page_number=processed.page_number,
            text="[תמלול עמוד זה נכשל]",
            success=False,
            error_message=str(e),
            api_call_ms=elapsed_ms,
            model_used=config.model,
        )
