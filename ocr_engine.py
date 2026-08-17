"""
ocr_engine.py — שליחת תמונה ל-Gemini וקבלת תמלול

האחריות של הקובץ הזה:
1. לטעון דוגמאות Few-Shot מהתיקייה (אם קיימות ואם הופעלו במפורש)
2. לבנות את הפרומפט המלא
3. לשלוח תמונה + פרומפט ל-Gemini API
4. לפרסר את התשובה לאובייקט מסודר
5. לדווח על שימוש בטוקנים ועלות משוערת
6. לטפל בשגיאות API מבלי לקרוס

הערה חשובה על Gemini 3:
כללי ההגדרה השתנו מול גרסאות 2.5 —
  * thinking_level ברירת מחדל במודלי Pro היא high, וטוקני החשיבה
    נספרים בתוך max_output_tokens. תקציב קטן = תמלול נקטע או תשובה ריקה.
  * temperature מתחת ל-1.0 עלולה לגרום ללופים ולביצועים מדורדרים.
    לכן איננו קובעים temperature בכלל — משתמשים בברירת המחדל.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
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
    truncated           — True אם התשובה נקטעה בגלל מגבלת טוקנים
    finish_reason       — סיבת הסיום שהמודל דיווח
    input_tokens        — טוקני קלט (תמונה + פרומפט)
    output_tokens       — טוקני פלט (הטקסט עצמו)
    thinking_tokens     — טוקני חשיבה פנימית (נספרים כפלט לחיוב)
    """
    page_number: int
    text: str
    uncertain_words: list[str] = field(default_factory=list)
    detected_language: str = "unknown"
    api_call_ms: int = 0
    model_used: str = ""
    success: bool = True
    error_message: str = ""
    truncated: bool = False
    finish_reason: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0

    @property
    def billable_output_tokens(self) -> int:
        """גוגל מחייבת על טוקני חשיבה בתעריף הפלט."""
        return self.output_tokens + self.thinking_tokens


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
    - קובץ תמונה (jpg/png/webp) + קובץ טקסט עם אותו שם בסיס.
    - דוגמה: sample_01.jpg ←→ sample_01.txt
    - אם קובץ .txt חסר או ריק — הזוג נדחה עם אזהרה.
    - נטענים לכל היותר max_samples זוגות.

    אזהרה: Few-Shot ברמת עמוד שלם מזהם את ההקשר —
    המודל לומד את *הנושא* של הדוגמאות ולא רק את צורת האותיות,
    ומשלים מהן תוכן לתמונה החדשה. מופעל רק בבקשה מפורשת.
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
            transcript = txt_path.read_text(encoding="utf-8").strip()
            # קובץ ריק גרוע מקובץ חסר — הוא מלמד את המודל "לתמונה כזו אין טקסט"
            if not transcript:
                logger.warning(f"Few-Shot: {txt_path.name} ריק — דילוג")
                continue

            img = Image.open(img_path).convert("RGB")
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
#
# שני עקרונות מנחים:
# 1. לומר למודל במפורש שהתוכן *אינו ידוע לו* — זו הגנה ישירה מפני
#    השלמה מהזיכרון, שהיא מקור ההזיות העיקרי בטקסט תורני.
# 2. לתת לו את *המשלב הלשוני* (לשון רבנית, קיצורים, כתיב חסר)
#    בלי לתת לו את *התוכן* — ידיעת המשלב עוזרת, ידיעת התוכן מזיקה.
SYSTEM_PROMPT = """You are a handwriting transcription specialist working on handwritten Hebrew Torah scholarship (chiddushei Torah — original novellae and commentary).

CRITICAL CONTEXT — READ THIS FIRST:
This is an ORIGINAL, UNPUBLISHED manuscript written by a private individual. You have never seen this text before and you cannot possibly know its content. Nothing in your training data contains it. Therefore:
  * NEVER complete a word, phrase, or sentence from memory.
  * NEVER fill in what "should" come next based on familiar phrasing.
  * NEVER assume the author is quoting something you recognize.
If you cannot read it in the image, you do not know it. Mark it uncertain instead.
YOU ARE NOT READING FOR MEANING — THIS IS THE MOST IMPORTANT INSTRUCTION:
Your task is not to understand this text. Your task is to copy the shapes of the letters that are on the page, one by one, exactly as they appear.
  * Do NOT use your knowledge of Torah, halacha, grammar, or context to decide what a word "should" be. Decide ONLY from the shapes you can see.
  * If the letters on the page form a word that makes no sense in the sentence — transcribe it anyway, exactly as written. A strange reading that matches the shapes is CORRECT. A sensible reading that does not match the shapes is WRONG.
  * NEVER smooth an odd reading into a fluent one. Never replace a puzzling cluster of letters with a phrase that "fits better".
  * Read the way a person copies an unfamiliar alphabet: shape, then shape, then shape. Meaning is irrelevant to your task.
Almost every serious error in this kind of work comes from understanding instead of copying.
STRICT RULES:
1. Transcribe ONLY what you can actually see written in the image. Do NOT add, invent, complete, or guess any content that is not clearly visible.
2. Go line by line, in the exact order the text appears in the image.
3. Transcribe the ENTIRE page. Do not stop early, do not summarize, do not skip sections. Continue until you reach the last written line on the page.
4. Preserve the original language of every word. The text may be Hebrew, Aramaic, English, or mixed. Do NOT translate anything.
5. Preserve paragraph breaks. Use a blank line between paragraphs.
6. If a word is unclear or you are not fully confident, write it as [word?] — replace "word" with your best guess. When in doubt, always prefer marking [word?] over writing a word that might be wrong. Over-marking uncertainty is always better than inventing content.
6a. UNCERTAINTY CALIBRATION — DO NOT SKIP THIS:
This handwriting is genuinely difficult. On a dense page you should normally end up marking SEVERAL words as [word?]. That is the expected, correct outcome, not a failure.
If you finish a full page of difficult handwriting and have marked NOTHING as uncertain, you have almost certainly over-committed: you resolved ambiguous shapes into confident words instead of flagging them. Go back and mark the ones you guessed.
A marked word costs the reader two seconds. An unmarked wrong word may never be caught. Always prefer the mark.
Use the exact form [word?] — with the guess inside the brackets, before the question mark.
7. If an entire section is completely unreadable, write [לא קריא].
7b. NEVER write English words in your output. The output must contain only Hebrew and Aramaic, plus the bracket markers listed here. For editorial observations use ONLY these exact Hebrew markers:
      [מחוק]        — text the writer struck through
      [לא קריא]     — a passage you cannot read at all
      [word?]       — an uncertain reading (keep the brackets and question mark)
    Do NOT write things like "crossed out", "strikethrough", "illegible", or any other English annotation.
8. Do NOT add punctuation that is not clearly written in the image.
9. Output ONLY the transcribed text. No explanations, no commentary, no headers.
10. Do NOT use surrounding context to guess unclear words. Each word must be readable on its own in the image.

RULES SPECIFIC TO RABBINIC HEBREW — these matter greatly here:
11. ABBREVIATIONS: Rabbinic Hebrew is full of abbreviations marked with an apostrophe (') or gershayim ("), for example forms like וכו' / ז"ל / עי"ש / הקב"ה / ר' . Copy every abbreviation EXACTLY as written, with its punctuation marks. NEVER expand an abbreviation to its full form. NEVER add an abbreviation mark that is not written.
11a. THE ABBREVIATION MARK IS PHYSICAL EVIDENCE — THIS IS CRITICAL:
An apostrophe or gershayim is a mark actually drawn on the page. The moment you see such a mark inside or at the end of a cluster of letters, that cluster IS an abbreviation. It is NOT an ordinary word.
  * When a mark is visible, transcribe the letters you actually see, with the mark, in order.
  * NEVER replace such a cluster with a common Hebrew word that merely resembles it. For example, a cluster written כמש"כ must NEVER be transcribed as the ordinary word כמעט. The visible mark overrides any familiar word your instinct suggests.
  * A familiar, fluent-sounding word must never win over a visible abbreviation mark. If the mark is there, the abbreviation is the correct reading.

11b. THESE MARKS ARE SMALL AND EASY TO MISS. Before finalising each short cluster of 2-5 letters, look specifically for a tiny apostrophe or gershayim above or after the letters. Missing them is the single most common transcription error in this genre.

11c. Abbreviations frequently used in this kind of text include forms such as:
מש"כ · כמש"כ · עי"ש · ע"ש · ע"י · ע"כ · ע"פ · עפ"י · אע"פ · וכו' · ז"ל · ח"ו · א"כ · מ"מ · י"ל · וי"ל · נ"ל · כנ"ל · הנ"ל · ד"ה · ד"א · פי' · תוס' · מח' · חז"ל · ארז"ל · רש"י · רמב"ן · עה"ת · סט"א · אכי"ר · בעהי"ת · ר' · ב'
This list exists ONLY to help you RECOGNISE such marks when they are physically present on the page. It is NOT a list of words to insert. Never write an abbreviation that you cannot actually see written.
11d. NEVER EXPAND AN ABBREVIATION, EVEN WHEN YOU ARE CERTAIN WHAT IT MEANS:
An abbreviation is copied as letters plus mark, never as the words it stands for.
  * במח' stays במח'. Writing במכירת instead is an ERROR, even though it is what the author meant.
  * המח' stays המח'. וכו' stays וכו'. פי' stays פי'.
Understanding what an abbreviation stands for is exactly the kind of "reading for meaning" that you must not do here. Copy the mark, copy the letters, move on.
12. SPELLING: Hebrew here may use defective or plene spelling (כתיב חסר / כתיב מלא) that differs from modern normative spelling. Copy the letters exactly as written. Do NOT normalize, modernize, or "correct" spelling.
13. SYNTAX: This genre is often terse, elliptical, and grammatically incomplete by design. Sentences may break off, lack a verb, or use unusual word order. Do NOT fix, reorder, smooth, or complete the syntax. Transcribe the broken text as broken.
14. QUOTATIONS: The author may quote a verse or a Talmudic passage. Transcribe ONLY the letters actually visible on the page. Do NOT extend a quotation beyond what is written, even if you recognize the source and know how it continues. A quotation cut off mid-word stays cut off mid-word.
15. Do NOT rephrase into modern Hebrew. Do NOT paraphrase. This is transcription, not translation or editing."""


def _load_writer_profile(path: str) -> str:
    """
    טוענת תיאור חופשי של מאפייני כתב היד, אם קיים קובץ כזה.

    למה זה עובד טוב יותר מ-Few-Shot ויזואלי?
    תיאור טקסטואלי מלמד את המודל על *צורת האותיות* בלי להכניס
    לתוך ההקשר תוכן של דוגמאות — ולכן אין זיהום ואין המצאה.

    דוגמה למה שכדאי לכתוב בקובץ:
        האות ב' נכתבת ללא זווית ונראית דומה ל-ו'
        המ' הסופית נשארת פתוחה למטה
        הכותב מחבר את ה-א' ל-אות שאחריה
    """
    p = Path(path)
    if not p.exists():
        return ""
    try:
        text = p.read_text(encoding="utf-8").strip()
        # שורות שמתחילות ב-# הן הערות ולא נשלחות למודל
        lines = [
            ln for ln in text.splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        profile = "\n".join(lines).strip()
        if profile:
            logger.info(f"נטען פרופיל כתב יד ({len(profile)} תווים)")
        return profile
    except Exception as e:
        logger.warning(f"שגיאה בטעינת פרופיל כתב היד: {e}")
        return ""


def _build_contents(
    image: Image.Image,
    few_shot_samples: list[FewShotSample],
    language_mode: str,
    writer_profile: str = "",
) -> list:
    """
    בונה את רשימת ה-contents לשליחה ל-Gemini.

    ללא Few-Shot (המצב המומלץ):
        [הוראה, תמונה]
    עם Few-Shot:
        [הוראה, דוגמה1_תמונה, דוגמה1_תמלול, ..., תמונה_חדשה]
    """
    contents = []

    # הוספת הנחיית שפה לפרומפט
    system = SYSTEM_PROMPT
    if language_mode == "he":
        system += "\n\nNote: The document is in Hebrew only."
    elif language_mode == "en":
        system += "\n\nNote: The document is in English only."

    # פרופיל כתב היד — תיאור צורת האותיות, ללא תוכן
    if writer_profile:
        system += (
            "\n\nKNOWN CHARACTERISTICS OF THIS WRITER'S HAND:\n"
            "The following notes describe how this specific person forms letters. "
            "Use them ONLY to interpret letter shapes — they say nothing about content.\n"
            f"{writer_profile}"
        )

    contents.append(system)

    # הוספת דוגמאות Few-Shot — לכיול סגנון כתיבה בלבד
    if few_shot_samples:
        contents.append(
            "The following examples show this person's handwriting style. "
            "Study them ONLY to learn how this person forms letters and words. "
            "WARNING: The content of these examples is completely unrelated to the new image. "
            "Do NOT assume any words, phrases, or topics from these examples appear in the new image."
        )
    for sample in few_shot_samples:
        contents.append(sample.image)
        contents.append(f"Example transcription (unrelated content):\n{sample.transcript}")

    # הוספת ההנחיה לתמלול + התמונה החדשה
    contents.append(
        "Now transcribe ONLY what is physically written in the NEW image below. "
        "Do NOT reuse or reference any content from the example images above — "
        "those were only for learning letter shapes. "
        "Transcribe every line from the top of the page to the very bottom:"
    )
    contents.append(image)

    return contents


# ============================================================
# בניית הגדרות הבקשה — עמיד לגרסאות SDK שונות
# ============================================================

def _build_gen_config(cfg: AppConfig) -> types.GenerateContentConfig:
    """
    בונה את GenerateContentConfig לפי ההגדרות.

    למה try/except?
    thinking_level ו-media_resolution הם פרמטרים חדשים של Gemini 3.
    בגרסאות SDK ישנות הם לא קיימים והבנייה תיפול.
    במקרה כזה מדלגים עליהם במקום לקרוס — עם אזהרה בלוג.

    שימי לב: איננו קובעים temperature.
    גוגל ממליצה במפורש להשאיר את ברירת המחדל (1.0) ב-Gemini 3;
    ערך נמוך יותר עלול לגרום ללופים ולעצירה מוקדמת.
    """
    kwargs: dict = {"max_output_tokens": cfg.max_output_tokens}

    # ── כיבוי Automatic Function Calling ──
    # איננו משתמשים בכלים, ובלי הכיבוי ה-SDK מדפיס אזהרה בכל בקשה.
    try:
        kwargs["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(
            disable=True
        )
    except AttributeError:
        pass

    # ── רזולוציית מדיה ──
    # קובעת כמה טוקנים מוקצים לתמונה. לכתב יד קשה רוצים גבוה.
    try:
        kwargs["media_resolution"] = getattr(
            types.MediaResolution,
            f"MEDIA_RESOLUTION_{cfg.media_resolution.upper()}",
        )
    except AttributeError:
        logger.warning(
            f"media_resolution='{cfg.media_resolution}' לא נתמך ב-SDK המותקן — מדלג. "
            "כדאי לעדכן: pip install -U google-genai"
        )

    # ── רמת חשיבה ──
    # low = פחות טוקני חשיבה, יותר תקציב לתמלול עצמו.
    # זו ההמלצה של גוגל למשימות חילוץ ותמלול.
    try:
        kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_level=cfg.thinking_level.upper()
        )
    except (AttributeError, TypeError, ValueError):
        logger.warning(
            "thinking_level לא נתמך ב-SDK המותקן — מדלג. "
            "כדאי לעדכן: pip install -U google-genai"
        )

    try:
        return types.GenerateContentConfig(**kwargs)
    except TypeError as e:
        # נפילה אחורה: רק המגבלה הבסיסית
        logger.warning(f"הגדרות מתקדמות נדחו על ידי ה-SDK ({e}) — משתמש בהגדרות בסיסיות")
        return types.GenerateContentConfig(max_output_tokens=cfg.max_output_tokens)


# ============================================================
# לקוח Gemini — נוצר פעם אחת בלבד
# ============================================================

def _explain_api_error(err: Exception) -> tuple[str, str]:
    """
    מתרגמת שגיאת API להודעה מובנת.
    מחזירה: (הודעה_קצרה, הסבר_מפורט)

    מטרה מרכזית: לזהות חסימה של מסנן תוכן (NetFree וכדומה),
    שנראית כמו שגיאת API אבל היא בכלל לא הגיעה לגוגל.
    """
    raw = str(err)

    # ── חסימת NetFree ──
    if "blockByNetFree" in raw or "netfree.link" in raw:
        block_url = ""
        m = re.search(r"(//netfree\.link/block/#[^'\"}\s]+)", raw)
        if m:
            block_url = "https:" + m.group(1)

        detail = (
            "הבקשה נחסמה על ידי מסנן האינטרנט NetFree ולא הגיעה כלל לגוגל.\n"
            "    זו אינה שגיאה בקוד ואינה שגיאה של Gemini.\n"
            "    הכתובת שצריך לאשר: generativelanguage.googleapis.com\n"
        )
        if block_url:
            detail += f"    לבקשת אישור, פתחי בדפדפן:\n    {block_url}"
        else:
            detail += "    לבקשת אישור, פני לתמיכה של NetFree."

        return ("נחסם על ידי NetFree", detail)

    # ── מסננים אחרים / חסימת רשת ──
    if "418" in raw and "block" in raw.lower():
        return ("נחסם על ידי מסנן רשת", f"הבקשה נחסמה לפני שהגיעה לגוגל.\n    {raw[:300]}")

    # ── שגיאות API אמיתיות ──
    if "429" in raw:
        return ("מכסה חרגה (429)", "יותר מדי בקשות או שהמכסה נגמרה. נסי שוב בעוד כמה דקות.")
    if "401" in raw or "403" in raw or "API_KEY" in raw.upper():
        return ("בעיית מפתח API", "המפתח שגוי, פג, או שאין לו הרשאה למודל הזה.")
    if "404" in raw:
        return ("מודל לא נמצא", "שם המודל שגוי או שאינו זמין לחשבון שלך.")

    return (raw[:200], raw)


@lru_cache(maxsize=4)
def _get_client(api_key: str) -> genai.Client:
    """
    מחזירה לקוח Gemini, ויוצרת אותו רק בפעם הראשונה.

    בגרסה הקודמת נוצר לקוח חדש לכל עמוד — בזבוז מיותר
    של חיבורים ואתחולים. lru_cache שומרת את המופע לפי המפתח.
    """
    return genai.Client(api_key=api_key)


# ============================================================
# פירסור תשובת Gemini
# ============================================================

def _detect_language(text: str) -> str:
    """זיהוי שפה לפי תווים בטקסט."""
    has_hebrew = bool(re.search(r'[֐-׿]', text))
    has_latin = bool(re.search(r'[a-zA-Z]', text))

    if has_hebrew and has_latin:
        return "mixed"
    if has_hebrew:
        return "he"
    if has_latin:
        return "en"
    return "unknown"


def _extract_usage(response) -> tuple[int, int, int]:
    """
    שולפת נתוני שימוש בטוקנים מהתשובה.
    מחזירה: (קלט, פלט, חשיבה)

    usage_metadata הוא המקור האמיתי לחיוב — לא הערכה.
    """
    try:
        um = response.usage_metadata
        return (
            int(getattr(um, "prompt_token_count", 0) or 0),
            int(getattr(um, "candidates_token_count", 0) or 0),
            int(getattr(um, "thoughts_token_count", 0) or 0),
        )
    except Exception:
        return (0, 0, 0)


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

    writer_profile = _load_writer_profile(config.writer_profile_path)
    contents = _build_contents(
        processed.image, samples, config.language_mode, writer_profile
    )
    gen_config = _build_gen_config(config)
    client = _get_client(config.api_key)

    logger.info(f"שולח עמוד {processed.page_number} ל-Gemini ({config.model})...")
    start = time.monotonic()

    try:
        response = client.models.generate_content(
            model=config.model,
            contents=contents,
            config=gen_config,
        )
        elapsed_ms = int((time.monotonic() - start) * 1000)

        raw_text = response.text or ""
        in_tok, out_tok, think_tok = _extract_usage(response)

        # ── בדיקת סיבת הסיום ──
        finish_reason = ""
        try:
            finish_reason = str(response.candidates[0].finish_reason or "")
        except Exception:
            pass

        truncated = "MAX_TOKENS" in finish_reason.upper()

        # אם התשובה ריקה — ננסה לשחזר טקסט ישירות מה-parts
        if not raw_text:
            try:
                parts = response.candidates[0].content.parts or []
                raw_text = "".join(
                    p.text for p in parts if getattr(p, "text", None)
                )
            except Exception as e:
                logger.debug(f"שגיאה בשליפת parts: {e}")

        # ── אזהרות ברורות למשתמשת ──
        if truncated:
            logger.error(
                f"עמוד {processed.page_number}: התמלול נקטע! "
                f"נגמר תקציב הטוקנים ({config.max_output_tokens}). "
                f"חשיבה={think_tok} טוקנים, פלט={out_tok} טוקנים. "
                f"פתרון: להעלות MAX_OUTPUT_TOKENS ב-.env, "
                f"או להוריד THINKING_LEVEL ל-'low'."
            )
        elif not raw_text:
            logger.error(
                f"עמוד {processed.page_number}: תשובה ריקה לגמרי. "
                f"finish_reason={finish_reason}, חשיבה={think_tok} טוקנים. "
                f"אם מספר טוקני החשיבה גבוה — התקציב נשרף על חשיבה."
            )

        logger.info(
            f"עמוד {processed.page_number}: התקבל תמלול "
            f"({len(raw_text)} תווים, {elapsed_ms}ms) | "
            f"טוקנים: קלט={in_tok} פלט={out_tok} חשיבה={think_tok}"
        )

        return TranscriptionResult(
            page_number=processed.page_number,
            text=raw_text.strip(),
            uncertain_words=re.findall(r'\[([^\]]+)\?]', raw_text),
            detected_language=_detect_language(raw_text),
            api_call_ms=elapsed_ms,
            model_used=config.model,
            success=bool(raw_text),
            error_message="" if raw_text else f"תשובה ריקה (finish_reason={finish_reason})",
            truncated=truncated,
            finish_reason=finish_reason,
            input_tokens=in_tok,
            output_tokens=out_tok,
            thinking_tokens=think_tok,
        )

    except Exception as e:
        elapsed_ms = int((time.monotonic() - start) * 1000)

        short, detail = _explain_api_error(e)
        logger.error(f"עמוד {processed.page_number}: {short}\n    {detail}")
        logger.debug(f"שגיאה מקורית: {e}")

        # במקום לקרוס — מחזירים placeholder
        # document_writer יציג את זה בצורה ברורה במסמך
        return TranscriptionResult(
            page_number=processed.page_number,
            text=f"[תמלול עמוד זה נכשל — {short}]",
            success=False,
            error_message=short,
            api_call_ms=elapsed_ms,
            model_used=config.model,
        )


# ============================================================
# חישוב עלות
# ============================================================

def estimate_cost(results: list[TranscriptionResult], config: AppConfig) -> dict:
    """
    מחשבת עלות משוערת של הריצה לפי נתוני השימוש האמיתיים.

    שימי לב: גוגל מחייבת על טוקני חשיבה בתעריף הפלט —
    לכן הם נכללים בחישוב.

    התעריפים נלקחים מ-.env (PRICE_INPUT_PER_M / PRICE_OUTPUT_PER_M)
    ויש לעדכן אותם אם התמחור משתנה.
    """
    in_tok = sum(r.input_tokens for r in results)
    out_tok = sum(r.billable_output_tokens for r in results)

    cost_in = in_tok / 1_000_000 * config.price_input_per_m
    cost_out = out_tok / 1_000_000 * config.price_output_per_m

    return {
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "thinking_tokens": sum(r.thinking_tokens for r in results),
        "cost_input_usd": cost_in,
        "cost_output_usd": cost_out,
        "cost_total_usd": cost_in + cost_out,
        "cost_per_page_usd": (cost_in + cost_out) / len(results) if results else 0.0,
    }
