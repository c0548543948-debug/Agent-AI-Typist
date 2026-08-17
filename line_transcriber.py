"""
line_transcriber.py — תמלול שורה אחר שורה

למה בכלל:
בתמלול ברמת עמוד, כל שורה מקבלת נתח קטן מתקציב הפיקסלים שהמודל רואה.
גרש הוא כמה פיקסלים בודדים, ובעמוד שלם הוא פשוט לא מגיע.
כשכל שורה נשלחת לבדה, אותו גרש מקבל פי כמה רזולוציה.

יתרון שני, לא פחות חשוב: בשורה בודדת אין מרחב להמציא.
כשהמודל לא הצליח לקרוא עמוד שלם הוא חיבר דיון תורני שלם.
כשהוא לא מצליח לקרוא שורה — הוא יכול לטעות במילה, לא בעמוד.

עלות — והנקודה שצריך לשים לב אליה:
הפרומפט נשלח מחדש בכל בקשה. פרומפט של 7,000 תווים כפול 32 שורות
הוא בזבוז אמיתי. לכן כאן משתמשים בפרומפט מקוצר שמכיל רק את הכללים
שבאמת משנים ברמת שורה, ואפשר גם לשלוח כמה שורות בבקשה אחת.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from google.genai import types
from loguru import logger

from config import AppConfig
from line_segmenter import TextLine
from ocr_engine import _get_client, _explain_api_error, _load_writer_profile


# ============================================================
# פרומפט מקוצר לשורה בודדת
# ============================================================

LINE_PROMPT = """You are transcribing ONE LINE from a handwritten Hebrew manuscript of original Torah scholarship.

You have never seen this text. You cannot know what it says. Read only what is drawn on the page.

RULES:
1. Copy the SHAPES of the letters, one by one. Do NOT read for meaning. Do NOT use Torah knowledge, grammar or context to decide what a word "should" be. A strange reading that matches the shapes is correct; a sensible reading that does not match the shapes is wrong.
2. Output ONLY the text of this single line. No explanation, no line number, nothing else.
3. An apostrophe (') or gershayim (") is a mark drawn on the page. If you see one, that cluster is an ABBREVIATION — copy the letters plus the mark exactly. Never replace it with an ordinary Hebrew word that resembles it, and never expand it into the words it stands for. These marks are small: look for them.
4. Do not fix spelling, grammar or word order. Do not add punctuation that is not written. Do not translate.
5. If a word is unclear, write it as [word?] with your best guess inside. Marking is cheap; a silent wrong word is not. Expect to mark something on a hard line.
6. If the line is blank or unreadable, output exactly: [לא קריא]
7. Write only Hebrew and Aramaic. Use [מחוק] for struck-through text. Never write English."""


# ============================================================
# תוצאה
# ============================================================

@dataclass
class LineResult:
    """תוצאת תמלול של שורה אחת."""
    index: int
    text: str
    success: bool = True
    error: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    ms: int = 0

    @property
    def billable_output_tokens(self) -> int:
        return self.output_tokens + self.thinking_tokens


@dataclass
class PageLinesResult:
    """כל השורות של עמוד."""
    page_number: int
    lines: list[LineResult] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(l.text for l in self.lines if l.text.strip())

    @property
    def input_tokens(self) -> int:
        return sum(l.input_tokens for l in self.lines)

    @property
    def output_tokens(self) -> int:
        return sum(l.billable_output_tokens for l in self.lines)

    @property
    def thinking_tokens(self) -> int:
        return sum(l.thinking_tokens for l in self.lines)

    @property
    def ms(self) -> int:
        return sum(l.ms for l in self.lines)


# ============================================================
# הגדרות הבקשה לשורה
# ============================================================

def _line_config(cfg: AppConfig) -> types.GenerateContentConfig:
    """
    הגדרות לבקשת שורה בודדת.

    תקציב פלט קטן — שורה היא כמה עשרות מילים, לא עמוד.
    רמת חשיבה נמוכה — היחידה קטנה, ומכפילים אותה בעשרות שורות.
    """
    kwargs: dict = {"max_output_tokens": cfg.line_max_output_tokens}

    try:
        kwargs["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(disable=True)
    except AttributeError:
        pass
    try:
        kwargs["media_resolution"] = getattr(
            types.MediaResolution, f"MEDIA_RESOLUTION_{cfg.media_resolution.upper()}"
        )
    except AttributeError:
        pass
    try:
        kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_level=cfg.line_thinking_level.upper()
        )
    except (AttributeError, TypeError, ValueError):
        pass

    try:
        return types.GenerateContentConfig(**kwargs)
    except TypeError:
        return types.GenerateContentConfig(max_output_tokens=cfg.line_max_output_tokens)


# ============================================================
# תמלול שורה
# ============================================================

_CLEAN = re.compile(r'^\s*(?:\d+[.):]\s*)?')      # מספור שורה שהמודל מוסיף לפעמים


def transcribe_line(
    line: TextLine,
    cfg: AppConfig,
    client=None,
    writer_profile: str = "",
) -> LineResult:
    """מתמללת שורה אחת ומחזירה את הטקסט שלה."""
    client = client or _get_client(cfg.api_key)

    prompt = LINE_PROMPT
    if writer_profile:
        prompt += (
            "\n\nNotes on this writer's letter shapes (use ONLY to read shapes, "
            f"they say nothing about content):\n{writer_profile}"
        )

    start = time.monotonic()
    try:
        response = client.models.generate_content(
            model=cfg.model,
            contents=[prompt, line.image],
            config=_line_config(cfg),
        )
        elapsed = int((time.monotonic() - start) * 1000)

        text = (response.text or "").strip()
        text = _CLEAN.sub("", text).strip()

        um = getattr(response, "usage_metadata", None)
        return LineResult(
            index=line.index,
            text=text,
            success=bool(text),
            error="" if text else "תשובה ריקה",
            input_tokens=int(getattr(um, "prompt_token_count", 0) or 0) if um else 0,
            output_tokens=int(getattr(um, "candidates_token_count", 0) or 0) if um else 0,
            thinking_tokens=int(getattr(um, "thoughts_token_count", 0) or 0) if um else 0,
            ms=elapsed,
        )

    except Exception as e:
        short, detail = _explain_api_error(e)
        logger.error(f"שורה {line.index}: {short}\n    {detail}")
        return LineResult(
            index=line.index,
            text=f"[שורה {line.index} נכשלה]",
            success=False,
            error=short,
            ms=int((time.monotonic() - start) * 1000),
        )


def transcribe_lines(
    lines: list[TextLine],
    cfg: AppConfig,
    page_number: int = 1,
) -> PageLinesResult:
    """
    מתמללת את כל שורות העמוד, אחת אחרי השנייה.

    הערה על מקביליות: אפשר להריץ כמה שורות במקביל ולקצר מאוד את הזמן.
    כאן זה טורי בכוונה — כדי שהמדידה הראשונה תהיה פשוטה וברורה,
    ושלא נתנגש במגבלות קצב של ה-API בזמן שאנחנו עוד בודקים אם זה בכלל עוזר.
    """
    from tqdm import tqdm

    client = _get_client(cfg.api_key)
    profile = _load_writer_profile(cfg.writer_profile_path)

    result = PageLinesResult(page_number=page_number)
    for ln in tqdm(lines, desc=f"עמוד {page_number}", unit="שורה"):
        result.lines.append(transcribe_line(ln, cfg, client, profile))

    ok = sum(1 for l in result.lines if l.success)
    logger.info(
        f"עמוד {page_number}: {ok}/{len(lines)} שורות | "
        f"טוקנים קלט={result.input_tokens} פלט={result.output_tokens} "
        f"חשיבה={result.thinking_tokens} | {result.ms/1000:.0f} שניות"
    )
    return result