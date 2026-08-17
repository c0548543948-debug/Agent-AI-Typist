"""
punctuator.py — הוספת פיסוק לטקסט מתומלל, בבטחה

למה זה מודול נפרד ולא חלק מהפרומפט של התמלול:

לטקסט תורני בכתב יד כמעט אין פיסוק. הוספת פסיקים הופכת אותו לקריא —
אבל היא **פעולת עריכה ולא פעולת תמלול**. הפסיק לא נמצא בדף.

אם נוסיף פיסוק בשלב התמלול, הוא ייכנס לשכבה הדיפלומטית — זו שממנה
מאמנים את המודל. והמודל ילמד שבמקום מסוים בכתב יש פסיק, כשאין שם כלום.
זהו בדיוק אותו זיהום שנוצר כשמתקנים שגיאות של הכותב.

לכן שתי שכבות:
    שכבה דיפלומטית  — מה שכתוב בדף. ללא פיסוק. זו שכבת האימון.
    שכבה קריאה      — עם פיסוק. זו שהמשתמש מקבל.

הערובה:
אחרי הוספת הפיסוק המודול **מוריד את כל הפיסוק חזרה ומשווה למקור**.
אם אפילו אות אחת השתנתה — התוצאה נדחית והטקסט המקורי מוחזר כמו שהוא.
כלומר אי אפשר שהמעבר הזה ישנה תוכן, גם אם המודל ישתגע.
"""

from __future__ import annotations

import re

from google.genai import types
from loguru import logger

from config import AppConfig


# ============================================================
# מה מותר להוסיף
# ============================================================

# תווי הפיסוק שהמודל מורשה להכניס.
# שימי לב: גרש (') וגרשיים (") *אינם* ברשימה —
# הם חלק מקיצורים בלשון רבנית (ז"ל, וכו', עי"ש) ולא סימני פיסוק.
# אסור בתכלית האיסור לתת למודל להוסיף או להזיז אותם.
ALLOWED_PUNCT = ',.;:!?—'

_STRIP_RE = re.compile(f'[{re.escape(ALLOWED_PUNCT)}]')
_WS_RE = re.compile(r'\s+')


PUNCTUATION_PROMPT = f"""You are adding punctuation to a Hebrew text that was transcribed from a handwritten manuscript of Torah scholarship.

YOUR ONLY PERMITTED ACTION is to insert punctuation marks from this exact set:
{ALLOWED_PUNCT}

You may also insert line breaks between sentences or between distinct arguments.

ABSOLUTELY FORBIDDEN:
1. Do NOT add, remove, change, or reorder ANY Hebrew or Aramaic letter. Not one.
2. Do NOT touch apostrophes (') or gershayim ("). In this text they mark abbreviations such as forms like וכו' or ז"ל — they are NOT punctuation. Leave every one exactly where it is.
3. Do NOT expand abbreviations.
4. Do NOT fix spelling, grammar, or word order.
5. Do NOT add or remove words.
6. Do NOT translate or rephrase anything.
7. Do NOT add explanatory notes, brackets, or commentary.
8. Do NOT remove markers of the form [word?] — leave them exactly as they are, including the brackets and question mark.

If a sentence boundary is unclear, prefer adding nothing over guessing.

The output must be the input text, character for character identical, with only punctuation marks inserted between existing characters.

Output ONLY the punctuated text. No explanation."""


# ============================================================
# אימות
# ============================================================

def _skeleton(text: str) -> str:
    """
    מחזירה את "שלד" הטקסט: כל התווים למעט פיסוק מותר ורווחים.
    שני טקסטים עם אותו שלד מכילים בהכרח בדיוק את אותן אותיות
    באותו סדר — כלומר לא נוסף ולא נגרע תוכן.
    """
    return _WS_RE.sub('', _STRIP_RE.sub('', text))


def verify_only_punctuation_changed(original: str, punctuated: str) -> tuple[bool, str]:
    """
    בודקת שהשינוי היחיד היה הוספת פיסוק.
    מחזירה: (תקין, הסבר)
    """
    a, b = _skeleton(original), _skeleton(punctuated)
    if a == b:
        return True, ""

    # מוצאים את המקום הראשון שנשבר, כדי שההודעה תהיה שימושית
    n = min(len(a), len(b))
    i = next((k for k in range(n) if a[k] != b[k]), n)
    ctx_a = a[max(0, i - 25):i + 25]
    ctx_b = b[max(0, i - 25):i + 25]
    return False, (
        f"התוכן השתנה בתו {i} מתוך {len(a)}.\n"
        f"    מקור: ...{ctx_a}...\n"
        f"    פלט:  ...{ctx_b}..."
    )


# ============================================================
# הפונקציה הראשית
# ============================================================

def add_punctuation(text: str, config: AppConfig, client=None) -> tuple[str, bool]:
    """
    מוסיפה פיסוק לטקסט מתומלל.

    מחזירה: (טקסט, האם הצליח)
    אם האימות נכשל — מוחזר הטקסט המקורי ללא שינוי, וההצלחה False.
    כלומר במקרה הגרוע לא קורה כלום. אין מצב שבו תוכן נפגע.
    """
    if not text.strip():
        return text, False

    if client is None:
        from ocr_engine import _get_client
        client = _get_client(config.api_key)

    # לפיסוק אין צורך בחשיבה עמוקה — זו משימה תחבירית פשוטה.
    # thinking נמוך חוסך כאן כסף בלי לפגוע באיכות.
    kwargs = {"max_output_tokens": config.max_output_tokens}
    try:
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_level="LOW")
    except (AttributeError, TypeError, ValueError):
        pass

    try:
        response = client.models.generate_content(
            model=config.model,
            contents=[PUNCTUATION_PROMPT, f"TEXT TO PUNCTUATE:\n\n{text}"],
            config=types.GenerateContentConfig(**kwargs),
        )
        result = (response.text or "").strip()
    except Exception as e:
        logger.warning(f"הוספת פיסוק נכשלה: {e}")
        return text, False

    if not result:
        logger.warning("הוספת פיסוק החזירה תשובה ריקה — הטקסט נשאר כפי שהוא")
        return text, False

    ok, why = verify_only_punctuation_changed(text, result)
    if not ok:
        logger.error(
            "הוספת הפיסוק נדחתה — המודל שינה תוכן ולא רק פיסוק.\n"
            f"    {why}\n"
            "    הטקסט המקורי נשמר ללא שינוי."
        )
        return text, False

    added = len(_STRIP_RE.findall(result)) - len(_STRIP_RE.findall(text))
    logger.info(f"פיסוק נוסף בבטחה ({added} סימנים). התוכן לא שונה.")
    return result, True


# ============================================================
# בדיקה עצמית
# ============================================================

if __name__ == "__main__":
    src = 'ובין תפל ולבן וברש"י אמר רשב"י חזרנו על כל המקרא ולא מצינו מקום ששמו תופל'

    cases = [
        ("פיסוק בלבד — אמור לעבור",
         'ובין תפל ולבן, וברש"י אמר רשב"י: חזרנו על כל המקרא, ולא מצינו מקום ששמו תופל.'),
        ("מילה נוספה — אמור להידחות",
         'ובין תפל ולבן, וברש"י אמר רשב"י: חזרנו על כל המקרא הקדוש, ולא מצינו מקום ששמו תופל.'),
        ("קיצור נפתח — אמור להידחות",
         'ובין תפל ולבן, וברשי אמר רשבי: חזרנו על כל המקרא, ולא מצינו מקום ששמו תופל.'),
        ("אות שונתה — אמור להידחות",
         'ובין תפל ולבן, וברש"י אמר רשב"י: חזרנו על כל המקרה, ולא מצינו מקום ששמו תופל.'),
        ("שורות חדשות — אמור לעבור",
         'ובין תפל ולבן,\nוברש"י אמר רשב"י:\nחזרנו על כל המקרא, ולא מצינו מקום ששמו תופל.'),
    ]

    print("בדיקת מנגנון האימות:\n")
    for name, cand in cases:
        ok, why = verify_only_punctuation_changed(src, cand)
        print(f"  [{'עבר ' if ok else 'נדחה'}]  {name}")
        if not ok:
            print(f"          {why.splitlines()[0]}")
