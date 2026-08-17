"""
config.py — הגדרות מרכזיות של AI-Typist Agent
כל ערכי ברירת המחדל נמצאים כאן.
ניתן לדרוס כל הגדרה דרך קובץ .env או פרמטר CLI.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# טעינת משתני סביבה מקובץ .env (אם קיים)
load_dotenv()

# ===== נתיבים =====
BASE_DIR = Path(__file__).parent
LOGS_DIR = BASE_DIR / "logs"
FEW_SHOT_DIR = BASE_DIR / "few_shot"

# ודא שתיקיות קיימות
LOGS_DIR.mkdir(exist_ok=True)


def _env_bool(name: str, default: bool) -> bool:
    """קורא משתנה סביבה בוליאני בצורה סובלנית."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class AppConfig(BaseModel):
    """הגדרות המערכת — נטענות מ-.env עם ערכי ברירת מחדל"""

    # ===== Gemini API =====
    api_key: str = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    model: str = Field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
    )
    # חשוב: להשתמש במודל *הבנה* ולא במודל *יצירת תמונות*.
    # gemini-3-pro-image הוא מודל ליצירת ועריכת תמונות — הוא ימציא טקסט.

    # ===== הגדרות בקשה (Gemini 3) =====
    max_output_tokens: int = Field(
        default_factory=lambda: int(os.getenv("MAX_OUTPUT_TOKENS", "65536"))
    )
    # ב-Gemini 3 טוקני החשיבה נספרים בתוך התקציב הזה.
    # תקציב קטן מדי (למשל 8192) גורם לתמלול שנקטע באמצע העמוד.
    #
    # 65536 הוא המקסימום של gemini-3.1-pro. חשוב להבין:
    # התקציב הוא *תקרה* ולא הזמנה — משלמים רק על טוקנים שנוצרו בפועל.
    # לכן אין שום עלות בהגדרת תקרה גבוהה, ויש רק תועלת: אין קטיעה.

    thinking_level: str = Field(
        default_factory=lambda: os.getenv("THINKING_LEVEL", "high")
    )
    # "minimal" | "low" | "medium" | "high"
    #
    # גוגל ממליצה low לתמלול — אבל ההמלצה הזו נכתבה למסמכים רגילים.
    # לכתב יד קשה מאוד פענוח אות מטושטשת *הוא* בעיה שדורשת היגיון,
    # ובבדיקה בפועל high נתן תמלול נאמן יותר.
    # לריצות זולות על כתב יד קריא: --thinking low

    media_resolution: str = Field(
        default_factory=lambda: os.getenv("MEDIA_RESOLUTION", "high")
    )
    # "low" | "medium" | "high"
    # קובע כמה טוקנים מוקצים לתמונה. לכתב יד קשה — high.

    # ===== עיבוד תמונה =====
    render_dpi: int = Field(default_factory=lambda: int(os.getenv("RENDER_DPI", "300")))
    # DPI גבוה = תמונה ברורה יותר, חשוב במיוחד לכתב יד

    # ===== תמלול =====
    confidence_threshold: float = Field(
        default_factory=lambda: float(os.getenv("CONFIDENCE_THRESHOLD", "0.80"))
    )
    # מתחת לסף זה, מילה מסומנת כ-[ניחוש?]

    language_mode: str = Field(
        default_factory=lambda: os.getenv("LANGUAGE_MODE", "auto")
    )
    # "he" | "en" | "auto"
    # auto = Gemini מזהה שפה לכל פסקה בנפרד

    # ===== Few-Shot =====
    use_few_shot: bool = Field(
        default_factory=lambda: _env_bool("USE_FEW_SHOT", False)
    )
    # כבוי כברירת מחדל!
    # Few-Shot ברמת עמוד שלם מזהם את ההקשר — המודל לומד את *הנושא*
    # של הדוגמאות ומשלים ממנו תוכן לתמונה החדשה.
    # להפעלה חד-פעמית: python main.py --few-shot

    max_few_shot_samples: int = Field(
        default_factory=lambda: int(os.getenv("MAX_FEW_SHOT_SAMPLES", "5"))
    )
    few_shot_dir: str = Field(
        default_factory=lambda: os.getenv("FEW_SHOT_DIR", str(FEW_SHOT_DIR))
    )

    # ===== תמלול שורה אחר שורה =====
    by_lines: bool = Field(
        default_factory=lambda: _env_bool("BY_LINES", False)
    )
    # מחלק את העמוד לשורות ומתמלל כל אחת בנפרד.
    # יתרון: רזולוציה גבוהה בהרבה לכל שורה, ואין מרחב להמצאה.
    # חיסרון: קריאה נפרדת לכל שורה — יקר ואיטי יותר.

    line_max_output_tokens: int = Field(
        default_factory=lambda: int(os.getenv("LINE_MAX_OUTPUT_TOKENS", "2048"))
    )
    # שורה היא כמה עשרות מילים. תקציב עמוד שלם כאן הוא בזבוז.

    line_thinking_level: str = Field(
        default_factory=lambda: os.getenv("LINE_THINKING_LEVEL", "low")
    )
    # ברמת שורה היחידה קטנה, וכופלים אותה בעשרות שורות.
    # חשיבה גבוהה כאן מייקרת פי כמה בלי תועלת ברורה.

    # ===== פיסוק =====
    add_punctuation: bool = Field(
        default_factory=lambda: _env_bool("ADD_PUNCTUATION", True)
    )
    # מוסיף פיסוק במעבר שני ונפרד, אחרי התמלול.
    #
    # למה נפרד: פסיק אינו נמצא בדף. הוספתו היא עריכה ולא תמלול.
    # אם היא תיכנס לשלב התמלול, הפיסוק ייכנס לנתוני האימון
    # והמודל ילמד לראות פסיקים במקום שאין בו כלום.
    #
    # השכבה הדיפלומטית (ללא פיסוק) נשמרת תמיד לקובץ נפרד,
    # והיא זו שמשמשת לאימון. הפיסוק מופיע רק במסמך למשתמש.
    # לכיבוי: --no-punctuate

    # ===== פרופיל כתב היד =====
    writer_profile_path: str = Field(
        default_factory=lambda: os.getenv(
            "WRITER_PROFILE", str(BASE_DIR / "writer_profile.txt")
        )
    )
    # קובץ טקסט חופשי שמתאר את מאפייני כתב היד (צורת אותיות בלבד).
    # התוכן נוסף לפרומפט אוטומטית. שורות שמתחילות ב-# הן הערות.
    # זו חלופה בטוחה ל-Few-Shot: מלמדת צורת אותיות בלי להכניס תוכן.

    # ===== תמחור (לדיווח עלות בלבד) =====
    price_input_per_m: float = Field(
        default_factory=lambda: float(os.getenv("PRICE_INPUT_PER_M", "2.0"))
    )
    price_output_per_m: float = Field(
        default_factory=lambda: float(os.getenv("PRICE_OUTPUT_PER_M", "12.0"))
    )
    # דולר למיליון טוקנים. ברירת המחדל לפי תמחור gemini-3.1-pro.
    # שימי לב: גוגל מחייבת על טוקני חשיבה בתעריף הפלט.
    # לעדכון: https://ai.google.dev/gemini-api/docs/pricing

    # ===== מסמך פלט =====
    default_font: str = "Arial"
    default_font_pt: int = 12             # גודל גופן בנקודות
    line_spacing: float = 1.15
    uncertainty_color: str = "FFFF00"    # צהוב
    output_merge: bool = Field(
        default_factory=lambda: _env_bool("OUTPUT_MERGE", True)
    )
    # True = כל העמודים בקובץ אחד, False = קובץ לכל עמוד

    # ===== לוגים =====
    log_level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    # ===== Web UI =====
    web_port: int = Field(default_factory=lambda: int(os.getenv("WEB_PORT", "5000")))
    max_upload_mb: int = 50

    # ===== תאימות לאחור =====
    @property
    def default_font_size(self) -> int:
        """
        גודל גופן ביחידות half-points, כפי ש-document_writer מצפה.
        נשמר לתאימות עם קוד קיים.
        """
        return self.default_font_pt * 2

    def validate_api_key(self) -> None:
        """בדיקה שמפתח API הוגדר"""
        if not self.api_key:
            raise ValueError(
                "מפתח GEMINI_API_KEY חסר.\n"
                "1. העתיקי את .env.example ל-.env\n"
                "2. הכניסי את המפתח שלך מ-https://aistudio.google.com/app/apikey"
            )


# instance גלובלי — מיובא בכל המודולים
config = AppConfig()
