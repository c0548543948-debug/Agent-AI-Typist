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


class AppConfig(BaseModel):
    """הגדרות המערכת — נטענות מ-.env עם ערכי ברירת מחדל"""

    # ===== Gemini API =====
    api_key: str = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    model: str = Field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))

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
    max_few_shot_samples: int = Field(
        default_factory=lambda: int(os.getenv("MAX_FEW_SHOT_SAMPLES", "5"))
    )
    few_shot_dir: str = Field(
        default_factory=lambda: os.getenv("FEW_SHOT_DIR", str(FEW_SHOT_DIR))
    )

    # ===== מסמך פלט =====
    default_font: str = "Arial"
    default_font_size: int = 24          # יחידות docx (half-points) — 24 = 12pt
    line_spacing: float = 1.15
    uncertainty_color: str = "FFFF00"    # צהוב
    output_merge: bool = Field(
        default_factory=lambda: os.getenv("OUTPUT_MERGE", "true").lower() == "true"
    )
    # True = כל העמודים בקובץ אחד, False = קובץ לכל עמוד

    # ===== לוגים =====
    log_level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    # ===== Web UI =====
    web_port: int = Field(default_factory=lambda: int(os.getenv("WEB_PORT", "5000")))
    max_upload_mb: int = 50

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
