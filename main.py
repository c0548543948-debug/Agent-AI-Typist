"""
main.py — נקודת הכניסה של AI-Typist Agent

האחריות של הקובץ הזה:
1. לקרוא פרמטרים מהמשתמש (CLI או הפעלת Web)
2. לאתחל לוגים והגדרות
3. לתאם את הרצף: תמונות → תמלול → מסמך Word
4. להפעיל את שרת ה-Web במצב --web

הקובץ הזה לא מכיל לוגיקה עסקית — רק מחבר בין המודולים.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from config import AppConfig, config as default_config
from image_processor import prepare_images
from ocr_engine import transcribe_image, load_few_shot_samples
from document_writer import create_docx


# ============================================================
# אתחול לוגים
# ============================================================

def setup_logging(log_level: str) -> None:
    """
    מגדירה את מערכת הלוגים (loguru).

    loguru מחליפה את ה-logging הסטנדרטי של Python.
    היא פשוטה יותר לשימוש ומדפיסה לוגים יפים יותר.

    אנחנו כותבים לשני מקומות:
    1. Terminal — כדי שהמשתמש יראה התקדמות בזמן אמת
    2. קובץ בתיקיית logs/ — כדי שאפשר יהיה לבדוק מה קרה אחרי הריצה
    """
    # הסרת ה-handler הברירת מחדל
    logger.remove()

    # הדפסה לטרמינל
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        colorize=True,
    )

    # כתיבה לקובץ — שם הקובץ כולל חותמת זמן
    log_file = Path("logs") / "{time:YYYY-MM-DD_HH-mm-ss}.log"
    logger.add(
        str(log_file),
        level="DEBUG",   # בקובץ — כל הפרטים, כולל DEBUG
        encoding="utf-8",
        rotation="10 MB",  # קובץ חדש אם הנוכחי עולה על 10MB
    )


# ============================================================
# הרצת תמלול — הלוגיקה המרכזית
# ============================================================

def run_transcription(
    input_path: str,
    output_path: str,
    cfg: AppConfig,
    few_shot_dir: str | None = None,
    skip_preprocess: bool = False,
) -> None:
    """
    מריצה את כל תהליך התמלול מקצה לקצה.

    1. עיבוד קובץ הקלט לתמונות
    2. טעינת דוגמאות Few-Shot (אם יש)
    3. תמלול כל עמוד עם Gemini
    4. יצירת קובץ Word

    few_shot_dir — תיקייה חלופית לדוגמאות (דורס את ברירת המחדל)
    """
    logger.info(f"קלט: {input_path}")
    logger.info(f"פלט: {output_path}")
    logger.info(f"מצב שפה: {cfg.language_mode}")

    # ── שלב א: עיבוד תמונות ──
    logger.info("שלב 1/3: מעבד תמונות...")
    try:
        images = prepare_images(input_path, dpi=cfg.render_dpi, skip_preprocess=skip_preprocess)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logger.error(f"שגיאה בעיבוד תמונות: {e}")
        sys.exit(1)

    logger.info(f"עמודים לתמלול: {len(images)}")

    # ── שלב ב: טעינת Few-Shot ──
    fs_dir = few_shot_dir or cfg.few_shot_dir
    few_shot = load_few_shot_samples(fs_dir, cfg.max_few_shot_samples)

    # ── שלב ג: תמלול עמוד אחר עמוד ──
    logger.info("שלב 2/3: שולח ל-Gemini...")
    results = []

    # tqdm מציג סרגל התקדמות בטרמינל
    # כך המשתמש רואה כמה עמודים נותרו ולא חושב שהתוכנית קפאה
    for processed_img in tqdm(images, desc="מתמלל", unit="עמוד"):
        result = transcribe_image(processed_img, cfg, few_shot)
        results.append(result)

        if result.success:
            uncertain_str = (
                f" | {len(result.uncertain_words)} מילים לא-ודאיות"
                if result.uncertain_words else ""
            )
            logger.debug(f"עמוד {result.page_number}: הושלם{uncertain_str}")
        else:
            logger.warning(
                f"עמוד {result.page_number}: נכשל — {result.error_message}"
            )

    # ── שלב ד: יצירת מסמך Word ──
    logger.info("שלב 3/3: יוצר מסמך Word...")
    create_docx(results, output_path, cfg)

    # סיכום
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful
    total_uncertain = sum(len(r.uncertain_words) for r in results)
    total_sec = sum(r.api_call_ms for r in results) / 1000

    logger.info("─" * 40)
    logger.info(f"הושלם בהצלחה!")
    logger.info(f"עמודים: {successful}/{len(results)}" + (f" ({failed} נכשלו)" if failed else ""))
    logger.info(f"מילים לא-ודאיות: {total_uncertain}")
    logger.info(f"זמן עיבוד API: {total_sec:.1f} שניות")
    logger.info(f"קובץ נשמר: {output_path}")


# ============================================================
# פרמטרי CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    """
    מגדירה את פרמטרי שורת הפקודה.

    argparse היא ספריית Python סטנדרטית לניהול פרמטרי CLI.
    היא גם יוצרת אוטומטית הודעת --help למשתמש.
    """
    parser = argparse.ArgumentParser(
        description="AI-Typist Agent — המרת כתב יד לקובץ Word",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
דוגמאות שימוש:
  python main.py --web
  python main.py -i scan.pdf -o output.docx
  python main.py -i photo.jpg -o output.docx --language he
  python main.py -i scan.pdf --few-shot-dir ./my_samples
        """,
    )

    # ── מצב Web ──
    parser.add_argument(
        "--web",
        action="store_true",
        help="הפעל ממשק Web (במקום CLI)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="פורט שרת ה-Web (ברירת מחדל: 5000)",
    )

    # ── קלט ופלט ──
    parser.add_argument(
        "-i", "--input",
        help="נתיב קובץ קלט (PDF / JPG / PNG / WEBP)",
    )
    parser.add_argument(
        "-o", "--output",
        help="נתיב קובץ Word פלט (ברירת מחדל: שם_קלט_transcribed.docx)",
    )

    # ── הגדרות תמלול ──
    parser.add_argument(
        "--language",
        choices=["he", "en", "auto"],
        default=None,
        help="מצב שפה: he=עברית, en=אנגלית, auto=זיהוי אוטומטי (ברירת מחדל: auto)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="סף ביטחון 0.0–1.0 (ברירת מחדל: 0.80)",
    )
    parser.add_argument(
        "--few-shot-dir",
        default=None,
        help="תיקיית דוגמאות כיול Few-Shot",
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="דלג על עיבוד תמונה מקדים (deskew, ניגודיות)",
    )

    # ── לוגים ──
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="רמת פירוט הלוגים (ברירת מחדל: INFO)",
    )

    return parser


def _build_output_path(input_path: str) -> str:
    """
    בונה נתיב פלט אוטומטי אם המשתמש לא ציין -o.
    דוגמה: scan.pdf → scan_transcribed.docx
    """
    p = Path(input_path)
    return str(p.parent / f"{p.stem}_transcribed.docx")


# ============================================================
# נקודת כניסה
# ============================================================

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # בניית config — דורס ערכי ברירת מחדל לפי פרמטרי CLI
    cfg = default_config

    if args.language:
        cfg.language_mode = args.language
    if args.confidence:
        cfg.confidence_threshold = args.confidence
    if args.log_level:
        cfg.log_level = args.log_level
    if args.port:
        cfg.web_port = args.port

    # אתחול לוגים
    setup_logging(cfg.log_level)
    logger.info("AI-Typist Agent מופעל")

    # ── מצב Web ──
    if args.web:
        logger.info(f"מפעיל ממשק Web על פורט {cfg.web_port}...")
        # ה-import כאן (ולא בראש הקובץ) כדי שהפעלת CLI לא תדרוש Flask
        from web_ui.app import create_app
        app = create_app(cfg)
        app.run(
            host="127.0.0.1",
            port=cfg.web_port,
            debug=False,
        )
        return

    # ── מצב CLI ──
    if not args.input:
        logger.error("חסר קובץ קלט. השתמשי ב: -i קובץ.pdf, או --web לממשק גרפי")
        parser.print_help()
        sys.exit(1)

    # בדיקת מפתח API
    try:
        cfg.validate_api_key()
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    output = args.output or _build_output_path(args.input)

    run_transcription(
        input_path=args.input,
        output_path=output,
        cfg=cfg,
        few_shot_dir=args.few_shot_dir,
        skip_preprocess=args.no_preprocess,
    )


if __name__ == "__main__":
    main()
