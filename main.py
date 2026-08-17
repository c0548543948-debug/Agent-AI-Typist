"""
main.py — נקודת הכניסה של AI-Typist Agent

האחריות של הקובץ הזה:
1. לקרוא פרמטרים מהמשתמש (CLI או הפעלת Web)
2. לאתחל לוגים והגדרות
3. לתאם את הרצף: תמונות → תמלול → מסמך Word
4. לדווח על שימוש בטוקנים ועלות
5. להפעיל את שרת ה-Web במצב --web

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
from ocr_engine import transcribe_image, load_few_shot_samples, estimate_cost
from document_writer import create_docx


# ============================================================
# אתחול לוגים
# ============================================================

def setup_logging(log_level: str) -> None:
    """
    מגדירה את מערכת הלוגים (loguru).

    אנחנו כותבים לשני מקומות:
    1. Terminal — כדי שהמשתמש יראה התקדמות בזמן אמת
    2. קובץ בתיקיית logs/ — כדי שאפשר יהיה לבדוק מה קרה אחרי הריצה
    """
    logger.remove()

    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        colorize=True,
    )

    log_file = Path("logs") / "{time:YYYY-MM-DD_HH-mm-ss}.log"
    logger.add(
        str(log_file),
        level="DEBUG",
        encoding="utf-8",
        rotation="10 MB",
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
    auto_orient: bool = False,
) -> None:
    """
    מריצה את כל תהליך התמלול מקצה לקצה.

    1. עיבוד קובץ הקלט לתמונות
    2. טעינת דוגמאות Few-Shot (רק אם cfg.use_few_shot דלוק)
    3. תמלול כל עמוד עם Gemini
    4. יצירת קובץ Word
    5. דיווח שימוש ועלות
    """
    logger.info(f"קלט: {input_path}")
    logger.info(f"פלט: {output_path}")
    logger.info(f"מודל: {cfg.model}")
    logger.info(
        f"מצב שפה: {cfg.language_mode} | "
        f"חשיבה: {cfg.thinking_level} | "
        f"רזולוציה: {cfg.media_resolution} | "
        f"תקציב פלט: {cfg.max_output_tokens}"
    )

    # ── שלב א: עיבוד תמונות ──
    logger.info("שלב 1/3: מעבד תמונות...")
    try:
        images = prepare_images(
            input_path,
            dpi=cfg.render_dpi,
            skip_preprocess=skip_preprocess,
            auto_orient=auto_orient,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logger.error(f"שגיאה בעיבוד תמונות: {e}")
        sys.exit(1)

    logger.info(f"עמודים לתמלול: {len(images)}")

    # ── שלב ב: טעינת Few-Shot (רק בבקשה מפורשת) ──
    if cfg.use_few_shot:
        fs_dir = few_shot_dir or cfg.few_shot_dir
        logger.warning(
            "Few-Shot מופעל. שימי לב: דוגמאות ברמת עמוד שלם עלולות "
            "לגרום למודל להשלים תוכן מהן במקום לתמלל את התמונה."
        )
        few_shot = load_few_shot_samples(fs_dir, cfg.max_few_shot_samples)
    else:
        few_shot = []
        logger.info("Few-Shot כבוי (מומלץ). להפעלה: --few-shot")

    # ── שלב ג: תמלול עמוד אחר עמוד ──
    logger.info("שלב 2/3: שולח ל-Gemini...")
    results = []

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

    # ── שלב ד: שמירת השכבה הדיפלומטית ──
    # זו השכבה שמשמשת לאימון: בדיוק מה שנקרא מהדף, בלי פיסוק שנוסף.
    # נשמרת *לפני* מעבר הפיסוק, כי אחריו היא כבר לא דיפלומטית.
    diplomatic_path = Path(output_path).with_name(
        Path(output_path).stem + "_diplomatic.txt"
    )
    _save_diplomatic(results, diplomatic_path)

    # ── שלב ה: פיסוק (מעבר נפרד) ──
    if cfg.add_punctuation:
        logger.info("מוסיף פיסוק (מעבר נפרד, עם אימות שהתוכן לא משתנה)...")
        _punctuate_results(results, cfg)

    # ── שלב ו: יצירת מסמך Word ──
    logger.info("שלב 3/3: יוצר מסמך Word...")
    create_docx(results, output_path, cfg)

    _print_summary(results, output_path, cfg)


def _save_diplomatic(results: list, path: Path) -> None:
    """
    שומרת את התמלול הגולמי — בדיוק כפי שנקרא מהדף.

    זהו הקובץ שישמש לאימון מודל בעתיד, ולכן הוא לא עובר פיסוק,
    לא עובר תיקון שגיאות של הכותב, ולא עובר שום עריכה.
    """
    try:
        blocks = []
        for r in results:
            blocks.append(f"### עמוד {r.page_number}\n{r.text}")
        path.write_text("\n\n".join(blocks), encoding="utf-8")
        logger.info(f"שכבה דיפלומטית (לאימון) נשמרה: {path.name}")
    except Exception as e:
        logger.warning(f"שמירת השכבה הדיפלומטית נכשלה: {e}")


def _punctuate_results(results: list, cfg: AppConfig) -> None:
    """
    מריצה את מעבר הפיסוק על כל עמוד שהצליח.
    אם האימות נכשל בעמוד מסוים — אותו עמוד נשאר בלי פיסוק,
    והשאר ממשיכים. אין מצב שתוכן נפגע.
    """
    from punctuator import add_punctuation

    ok_count = 0
    for r in results:
        if not r.success or not r.text.strip():
            continue
        r.text, ok = add_punctuation(r.text, cfg)
        if ok:
            ok_count += 1
        else:
            logger.warning(f"עמוד {r.page_number}: נשאר בלי פיסוק")

    logger.info(f"פיסוק הושלם ב-{ok_count} עמודים")


def _print_summary(results: list, output_path: str, cfg: AppConfig) -> None:
    """מדפיסה סיכום ריצה כולל שימוש בטוקנים ועלות משוערת."""
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful
    truncated = [r.page_number for r in results if r.truncated]
    total_uncertain = sum(len(r.uncertain_words) for r in results)
    total_sec = sum(r.api_call_ms for r in results) / 1000

    usage = estimate_cost(results, cfg)

    logger.info("─" * 50)
    logger.info("הושלם!")
    logger.info(f"עמודים: {successful}/{len(results)}" + (f" ({failed} נכשלו)" if failed else ""))
    logger.info(f"מילים לא-ודאיות: {total_uncertain}")
    logger.info(f"זמן עיבוד API: {total_sec:.1f} שניות")
    logger.info("─" * 50)
    logger.info("שימוש בטוקנים:")
    logger.info(f"  קלט:   {usage['input_tokens']:>9,}")
    logger.info(f"  פלט:   {usage['output_tokens']:>9,}  (כולל חשיבה)")
    logger.info(f"  חשיבה: {usage['thinking_tokens']:>9,}")
    logger.info("עלות משוערת:")
    logger.info(f"  קלט:    ${usage['cost_input_usd']:.4f}")
    logger.info(f"  פלט:    ${usage['cost_output_usd']:.4f}")
    logger.info(f"  סה\"כ:   ${usage['cost_total_usd']:.4f}")
    logger.info(f"  לעמוד:  ${usage['cost_per_page_usd']:.4f}")
    logger.info("─" * 50)

    # אזהרה בולטת אם תמלול נקטע
    if truncated:
        logger.error(
            f"אזהרה: העמודים {truncated} נקטעו באמצע — נגמר תקציב הטוקנים.\n"
            f"    התקציב הנוכחי: {cfg.max_output_tokens}\n"
            f"    פתרון: העלי MAX_OUTPUT_TOKENS ב-.env (למשל ל-65536),\n"
            f"    או הורידי THINKING_LEVEL ל-'minimal'."
        )

    logger.info(f"קובץ נשמר: {output_path}")


# ============================================================
# פרמטרי CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    """מגדירה את פרמטרי שורת הפקודה."""
    parser = argparse.ArgumentParser(
        description="AI-Typist Agent — המרת כתב יד לקובץ Word",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
דוגמאות שימוש:
  python main.py --web
  python main.py -i scan.pdf -o output.docx
  python main.py -i photo.jpg -o output.docx --language he
  python main.py -i scan.pdf --thinking minimal --max-tokens 65536
  python main.py -i scan.pdf --few-shot          (לא מומלץ)
        """,
    )

    # ── מצב Web ──
    parser.add_argument("--web", action="store_true", help="הפעל ממשק Web (במקום CLI)")
    parser.add_argument("--port", type=int, default=None, help="פורט שרת ה-Web (ברירת מחדל: 5000)")

    # ── קלט ופלט ──
    parser.add_argument("-i", "--input", help="נתיב קובץ קלט (PDF / JPG / PNG / WEBP)")
    parser.add_argument("-o", "--output", help="נתיב קובץ Word פלט (ברירת מחדל: שם_קלט_transcribed.docx)")

    # ── הגדרות תמלול ──
    parser.add_argument(
        "--model",
        default=None,
        help="שם מודל Gemini (ברירת מחדל: מ-.env). חייב להיות מודל הבנה, לא יצירת תמונות",
    )
    parser.add_argument(
        "--language",
        choices=["he", "en", "auto"],
        default=None,
        help="מצב שפה: he=עברית, en=אנגלית, auto=זיהוי אוטומטי (ברירת מחדל: auto)",
    )
    parser.add_argument(
        "--thinking",
        choices=["minimal", "low", "medium", "high"],
        default=None,
        help="רמת חשיבה של המודל (ברירת מחדל: low). גבוה = פחות תקציב לתמלול",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="תקציב טוקני פלט (ברירת מחדל: 32768). כולל טוקני חשיבה",
    )
    parser.add_argument(
        "--resolution",
        choices=["low", "medium", "high"],
        default=None,
        help="רזולוציית התמונה שנשלחת למודל (ברירת מחדל: high)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="סף ביטחון 0.0–1.0 (ברירת מחדל: 0.80)",
    )

    # ── Few-Shot ──
    fs_group = parser.add_mutually_exclusive_group()
    fs_group.add_argument(
        "--few-shot",
        dest="few_shot",
        action="store_true",
        default=None,
        help="הפעל דוגמאות Few-Shot (כבוי כברירת מחדל — עלול לגרום להמצאת תוכן)",
    )
    fs_group.add_argument(
        "--no-few-shot",
        dest="few_shot",
        action="store_false",
        help="כבה דוגמאות Few-Shot במפורש",
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
    parser.add_argument(
        "--auto-orient",
        action="store_true",
        help="נסה לזהות סיבוב של 90 מעלות (כבוי כברירת מחדל — EXIF מטפל בזה)",
    )

    # ── פיסוק ──
    p_group = parser.add_mutually_exclusive_group()
    p_group.add_argument(
        "--punctuate",
        dest="punctuate",
        action="store_true",
        default=None,
        help="הוסף פיסוק במעבר נפרד (ברירת מחדל: דלוק)",
    )
    p_group.add_argument(
        "--no-punctuate",
        dest="punctuate",
        action="store_false",
        help="בלי פיסוק — פלט דיפלומטי בלבד",
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

    # model_copy — כדי לא לשנות את הסינגלטון הגלובלי.
    # בגרסה הקודמת cfg הצביע על אותו אובייקט ושינויי CLI "דבקו" בו.
    cfg = default_config.model_copy(deep=True)

    if args.model:
        cfg.model = args.model
    if args.language:
        cfg.language_mode = args.language
    if args.thinking:
        cfg.thinking_level = args.thinking
    if args.max_tokens:
        cfg.max_output_tokens = args.max_tokens
    if args.resolution:
        cfg.media_resolution = args.resolution
    if args.confidence:
        cfg.confidence_threshold = args.confidence
    if args.few_shot is not None:
        cfg.use_few_shot = args.few_shot
    if args.punctuate is not None:
        cfg.add_punctuation = args.punctuate
    if args.log_level:
        cfg.log_level = args.log_level
    if args.port:
        cfg.web_port = args.port

    setup_logging(cfg.log_level)
    logger.info("AI-Typist Agent מופעל")

    # ── אזהרה על מודל שגוי ──
    # מודלי יצירת תמונות ימציאו טקסט במקום לתמלל.
    if "image" in cfg.model.lower():
        logger.error(
            f"המודל '{cfg.model}' נראה כמודל ליצירת תמונות ולא להבנת תמונות.\n"
            f"    מודלים כאלה ממציאים טקסט במקום לתמלל אותו.\n"
            f"    מומלץ: gemini-3.1-pro-preview"
        )

    # ── מצב Web ──
    if args.web:
        logger.info(f"מפעיל ממשק Web על פורט {cfg.web_port}...")
        # ה-import כאן (ולא בראש הקובץ) כדי שהפעלת CLI לא תדרוש Flask
        try:
            from web_ui.app import create_app
        except ImportError as e:
            logger.error(
                f"ממשק ה-Web אינו זמין: {e}\n"
                f"    הקובץ web_ui/app.py חסר. השתמשי במצב CLI בינתיים:\n"
                f"    python main.py -i קובץ.jpg -o פלט.docx"
            )
            sys.exit(1)
        app = create_app(cfg)
        app.run(host="127.0.0.1", port=cfg.web_port, debug=False)
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
        auto_orient=args.auto_orient,
    )


if __name__ == "__main__":
    main()
