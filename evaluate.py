"""
evaluate.py — מדידת דיוק התמלול מול תמלול אמת

למה הקובץ הזה קיים:
עד עכשיו העבודה התנהלה לפי תחושה — "יצא יותר טוב", "יצא גרוע".
זה לא מספיק. בלי מדידה אין דרך לדעת אם שינוי בפרומפט, במודל או
בעיבוד התמונה שיפר או הרע, ואין דרך להשוות לכלים אחרים.

הקובץ הזה הופך תחושה למספר.

איך משתמשים:
    # השוואת פלט בודד לתמלול אמת
    python evaluate.py --output תמלול.docx --reference אמת.txt

    # מעבר על תיקייה שלמה של זוגות (תמונה + txt באותו שם)
    python evaluate.py --pairs "C:/.../דוגמות אבא להקלדה"

המדדים:
  CER — שיעור שגיאת תווים. המדד המקובל בתחום זיהוי כתב יד.
        5 אחוזים ומטה נחשב תוצאה מעולה בכתב יד מורכב.
  WER — שיעור שגיאת מילים. תמיד גבוה מ-CER, ומשקף טוב יותר
        כמה עבודת הגהה נדרשת בפועל.
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np


# ============================================================
# נרמול טקסט
# ============================================================

# שני סגנונות של סימון אי-ודאות שהמודל מייצר בפועל:
#   [מילה?]  — הצורה שביקשנו
#   מילה[?]  — צורה שהוא נוטה אליה לפעמים
UNCERTAIN = re.compile(r'\[([^\]\[]+?)\?\]')
UNCERTAIN_SUFFIX = re.compile(r'\[\?\]')
UNREADABLE = re.compile(r'\[unreadable\]|\[לא קריא\]|\[מחוק\]|\[crossed out\]|\[strikethrough\]')
NIKUD = re.compile(r'[\u0591-\u05C7]')     # טעמים וניקוד


def normalize(text: str, strip_punct: bool = False) -> str:
    """
    מנרמלת טקסט לפני השוואה, כדי שלא נמדוד הבדלים שלא מעניינים אותנו.

    מה נעשה:
    - סימון [מילה?] מוחלף במילה עצמה — זה מה שהמודל קרא
    - סימון [לא קריא] מוסר
    - ניקוד וטעמים מוסרים (אם מופיעים רק בצד אחד)
    - צורות סופיות של אותיות מאוחדות לצורה הרגילה
      (הבחנה גרפית בלבד; לא רוצים לספור אותה כשגיאה)
    - גרשיים למיניהם מאוחדים לתו אחד
    - רווחים מאוחדים
    """
    t = unicodedata.normalize("NFKC", text)
    t = UNCERTAIN.sub(r'\1', t)
    t = UNCERTAIN_SUFFIX.sub('', t)
    t = UNREADABLE.sub(' ', t)
    t = NIKUD.sub('', t)

    # איחוד סוגי גרשיים ומקפים
    t = t.replace('״', '"').replace('”', '"').replace('“', '"')
    t = t.replace('׳', "'").replace('’', "'").replace('‘', "'")
    t = t.replace('־', '-').replace('–', '-').replace('—', '-')

    # אותיות סופיות → צורה רגילה
    finals = {'ך': 'כ', 'ם': 'מ', 'ן': 'נ', 'ף': 'פ', 'ץ': 'צ'}
    t = ''.join(finals.get(c, c) for c in t)

    if strip_punct:
        t = re.sub(r'[^\w\s\'"]', ' ', t)

    return re.sub(r'\s+', ' ', t).strip()


# ============================================================
# מרחק עריכה
# ============================================================

def edit_distance(a: list, b: list) -> int:
    """
    מרחק לוינשטיין בין שתי סדרות.
    ממומש עם שתי שורות בלבד כדי לא לצרוך זיכרון על טקסטים ארוכים.
    """
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)

    prev = np.arange(len(b) + 1)
    for i, ca in enumerate(a, 1):
        cur = np.empty(len(b) + 1, dtype=np.int64)
        cur[0] = i
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1,          # מחיקה
                         cur[j - 1] + 1,       # הוספה
                         prev[j - 1] + (ca != cb))   # החלפה
        prev = cur
    return int(prev[-1])


# ============================================================
# חישוב המדדים
# ============================================================

def _lenient_variants(text: str) -> str:
    """
    נרמול מקל: מבטל הבדלים שאינם שגיאות קריאה אמיתיות.

    - כתיב חסר ומלא: מידה / מדה, חסרן / חסרון. אותה מילה, שני איותים.
    - סימני קו ושוויון שהכותב משתמש בהם ומקלידים לא תמיד מעתיקים.
    - פיסוק.

    זה לא מחליף את המדידה המחמירה — הוא משלים אותה.
    הפער בין השתיים מראה כמה מהשגיאות הן ענייני איות ולא קריאה.
    """
    t = normalize(text)
    t = re.sub(r'[=\-–—]', ' ', t)
    t = re.sub(r'[()\.,;:!?]', ' ', t)
    t = re.sub(r'[יו]', '', t)
    return re.sub(r'\s+', ' ', t).strip()


def _lenient_cer(hypothesis: str, reference: str) -> float:
    a, b = _lenient_variants(hypothesis), _lenient_variants(reference)
    if not b:
        return 0.0
    return edit_distance(list(a), list(b)) / len(b)


def score(hypothesis: str, reference: str) -> dict:
    """
    משווה תמלול מול אמת ומחזירה מדדים.

    hypothesis — מה שהכלי הפיק
    reference  — התמלול הנכון
    """
    h_norm = normalize(hypothesis)
    r_norm = normalize(reference)

    cer_dist = edit_distance(list(h_norm), list(r_norm))
    wer_dist = edit_distance(h_norm.split(), r_norm.split())

    ref_chars = max(1, len(r_norm))
    ref_words = max(1, len(r_norm.split()))

    return {
        "cer": cer_dist / ref_chars,
        "wer": wer_dist / ref_words,
        "ref_chars": len(r_norm),
        "hyp_chars": len(h_norm),
        "ref_words": ref_words,
        "hyp_words": len(h_norm.split()),
        "uncertain_marked": (len(UNCERTAIN.findall(hypothesis))
                             + len(UNCERTAIN_SUFFIX.findall(hypothesis))),
        "cer_lenient": _lenient_cer(hypothesis, reference),
        "length_ratio": len(h_norm) / ref_chars,
    }


# ============================================================
# קריאת קבצים
# ============================================================

def read_text(path: Path) -> str:
    """קוראת טקסט מ-txt או מ-docx."""
    if path.suffix.lower() == ".docx":
        from docx import Document
        doc = Document(str(path))
        parts = [p.text for p in doc.paragraphs]
        # מסירים את טבלת הסיכום ומפרידי העמודים שהכלי מוסיף
        parts = [p for p in parts if not p.startswith("───")]
        return "\n".join(parts)
    return path.read_text(encoding="utf-8")


# ============================================================
# דיווח
# ============================================================

def verdict(cer: float) -> str:
    """מתרגמת אחוז שגיאה למשפט מובן."""
    if cer <= 0.05:
        return "מעולה — ברמה של מודל מאומן על הכתב"
    if cer <= 0.15:
        return "טוב — הגהה מהירה תספיק"
    if cer <= 0.30:
        return "בינוני — שימושי כטיוטה, דורש הגהה יסודית"
    if cer <= 0.60:
        return "חלש — חוסך מעט מאוד מול הקלדה מאפס"
    return "כשל — הפלט אינו קשור למקור"


def report(name: str, s: dict) -> None:
    print(f"\n{'='*58}")
    print(f"  {name}")
    print(f"{'='*58}")
    print(f"  שגיאת תווים  (CER)   {s['cer']*100:6.2f}%")
    print(f"  שגיאת מילים  (WER)   {s['wer']*100:6.2f}%")
    print(f"  שגיאת תווים מקלה     {s['cer_lenient']*100:6.2f}%   "
          f"(בלי כתיב חסר/מלא ופיסוק)")
    print(f"  {'-'*54}")
    print(f"  תווים   אמת {s['ref_chars']:>6}   פלט {s['hyp_chars']:>6}")
    print(f"  מילים   אמת {s['ref_words']:>6}   פלט {s['hyp_words']:>6}")
    print(f"  יחס אורך            {s['length_ratio']:6.2f}")
    print(f"  סומנו כלא-ודאיות    {s['uncertain_marked']:>6}")
    print(f"  {'-'*54}")
    print(f"  הערכה: {verdict(s['cer'])}")

    if s['length_ratio'] < 0.7:
        print("\n  אזהרה: הפלט קצר משמעותית מהאמת —")
        print("          ייתכן שהתמלול נקטע או שדולגו שורות.")
    elif s['length_ratio'] > 1.3:
        print("\n  אזהרה: הפלט ארוך משמעותית מהאמת —")
        print("          חשד להמצאת תוכן.")


# ============================================================
# מעבר על תיקיית זוגות
# ============================================================

def run_pairs(folder: Path, out_dir: Path | None, by_lines: bool = False) -> None:
    """
    מריצה את הכלי על כל תמונה בתיקייה שיש לה קובץ txt באותו שם,
    ומודדת את התוצאה מול אותו txt.

    זה מייצר טבלת דיוק על כל הדוגמאות בבת אחת — כך שכל שינוי
    בקוד נמדד על כמה עמודים ולא על אחד.
    """
    from config import config
    from image_processor import prepare_images
    from ocr_engine import transcribe_image
    from line_segmenter import segment_lines
    from line_transcriber import transcribe_lines

    out_dir = out_dir or Path("eval_out")
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".webp"}
    pairs = []
    for img in sorted(folder.iterdir()):
        if img.suffix.lower() in exts:
            ref = img.with_suffix(".txt")
            if ref.exists() and ref.read_text(encoding="utf-8").strip():
                pairs.append((img, ref))

    if not pairs:
        print(f"לא נמצאו זוגות תמונה+txt בתיקייה: {folder}")
        return

    print(f"נמצאו {len(pairs)} זוגות. מריץ...")

    rows = []
    for img, ref in pairs:
        processed = prepare_images(str(img), dpi=config.render_dpi)
        if by_lines:
            chunks = []
            for pg in processed:
                lines = segment_lines(pg.image)
                chunks.append(transcribe_lines(lines, config, pg.page_number).text)
            text = "\n\n".join(chunks)
        else:
            text = "\n\n".join(
                transcribe_image(p, config, []).text for p in processed
            )
        (out_dir / f"{img.stem}.txt").write_text(text, encoding="utf-8")

        s = score(text, ref.read_text(encoding="utf-8"))
        report(img.name, s)
        rows.append((img.name, s))

    # סיכום
    print(f"\n{'='*58}")
    print("  סיכום כל הדוגמאות")
    print(f"{'='*58}")
    print(f"  {'עמוד':<28}{'CER':>9}{'WER':>9}")
    for name, s in rows:
        print(f"  {name[:27]:<28}{s['cer']*100:8.2f}%{s['wer']*100:8.2f}%")
    mean_cer = sum(s['cer'] for _, s in rows) / len(rows)
    mean_wer = sum(s['wer'] for _, s in rows) / len(rows)
    print(f"  {'-'*46}")
    print(f"  {'ממוצע':<28}{mean_cer*100:8.2f}%{mean_wer*100:8.2f}%")
    print(f"\n  הערכה כוללת: {verdict(mean_cer)}")
    print(f"  הפלטים נשמרו ב: {out_dir}")


# ============================================================
# נקודת כניסה
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="מדידת דיוק תמלול מול תמלול אמת",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
דוגמאות:
  python evaluate.py --output תמלול.docx --reference אמת.txt
  python evaluate.py --pairs "C:/Users/.../דוגמות אבא להקלדה"
        """,
    )
    ap.add_argument("--output", help="קובץ הפלט של הכלי (docx או txt)")
    ap.add_argument("--reference", help="קובץ התמלול הנכון (txt)")
    ap.add_argument("--pairs", help="תיקייה עם זוגות תמונה + txt באותו שם")
    ap.add_argument("--out-dir", default=None, help="לאן לשמור פלטים במצב pairs")
    ap.add_argument("--by-lines", action="store_true",
                    help="חתוך לשורות ותמלל כל שורה בנפרד")
    args = ap.parse_args()

    if args.pairs:
        run_pairs(Path(args.pairs),
                  Path(args.out_dir) if args.out_dir else None,
                  by_lines=args.by_lines)
        return

    if not (args.output and args.reference):
        ap.print_help()
        sys.exit(1)

    hyp = read_text(Path(args.output))
    ref = read_text(Path(args.reference))
    report(Path(args.output).name, score(hyp, ref))


if __name__ == "__main__":
    main()
