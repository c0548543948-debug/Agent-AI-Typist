"""
corpus.py — מסד הנתונים המקומי של הפרויקט

זהו הנכס. הקוד ניתן לכתיבה מחדש בשבוע; הקורפוס לא.

מה נשמר כאן:
    עמודים          — כל תמונה שהועלתה
    שורות           — תמונת השורה, מיקומה בעמוד, ותמונת הקשר
    ניבויים         — מה שהמודל תמלל, כולל מודל וזמן
    תיקונים         — מה שאת אישרת או תיקנת

ההבחנה החשובה ביותר במסד הזה היא בשדה `kind` בטבלת התיקונים:

    'fix'  — התמלול היה שגוי. הטקסט המתוקן הוא מה שכתוב בדף.
             ← נכנס לנתוני האימון.

    'sic'  — התמלול נכון, אבל בדף עצמו יש שגיאה של הכותב.
             ← הטקסט לאימון נשאר כפי שהוא בדף. הצורה המתוקנת
               נשמרת בנפרד בשדה note, ומופיעה רק בשכבה הקריאה.

בלי ההבחנה הזו כל תיקון של שגיאת כותב היה מלמד את המודל
לראות אותיות שאינן קיימות בתמונה.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS pages (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    source_path   TEXT NOT NULL,
    image_path    TEXT NOT NULL,
    writer        TEXT DEFAULT 'default',
    n_lines       INTEGER DEFAULT 0,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS lines (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    page_id            INTEGER NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
    idx                INTEGER NOT NULL,
    x0 INTEGER, y0 INTEGER, x1 INTEGER, y1 INTEGER,
    image_path         TEXT NOT NULL,
    context_image_path TEXT,
    UNIQUE(page_id, idx)
);

CREATE TABLE IF NOT EXISTS predictions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    line_id     INTEGER NOT NULL REFERENCES lines(id) ON DELETE CASCADE,
    text        TEXT NOT NULL,
    model       TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS corrections (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    line_id     INTEGER NOT NULL UNIQUE REFERENCES lines(id) ON DELETE CASCADE,
    text        TEXT NOT NULL,          -- מה שכתוב בדף. זו שכבת האימון.
    kind        TEXT NOT NULL,          -- 'fix' | 'sic' | 'confirm'
    note        TEXT,                   -- ב-'sic': הצורה המתוקנת לקריאה
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_lines_page ON lines(page_id);
CREATE INDEX IF NOT EXISTS idx_pred_line  ON predictions(line_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class LineRow:
    """שורה כפי שהיא מוצגת בממשק ההגהה."""
    id: int
    page_id: int
    idx: int
    image_path: str
    context_image_path: str | None
    prediction: str
    correction: str | None
    kind: str | None
    note: str | None

    @property
    def reviewed(self) -> bool:
        return self.correction is not None


class Corpus:
    """
    עטיפה דקה סביב SQLite.

    למה SQLite ולא קבצים: צריך לשאול שאלות כמו "מה השורה הבאה
    שלא הוגהה" ו"כמה מילים כבר נאספו", ולעשות את זה בלי לסרוק
    תיקיות. וזה קובץ אחד שאפשר לגבות בהעתקה.
    """

    def __init__(self, root: str | Path = "corpus_data"):
        self.root = Path(root)
        self.images = self.root / "lines"
        self.pages_dir = self.root / "pages"
        for d in (self.root, self.images, self.pages_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "corpus.db"
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ========================================================
    # כתיבה
    # ========================================================

    def add_page(self, source_path: str, page_image, writer: str = "default") -> int:
        """שומרת עמוד מעובד ומחזירה את המזהה שלו."""
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO pages (source_path, image_path, writer, created_at)"
                " VALUES (?,?,?,?)",
                (source_path, "", writer, _now()),
            )
            page_id = int(cur.lastrowid)

        path = self.pages_dir / f"page{page_id:05d}.png"
        page_image.save(path)
        with self._conn() as c:
            c.execute("UPDATE pages SET image_path=? WHERE id=?", (str(path), page_id))
        return page_id

    def add_lines(self, page_id: int, lines) -> list[int]:
        """שומרת את שורות העמוד ואת תמונותיהן."""
        ids = []
        for ln in lines:
            img_path = self.images / f"p{page_id:05d}_l{ln.index:03d}.png"
            ln.image.save(img_path)

            ctx_path = None
            if getattr(ln, "context_image", None) is not None:
                ctx_path = self.images / f"p{page_id:05d}_l{ln.index:03d}_ctx.png"
                ln.context_image.save(ctx_path)

            x0, y0, x1, y1 = ln.box
            with self._conn() as c:
                cur = c.execute(
                    "INSERT OR REPLACE INTO lines"
                    " (page_id, idx, x0,y0,x1,y1, image_path, context_image_path)"
                    " VALUES (?,?,?,?,?,?,?,?)",
                    (page_id, ln.index, x0, y0, x1, y1,
                     str(img_path), str(ctx_path) if ctx_path else None),
                )
                ids.append(int(cur.lastrowid))

        with self._conn() as c:
            c.execute("UPDATE pages SET n_lines=? WHERE id=?", (len(lines), page_id))
        return ids

    def set_prediction(self, line_id: int, text: str, model: str = "") -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO predictions (line_id, text, model, created_at)"
                " VALUES (?,?,?,?)",
                (line_id, text, model, _now()),
            )

    def save_correction(
        self, line_id: int, text: str, kind: str = "fix", note: str | None = None
    ) -> None:
        """
        שומרת תיקון.

        kind='confirm' — התמלול היה נכון כפי שהוא
        kind='fix'     — התמלול תוקן. הטקסט החדש הוא מה שבדף
        kind='sic'     — בדף יש שגיאה של הכותב. text נשאר כמו בדף,
                         note מכיל את הצורה המתוקנת לקריאה בלבד
        """
        if kind not in {"confirm", "fix", "sic"}:
            raise ValueError(f"סוג תיקון לא מוכר: {kind}")
        with self._conn() as c:
            c.execute(
                "INSERT INTO corrections (line_id, text, kind, note, created_at)"
                " VALUES (?,?,?,?,?)"
                " ON CONFLICT(line_id) DO UPDATE SET"
                " text=excluded.text, kind=excluded.kind,"
                " note=excluded.note, created_at=excluded.created_at",
                (line_id, text, kind, note, _now()),
            )

    # ========================================================
    # קריאה
    # ========================================================

    def _row_to_line(self, r: sqlite3.Row) -> LineRow:
        return LineRow(
            id=r["id"], page_id=r["page_id"], idx=r["idx"],
            image_path=r["image_path"], context_image_path=r["context_image_path"],
            prediction=r["prediction"] or "", correction=r["correction"],
            kind=r["kind"], note=r["note"],
        )

    _SELECT = """
        SELECT l.*,
               (SELECT text FROM predictions p WHERE p.line_id=l.id
                ORDER BY p.id DESC LIMIT 1) AS prediction,
               c.text AS correction, c.kind AS kind, c.note AS note
        FROM lines l
        LEFT JOIN corrections c ON c.line_id = l.id
    """

    def page_lines(self, page_id: int) -> list[LineRow]:
        with self._conn() as c:
            rows = c.execute(
                self._SELECT + " WHERE l.page_id=? ORDER BY l.idx", (page_id,)
            ).fetchall()
        return [self._row_to_line(r) for r in rows]

    def line(self, line_id: int) -> LineRow | None:
        with self._conn() as c:
            r = c.execute(self._SELECT + " WHERE l.id=?", (line_id,)).fetchone()
        return self._row_to_line(r) if r else None

    def pages(self) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                """SELECT p.*,
                          (SELECT COUNT(*) FROM lines l
                           JOIN corrections c ON c.line_id=l.id
                           WHERE l.page_id=p.id) AS reviewed
                   FROM pages p ORDER BY p.id DESC"""
            ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict:
        """כמה נאסף עד כה — המספר שקובע מתי אפשר לאמן."""
        with self._conn() as c:
            r = c.execute(
                """SELECT
                     (SELECT COUNT(*) FROM pages) AS pages,
                     (SELECT COUNT(*) FROM lines) AS lines,
                     (SELECT COUNT(*) FROM corrections) AS reviewed,
                     (SELECT COUNT(*) FROM corrections WHERE kind='sic') AS sic"""
            ).fetchone()
            words = c.execute(
                "SELECT COALESCE(SUM(LENGTH(text)-LENGTH(REPLACE(text,' ',''))+1),0) AS w"
                " FROM corrections"
            ).fetchone()["w"]
        d = dict(r)
        d["words"] = int(words or 0)
        d["target_words"] = 10000
        d["progress"] = min(100, round(100 * d["words"] / 10000, 1))
        return d

    # ========================================================
    # ייצוא לאימון
    # ========================================================

    def export_training(self, out_dir: str | Path) -> dict:
        """
        מייצאת את הזוגות המאושרים בפורמט שמודלים פתוחים אוכלים:
        תיקיית תמונות שורה + קובץ אחד שמקשר תמונה לטקסט.

        זה הפורמט הפשוט ביותר, ומכאן ההמרה לכל פורמט אחר
        (Transkribus, JSONL של Vertex) היא סקריפט קצר.

        חשוב: מיוצא הטקסט הדיפלומטי בלבד — מה שכתוב בדף.
        שורות מסוג 'sic' מיוצאות עם הטקסט השגוי שבדף, לא עם התיקון.
        """
        out = Path(out_dir)
        (out / "lines").mkdir(parents=True, exist_ok=True)

        import shutil
        with self._conn() as c:
            rows = c.execute(
                """SELECT l.image_path, c.text, c.kind
                   FROM corrections c JOIN lines l ON l.id=c.line_id
                   ORDER BY l.page_id, l.idx"""
            ).fetchall()

        pairs = []
        for i, r in enumerate(rows, 1):
            src = Path(r["image_path"])
            if not src.exists():
                continue
            dst = out / "lines" / f"{i:05d}.png"
            shutil.copyfile(src, dst)
            pairs.append((f"lines/{dst.name}", r["text"].replace("\n", " ").strip()))

        with open(out / "labels.txt", "w", encoding="utf-8") as f:
            for path, text in pairs:
                f.write(f"{path}\t{text}\n")

        words = sum(len(t.split()) for _, t in pairs)
        return {"lines": len(pairs), "words": words, "out_dir": str(out)}
