"""
web_ui/app.py — ממשק ההגהה

זהו הקובץ שהיה חסר בפרויקט מההתחלה, וזה גם לב המוצר.

למה הוא הלב ולא מנוע הזיהוי:
דיוק של מאה אחוזים לא קיים ולא יהיה. לכן ההשוואה הנכונה איננה
מול שלמות אלא מול הקלדה מאפס. מסך שמציג תמונת שורה מעל הטקסט
שלה, ומאפשר לתקן במקלדת בלי לעזוב אותה, הופך עמוד של שעה
לעמוד של דקות.

ובמקביל — כל תיקון נשמר כזוג (תמונת שורה, טקסט נכון).
המשתמשת סתם עובדת; הקורפוס נבנה מעצמו.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from flask import (
    Flask, jsonify, redirect, render_template,
    request, send_file, url_for,
)
from loguru import logger


def create_app(cfg=None):
    from config import config as default_config
    cfg = cfg or default_config

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = cfg.max_upload_mb * 1024 * 1024

    from corpus import Corpus
    corpus = Corpus(Path(cfg.writer_profile_path).parent / "corpus_data")

    # ========================================================
    # מסכים
    # ========================================================

    @app.route("/")
    def index():
        return render_template(
            "index.html", pages=corpus.pages(), stats=corpus.stats()
        )

    @app.route("/review/<int:page_id>")
    def review(page_id: int):
        lines = corpus.page_lines(page_id)
        if not lines:
            return redirect(url_for("index"))
        return render_template(
            "review.html",
            page_id=page_id,
            lines=lines,
            stats=corpus.stats(),
            done=sum(1 for l in lines if l.reviewed),
        )

    # ========================================================
    # העלאה ועיבוד
    # ========================================================

    @app.route("/upload", methods=["POST"])
    def upload():
        f = request.files.get("file")
        if not f or not f.filename:
            return redirect(url_for("index"))

        suffix = Path(f.filename).suffix or ".jpg"
        tmp = Path(tempfile.gettempdir()) / f"upload_{f.filename}"
        f.save(tmp)

        from image_processor import prepare_images
        from line_segmenter import segment_lines
        from line_transcriber import transcribe_lines

        first_page_id = None
        for page in prepare_images(str(tmp), dpi=cfg.render_dpi):
            lines = segment_lines(page.image)
            if not lines:
                continue

            page_id = corpus.add_page(f.filename, page.image)
            first_page_id = first_page_id or page_id
            line_ids = corpus.add_lines(page_id, lines)

            # תמלול ראשוני — הטיוטה שממנה מתחילה ההגהה
            result = transcribe_lines(lines, cfg, page.page_number)
            for lid, lr in zip(line_ids, result.lines):
                corpus.set_prediction(lid, lr.text, cfg.model)

            logger.info(f"עמוד {page_id}: {len(lines)} שורות מוכנות להגהה")

        if first_page_id is None:
            return redirect(url_for("index"))
        return redirect(url_for("review", page_id=first_page_id))

    # ========================================================
    # שמירת תיקון
    # ========================================================

    @app.route("/api/save", methods=["POST"])
    def api_save():
        data = request.get_json(force=True)
        corpus.save_correction(
            line_id=int(data["line_id"]),
            text=data["text"].strip(),
            kind=data.get("kind", "fix"),
            note=(data.get("note") or "").strip() or None,
        )
        return jsonify({"ok": True, "stats": corpus.stats()})

    # ========================================================
    # תמונות
    # ========================================================

    @app.route("/img/line/<int:line_id>")
    def img_line(line_id: int):
        ln = corpus.line(line_id)
        return send_file(ln.image_path) if ln else ("", 404)

    @app.route("/img/context/<int:line_id>")
    def img_context(line_id: int):
        ln = corpus.line(line_id)
        if not ln:
            return "", 404
        return send_file(ln.context_image_path or ln.image_path)

    # ========================================================
    # ייצוא
    # ========================================================

    @app.route("/export")
    def export():
        out = Path(corpus.root) / "training_export"
        info = corpus.export_training(out)
        return render_template("index.html", pages=corpus.pages(),
                               stats=corpus.stats(), export=info)

    return app


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import config
    create_app(config).run(host="127.0.0.1", port=config.web_port, debug=False)
