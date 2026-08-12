# AI-Typist Agent

המרה אוטומטית של כתב יד (תמונות / PDF סרוקים) לקובץ Word מפורמט.

## התקנה

```bash
pip install -r requirements.txt
```

> דרוש גם `poppler` להמרת PDF:
> - Windows: הורידי מ-https://github.com/oschwartz10612/poppler-windows ופתחי ל-`C:\poppler`
> - Mac: `brew install poppler`

## הגדרת API Key

```bash
cp .env.example .env
# פתחי .env והכניסי את המפתח מ-https://aistudio.google.com/app/apikey
```

## הרצה

### ממשק Web (מומלץ)
```bash
python main.py --web
```
פתחי דפדפן על http://localhost:5000

### שורת פקודה
```bash
python main.py -i scan.pdf -o output.docx
python main.py -i photo.jpg -o output.docx --language he
python main.py -i scan.pdf --few-shot-dir ./few_shot
```

## כיול לכתב יד ספציפי (Few-Shot)

בתיקיית `few_shot/` הכניסי זוגות:
- `sample_01_image.jpg` — תמונה של כתב יד
- `sample_01_transcript.txt` — התמלול הנכון (UTF-8)

המערכת תשתמש בהם אוטומטית כדוגמאות לגמיני.
