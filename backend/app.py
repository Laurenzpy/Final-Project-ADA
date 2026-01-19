from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from backend.final_emoji_translator import FinalEmojiTranslator, HybridConfig


BASE_DIR = Path(__file__).resolve().parent.parent  # Projekt-Root
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI()

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR)) if TEMPLATES_DIR.exists() else None

_TRANSLATOR: Optional[FinalEmojiTranslator] = None


def get_translator() -> FinalEmojiTranslator:
    """
    Baut den Translator genau so, wie final_emoji_translator.py ihn erwartet.
    Wichtig: TM nur aus Stage 1–4 (kein Leakage).
    """
    global _TRANSLATOR
    if _TRANSLATOR is not None:
        return _TRANSLATOR

    cfg = HybridConfig(
        tm_train_paths=[
            str(BASE_DIR / "data" / "emoji_dataset_stage1_e2t.csv"),
            str(BASE_DIR / "data" / "emoji_dataset_stage2_e2t.csv"),
            str(BASE_DIR / "data" / "emoji_dataset_stage3_e2t.csv"),
            str(BASE_DIR / "data" / "emoji_dataset_stage4_e2t.csv"),
        ],
        # passt zu HybridConfig in final_emoji_translator.py
        t5_model_dir=str(BASE_DIR / "artifacts" / "t5_e2t"),
        retrieval_high_conf=0.60,
        retrieval_low_conf=0.35,
        t5_fallback_below_conf=0.55,
        enable_t5_fallback=True,
        device="auto",
        max_new_tokens=32,
        num_beams=4,
    )

    _TRANSLATOR = FinalEmojiTranslator(cfg)
    return _TRANSLATOR


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    if templates is not None and (TEMPLATES_DIR / "index.html").exists():
        return templates.TemplateResponse("index.html", {"request": request})
    return HTMLResponse("<h1>Emoji Translator Backend läuft</h1>")


@app.post("/api/translate")
async def api_translate(request: Request):
    """
    Akzeptiert mehrere Body-Formate (Frontend-sicher):
      - {"emoji_sequence": "..."}
      - {"text": "..."}
      - {"input": "..."}
      - {"emoji": "..."}
      - raw string body

    Liefert:
      - prediction (neues Backend-Feld)
      - output/result/translation/text (Legacy/Frontend-kompatibel)
    """
    try:
        body = await request.body()
        if not body:
            return JSONResponse({"error": "empty body"}, status_code=400)

        # JSON oder raw string
        try:
            payload = json.loads(body.decode("utf-8"))
            if not isinstance(payload, dict):
                payload = {"emoji_sequence": str(payload)}
        except Exception:
            payload = {"emoji_sequence": body.decode("utf-8")}

        emoji_seq = (
            payload.get("emoji_sequence")
            or payload.get("text")
            or payload.get("input")
            or payload.get("emoji")
        )

        if not emoji_seq or not isinstance(emoji_seq, str):
            return JSONResponse(
                {"error": "missing input. Use emoji_sequence/text/input"},
                status_code=422,
            )

        tr = get_translator()
        out = tr.translate(emoji_seq)  # liefert dict mit "prediction", "mode", "retrieval_score", ...

        # Frontend-Kompatibilität: viele UIs erwarten "output" oder "result"
        result_text = out.get("prediction", "")

        response = {
            # Legacy keys (damit UI NICHT "Error" zeigt)
            "output": result_text,
            "result": result_text,
            "translation": result_text,
            "text": result_text,

            # unser Standard / Debug-Infos
            **out,
        }
        return JSONResponse(response)

    except Exception as e:
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)