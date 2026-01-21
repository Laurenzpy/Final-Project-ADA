from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from backend.final_emoji_translator import FinalEmojiTranslator, HybridConfig

BASE_DIR = Path(__file__).resolve().parent.parent  # project root
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI()

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR)) if TEMPLATES_DIR.exists() else None

_TRANSLATOR: Optional[FinalEmojiTranslator] = None


def get_translator() -> FinalEmojiTranslator:
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
        ranker_model_dir=str(BASE_DIR / "artifacts" / "t5_ranker"),
        use_ranker=True,
        require_ranker=True,

        top_k=8,
        compute_retrieval_debug=False,  # demo: keep response smaller
        device="auto",
        max_src_length=256,
        max_new_tokens=4,
        num_beams=4,
    )

    _TRANSLATOR = FinalEmojiTranslator(cfg)
    return _TRANSLATOR


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    if templates is not None and (TEMPLATES_DIR / "index.html").exists():
        return templates.TemplateResponse("index.html", {"request": request})
    return HTMLResponse("<h1>Emoji Translator Backend läuft (RAG-Ranker)</h1>")


@app.post("/api/translate")
async def api_translate(request: Request):
    try:
        body = await request.body()
        if not body:
            return JSONResponse({"error": "empty body"}, status_code=400)

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
            return JSONResponse({"error": "missing input (emoji_sequence/text/input)"}, status_code=422)

        tr = get_translator()
        out = tr.translate(emoji_seq)

        result_text = out.get("prediction", "")
        response = {
            "output": result_text,
            "result": result_text,
            "translation": result_text,
            "text": result_text,
            **out,
        }
        return JSONResponse(response)

    except Exception as e:
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)
    

def get_translator() -> FinalEmojiTranslator:
    global _TRANSLATOR
    if _TRANSLATOR is not None:
        return _TRANSLATOR

    cfg = HybridConfig(
        # ✅ use training-only final memory (Stage1-4 + Stage5_train)
        tm_train_paths=[str(BASE_DIR / "data" / "final_train_e2t.csv")],
        # ✅ your trained T5 generator
        t5_model_dir=str(BASE_DIR / "artifacts" / "t5_e2t"),
        enable_t5=True,
        device="auto",        # ✅ now supports mps in final_emoji_translator.py
        max_new_tokens=64,
        num_beams=4,
    )

    _TRANSLATOR = FinalEmojiTranslator(cfg)
    return _TRANSLATOR