from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from backend.emoji_translator import build_translator

# Uvicorn looks for this variable: "app"
app = FastAPI(title="Emoji Translator")

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

translator = build_translator()


class TranslateRequest(BaseModel):
    emoji_sequence: str
    instructions: str | None = None


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/translate")
def translate(req: TranslateRequest):
    seq = (req.emoji_sequence or "").strip()
    if not seq:
        return JSONResponse({"ok": False, "result": "Please enter 1–6 emojis."})

    result = translator.translate(seq)
    return JSONResponse({"ok": True, "result": result})
