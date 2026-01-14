<!-- Copilot instructions for contributors and AI coding agents -->
# Emoji Translator — Copilot instructions

Short, actionable guidance to help an AI agent be immediately productive in this repo.

- Repo purpose: translate short emoji sequences (1–6 emojis) into concise English text using a mix of exact DB lookups, semantic-similarity (SentenceTransformer) retrieval, and an LLM fallback (Ollama/Llama).

Key places to look
- `backend/app.py` — FastAPI app; exposes the web UI and API. Uvicorn expects `app` here.
- `backend/emoji_translator.py` (and `test_emoji_translator.py`) — builds the runtime translator via `build_translator()` and implements the DB / SIM / LLM cascade. Refer here for how inputs are normalized and how outputs are labeled (🎯 [DB], 💡 [SIM], 🦙 [LLM]).
- `train_bart_emoji2text.py` — HuggingFace/transformers training pipeline (CLI style).
- `train_emoji2text.py`, `Model.py` — custom PyTorch training and inference utilities (seq2seq transformer / GRU variants).
- `merged_emoji_sample.csv`, `emoji_dataset_stage*.csv` — canonical data inputs. Stage CSVs must contain columns: `input` (English) and `output` (emoji sequence).
- `emoji2text_model/` and `checkpoints/` — trained model artifacts and saved vocab JSONs (`src_vocab.json`, `tgt_vocab.json`).

Why things are structured this way (big picture)
- The runtime translator favors exact match first (fast + deterministic), then a lightweight similarity stage (SentenceTransformer embeddings), and finally an LLM call as a graceful fallback. This keeps latency and variability bounded for frequent sequences.
- Training and research code are separated: HuggingFace scripts for large pretrained seq2seq models (`train_bart_emoji2text.py`) and smaller/custom PyTorch scripts for lightweight experiments (`train_emoji2text.py`, `Model.py`).

Developer workflows (quick commands)
- Run dev server (FastAPI + Uvicorn):
  uvicorn backend.app:app --reload --port 8000

- Test the JSON API (example):
  curl -s -X POST -H "Content-Type: application/json" \
    -d '{"emoji_sequence":"🎉🎂"}' http://localhost:8000/api/translate

- Train (HuggingFace BART):
  python train_bart_emoji2text.py --data_dir . --defs_csv merged_emoji_sample.csv --out_dir ./emoji2text_bart

- Train (custom PyTorch):
  python train_emoji2text.py

- Run lightweight translator tests (script-style):
  python test_emoji_translator.py

Project-specific conventions & gotchas
- Emoji splitting: code uses `regex.findall(r"\X", s)` to split grapheme clusters. Do not replace with naive character iteration — it breaks multi-codepoint emoji (flags, family sequences, variation selectors).
- Stage CSV format: expected columns are `input` (English) and `output` (emoji sequence). Stage files are often space-separated tokens ("🎵 💃"); training code accepts both space-separated and no-space (uses grapheme split).
- Normalization: functions like `_normalize_seq_string` remove variation selectors and skin tones so keys may exist both as raw and normalized forms — reuse these helpers when changing matching logic.
- LLM integration: the translator calls a local Ollama endpoint at `http://localhost:11434/api/generate` by default. If Ollama is not available, LLM fallback may fail or return empty — tests assume Ollama is running for LLM outputs. Consider mocking or feature-flagging LLM calls in unit tests.
- Vocab / model artifacts: `emoji2text_model/` stores `model.pt`, `src_vocab.json`, `tgt_vocab.json`. `checkpoints/` contains other .pt files. Keep naming consistent if you add new artifacts.

Integration points and external deps
- Ollama (LLM): default endpoint `http://localhost:11434`. Configurable via `EmojiTranslator(ollama_url=..., ollama_model=...)`.
- SentenceTransformer: embedding model `all-MiniLM-L6-v2` is used for the SIM stage.
- PyTorch / Transformers / Datasets: training pipelines rely on `torch`, `transformers`, and `datasets` (see `train_bart_emoji2text.py`).
- `regex` (PyPI package) is required for correct grapheme splitting.

When changing behavior, test these edge cases
- Inputs with variation selectors or skin-tone modifiers.
- Inputs with non-emoji characters (translator should return a friendly error like "Please enter 1–6 emojis.").
- New emoji sequences not in stage CSVs — exercise the SIM and LLM fallback.

Useful code examples to reference
- `build_translator()` in `backend/emoji_translator.py` — shows how metadata (`merged_emoji_sample.csv`) and `emoji_dataset_stage*.csv` are loaded at startup.
- `/api/translate` in `backend/app.py` — minimal FastAPI wrapper to call `translator.translate()` and return JSON.

If you need more context or want me to: (a) add example unit tests for mocking the Ollama call, (b) produce a requirements.txt, or (c) add a short README dev section — tell me which and I'll add it.

---
Please review: is anything missing or unclear (e.g., specific run flags, expected dataset locations, or CI/test commands)? I can iterate the file based on your feedback.
