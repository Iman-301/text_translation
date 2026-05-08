from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from seq2seq_backend import Seq2SeqBackendError, Seq2SeqConfig, translate_with_seq2seq


def _repo_root() -> Path:
    # This file lives in <repo>/src/
    return Path(__file__).resolve().parent.parent


ROOT = _repo_root()
UI_DIR = (ROOT / "src" / "web_ui").resolve()
STATIC_DIR = (UI_DIR / "static").resolve()


app = FastAPI(title="Scratch Translator", version="1.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(str(UI_DIR / "index.html"))


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    to: Literal["fr", "es", "de"] = "fr"
    method: Literal["greedy", "beam_search"] = "greedy"
    beam_width: int = Field(5, ge=2, le=12)
    max_length: int = Field(50, ge=4, le=128)
    device: Literal["cpu", "cuda"] = "cpu"
    checkpoint_path: str = "seq2seq/models/checkpoints/best_model.pt"
    vocab_dir: str = "seq2seq/data/vocab"


@app.post("/api/translate")
def translate(payload: TranslateRequest) -> JSONResponse:
    try:
        ckpt = Path(payload.checkpoint_path)
        vocab = Path(payload.vocab_dir)
        cfg = Seq2SeqConfig(
            checkpoint_path=(ROOT / ckpt).resolve() if not ckpt.is_absolute() else ckpt,
            vocab_dir=(ROOT / vocab).resolve() if not vocab.is_absolute() else vocab,
            target_lang=payload.to,
            method=payload.method,
            beam_width=int(payload.beam_width),
            max_length=min(int(payload.max_length), 128),
            device=payload.device,
        )
        out, meta = translate_with_seq2seq(payload.text, cfg)
        return JSONResponse({"ok": True, "translation": out, "meta": meta})
    except (Seq2SeqBackendError, ValueError) as err:
        return JSONResponse({"ok": False, "error": str(err)}, status_code=400)
    except Exception as err:  # safety net for UI friendliness
        return JSONResponse({"ok": False, "error": f"Unexpected error: {err}"}, status_code=500)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True}

