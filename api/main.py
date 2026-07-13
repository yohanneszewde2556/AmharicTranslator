"""
Phase 6 — FastAPI REST Deployment
Wraps the trained Amharic-English NMT model in a REST API.

Endpoints:
  GET  /health      — liveness check, confirms model is loaded
  GET  /languages   — lists supported language pairs
  POST /translate   — translates Amharic text to English

Usage:
  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""
import torch
import sentencepiece as spm
import time
import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Allow imports from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import MAX_LENGTH
from src.inference import load_model, translate
from api.models import (
    TranslationRequest,
    TranslationResponse,
    HealthResponse,
    LanguagesResponse
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPM_MODEL      = os.path.join(BASE_DIR, "data", "processed", "am_en_bpe.model")
CHECKPOINT     = os.path.join(BASE_DIR, "checkpoints", "best_model.pt")

# ── Global app state ──────────────────────────────────────────────────────────
# Model and tokenizer are loaded once at startup and reused for every request.
# This is critical — loading the model per-request would be extremely slow.
app_state = {
    "model"       : None,
    "sp"          : None,
    "device"      : None,
    "model_loaded": False
}


# ── Lifespan: load model at startup ───────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once when the API server starts.
    Loads the model and tokenizer into global app_state.
    """
    print("=" * 55)
    print("Amharic-English NMT API — Starting up")
    print("=" * 55)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    app_state["device"] = device
    print(f"Device: {device}")

    # Check required files exist
    if not os.path.exists(CHECKPOINT):
        print(f"WARNING: No checkpoint found at {CHECKPOINT}")
        print("API will start but /translate will return 503 until model is loaded.")
        app_state["model_loaded"] = False
    elif not os.path.exists(SPM_MODEL):
        print(f"WARNING: Tokenizer not found at {SPM_MODEL}")
        app_state["model_loaded"] = False
    else:
        # Load model weights
        model = load_model(CHECKPOINT, device)

        # Wrap in torch.no_grad context for inference — no gradient tracking needed
        model.eval()

        # Load tokenizer
        sp = spm.SentencePieceProcessor()
        sp.load(SPM_MODEL)

        app_state["model"]        = model
        app_state["sp"]           = sp
        app_state["model_loaded"] = True

        print(f"Model loaded successfully from {CHECKPOINT}")
        print(f"Tokenizer loaded from {SPM_MODEL}")
        print("API is ready to serve requests.")

    print("=" * 55)

    yield  # API runs here

    # Shutdown: clean up GPU memory
    print("Shutting down API...")
    if app_state["model"] is not None:
        del app_state["model"]
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print("Shutdown complete.")


# ── FastAPI app instance ───────────────────────────────────────────────────────
app = FastAPI(
    title="Amharic-English NMT API",
    description="Neural Machine Translation API for Amharic to English translation",
    version="1.0.0",
    lifespan=lifespan
)

# Allow cross-origin requests (needed if a frontend on a different port calls this)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """
    Liveness check.
    Returns whether the API is running and the model is loaded.
    """
    return HealthResponse(
        status="ok",
        model_loaded=app_state["model_loaded"],
        device=str(app_state["device"])
    )


@app.get("/languages", response_model=LanguagesResponse, tags=["System"])
def languages():
    """
    Returns the supported translation language pairs.
    Currently only Amharic → English is supported.
    """
    return LanguagesResponse()


@app.post("/translate", response_model=TranslationResponse, tags=["Translation"])
def translate_text(request: TranslationRequest):
    """
    Translates Amharic text to English using beam search decoding.

    - **text**: Amharic input sentence (max 512 characters)
    - **source_lang**: must be "am"
    - **target_lang**: must be "en"
    """
    # Check model is ready
    if not app_state["model_loaded"]:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Check server logs for details."
        )

    # Reject empty text after stripping
    text = request.text.strip()
    if not text:
        raise HTTPException(
            status_code=400,
            detail="Input text cannot be empty."
        )

    # Run inference — wrapped in no_grad for speed and memory efficiency
    start = time.perf_counter()
    try:
        with torch.no_grad():
            result = translate(
                text=text,
                model=app_state["model"],
                sp=app_state["sp"],
                device=app_state["device"],
                method="beam",
                beam_width=5,
                max_len=MAX_LENGTH
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Translation failed: {str(e)}"
        )

    compute_time_ms = (time.perf_counter() - start) * 1000

    return TranslationResponse(
        translation=result,
        source_lang=request.source_lang,
        target_lang=request.target_lang,
        compute_time_ms=round(compute_time_ms, 2)
    )


# ── Run directly ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False  # set True during development only
    )
