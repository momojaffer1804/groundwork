"""
Phase 6 -- FastAPI wrapper around the Groundwork pipeline.

Loads all models ONCE at server startup (not per-request) and exposes
a single /ask endpoint that runs the full pipeline: parse -> chunk ->
retrieve -> rerank -> read -> threshold check.

Run with: uvicorn api.main:app --reload
Then visit http://127.0.0.1:8000/docs for interactive API docs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from parser import extract_text
from chunker import chunk_text
from qa_engine import answer_question_v3, _load_model
from retriever import _get_model as _load_retriever_model
from reranker import _get_reranker

# Evidence-based threshold from the 12-question eval set (see
# evaluation/eval_runner.py for how this was picked).
REFUSAL_THRESHOLD = 0.15

app = FastAPI(title="Groundwork", description="Grounded QA over research papers")

# In-memory cache: paper path -> list of chunks, so we don't re-parse
# and re-chunk the same paper on every question. Simple dict is fine
# at this scale (a handful of papers) -- no need for a real cache/DB.
_chunk_cache = {}


@app.on_event("startup")
def load_models_on_startup():
    """
    Load all three models once when the server starts, not on every
    request. This is the whole point of running as a server instead
    of a one-off script -- avoids the multi-second reload cost we
    kept hitting during CLI testing.
    """
    print("Loading models (this happens once, at startup)...")
    _load_model()
    _load_retriever_model()
    _get_reranker()
    print("Models loaded. Ready.")


class AskRequest(BaseModel):
    paper_path: str
    question: str


class AskResponse(BaseModel):
    question: str
    answer: str | None
    confidence: float
    refused: bool
    reason: str | None = None


def _get_chunks(paper_path: str) -> list:
    if paper_path not in _chunk_cache:
        if not Path(paper_path).exists():
            raise HTTPException(status_code=404, detail=f"No PDF found at {paper_path}")
        text = extract_text(paper_path)
        _chunk_cache[paper_path] = chunk_text(text)
    return _chunk_cache[paper_path]


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    chunks = _get_chunks(request.paper_path)

    result, elapsed = answer_question_v3(request.question, chunks, verbose=False)

    if result["answer"] is None:
        return AskResponse(
            question=request.question,
            answer=None,
            confidence=0.0,
            refused=True,
            reason="No answer found in the document.",
        )

    if result["score"] < REFUSAL_THRESHOLD:
        return AskResponse(
            question=request.question,
            answer=None,
            confidence=result["score"],
            refused=True,
            reason=f"Confidence {result['score']:.4f} below threshold {REFUSAL_THRESHOLD}.",
        )

    return AskResponse(
        question=request.question,
        answer=result["answer"],
        confidence=result["score"],
        refused=False,
    )


@app.get("/health")
def health():
    return {"status": "ok"}