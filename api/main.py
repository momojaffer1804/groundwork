import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel

from parser import extract_text_from_bytes
from chunker import chunk_text
from qa_engine import answer_question_v3, _load_model
from retriever import _get_model as _load_retriever_model
from reranker import _get_reranker

REFUSAL_THRESHOLD = 0.15

app = FastAPI(title="Groundwork", description="Grounded QA over research papers")

_chunk_cache = {}


@app.on_event("startup")
def load_models_on_startup():
    print("Loading models (this happens once, at startup)...")
    _load_model()
    _load_retriever_model()
    _get_reranker()
    print("Models loaded. Ready.")


class AskRequest(BaseModel):
    paper_id: str
    question: str


class AskResponse(BaseModel):
    question: str
    answer: str | None
    confidence: float
    refused: bool
    reason: str | None = None


class UploadResponse(BaseModel):
    paper_id: str
    filename: str
    num_chunks: int


@app.post("/upload", response_model=UploadResponse)
async def upload_paper(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    paper_id = str(uuid.uuid4())
    contents = await file.read()

    text = extract_text_from_bytes(contents)
    chunks = chunk_text(text)
    _chunk_cache[paper_id] = chunks

    return UploadResponse(paper_id=paper_id, filename=file.filename, num_chunks=len(chunks))


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    if request.paper_id not in _chunk_cache:
        raise HTTPException(status_code=404, detail="Unknown paper_id. Upload a PDF via /upload first.")
    chunks = _chunk_cache[request.paper_id]

    result, elapsed = answer_question_v3(request.question, chunks, verbose=False)

    if result["answer"] is None:
        return AskResponse(question=request.question, answer=None, confidence=0.0, refused=True, reason="No answer found in the document.")

    if result["score"] < REFUSAL_THRESHOLD:
        return AskResponse(question=request.question, answer=None, confidence=result["score"], refused=True, reason=f"Confidence {result['score']:.4f} below threshold {REFUSAL_THRESHOLD}.")

    return AskResponse(question=request.question, answer=result["answer"], confidence=result["score"], refused=False)


@app.get("/health")
def health():
    return {"status": "ok"}