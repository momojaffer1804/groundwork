"""
Phase 3 retriever -- bi-encoder embedding retrieval.

Replaces the brute-force "run the reader over every chunk" approach
from Phase 1 with a proper retrieval step: embed all chunks once,
embed the question, and use cosine similarity to narrow down to the
top-k most relevant chunks before the reader ever runs.

This is the direct fix for the failure mode we saw in Phase 1: the
reader picking a wrong-but-confident answer ("beam search") because
it had no way to know which of 23 windows was actually about the
right topic. Retrieval solves that by scoring relevance BEFORE
reading, using semantic similarity rather than the reader's own
(unreliable, out of context) confidence.
"""

from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
TOP_K = 10

_model = None


def _get_model():
    global _model
    if _model is None:
        print(f"Loading retriever model {MODEL_NAME}...")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def retrieve_top_k(question: str, chunks: list[str], k: int = TOP_K):
    """
    Embed all chunks and the question, score by cosine similarity,
    and return the top-k (chunk_text, score) pairs sorted by
    relevance, highest first.
    """
    model = _get_model()

    # normalize_embeddings=True makes cosine similarity equivalent to
    # a plain dot product, which sentence-transformers' built-in
    # similarity() computes efficiently for us.
    chunk_embeddings = model.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
    question_embedding = model.encode([question], normalize_embeddings=True, show_progress_bar=False)

    scores = model.similarity(question_embedding, chunk_embeddings)[0]  # shape: (num_chunks,)

    # Pair each chunk with its score, sort descending, take top k
    scored_chunks = list(zip(chunks, scores.tolist()))
    scored_chunks.sort(key=lambda pair: pair[1], reverse=True)

    return scored_chunks[:k]


if __name__ == "__main__":
    # Usage: python pipeline/retriever.py sample_papers/attention.pdf "your question"
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parent))
    from parser import extract_text
    from chunker import chunk_text

    if len(sys.argv) != 3:
        print('Usage: python pipeline/retriever.py <path_to_pdf> "<question>"')
        sys.exit(1)

    pdf_path, question = sys.argv[1], sys.argv[2]

    text = extract_text(pdf_path)
    chunks = chunk_text(text)
    print(f"Document split into {len(chunks)} chunks.\n")

    top_chunks = retrieve_top_k(question, chunks)

    print(f"Top {len(top_chunks)} chunks for: {question!r}\n")
    for rank, (chunk, score) in enumerate(top_chunks, start=1):
        preview = chunk[:150].replace("\n", " ")
        print(f"#{rank}  score={score:.4f}  {preview}...")