"""
Phase 4 reranker -- cross-encoder reranking via bge-reranker-v2-m3.

Sits between the retriever and the reader: takes the retriever's top-k
chunks (relevance judged by comparing separate embeddings) and rescores
them with a cross-encoder, which reads the question and each chunk
TOGETHER in a single forward pass. This gives a much sharper relevance
signal than bi-encoder similarity alone.

Why this matters concretely (from actual debugging on this project):
bi-encoder retrieval on "How many attention heads does the base model
use?" ranked the chunk containing "we employ h = 8 parallel attention
layers" at #1 -- but a DIFFERENT chunk merely mentioning "layer 5 of 6"
had higher READER confidence and won the final answer, because reader
confidence alone isn't reliable relevance judgment. A cross-encoder is
built specifically to make this kind of fine-grained "does this chunk
actually answer this question" distinction that a bi-encoder can't.
"""

from sentence_transformers import CrossEncoder

MODEL_NAME = "BAAI/bge-reranker-v2-m3"
TOP_K_AFTER_RERANK = 3

_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        print(f"Loading reranker model {MODEL_NAME}...")
        _reranker = CrossEncoder(MODEL_NAME)
    return _reranker


def rerank(question: str, retrieved_chunks: list, top_k: int = TOP_K_AFTER_RERANK):
    """
    Rerank retrieved (chunk, retrieval_score) pairs using a cross-encoder.

    Takes the retriever's output directly (list of (chunk_text, retrieval_score)
    tuples) and returns a new list of (chunk_text, rerank_score) tuples,
    sorted by the cross-encoder's score, keeping only the top_k.

    The original retrieval_score is intentionally dropped here -- the
    cross-encoder score is a strictly better relevance signal for what
    happens next (the reader), so there's no reason to keep mixing in
    the weaker bi-encoder score once reranking has happened.
    """
    reranker = _get_reranker()

    chunk_texts = [chunk for chunk, _ in retrieved_chunks]
    pairs = [[question, chunk] for chunk in chunk_texts]

    rerank_scores = reranker.predict(pairs)

    scored = list(zip(chunk_texts, rerank_scores))
    scored.sort(key=lambda pair: pair[1], reverse=True)

    return scored[:top_k]


if __name__ == "__main__":
    # Usage: python pipeline/reranker.py sample_papers/attention.pdf "your question"
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parent))
    from parser import extract_text
    from chunker import chunk_text
    from retriever import retrieve_top_k

    if len(sys.argv) != 3:
        print('Usage: python pipeline/reranker.py <path_to_pdf> "<question>"')
        sys.exit(1)

    pdf_path, question = sys.argv[1], sys.argv[2]

    text = extract_text(pdf_path)
    chunks = chunk_text(text)

    retrieved = retrieve_top_k(question, chunks)
    print(f"Retrieved {len(retrieved)} candidates from bi-encoder.\n")

    reranked = rerank(question, retrieved)
    print(f"Top {len(reranked)} after cross-encoder reranking:\n")
    for rank, (chunk, score) in enumerate(reranked, start=1):
        preview = chunk[:150].replace("\n", " ")
        print(f"#{rank}  rerank_score={score:.4f}  {preview}...")