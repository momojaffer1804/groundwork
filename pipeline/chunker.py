
# Phase 3 chunker : token-aware chunking via Chonkie.

# Replaces the crude character-based windowing in qa_engine.py (Phase 1)
#with proper token-aware chunks. 
# This matters because the reader model has a hard 512-token limit 
# and character counts are a rough proxy for token counts and can overshoot or undershoot
#  depending on the text (equations, symbols, and dense technical terms tokenize differently
#than plain prose).




from chonkie import TokenChunker
# Keep chunks comfortably under the reader's 512-token limit so that
# the question itself + special tokens still fit when the reader
# tokenizes question+context together later.
from density import numeric_density, HIGH_DENSITY_THRESHOLD


CHUNK_SIZE_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 50

# smaller chunks for number-heavy sections -- found that dense stats/hyperparam
# text (beam size, penalties, citations all crammed together) trips up the
# reader even when retrieval gets the right chunk. tested the same sentence
# alone and it worked fine, so it's a "too much going on" problem, not a
# retrieval problem. giving it a tighter window here instead.
DENSE_CHUNK_SIZE_TOKENS = 150
DENSE_CHUNK_OVERLAP_TOKENS = 25
_chunker = None
_dense_chunker = None


def _get_chunker():
    global _chunker
    if _chunker is None:
        _chunker = TokenChunker(
            chunk_size=CHUNK_SIZE_TOKENS,
            chunk_overlap=CHUNK_OVERLAP_TOKENS,
        )
    return _chunker


def _get_dense_chunker():
    global _dense_chunker
    if _dense_chunker is None:
        _dense_chunker = TokenChunker(
            chunk_size=DENSE_CHUNK_SIZE_TOKENS,
            chunk_overlap=DENSE_CHUNK_OVERLAP_TOKENS,
        )
    return _dense_chunker


def chunk_text(text: str) -> list[str]:
    chunker = _get_chunker()
    chunks = chunker.chunk(text)
    return [c.text for c in chunks]


def chunk_text_adaptive(text: str) -> list[str]:
    """
    Same as chunk_text, but re-splits any chunk that's too number-dense
    into smaller pieces. Normal prose stays at 400 tokens, only the
    cluttered sections get shrunk.
    """
    base_chunks = chunk_text(text)
    dense_chunker = _get_dense_chunker()

    final_chunks = []
    for chunk in base_chunks:
        if numeric_density(chunk) >= HIGH_DENSITY_THRESHOLD:
            sub_chunks = dense_chunker.chunk(chunk)
            final_chunks.extend(c.text for c in sub_chunks)
        else:
            final_chunks.append(chunk)

    return final_chunks
if __name__ == "__main__":
    # Quick sanity check: python pipeline/chunker.py sample_papers/attention.pdf
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parent))
    from parser import extract_text

    if len(sys.argv) != 2:
        print("Usage: python pipeline/chunker.py <path_to_pdf>")
        sys.exit(1)

    text = extract_text(sys.argv[1])
    chunks = chunk_text(text)

    print(f"Document produced {len(chunks)} chunks.\n")
    print(f"Document produced {len(chunks)} chunks (target ~{CHUNK_SIZE_TOKENS} tokens each, actual char length varies).\n")
    for i, c in enumerate(chunks[:3]):
        print(f"--- Chunk {i} ({len(c)} chars, not tokens) ---")