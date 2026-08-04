
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
CHUNK_SIZE_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 50

_chunker = None


def _get_chunker():
    global _chunker
    if _chunker is None:
        _chunker = TokenChunker(
            chunk_size=CHUNK_SIZE_TOKENS,
            chunk_overlap=CHUNK_OVERLAP_TOKENS,
        )
    return _chunker


def chunk_text(text: str) -> list[str]:
    """
    Split text into overlapping, token-aware chunks.
    Returns a plain list of chunk strings (text only -- no metadata
    yet, that gets added when we wire this into the retriever).
    """
    chunker = _get_chunker()
    chunks = chunker.chunk(text)
    return [c.text for c in chunks]


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