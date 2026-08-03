# Phase 1 QA engine -- naive extractive QA, no retriever yet.

#Goal for this phase: 
# 1.prove the reader model itself works and can pull a correct answer out of paper text. 
# 2.no retriever/reranker
# 3. we do : split the whole paper into fixed-size windows and run the QA model over EVERY window,
#keeping whichever answer has the highest confidence score.
# O(n) and slow and phase 3 replaces this with embedding based retrieval ..we do this step to measure improvement


import sys
import time

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from parser import extract_text

# Small, fast reader model, well-suited to CPU for Phase 1
MODEL_NAME = "deepset/deberta-v3-base-squad2"

# deberta-v3-base-squad2 has a 512 token context limit, so we chunk
# crudely by characters here (real token-aware chunking comes in
# Phase 3 with Chonkie). ~2000 chars keeps us safely under the limit
# for most papers' sentence lengths.
WINDOW_SIZE_CHARS = 2000
WINDOW_OVERLAP_CHARS = 200

_tokenizer = None
_model = None


def _load_model():
    """Load tokenizer + model once, lazily, and cache them."""
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        print(f"Loading {MODEL_NAME} (first run downloads ~700MB, then it's cached)...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
        _model.eval()
    return _tokenizer, _model


def make_windows(text: str, size: int = WINDOW_SIZE_CHARS, overlap: int = WINDOW_OVERLAP_CHARS):
    """Split text into overlapping character windows."""
    windows = []
    start = 0
    while start < len(text):
        end = start + size
        windows.append(text[start:end])
        start += size - overlap
    return windows


def _answer_single_window(question: str, context: str, tokenizer, model):
    """
    Run extractive QA on one window by hand: tokenize question+context,
    forward pass, take the argmax start/end logits, decode the span,
    and combine start/end scores into a single confidence number.
    """
    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    start_logits = outputs.start_logits[0]
    end_logits = outputs.end_logits[0]

    start_idx = int(torch.argmax(start_logits))
    end_idx = int(torch.argmax(end_logits))

    # Guard against a nonsensical span (end before start)
    if end_idx < start_idx:
        end_idx = start_idx

    # Confidence: softmax probability of the chosen start position times
    # the chosen end position -- same spirit as what pipeline() used to
    # report, just computed by hand.
    start_prob = torch.softmax(start_logits, dim=0)[start_idx].item()
    end_prob = torch.softmax(end_logits, dim=0)[end_idx].item()
    score = start_prob * end_prob

    answer_tokens = inputs["input_ids"][0][start_idx:end_idx + 1]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    return {"answer": answer.strip(), "score": score}


def answer_question(question: str, full_text: str, verbose: bool = True):
    """
    Naive O(N) extractive QA: run the reader over every window in the
    document and keep the highest-confidence answer.
    """
    tokenizer, model = _load_model()

    windows = make_windows(full_text)
    if verbose:
        print(f"Document split into {len(windows)} windows (naive char-based).")
        print(f"Running QA model over all {len(windows)} windows...\n")

    best = {"answer": None, "score": -1.0, "window_idx": None}
    start_time = time.time()

    for i, window in enumerate(windows):
        if not window.strip():
            continue
        result = _answer_single_window(question, window, tokenizer, model)
        if verbose:
            print(f"  window {i+1}/{len(windows)}: score={result['score']:.4f}  answer={result['answer']!r}")

        # squad2-style readers predict "no answer in this window" by
        # pointing at [CLS], which decodes to an empty string -- and
        # they can be VERY confident about that. An empty answer is
        # never a real answer, so it should never win the cross-window
        # comparison no matter how high its score is.
        if not result["answer"]:
            continue

        if result["score"] > best["score"]:
            best = {"answer": result["answer"], "score": result["score"], "window_idx": i}
    elapsed = time.time() - start_time

    if verbose:
        print(f"\nDone in {elapsed:.1f}s across {len(windows)} windows "
              f"({elapsed / max(len(windows), 1):.2f}s/window avg).")

    return best, elapsed


if __name__ == "__main__":
    # Usage: python pipeline/qa_engine.py sample_papers/your_paper.pdf "your question"
    if len(sys.argv) != 3:
        print('Usage: python pipeline/qa_engine.py <path_to_pdf> "<question>"')
        sys.exit(1)

    pdf_path, question = sys.argv[1], sys.argv[2]

    print(f"Parsing {pdf_path}...")
    text = extract_text(pdf_path)
    print(f"Extracted {len(text)} characters.\n")

    result, elapsed = answer_question(question, text)

    print("\n===== RESULT =====")
    print(f"Question: {question}")
    print(f"Answer:   {result['answer']}")
    print(f"Score:    {result['score']:.4f}")
    print(f"Found in window #{result['window_idx']}")
    print(f"Total latency: {elapsed:.1f}s  <-- write this down, Phase 3 should beat it")