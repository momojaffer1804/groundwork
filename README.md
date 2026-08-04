# Groundwork

Groundwork answers questions about a research paper using only the paper itself.
If the answer isn't in the text, it says so instead of guessing.

Most "chat with your PDF" tools will confidently make something up when they
don't actually know. This one is built around refusing to do that - every
answer comes with the exact source paragraph it came from and a confidence
score, and below a tuned threshold it just tells you it can't find a reliable
answer.

## How it works

1. Parse the PDF into clean text (currently PyMuPDF, moving to Docling for
   better table/layout handling)
2. Split into overlapping chunks
3. Embed chunks and retrieve the ones most relevant to the question
4. Rerank the retrieved chunks with a cross-encoder
5. Run extractive QA on the top few chunks
6. Check the answer's confidence against a threshold - return it if it clears
   the bar, refuse if it doesn't

## Status

Currently Phase 1: a naive brute-force baseline (no retrieval yet - the
reader runs over every chunk in the document and the highest-confidence
answer wins). This exists specifically to have a "before" number to compare
against once retrieval is added in Phase 3.

Baseline so far: ~17s per question, and it gets some answers wrong when the
correct chunk doesn't score highest against 20+ competing chunks with no
retrieval to narrow the field first. That's expected - it's the exact
problem the next phase is meant to fix, and it's a genuine limitation of
brute-force QA, not a bug.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate  # venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Running it

```bash
python pipeline/parser.py sample_papers/your_paper.pdf
python pipeline/qa_engine.py sample_papers/your_paper.pdf "your question"
```

## Why

Reading a 30-page paper to find one detail is slow. Asking a general chatbot
is fast but it hallucinates. This is meant to be the option that's both fast
and honest about what it actually knows.
