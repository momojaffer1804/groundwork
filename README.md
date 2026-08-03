# Groundwork

Grounded, hallucination-free question answering over research papers.
Answers only from the uploaded PDF's content — refuses when it can't
find a reliable answer, backed by a confidence threshold tuned on a
labeled eval set.

## Status
Phase 1 — basic end-to-end loop (in progress)

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Architecture, phases, and eval methodology
See project blueprint (not yet in-repo — paste in once finalized).
