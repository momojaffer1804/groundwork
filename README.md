# Groundwork

Groundwork answers questions about a research paper using only the paper itself.
If the answer isn't in the text, it refuses instead of guessing.

Most "chat with your PDF" tools will confidently state something that isn't
true when they don't actually know. This project is built around reducing
that: every answer comes with a confidence score, low-confidence answers are
refused, and a verification step rejects answers that don't actually appear
in the source text they claim to come from.

This does not fully eliminate incorrect answers. See "Known limitations"
below for the specific case where it still fails, and why.

## How it works

1. Upload a PDF - it's parsed directly in memory, never saved to disk
2. Split the text into overlapping chunks (Chonkie, token-based)
3. Embed all chunks and the question, retrieve the top 10 most relevant chunks
   using a bi-encoder (bge-small-en-v1.5)
4. Rerank those 10 down to the top 3 using a cross-encoder (bge-reranker-v2-m3),
   which reads the question and each chunk together for a sharper relevance check
5. Run extractive QA (DeBERTa-v3) on the top 3 chunks
6. Before accepting an answer: check it clears a minimum rerank relevance
   score, check it actually appears verbatim in the chunk it supposedly came
   from, and check the combined confidence score against a threshold. If any
   of these fail, the system refuses instead of returning the answer.

## Why retrieval and reranking, not just one model

A naive setup - run the QA model over every chunk in the paper and take the
most confident answer - fails in a specific, repeatable way: the reader's
own confidence isn't a reliable signal for correctness on its own. During
development, a question about attention head count returned "beam search"
instead of the actual answer, because a wrong chunk scored higher confidence
than the right one.

Retrieval narrows the field to relevant chunks before the reader ever runs.
Reranking then sharpens that further using both the question and the chunk
together, not just embedding similarity. Adding both fixed that specific
wrong-answer problem and cut average latency from about 16.7 seconds to
about 1.7 seconds per question, since the reader only runs on 3 chunks
instead of scanning the whole document.

## Evaluation

Tested on 32 hand-labeled questions across 3 different papers (a machine
learning architecture paper, a computer vision paper, and a medical review
paper), split between answerable and adversarial (asking about things not
in the paper, to check the system refuses instead of making something up).

The refusal threshold was picked by sweeping a range of values against this
data and checking where correct answers, wrong answers, and refusals
actually separated, not by guessing a number.

## Known limitations

**Extractive answers only.** The reader pulls an exact span of text from the
paper, it doesn't generate or rephrase. Questions with a single clear answer
in the text work well. Questions that need combining information across
multiple sentences generally don't, since the model has no way to
synthesize an answer that doesn't already exist as a contiguous span in the
source.

**Confidence does not fully predict correctness.** This is the main open
problem. Found and confirmed on two separate papers: the reader can be
highly confident about a real, verbatim number from the source text that is
still the wrong answer to the specific question asked, when the paper
mentions multiple similar numbers close together or in similar sentences.
Example: a paper stating a model was tested with "100 and 1000 layers" in
one place, and a different, more specific number in another place further
away - the reader returned "1000" with high confidence when the actual
answer the question was looking for was elsewhere in the document.

A verification step was added that rejects answers not found verbatim in
their source chunk. This catches fabricated answers reliably. It does not
catch this specific failure, because the wrong answer in this case is a
real number that genuinely exists in the source text, just not the one the
question was asking about. This was tested directly and confirmed: the
verification check correctly left the wrong-but-real answer in place,
because there was nothing to catch.

Two other fixes were attempted for this problem and did not work: shrinking
chunk size in number-dense regions (made results worse, reverted), and
requiring a minimum reranker relevance score before accepting an answer
(had no effect on this specific failure, the relevance score was already
high for the wrong chunk). Both are documented in NOTES.md.

**Small eval set.** 32 questions across 3 papers is enough to catch and
characterize real failure patterns, not enough to make strong statistical
generalization claims.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate   # venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Running it

Two processes need to run at the same time.

Start the API:
```bash
uvicorn api.main:app --reload --reload-dir api --reload-dir pipeline
```

In a separate terminal, start the UI:
```bash
streamlit run app.py
```

Upload a PDF and ask a question. The API can also be tested directly at
`http://127.0.0.1:8000/docs`.

## Tech stack

- Parsing: PyMuPDF
- Chunking: Chonkie
- Retrieval: bge-small-en-v1.5 (sentence-transformers)
- Reranking: bge-reranker-v2-m3 (cross-encoder)
- Reading: deberta-v3-base-squad2
- API: FastAPI
- UI: Streamlit

## What would actually fix the remaining problem

Confidence-based refusal and verbatim-presence checking both have a hard
ceiling: neither can distinguish between two real, verbatim answers in the
source text when only one of them correctly answers the specific question.
Closing this gap would need one of:

- A generative reader that reasons across the full retrieved context
  instead of extracting a single span, so it can weigh which number
  actually corresponds to which described entity
- An independent verification model that checks whether the retrieved
  evidence specifically supports the exact answer given, not just that the
  answer text exists somewhere nearby
- Better retrieval that surfaces the specific, most relevant mention
  instead of a topically similar one

None of these were built in this version. This was a learning project built
to understand retrieval-augmented QA end to end, including where it breaks
and why - not a production system.