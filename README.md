# Groundwork

Groundwork answers questions about a research paper using only the paper itself.
If the answer isn't in the text, it says so instead of guessing.

Most "chat with your PDF" tools will confidently make something up when they
don't actually know. This one is built around refusing to do that - every
answer comes with a confidence score, and below a tuned threshold it just
tells you it can't find a reliable answer.

## How it works

1. Upload a PDF - it's parsed directly in memory, never saved to disk
2. Split the text into overlapping chunks (Chonkie, token-based)
3. Embed all chunks and the question, retrieve the top 10 most relevant chunks
   using a bi-encoder (bge-small-en-v1.5)
4. Rerank those 10 down to the top 3 using a cross-encoder (bge-reranker-v2-m3),
   which reads the question and each chunk together for a sharper relevance check
5. Run extractive QA (DeBERTa-v3) on the top 3 chunks
6. Combine the reader's confidence with the reranker's score, check it against
   a threshold, and either return the answer or refuse

## Why retrieval and reranking, not just one model

A naive setup - run the QA model over every chunk in the paper and take the
most confident answer - fails in a specific, repeatable way: the reader's
own confidence isn't a reliable signal for correctness on its own. During
development, a question about attention head count returned "beam search"
instead of the actual answer, because a wrong chunk scored higher confidence
than the right one.

Retrieval narrows the field to relevant chunks before the reader ever runs.
Reranking then sharpens that further using both the question and the chunk
together, not just embedding similarity. Adding both fixed the wrong-answer
problem and cut average latency from about 16.7 seconds to about 1.7 seconds
per question, since the reader only runs on 3 chunks instead of scanning the
whole document.

## Evaluation

Tested on 12 hand-labeled questions against one paper: 8 answerable, 4
adversarial (asking about things not in the paper, to check the system
refuses instead of hallucinating).

Results:
- 7 out of 8 answerable questions answered correctly
- 4 out of 4 adversarial questions correctly refused, no false positives
- Average latency: about 1.7 seconds per question

The refusal threshold (0.15) was picked by looking at the actual score gap
in this data: adversarial questions scored no higher than 0.05, every
correct answer scored 0.27 or higher. This is evidence-based but still a
small sample, not a full precision-recall sweep across many papers.

Informally tested on a second, different paper as well (outside the 12-question
set) - it answered a factual question correctly and correctly refused an
out-of-scope question, which is a reasonable sign the system isn't just
overfit to one paper's phrasing.

## Known limitations

**Extractive answers only.** The reader pulls an exact span of text from the
paper, it doesn't generate or rephrase. Questions with a single clear answer
in the text work well. Questions that need combining information from
multiple sentences, or a "why" explanation, generally don't - the model has
no way to synthesize an answer that doesn't already exist as a contiguous
span in the source text.

**Reader confusion in dense numeric text.** Found during testing: a question
about beam search width failed even though retrieval and reranking correctly
identified the right chunk with a large margin. The chunk contained several
numbers close together (beam size, length penalty, a citation number, an
input length offset), and the reader picked the wrong one. Tested the same
sentence in isolation and the reader got it right immediately, confirming
this is a reader limitation, not a retrieval or parsing bug. Smaller chunk
sizes might reduce this - not yet tested.

**Small eval set.** 12 questions on one paper is enough to catch obvious
failures and pick a rough threshold, not enough to make strong generalization
claims. A larger, multi-paper eval set is the natural next step.

### Dense-numeric question failure (known, diagnosed, not fixed)

One question in the eval set ("What is beam search width used for
inference?") fails even though retrieval and reranking correctly rank
the right

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

## What's next

- Larger, multi-paper evaluation set
- Proper threshold calibration (temperature scaling or similar) instead of
  the current evidence-based but manually picked value
- Smaller chunk sizes to test against the dense-numeric-text failure mode
- Possibly swap the parser to Grobid, which is built specifically for
  academic papers, if a wider set of test papers shows PyMuPDF struggling
  on tables or multi-column layouts