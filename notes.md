# Known Issues & Future Fixes

## 1. Extractive reader gets confused by dense numbers

### The Problem
* **Question:** "beam size used for inference" (Expected answer: `4`).
* **Result:** Model failed to extract `4`.
* **What went right:** Retrieval and reranker worked perfectly. The right chunk was ranked #1 (`0.8939` score vs #2 at `0.0368`).

### Root Cause
* **Too many numbers in one spot:** The ~400-token chunk had multiple numbers packed close together (beam size `4`, length penalty `0.6`, citation `[38]`, and offset `+50`). 
* **Proof:** When fed *only* the single target sentence without the rest of the chunk, the reader extracted `4` immediately with **99.18% confidence**.
* **Conclusion:** The pipeline isn't broken. Extractive QA models just struggle to map the right number when several numerical values sit together in dense text.

### Planned Test (Deferred)
* **Idea:** Lower chunk size from ~400 tokens down to **200–250 tokens** to keep numbers separated.
* **Next step:** Run the evaluation dataset before and after changing chunk size to verify if accuracy improves without breaking questions that need broader context.