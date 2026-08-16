# Known Issues & Future Fixes

## 1. extractive reader gets confused by dense numbers

question: "beam size used for inference" (answer should be 4)
result: model fails to extract it every time

retrieval and reranker both worked fine here, not their fault. right chunk
ranked #1 with 0.8939 vs #2 at 0.0368, so it's not a search problem.

root cause: too many numbers crammed in the same chunk. beam size 4,
length penalty 0.6, citation [38] x2, +50 offset, all sitting close
together. confirmed this by feeding the reader just the one sentence
alone with nothing else around it, it got 4 instantly at 99% confidence.
so the model can do it, it just gets lost when there's too much numeric
noise nearby in the same window.

tried fixing with adaptive chunking, results below.

## adaptive chunking, tried it, reverted

idea: detect number-dense chunks (density.py, threshold 1.2) and split
just those into smaller pieces, leave normal prose at 400 tokens like
before.

first attempt was 150 tokens for dense chunks, then dropped to 80 to see
if smaller helps more.

at 80 tokens the reader basically gave up, all three ranked chunks came
back with empty answers and near 100% confidence that there's no answer
there at all (rank1 score 1.0000, answer ''). so going smaller made it
worse, not better. probably needs enough surrounding text to even
recognize it as a spot where an answer exists, and 80 tokens isn't enough
for that.

ran the full eval set comparing fixed chunking vs adaptive, 12 questions
total:

- beam search width: still fails on both, not fixed
- dataset question (en-de): fixed gets it right, "WMT 2014" at 0.9552.
  adaptive gets it wrong, "newstest2013" at 0.8440. this is a real
  regression, not just a lower score on the same answer
- everything else answerable: same answer either way, scores moved up or
  down a bit but nothing flipped correct to wrong or wrong to correct
- all 4 adversarial questions still refuse correctly on both, no change
  there

so net result: adaptive chunking didn't fix the thing it was built for
and broke something that was already working. calling it here, reverting
to plain chunk_text as the production path. keeping chunk_text_adaptive
and density.py in the repo since it's real tested work, just not wiring
it into qa_engine or the api.

beam search width stays a known limitation. this looks like a reader
model ceiling at this point, not something fixable by messing with chunk
size more. would need actual fine-tuning to fix properly and that's not
realistic here, no gpu, no training data, not worth the time against the
deadline.