# test_questions.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

from parser import extract_text
from chunker import chunk_text
from qa_engine import answer_question_v2

# Temporary Threshold
REFUSAL_THRESHOLD = 0.30

text = extract_text("sample_papers/attention.pdf")
chunks = chunk_text(text)

questions = [
    "How many attention heads does the base model use?",
    "What dataset was used for English-to-German translation?",
    "What was the training time?",
    "What optimizer parameters were used?",
    "What optimizer did the BERT paper use?",
    "Who wrote the PyTorch framework?",
    "What tool was used to implement the model?",
]

for q in questions:
    result, elapsed = answer_question_v2(q, chunks, verbose=False)
    print(f"\nQ: {q}")

    if result["answer"] is None:
        print(f"A: [no answer found]  ({elapsed:.1f}s)")
    elif result["score"] < REFUSAL_THRESHOLD:
        print(f"A: [refused - combined score {result['score']:.4f} below threshold {REFUSAL_THRESHOLD}]  "
              f"(would-have-said: {result['answer']!r}, {elapsed:.1f}s)")
    else:
        print(f"A: {result['answer']!r}  (combined={result['score']:.4f}, "
              f"retrieval={result['retrieval_score']:.4f}, {elapsed:.1f}s)")