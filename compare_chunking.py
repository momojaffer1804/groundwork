import sys
sys.path.insert(0, "pipeline")
import json
from parser import extract_text
from chunker import chunk_text, chunk_text_adaptive
from qa_engine import answer_question_v3

with open("evaluation/test_questions.json") as f:
    eval_set = json.load(f)

text = extract_text(eval_set["paper"])
fixed_chunks = chunk_text(text)
adaptive_chunks = chunk_text_adaptive(text)

print(f"fixed: {len(fixed_chunks)} chunks, adaptive: {len(adaptive_chunks)} chunks\n")

for item in eval_set["questions"]:
    q = item["question"]
    result_fixed, _ = answer_question_v3(q, fixed_chunks, verbose=False)
    result_adaptive, _ = answer_question_v3(q, adaptive_chunks, verbose=False)
    print(f"Q: {q}")
    print(f"  fixed:    {result_fixed['answer']!r} (score={result_fixed['score']:.4f})")
    print(f"  adaptive: {result_adaptive['answer']!r} (score={result_adaptive['score']:.4f})")
    print()