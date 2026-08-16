"""
Phase 5 eval runner (scaled down) -- runs test_questions.json through
the full pipeline and reports precision/recall/refusal stats.

Uses a simple substring match for correctness on answerable questions
(expected_answer text found in the returned answer, case-insensitive).
This is crude compared to a real scoring metric, but honest and good
enough for a small hand-labeled set -- flag this limitation in the
README rather than pretending it's more rigorous than it is.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

from parser import extract_text
from chunker import chunk_text
from qa_engine import answer_question_v3

# Placeholder threshold -- NOT calibrated yet. A real Phase 5 would sweep
# this against the results below and pick the value that best separates
# correct from incorrect/refused. For now this just proves the mechanism.
REFUSAL_THRESHOLD = 0.30


def load_eval_set(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_correct(answer: str, expected: str) -> bool:
    if answer is None or expected is None:
        return False
    return expected.strip().lower() in answer.strip().lower()


def run_eval(eval_set_path: str = "evaluation/test_questions.json"):
    eval_set = load_eval_set(eval_set_path)
    pdf_path = eval_set["paper"]
    questions = eval_set["questions"]

    text = extract_text(pdf_path)
    chunks = chunk_text(text)
    print(f"Loaded {pdf_path}: {len(chunks)} chunks.\n")

    results = []
    for item in questions:
        q = item["question"]
        expected = item["expected_answer"]
        should_answer = item["answerable"]

        result, elapsed = answer_question_v3(q, chunks, verbose=False)

        if result["answer"] is None or result["score"] < REFUSAL_THRESHOLD:
            given_answer = None
            refused = True
        else:
            given_answer = result["answer"]
            refused = False

        correct = is_correct(given_answer, expected) if should_answer else None

        results.append({
            "question": q,
            "should_answer": should_answer,
            "refused": refused,
            "given_answer": given_answer,
            "expected": expected,
            "correct": correct,
            "score": float(result["score"]),
            "reader_score": float(result.get("reader_score") or -1.0),
            "rerank_score": float(result.get("rerank_score") or -1.0),
            "elapsed": elapsed,
        })

        status = (
            "CORRECT" if should_answer and correct
            else "WRONG" if should_answer and not correct and not refused
            else "MISSED (refused a good question)" if should_answer and refused
            else "CORRECTLY REFUSED" if not should_answer and refused
            else "FALSE POSITIVE (answered an unanswerable question)"
        )
        print(f"[{status}] {q}")
        print(f"  -> {given_answer!r}  (score={result['score']:.4f}, {elapsed:.1f}s)\n")

    answerable = [r for r in results if r["should_answer"]]
    unanswerable = [r for r in results if not r["should_answer"]]

    correct_count = sum(1 for r in answerable if r["correct"])
    missed_count = sum(1 for r in answerable if r["refused"])
    wrong_count = len(answerable) - correct_count - missed_count

    false_positive_count = sum(1 for r in unanswerable if not r["refused"])
    correctly_refused_count = len(unanswerable) - false_positive_count

    print("===== SUMMARY =====")
    print(f"Answerable questions: {len(answerable)}")
    print(f"  Correct:  {correct_count}/{len(answerable)}")
    print(f"  Wrong:    {wrong_count}/{len(answerable)}")
    print(f"  Missed (wrongly refused): {missed_count}/{len(answerable)}")
    print(f"Unanswerable questions: {len(unanswerable)}")
    print(f"  Correctly refused: {correctly_refused_count}/{len(unanswerable)}")
    print(f"  False positives (answered anyway): {false_positive_count}/{len(unanswerable)}")

    avg_latency = sum(r["elapsed"] for r in results) / len(results)
    print(f"\nAvg latency per question: {avg_latency:.1f}s")

    # save raw results for offline threshold sweeping (no rerun needed)
    os.makedirs("evaluation/results", exist_ok=True)
    out_name = Path(eval_set_path).stem + "_raw.json"
    with open(f"evaluation/results/{out_name}", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved raw results to evaluation/results/{out_name}")

    return results


if __name__ == "__main__":
    run_eval()