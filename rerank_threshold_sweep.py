"""
Offline sweep for MIN_RERANK_SCORE -- the minimum reranker relevance
score required before trusting any reader answer from a chunk.

Goal: find a cutoff that catches false positives (confidently wrong
answers to questions the paper doesn't actually address, e.g. the
MNIST case) WITHOUT wrongly refusing questions that were already
being answered correctly.

Uses already-saved raw results (rerank_score field) -- no model calls,
no rerun needed. Requires eval_runner.py to have saved rerank_score
(added after the original *_raw.json files, so make sure you reran
run_full_eval.py after that change before using this).
"""

import json
from pathlib import Path

RESULTS_DIR = Path("evaluation/results")
RERANK_THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]


def load_all_results():
    all_results = []
    for f in RESULTS_DIR.glob("*_raw.json"):
        with open(f, "r", encoding="utf-8") as fh:
            all_results.extend(json.load(fh))
    return all_results


def evaluate_at_rerank_threshold(results, threshold):
    """
    Simulates: would this question still be answered (not refused) if
    we ALSO required rerank_score >= threshold, on top of whatever the
    existing refusal logic already decided?

    A question that was already refused stays refused. A question that
    was answered gets re-checked: if its rerank_score falls below the
    new threshold, it now becomes refused too.
    """
    correct = wrong = missed = false_positive = correctly_refused = 0

    for r in results:
        should_answer = r["should_answer"]
        was_refused = r["refused"]
        rerank_score = r.get("rerank_score", -1.0)

        if was_refused:
            newly_refused = True
        else:
            newly_refused = rerank_score < threshold

        if should_answer:
            if newly_refused:
                missed += 1
            elif r["correct"]:
                correct += 1
            else:
                wrong += 1
        else:
            if newly_refused:
                correctly_refused += 1
            else:
                false_positive += 1

    return correct, wrong, missed, correctly_refused, false_positive


def main():
    results = load_all_results()
    print(f"Loaded {len(results)} question results.\n")

    if not any("rerank_score" in r for r in results):
        print("ERROR: no rerank_score field found in saved results.")
        print("Make sure eval_runner.py saves rerank_score, then rerun")
        print("run_full_eval.py before using this sweep.")
        return

    print(f"{'MinRerank':<12}{'Correct':<10}{'Wrong':<8}{'Missed':<8}{'RefusedOK':<11}{'FalsePos':<10}")
    for t in RERANK_THRESHOLDS:
        c, w, m, ro, fp = evaluate_at_rerank_threshold(results, t)
        print(f"{t:<12}{c:<10}{w:<8}{m:<8}{ro:<11}{fp:<10}")

    print("\nLooking for: FalsePos drops (esp. the MNIST case) while")
    print("Correct stays flat and Missed doesn't climb too much.")


if __name__ == "__main__":
    main()