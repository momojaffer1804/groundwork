"""
Offline threshold sweep -- no model calls, just re-evaluates
already-collected scores against different REFUSAL_THRESHOLD values
to find the cutoff that best separates correct answers from
wrong/refused/false-positive ones.
"""

import json
from pathlib import Path

RESULTS_DIR = Path("evaluation/results")
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]


def load_all_results():
    all_results = []
    for f in RESULTS_DIR.glob("*_raw.json"):
        with open(f, "r", encoding="utf-8") as fh:
            all_results.extend(json.load(fh))
    return all_results


def evaluate_at_threshold(results, threshold):
    correct = wrong = missed = false_positive = correctly_refused = 0
    for r in results:
        score = r["score"]
        should_answer = r["should_answer"]
        # recompute refusal decision at this threshold
        refused = score < threshold

        if should_answer:
            if refused:
                missed += 1
            elif r["correct"]:
                correct += 1
            else:
                wrong += 1
        else:
            if refused:
                correctly_refused += 1
            else:
                false_positive += 1

    return correct, wrong, missed, correctly_refused, false_positive


def main():
    results = load_all_results()
    print(f"Loaded {len(results)} question results.\n")

    print(f"{'Threshold':<10}{'Correct':<10}{'Wrong':<8}{'Missed':<8}{'RefusedOK':<11}{'FalsePos':<10}")
    for t in THRESHOLDS:
        c, w, m, ro, fp = evaluate_at_threshold(results, t)
        print(f"{t:<10}{c:<10}{w:<8}{m:<8}{ro:<11}{fp:<10}")


if __name__ == "__main__":
    main()