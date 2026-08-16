"""
Runs eval_runner.run_eval() across all three labeled eval sets
(attention, resnet, parkinson) and prints a combined summary on top
of the per-paper output eval_runner already gives.
"""

import sys
from pathlib import Path

# this file lives in evaluation/, pipeline/ is one level up
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for eval_runner import

from eval_runner import run_eval

# paths relative to project root, since eval_runner internally does
# extract_text(pdf_path) etc and expects root-relative paths — so we
# still run this script FROM project root, just the .py file itself
# sits in evaluation/
EVAL_SETS = [
    "evaluation/test_questions_attention.json",
    "evaluation/test_questions_resnet.json",
    "evaluation/test_questions_parkinson.json",
]
all_results = []

for path in EVAL_SETS:
    print(f"\n{'='*60}")
    print(f"RUNNING: {path}")
    print('='*60)
    results = run_eval(path)
    all_results.extend(results)

# combined summary across all papers
answerable = [r for r in all_results if r["should_answer"]]
unanswerable = [r for r in all_results if not r["should_answer"]]

correct_count = sum(1 for r in answerable if r["correct"])
missed_count = sum(1 for r in answerable if r["refused"])
wrong_count = len(answerable) - correct_count - missed_count

false_positive_count = sum(1 for r in unanswerable if not r["refused"])
correctly_refused_count = len(unanswerable) - false_positive_count

avg_latency = sum(r["elapsed"] for r in all_results) / len(all_results)

print(f"\n{'='*60}")
print("COMBINED SUMMARY (all 3 papers)")
print('='*60)
print(f"Total questions: {len(all_results)}")
print(f"Answerable: {len(answerable)}")
print(f"  Correct:  {correct_count}/{len(answerable)}")
print(f"  Wrong:    {wrong_count}/{len(answerable)}")
print(f"  Missed (wrongly refused): {missed_count}/{len(answerable)}")
print(f"Unanswerable: {len(unanswerable)}")
print(f"  Correctly refused: {correctly_refused_count}/{len(unanswerable)}")
print(f"  False positives: {false_positive_count}/{len(unanswerable)}")
print(f"\nAvg latency per question (all papers): {avg_latency:.1f}s")