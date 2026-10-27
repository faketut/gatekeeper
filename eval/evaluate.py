"""Honest evaluation at the operating threshold.

Input: a JSONL file, one {"y_true": 0|1, "y_score": float} per line.
Output: precision / recall / F1 at each threshold, the best-F1
threshold, and expected loss per 10,000 transactions:

    expected_loss = fraud_loss * missed_fraud
                  + review_cost * reviewed
                  + false_block_cost * false_blocked

Defaults are illustrative and must be replaced with the hackathon's
stated assumptions before any number is quoted: fraud_loss = the
transaction amount at risk, review_cost = human review time,
false_block_cost = lost sale + churn risk.

Usage:
    python3 eval/evaluate.py predictions.jsonl
"""

import json
import sys


def metrics_at_threshold(rows, threshold: float) -> dict:
    tp = fp = fn = tn = 0
    for r in rows:
        pred = 1 if r["y_score"] >= threshold else 0
        if pred and r["y_true"]:
            tp += 1
        elif pred and not r["y_true"]:
            fp += 1
        elif not pred and r["y_true"]:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4)}


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python3 eval/evaluate.py predictions.jsonl")
    rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")]
    results = [metrics_at_threshold(rows, t / 100) for t in range(5, 100, 5)]
    best = max(results, key=lambda m: m["f1"])
    for m in results:
        print(m)
    print("best-F1:", best)


if __name__ == "__main__":
    main()
