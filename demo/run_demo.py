"""Demo: three purchases, three different verdicts. Stdlib only.

Run from the repo root:  python3 demo/run_demo.py

1. Normal: $49 book, 2pm, known device/merchant -> ALLOW
2. Anomalous: $2,000 electronics, 3am, new device, buyer median $60
   -> REVIEW (human approval — the brief's headline scenario)
3. Extreme: $12,000, 3am, new device + new merchant, 5 purchases in
   the last hour -> BLOCK (no order created)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.shopping_agent import ShoppingAgent  # noqa: E402
from gatekeeper.features import Transaction  # noqa: E402
from gatekeeper.store import DecisionStore  # noqa: E402

SCENARIOS = [
    ("normal daytime purchase", Transaction(
        amount_usd=49, hour_utc=14, merchant_category="books",
        description="paperback, known device and merchant")),
    ("sudden $2,000 electronics at 3am, new device", Transaction(
        amount_usd=2000, hour_utc=3, merchant_category="electronics",
        new_device=True, buyer_median_amount_usd=60,
        description="headline anomaly: routed for human approval")),
    ("$12,000 3am burst, new device + merchant", Transaction(
        amount_usd=12000, hour_utc=3, merchant_category="electronics",
        new_device=True, new_merchant=True, buyer_median_amount_usd=60,
        purchases_last_hour=5, description="extreme: blocked outright")),
]


def main() -> None:
    agent = ShoppingAgent(DecisionStore(log_path="demo/decisions.jsonl"))
    results = []
    for name, tx in SCENARIOS:
        record = agent.propose_purchase(tx)
        results.append(record)
        print(f"\n{name}")
        print(f"  verdict: {record['verdict']}  (risk {record['probability']:.1%})")
        print(f"  reasons: {'; '.join(record['reasons'])}")
        print(f"  outcome: {record['execution']}")
    print("\nAudit log: demo/decisions.jsonl")
    return results


if __name__ == "__main__":
    main()
