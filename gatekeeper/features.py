"""Transaction features for the risk scorer.

These are behavioural features an agent-payment gate can actually see at
decision time: amount vs the buyer's own history, time of day, device /
merchant novelty, and purchase velocity. They deliberately mirror the
kinds of signals in IEEE-CIS (amount, card/product context, time gaps)
without pretending the demo scorer *is* the IEEE-CIS model — that model
is trained separately in risk/train_ieee.py.
"""

from dataclasses import dataclass


@dataclass
class Transaction:
    amount_usd: float
    hour_utc: int  # 0-23
    merchant_category: str
    new_device: bool = False
    new_merchant: bool = False
    buyer_median_amount_usd: float = 60.0
    purchases_last_hour: int = 1
    agent_id: str = "shopping-agent-1"
    description: str = ""


def extract_features(tx: Transaction) -> dict:
    """Return interpretable, bounded features in [0, 1]-ish ranges."""
    amount = max(tx.amount_usd, 0.0)
    if amount < 100:
        amount_tier = 0.0
    elif amount < 500:
        amount_tier = 0.5
    elif amount < 2000:
        amount_tier = 0.8
    elif amount < 5000:
        amount_tier = 1.0
    else:
        amount_tier = 1.3

    median = max(tx.buyer_median_amount_usd, 1.0)
    ratio = amount / median
    # log-scale deviation from the buyer's own normal, capped.
    import math

    deviation = min(max(math.log(max(ratio, 0.01)) / math.log(10), 0.0), 1.2)

    return {
        "is_night": 1.0 if (tx.hour_utc < 5 or tx.hour_utc >= 23) else 0.0,
        "new_device": 1.0 if tx.new_device else 0.0,
        "new_merchant": 1.0 if tx.new_merchant else 0.0,
        "amount_tier": amount_tier,
        "amount_deviation": deviation,
        "velocity": min(max(tx.purchases_last_hour - 1, 0) / 4.0, 1.0),
    }
