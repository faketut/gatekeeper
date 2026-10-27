"""Demo scorer: a small, explainable logistic model.

This is NOT the competition model. It exists so the gateway, agent and UI
can be built and demoed before IEEE-CIS training lands. Its weights are
chosen so the three demo scenarios land in three different buckets
(ALLOW / REVIEW / BLOCK); the real model in risk/train_ieee.py replaces
it, and its thresholds must be retuned on a precision-recall curve.

score_transaction returns (probability, contributions) so the UI can show
*why* a payment was held, not just that it was.
"""

import math

from .features import Transaction, extract_features

# weight, in logit units. Sums are kept small so one signal alone cannot
# block a payment — blocking needs a combination of anomalies.
WEIGHTS = {
    "bias": -2.8,
    "is_night": 0.6,
    "new_device": 0.7,
    "new_merchant": 0.4,
    "amount_tier": 0.7,
    "amount_deviation": 0.5,
    "velocity": 0.9,
}


def score_transaction(tx: Transaction) -> tuple[float, dict]:
    feats = extract_features(tx)
    logit = WEIGHTS["bias"]
    contributions = {"bias": WEIGHTS["bias"]}
    for name, value in feats.items():
        contrib = WEIGHTS[name] * value
        contributions[name] = round(contrib, 3)
        logit += contrib
    prob = 1.0 / (1.0 + math.exp(-logit))
    return round(prob, 4), contributions
