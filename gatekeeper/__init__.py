"""Gatekeeper core: stdlib-only scoring, features and policy.

Kept dependency-free on purpose so the demo and tests run with plain
``python3``. Heavy pieces (XGBoost training, FastAPI gateway, PayPal
client) live in risk/, gateway/ and gatekeeper/paypal_client.py and are
only imported when used.
"""

from .features import Transaction, extract_features
from .policy import Decision, decide
from .scoring import score_transaction

__all__ = ["Transaction", "extract_features", "Decision", "decide", "score_transaction"]
