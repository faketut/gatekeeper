"""Shopping agent: proposes, never executes directly.

The agent's only route to money is through Gatekeeper. ALLOW executes
(via the PayPal client, once wired); REVIEW queues for a human; BLOCK
creates no order at all. This separation *is* the product — the agent
code must never gain a direct PayPal call.

execute() is the single choke point: it only runs for ALLOW decisions
or REVIEW decisions a human has approved, and it goes through the
PayPal client passed in (real sandbox client in production, a mock
transport in tests/demos).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gatekeeper.features import Transaction  # noqa: E402
from gatekeeper.policy import decide  # noqa: E402
from gatekeeper.scoring import score_transaction  # noqa: E402
from gatekeeper.store import DecisionStore  # noqa: E402


class ShoppingAgent:
    def __init__(self, store: DecisionStore | None = None, paypal_client=None):
        self.store = store or DecisionStore()
        self.paypal_client = paypal_client

    def propose_purchase(self, tx: Transaction) -> dict:
        prob, contributions = score_transaction(tx)
        decision = decide(prob, contributions)
        record = self.store.add(tx.__dict__, decision)
        if decision.verdict == "ALLOW":
            record["execution"] = "approved by gate — execute() to create a PayPal order"
        elif decision.verdict == "REVIEW":
            record["execution"] = "held for human approval — no order created"
        else:
            record["execution"] = "blocked — no order created"
        return record

    def execute(self, record: dict) -> dict:
        """Create a PayPal order for an ALLOW or human-approved record."""
        if not (record["verdict"] == "ALLOW" or record.get("status") == "approved"):
            raise ValueError("refusing to execute an unapproved/blocked payment")
        if self.paypal_client is None:
            from gatekeeper.paypal_client import from_env

            self.paypal_client = from_env()
        amount = float(record["transaction"].get("amount_usd", 0))
        order = self.paypal_client.create_order(amount)
        record["paypal_order_id"] = order.get("id")
        record["paypal_order_status"] = order.get("status")
        record["status"] = "order_created"
        return record

    def capture(self, record: dict) -> dict:
        if not record.get("paypal_order_id"):
            raise ValueError("no PayPal order to capture")
        if self.paypal_client is None:
            raise ValueError("no PayPal client configured")
        result = self.paypal_client.capture_order(record["paypal_order_id"])
        record["paypal_capture"] = result
        record["status"] = "captured"
        return record
