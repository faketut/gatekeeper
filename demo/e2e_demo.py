"""End-to-end demo against a MOCK PayPal transport (venv deps required).

Run from the repo root with the venv:
    .venv/bin/python demo/e2e_demo.py

Flow: agent proposes a normal $49 purchase -> gate ALLOWs -> agent
executes (mock order created) -> capture (mock COMPLETED) -> a
simulated dispute is listed and an evidence packet is drafted for
merchant review. No real PayPal account, no real money. Swap the mock
client for gatekeeper.paypal_client.from_env() to run the same flow
against the real sandbox once credentials exist.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.shopping_agent import ShoppingAgent  # noqa: E402
from gatekeeper.disputes import DisputesClient, draft_evidence_packet  # noqa: E402
from gatekeeper.features import Transaction  # noqa: E402
from gatekeeper.paypal_client import PayPalClient  # noqa: E402
from gatekeeper.store import DecisionStore  # noqa: E402
from tests.mock_paypal import mock_transport  # noqa: E402


def main() -> None:
    client = PayPalClient("mock-id", "mock-secret", "https://mock.paypal",
                          transport=mock_transport())
    agent = ShoppingAgent(DecisionStore(log_path="demo/e2e-decisions.jsonl"),
                          paypal_client=client)
    record = agent.propose_purchase(Transaction(
        amount_usd=49, hour_utc=14, merchant_category="books"))
    print(f"propose: {record['verdict']} (risk {record['probability']:.1%})")
    agent.execute(record)
    print(f"order: {record['paypal_order_id']} ({record['paypal_order_status']})")
    agent.capture(record)
    print(f"capture: {record['paypal_capture']['status']}")
    disputes = DisputesClient(client).list_disputes()
    print(f"disputes listed: {[d['dispute_id'] for d in disputes['items']]}")
    packet = draft_evidence_packet(
        {"id": record["paypal_order_id"],
         "purchase_units": [{"amount": {"value": "49.00"}}]}, record)
    print(f"evidence packet: {packet['status']} (order {packet['order_id']})")


if __name__ == "__main__":
    main()
