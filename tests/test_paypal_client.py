"""Tests for the PayPal client + disputes client against a mock transport."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gatekeeper.disputes import DisputesClient, draft_evidence_packet  # noqa: E402
from gatekeeper.paypal_client import PayPalAPIError, PayPalClient  # noqa: E402
from tests.mock_paypal import mock_transport  # noqa: E402


def make_client():
    transport = mock_transport()
    return PayPalClient("id", "secret", "https://mock.paypal", transport=transport), transport


class TestPayPalClient(unittest.TestCase):
    def test_create_and_capture_and_token_cached(self):
        client, transport = make_client()
        order = client.create_order(49.00)
        self.assertEqual(order["id"], "MOCK-ORDER-1")
        self.assertEqual(order["status"], "CREATED")
        capture = client.capture_order(order["id"])
        self.assertEqual(capture["status"], "COMPLETED")
        # Token fetched once and reused across both calls.
        self.assertEqual(transport.state["token_calls"], 1)

    def test_disputes_list_and_draft(self):
        client, _ = make_client()
        disputes = DisputesClient(client)
        listed = disputes.list_disputes()
        self.assertEqual(listed["items"][0]["dispute_id"], "PP-D-MOCK1")
        packet = draft_evidence_packet(
            {"id": "MOCK-ORDER-1", "purchase_units": [{"amount": {"value": "49.00"}}]},
            {"verdict": "ALLOW", "probability": 0.057})
        self.assertEqual(packet["order_id"], "MOCK-ORDER-1")
        self.assertIn("awaiting merchant approval", packet["status"])


if __name__ == "__main__":
    unittest.main()
