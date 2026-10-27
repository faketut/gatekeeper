"""Gateway tests via FastAPI TestClient. Requires venv deps."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

import gateway.app as app_module  # noqa: E402
from gatekeeper.paypal_client import PayPalClient  # noqa: E402
from tests.mock_paypal import mock_transport  # noqa: E402


class TestGateway(unittest.TestCase):
    def setUp(self):
        app_module.store = __import__("gatekeeper.store", fromlist=["DecisionStore"]).DecisionStore(
            log_path="demo/test-decisions.jsonl")
        app_module.get_paypal_client = lambda: PayPalClient(
            "id", "secret", "https://mock.paypal", transport=mock_transport())
        self.client = TestClient(app_module.app)

    def test_propose_approve_execute_capture_flow(self):
        # Anomalous purchase -> REVIEW -> approve -> order -> capture.
        resp = self.client.post("/propose", json={
            "amount_usd": 2000, "hour_utc": 3, "merchant_category": "electronics",
            "new_device": True, "buyer_median_amount_usd": 60})
        self.assertEqual(resp.status_code, 200)
        record = resp.json()
        self.assertEqual(record["verdict"], "REVIEW")
        did = record["id"]
        self.assertEqual(self.client.post(f"/decisions/{did}/approve").status_code, 200)
        executed = self.client.post(f"/decisions/{did}/execute").json()
        self.assertEqual(executed["paypal_order_id"], "MOCK-ORDER-1")
        captured = self.client.post(f"/decisions/{did}/capture").json()
        self.assertEqual(captured["status"], "captured")

    def test_blocked_cannot_execute(self):
        resp = self.client.post("/propose", json={
            "amount_usd": 12000, "hour_utc": 3, "merchant_category": "electronics",
            "new_device": True, "new_merchant": True,
            "buyer_median_amount_usd": 60, "purchases_last_hour": 5})
        did = resp.json()["id"]
        self.assertEqual(resp.json()["verdict"], "BLOCK")
        self.assertEqual(self.client.post(f"/decisions/{did}/execute").status_code, 400)


if __name__ == "__main__":
    unittest.main()
