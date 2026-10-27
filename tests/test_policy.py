"""Stdlib tests for the gate. Run: python3 -m unittest discover -s tests -v"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gatekeeper.features import Transaction
from gatekeeper.policy import decide
from gatekeeper.scoring import score_transaction


class TestGate(unittest.TestCase):
    def test_thresholds(self):
        self.assertEqual(decide(0.10, {}).verdict, "ALLOW")
        self.assertEqual(decide(0.50, {}).verdict, "REVIEW")
        self.assertEqual(decide(0.90, {}).verdict, "BLOCK")

    def test_demo_scenarios_land_in_three_buckets(self):
        normal = Transaction(amount_usd=49, hour_utc=14, merchant_category="books")
        anomaly = Transaction(amount_usd=2000, hour_utc=3, merchant_category="electronics",
                               new_device=True, buyer_median_amount_usd=60)
        extreme = Transaction(amount_usd=12000, hour_utc=3, merchant_category="electronics",
                               new_device=True, new_merchant=True,
                               buyer_median_amount_usd=60, purchases_last_hour=5)
        self.assertEqual(decide(*score_transaction(normal)).verdict, "ALLOW")
        self.assertEqual(decide(*score_transaction(anomaly)).verdict, "REVIEW")
        self.assertEqual(decide(*score_transaction(extreme)).verdict, "BLOCK")

    def test_single_signal_alone_does_not_block(self):
        # One anomaly (night only) must not block a normal purchase.
        tx = Transaction(amount_usd=49, hour_utc=3, merchant_category="books")
        self.assertNotEqual(decide(*score_transaction(tx)).verdict, "BLOCK")


if __name__ == "__main__":
    unittest.main()
