"""Unit tests for the v4 engineered-feature transform (transform_generic)."""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gatekeeper.ieee_features import transform_generic


FE_MAPS = {
    "card1_freq": {"group": "card1", "kind": "freq", "map": {1234: 50.0},
                   "missing": 7.0, "default": 0.0},
    "amt_mean_card1": {"group": "card1", "kind": "mean", "map": {1234: 42.5},
                       "default": 30.0},
}


class TestFETransform(unittest.TestCase):
    def test_dt_and_v_missing_and_fe_maps(self):
        # TransactionDT = 90000s -> dt_day 1, dt_hour 1, dt_day_mod7 1
        row = {"TransactionAmt": 100.0, "TransactionDT": 90000, "card1": 1234,
               "V1": 1.0, "V2": None, "V3": float("nan"), "ProductCD": "W"}
        features = ["V1", "V2", "V3", "dt_day", "dt_hour", "dt_day_mod7",
                    "card1_freq", "amt_mean_card1", "V_missing_count", "TransactionAmt_log"]
        out = transform_generic(row, features, {"V1": 0.0, "V2": 0.0, "V3": 0.0},
                                {}, fe_maps=FE_MAPS)
        self.assertEqual(out[3:6], [1.0, 1.0, 1.0])
        self.assertEqual(out[6], 50.0)
        self.assertEqual(out[7], 42.5)
        self.assertEqual(out[8], 2.0)  # V2 and V3 missing
        self.assertEqual(out[9], math.log1p(100.0))

    def test_fe_unseen_and_missing_groups(self):
        features = ["card1_freq", "amt_mean_card1"]
        self.assertEqual(transform_generic({"card1": 9999}, features, {}, {},
                                           fe_maps=FE_MAPS), [0.0, 30.0])
        self.assertEqual(transform_generic({"card1": None}, features, {}, {},
                                           fe_maps=FE_MAPS), [7.0, 30.0])

    def test_joint_freq_and_ratio_diff(self):
        features = ["amt_ratio_card1", "amt_diff_card1", "card1_addr1_freq"]
        fe_maps = {
            "amt_mean_card1": {"group": "card1", "kind": "mean", "map": {1234: 50.0},
                               "default": 30.0},
            "card1_addr1_freq": {"groups": ["card1", "addr1"], "kind": "joint_freq",
                                 "map": {"1234|100": 12.0}, "missing": 5.0,
                                 "default": 0.0},
        }
        row = {"TransactionAmt": 100.0, "card1": 1234, "addr1": 100}
        out = transform_generic(row, features, {}, {}, fe_maps=fe_maps)
        self.assertEqual(out, [2.0, 50.0, 12.0])
        # Unseen joint pair -> default; missing group -> missing count.
        row2 = {"TransactionAmt": 60.0, "card1": 1234, "addr1": 999}
        out2 = transform_generic(row2, features, {}, {}, fe_maps=fe_maps)
        self.assertEqual(out2[2], 0.0)
        row3 = {"TransactionAmt": 60.0, "card1": 1234, "addr1": None}
        out3 = transform_generic(row3, features, {}, {}, fe_maps=fe_maps)
        self.assertEqual(out3[2], 5.0)

    def test_keep_nan(self):
        row = {"V1": None}
        self.assertEqual(transform_generic(row, ["V1"], {"V1": 3.0}, {}), [3.0])
        self.assertTrue(math.isnan(
            transform_generic(row, ["V1"], {"V1": 3.0}, {}, keep_nan=True)[0]))


if __name__ == "__main__":
    unittest.main()
