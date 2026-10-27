"""Unit tests for ensemble artifacts (v5 IEEEModel path)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gatekeeper.ieee_model import IEEEModel


class StubModel:
    """predict_proba returns a fixed positive-class probability."""

    def __init__(self, proba):
        self.proba = proba

    def predict_proba(self, X):
        return [[1.0 - self.proba, self.proba] for _ in X]


class StubCalibrator:
    """Identity calibrator (predict returns its input)."""

    def predict(self, X):
        return list(X)


class TestEnsemble(unittest.TestCase):
    def test_ensemble_averages_members(self):
        artifact = {
            "ensemble": True,
            "members": [
                {"model": StubModel(0.2), "keep_nan": False},
                {"model": StubModel(0.6), "keep_nan": True},
            ],
            "calibrator": StubCalibrator(),
            "medians": {"V1": 0.0},
            "cat_maps": {},
            "fe_maps": {},
            "thresholds": {"review": 0.35, "block": 0.75},
            "features": ["V1"],
        }
        model = IEEEModel(artifact)
        result = model.score({"V1": 1.0})
        # Mean raw proba = 0.4 -> REVIEW at the 0.35 threshold.
        self.assertEqual(result["raw_probability"], 0.4)
        self.assertEqual(result["probability"], 0.4)
        self.assertEqual(result["verdict"], "REVIEW")


if __name__ == "__main__":
    unittest.main()
