"""Training pipeline test on synthetic IEEE-schema data.

Generates 6,000 synthetic rows, trains, and checks that artifacts
exist, thresholds are ordered (review < block), and the loaded model
scores a row. This proves the pipeline runs; its numbers are synthetic
and must never be quoted as IEEE-CIS results.
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestTrainPipeline(unittest.TestCase):
    def test_synthetic_train_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            out_dir = Path(tmp) / "artifacts"
            subprocess.run([sys.executable, str(ROOT / "risk" / "synthetic_ieee.py"),
                            "--out", str(data_dir), "--n", "6000"],
                           check=True, capture_output=True, text=True)
            result = subprocess.run(
                [sys.executable, str(ROOT / "risk" / "train_ieee.py"),
                 "--data-dir", str(data_dir), "--out-dir", str(out_dir),
                 "--max-rows", "6000"],
                check=True, capture_output=True, text=True, timeout=600)
            self.assertIn("roc_auc", result.stdout)
            metrics = json.loads((out_dir / "metrics.json").read_text())
            self.assertIn("roc_auc", metrics)
            self.assertLess(metrics["review"]["threshold"], metrics["block"]["threshold"])
            sys.path.insert(0, str(ROOT))
            from gatekeeper.ieee_model import IEEEModel

            model = IEEEModel.load(str(out_dir / "model.pkl"))
            self.assertLess(model.thresholds["review"], model.thresholds["block"])
            scored = model.score({"TransactionAmt": 50, "ProductCD": "W",
                                  "card4": "visa", "card6": "credit"})
            self.assertIn(scored["verdict"], ("ALLOW", "REVIEW", "BLOCK"))


if __name__ == "__main__":
    unittest.main()
