"""Load a trained IEEE-CIS artifact and score one transaction.

Artifact format (written by risk/train_ieee.py, pickle dict):
  model, calibrator, medians, cat_maps, thresholds, metadata
The calibrator is a fitted sklearn IsotonicRegression; if it is absent
the raw model probability is returned and marked uncalibrated.
"""

import pickle

from .ieee_features import transform_generic, transform_row


class IEEEModel:
    def __init__(self, artifact: dict):
        # Ensemble artifacts carry "members" instead of a single "model".
        self.model = artifact.get("model")
        self.calibrator = artifact.get("calibrator")
        self.medians = artifact["medians"]
        self.cat_maps = artifact["cat_maps"]
        self.thresholds = artifact.get("thresholds", {})
        self.metadata = artifact.get("metadata", {})
        # Kaggle artifacts carry a top-level "features" list in training
        # order; legacy local artifacts do not (their metadata list is
        # in a different order than training) and must use transform_row.
        self.features = artifact.get("features")
        self.fe_maps = artifact.get("fe_maps", {})
        self.keep_nan = artifact.get("keep_nan", False)
        # v5 ensemble artifacts: members = [{"model", "keep_nan"}];
        # the single-model path keeps "model" for backwards compat.
        self.ensemble = bool(artifact.get("ensemble"))
        self.members = artifact.get("members", [])

    @classmethod
    def load(cls, path: str) -> "IEEEModel":
        with open(path, "rb") as f:
            return cls(pickle.load(f))

    def score(self, row: dict) -> dict:
        if self.ensemble:
            # Ensemble raw probability = mean of members' raw probas,
            # each transformed with that member's keep_nan flag; the
            # calibrator was fitted on exactly this mean on cal data.
            raws = []
            for member in self.members:
                features = transform_generic(row, self.features, self.medians, self.cat_maps,
                                             fe_maps=self.fe_maps,
                                             keep_nan=member.get("keep_nan", False))
                raws.append(float(member["model"].predict_proba([features])[0][1]))
            raw = sum(raws) / len(raws) if raws else 0.0
        elif self.features:
            features = transform_generic(row, self.features, self.medians, self.cat_maps,
                                         fe_maps=self.fe_maps, keep_nan=self.keep_nan)
            raw = float(self.model.predict_proba([features])[0][1])
        else:
            features = transform_row(row, self.medians, self.cat_maps)
            raw = float(self.model.predict_proba([features])[0][1])
        calibrated = float(self.calibrator.predict([raw])[0]) if self.calibrator else raw
        calibrated = min(max(calibrated, 0.0), 1.0)
        review = self.thresholds.get("review", 0.35)
        block = self.thresholds.get("block", 0.75)
        verdict = "BLOCK" if calibrated >= block else (
            "REVIEW" if calibrated >= review else "ALLOW")
        return {"probability": round(calibrated, 4), "raw_probability": round(raw, 4),
                "verdict": verdict, "calibrated": self.calibrator is not None}
