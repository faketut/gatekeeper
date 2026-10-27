"""Train the real risk model on IEEE-CIS Fraud Detection.

Data: train_transaction.csv in --data-dir (see scripts/download_ieee.py).
Identity files are NOT joined in this first model: the join drops
590K rows to ~144K and adds memory pressure on a small VM, and the
brief's own reference notes transaction-only models reach useful
performance. The exclusion is recorded in the model card.

Splits (stratified, seed-fixed): 70% train / 15% calibration / 15% test.
Thresholds are chosen on CALIBRATION only, never on test:
  review_threshold = threshold with recall >= 0.90 and best precision
  block_threshold  = threshold with precision >= 0.90 and best recall
  (fallbacks: max-recall / max-F1 threshold if a target is unreachable)
Metrics and expected loss are then reported on TEST only.

Memory: data is read in chunks, all fraud rows are kept, legitimate
rows are subsampled to --max-rows total, numeric columns are float32.
Defaults (150K rows, ~150 features) are sized for a small VM; raise
--max-rows only on a machine with the RAM for it.

Model: XGBoost if installed, else sklearn HistGradientBoosting
(gradient boosting either way; the model card records which ran).

Usage:
    python3 risk/train_ieee.py --data-dir data/ieee-cis --out-dir artifacts
    python3 risk/train_ieee.py --data-dir data/synthetic-ieee --out-dir artifacts-test
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gatekeeper.ieee_features import BASE_NUMERIC, CATEGORICAL, FEATURES, V_FEATURES  # noqa: E402

USECOLS = ["TransactionAmt"] + [c for c in FEATURES if c != "TransactionAmt_log"]
REVIEW_COST = 2.0       # $ per human review (stated assumption)
FALSE_BLOCK_COST = 15.0  # $ lost sale + churn per false block (stated assumption)


def _count(path: Path):
    import pandas as pd

    total = fraud = 0
    for chunk in pd.read_csv(path, usecols=["isFraud"], chunksize=100_000):
        total += len(chunk)
        fraud += int(chunk["isFraud"].sum())
    return total, fraud


def load_sampled(path: Path, max_rows: int, seed: int):
    import numpy as np
    import pandas as pd

    total, fraud = _count(path)
    legit = total - fraud
    # Robust to a file missing a few optional columns (e.g. an older
    # export): missing columns are filled with NaN and then medians.
    header = pd.read_csv(path, nrows=0).columns.tolist()
    usecols = [c for c in USECOLS + ["isFraud"] if c in header]
    if total <= max_rows:
        df = pd.read_csv(path, usecols=usecols)
        for col in USECOLS:
            if col not in df.columns:
                df[col] = float("nan")
        return df, {"total_rows": total, "fraud_rows": fraud, "sampled": False}
    # Prevalence-preserving sampling (fixed 2026-10-27): an earlier
    # version kept ALL fraud rows and subsampled only legitimate rows,
    # inflating the sample fraud rate to 13.8% vs the natural 3.5%.
    # That inflates precision and expected loss and miscalibrates
    # thresholds for real traffic, so both classes are now sampled at
    # the same rate, preserving the natural prevalence. Class
    # imbalance is handled by the model (scale_pos_weight / balanced
    # class weight), not by distorting the sample.
    keep_rate = max_rows / max(total, 1)
    keep_rate = min(max(keep_rate, 0.0), 1.0)
    rng = np.random.default_rng(seed)
    parts = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=50_000):
        keep = rng.random(len(chunk)) < keep_rate
        parts.append(chunk[keep])
    df = pd.concat(parts, ignore_index=True)
    for col in USECOLS:
        if col not in df.columns:
            df[col] = float("nan")
    return df, {"total_rows": total, "fraud_rows": fraud,
                "sampled": True, "sampled_rows": len(df),
                "sampled_fraud": int(df["isFraud"].sum())}


def _fit_preprocess(train):
    import pandas as pd

    medians = {}
    for col in BASE_NUMERIC + V_FEATURES:
        if col in train.columns:
            medians[col] = float(train[col].median()) if train[col].notna().any() else 0.0
        else:
            medians[col] = 0.0
    cat_maps = {}
    for col in CATEGORICAL:
        if col in train.columns:
            vals = train[col].fillna("MISSING").astype(str).unique()
            cat_maps[col] = {v: i for i, v in enumerate(sorted(vals))}
        else:
            cat_maps[col] = {}
    return medians, cat_maps


def _transform(df, medians, cat_maps):
    import numpy as np
    import pandas as pd

    from gatekeeper.ieee_features import transform_row

    rows = df.to_dict("records")
    X = np.array([transform_row(r, medians, cat_maps) for r in rows], dtype=np.float32)
    return X


def _metrics(y_true, proba, thresholds):
    from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score

    out = {}
    for name, t in thresholds.items():
        pred = (proba >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())
        out[name] = {
            "threshold": round(float(t), 4),
            "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }
    out["roc_auc"] = round(float(roc_auc_score(y_true, proba)), 4)
    out["average_precision"] = round(float(average_precision_score(y_true, proba)), 4)
    return out


def _choose_thresholds(y_cal, proba_cal):
    import numpy as np
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_cal, proba_cal)
    # precision_recall_curve returns len(thresholds) = len(precision)-1
    precision, recall = precision[:-1], recall[:-1]
    # Thresholds from precision_recall_curve are in increasing order and
    # recall falls as the threshold rises. The review point is the
    # HIGHEST threshold that still keeps recall >= 0.90 — the lowest
    # threshold would review almost everything (a bug caught in testing:
    # it produced review=0.0 and 2,931/3,000 reviews on synthetic data).
    candidates = [float(t) for r, t in zip(recall, thresholds) if r >= 0.90]
    if candidates:
        review = max(candidates)
    else:
        review = float(thresholds[int(np.argmax(recall))])
    block = None
    best_recall_at_precision = -1.0
    for p, r, t in zip(precision, recall, thresholds):
        if p >= 0.90 and r > best_recall_at_precision:
            best_recall_at_precision = r
            block = float(t)
    if block is None:
        f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
        block = float(thresholds[int(np.argmax(f1))])
    if review >= block:  # degenerate curve: keep review strictly below block
        review = block * 0.5
    return {"review": review, "block": block}


def _expected_loss(y_true, amounts, proba, thresholds):
    import numpy as np

    review, block = thresholds["review"], thresholds["block"]
    blocked = proba >= block
    reviewed = (proba >= review) & ~blocked
    missed = (~blocked) & (~reviewed) & (y_true == 1)
    false_blocked = blocked & (y_true == 0)
    loss = float(np.sum(amounts[missed])
                 + REVIEW_COST * int(reviewed.sum())
                 + FALSE_BLOCK_COST * int(false_blocked.sum()))
    no_gate = float(np.sum(amounts[y_true == 1]))
    n = len(y_true)
    return {"gated_loss_per_10k": round(loss * 10000 / n, 2),
            "no_gate_loss_per_10k": round(no_gate * 10000 / n, 2),
            "blocked": int(blocked.sum()), "reviewed": int(reviewed.sum()),
            "missed_fraud": int(missed.sum()), "false_blocked": int(false_blocked.sum()),
            "assumptions": {"review_cost_usd": REVIEW_COST,
                            "false_block_cost_usd": FALSE_BLOCK_COST,
                            "fraud_loss": "transaction amount",
                            "review_outcome": "review catches fraud (human resolves correctly)"}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", default="artifacts")
    parser.add_argument("--max-rows", type=int, default=150_000)
    parser.add_argument("--model", choices=["auto", "xgb", "hgb"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import numpy as np
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import train_test_split

    data_dir = Path(args.data_dir)
    tx_path = data_dir / "train_transaction.csv"
    if not tx_path.exists():
        raise SystemExit(
            f"{tx_path} not found. Run: python3 scripts/download_ieee.py "
            f"--out {args.data_dir} (needs Kaggle credentials, see that script).")
    print(f"Loading {tx_path} (max {args.max_rows} rows)…", flush=True)
    df, provenance = load_sampled(tx_path, args.max_rows, args.seed)
    print(f"Loaded {len(df)} rows, fraud rate {df['isFraud'].mean():.4f}", flush=True)

    train, temp = train_test_split(df, test_size=0.30, stratify=df["isFraud"],
                                   random_state=args.seed)
    cal, test = train_test_split(temp, test_size=0.50, stratify=temp["isFraud"],
                                 random_state=args.seed)
    medians, cat_maps = _fit_preprocess(train)
    X_train, y_train = _transform(train, medians, cat_maps), train["isFraud"].to_numpy()
    X_cal, y_cal = _transform(cal, medians, cat_maps), cal["isFraud"].to_numpy()
    X_test, y_test = _transform(test, medians, cat_maps), test["isFraud"].to_numpy()
    amounts_test = test["TransactionAmt"].to_numpy(dtype=float)
    del train, cal, test, df

    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    model_type = None
    if args.model in ("auto", "xgb"):
        try:
            from xgboost import XGBClassifier

            model = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8,
                                  scale_pos_weight=scale_pos_weight,
                                  tree_method="hist", n_jobs=2,
                                  random_state=args.seed)
            model_type = "xgboost"
        except ImportError:
            if args.model == "xgb":
                raise
            model_type = None
    if model_type is None:
        from sklearn.ensemble import HistGradientBoostingClassifier

        # Balanced class weight handles the natural 3.5% rate; the
        # sample itself is NOT distorted to do that job.
        model = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
            min_samples_leaf=20, l2_regularization=0.1,
            class_weight="balanced",
            early_stopping=True, validation_fraction=0.15,
            random_state=args.seed)
        model_type = "histgradientboosting"
    print(f"Training {model_type} on {len(y_train)} rows…", flush=True)
    model.fit(X_train, y_train)
    proba_cal_raw = model.predict_proba(X_cal)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(proba_cal_raw, y_cal)
    proba_cal = calibrator.predict(proba_cal_raw)
    proba_test = calibrator.predict(model.predict_proba(X_test)[:, 1])
    thresholds = _choose_thresholds(y_cal, proba_cal)
    metrics = _metrics(y_test, proba_test, thresholds)
    metrics["test_n"] = len(y_test)
    metrics["test_fraud"] = int(y_test.sum())
    metrics["expected_loss"] = _expected_loss(y_test, amounts_test, proba_test, thresholds)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Top-level "features" is the training order used by transform_row
    # (numeric BASE+V, then categorical, then log) — recorded so future
    # artifacts can use the generic inference path.
    from gatekeeper.ieee_features import BASE_NUMERIC as _BN, CATEGORICAL as _CAT, V_FEATURES as _V

    _train_order = _BN + _V + _CAT + ["TransactionAmt_log"]
    artifact = {"model": model, "calibrator": calibrator, "medians": medians,
                "cat_maps": cat_maps, "thresholds": thresholds, "features": _train_order,
                "metadata": {"model_type": model_type, "features": FEATURES,
                             "provenance": provenance, "data_dir": str(data_dir),
                             "synthetic": "synthetic" in str(data_dir)}}
    with open(out_dir / "model.pkl", "wb") as f:
        pickle.dump(artifact, f)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "model_card.md", "w", encoding="utf-8") as f:
        f.write(_model_card(artifact, metrics))
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {out_dir}/model.pkl, metrics.json, model_card.md")


def _model_card(artifact, metrics) -> str:
    meta, prov = artifact["metadata"], artifact["metadata"]["provenance"]
    return f"""# Model card — Gatekeeper risk model

- Model: {meta['model_type']} (gradient boosting)
- Data: {meta['data_dir']} — {'SYNTHETIC TEST DATA — NOT IEEE-CIS RESULTS' if meta['synthetic'] else 'IEEE-CIS train_transaction.csv'}
- Rows: {prov.get('sampled_rows', prov.get('total_rows'))} used of {prov['total_rows']} total (sampled: {prov['sampled']})
- Features: {len(meta['features'])} — interpretable fields + C/D + V1-V100. V101-V339 and identity join excluded (see train_ieee.py).
- Split: 70% train / 15% calibration / 15% test, stratified, seed 42. Thresholds chosen on calibration only.
- Test fraud rate: {metrics['test_fraud']}/{metrics['test_n']} = {metrics['test_fraud']/metrics['test_n']:.4f} (natural IEEE-CIS rate is ~0.035; sampling preserves it — see train_ieee.py).
- Thresholds: REVIEW >= {artifact['thresholds']['review']:.4f}, BLOCK >= {artifact['thresholds']['block']:.4f}
- Test ROC AUC: {metrics['roc_auc']}, average precision: {metrics['average_precision']}
- Test at REVIEW: precision {metrics['review']['precision']}, recall {metrics['review']['recall']}
- Test at BLOCK: precision {metrics['block']['precision']}, recall {metrics['block']['recall']}
- Expected loss (test, per 10K): gated ${metrics['expected_loss']['gated_loss_per_10k']} vs no-gate ${metrics['expected_loss']['no_gate_loss_per_10k']}
  (assumptions: review $2, false block $15, fraud loss = amount, review catches fraud)
- Caveats: thresholds and loss depend on the stated cost assumptions; a different
  merchant's costs change the operating point. Do not quote these numbers without
  the data provenance line above.
"""


if __name__ == "__main__":
    main()
