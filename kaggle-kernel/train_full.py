"""Gatekeeper full-data training — runs on Kaggle (CPU), not locally.

Input (attached as a Kaggle dataset, no download, no credentials in code):
    /kaggle/input/ieeecis-fraud-detection/train_transaction.csv
    /kaggle/input/ieeecis-fraud-detection/train_identity.csv (left-joined)

What it does, mirroring risk/train_ieee.py so numbers stay comparable:
  * Full 590K rows (no subsampling), all V1-V339, identity join.
  * Stratified 70/15/15 train / calibration / test split, seed 42.
  * Medians + categorical maps fitted on train only.
  * Gradient boosting: LightGBM if available, else HistGradientBoosting,
    with balanced class weight (imbalance handled by the model, NOT by
    distorting the sample — the local pipeline's first run made that
    mistake and was discarded).
  * Isotonic calibration fitted on calibration only.
  * Thresholds chosen on calibration only:
      REVIEW = highest threshold with recall >= 0.90
      BLOCK  = highest-recall threshold with precision >= 0.90
      (fallbacks: max-recall / max-F1)
  * Test-only metrics + expected loss (review $2, false block $15,
    fraud loss = transaction amount, review catches fraud).

Outputs (in /kaggle/working/, downloaded via `kaggle kernels output`):
    model.pkl, metrics.json, model_card.md, versions.json
The model.pkl format matches gatekeeper/ieee_model.py's generic path:
top-level "features" is the exact feature order used for training
(numeric + categorical + TransactionAmt_log), so local inference can
load it without the local V1-V100 feature list.
"""

import json
import math
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (average_precision_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

INPUT_DIR = "/kaggle/input/ieeecis-fraud-detection"
OUT_DIR = "/kaggle/working"
SEED = 42
REVIEW_COST = 2.0
FALSE_BLOCK_COST = 15.0

CATEGORICAL_KNOWN = (["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain"]
                     + [f"M{i}" for i in range(1, 10)]
                     + ["DeviceType", "DeviceInfo"])


def find_input_file(name):
    """Locate an attached dataset file anywhere under /kaggle/input.

    The v1 run assumed /kaggle/input/ieeecis-fraud-detection/ but Kaggle
    reported FileNotFoundError there, so discover the real mount point
    instead of guessing again — and log the input tree if it's missing.
    """
    for root, _dirs, files in os.walk("/kaggle/input"):
        if name in files:
            return os.path.join(root, name)
    print("INPUT TREE (/kaggle/input):")
    for root, dirs, files in os.walk("/kaggle/input"):
        print(" ", root, "dirs:", dirs[:10], "files:", files[:10])
    raise FileNotFoundError(f"{name} not found under /kaggle/input")


def load_data():
    tx = pd.read_csv(find_input_file("train_transaction.csv"))
    try:
        ident_path = find_input_file("train_identity.csv")
    except FileNotFoundError:
        ident_path = None
        print("train_identity.csv not attached — training without identity join")
    if ident_path and os.path.exists(ident_path):
        ident = pd.read_csv(ident_path)
        tx = tx.merge(ident, on="TransactionID", how="left")
        print(f"identity joined: {len(tx)} rows, {len(ident)} identity rows")
    print(f"loaded {len(tx)} rows, fraud rate {tx['isFraud'].mean():.4f}")
    return tx


def build_feature_lists(df):
    # Classify by dtype, not by a hand list: the v2 run proved some
    # identity columns (e.g. id_30/id_31-style browser/OS strings) hold
    # values like 'NotFound' mixed with NaN, so treating every unlisted
    # column as numeric crashes median(). Object/string columns are
    # categorical; numeric-dtype columns are numeric.
    drop = {"TransactionID", "TransactionDT", "isFraud"}
    cat_cols = [c for c in df.columns
                if c not in drop and not pd.api.types.is_numeric_dtype(df[c])]
    num_cols = [c for c in df.columns
                if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
    features = num_cols + cat_cols + ["TransactionAmt_log"]
    return num_cols, cat_cols, features


def fit_preprocess(train, num_cols, cat_cols):
    medians = {}
    for col in num_cols:
        medians[col] = float(train[col].median()) if train[col].notna().any() else 0.0
    cat_maps = {}
    for col in cat_cols:
        vals = train[col].fillna("MISSING").astype(str).unique()
        cat_maps[col] = {v: i for i, v in enumerate(sorted(vals))}
    return medians, cat_maps


def transform(df, num_cols, cat_cols, features, medians, cat_maps):
    """Vectorised transform to a float32 matrix in `features` order."""
    cols = []
    for col in num_cols:
        cols.append(df[col].fillna(medians[col]).astype(np.float32).to_numpy())
    for col in cat_cols:
        mapped = df[col].fillna("MISSING").astype(str).map(cat_maps[col])
        cols.append(mapped.fillna(-1).astype(np.float32).to_numpy())
    amt = df["TransactionAmt"].fillna(0.0).astype(float).to_numpy()
    cols.append(np.log1p(np.maximum(amt, 0.0)).astype(np.float32))
    X = np.column_stack(cols)
    assert X.shape[1] == len(features), (X.shape, len(features))
    return X


def choose_thresholds(y_cal, proba_cal):
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_cal, proba_cal)
    precision, recall = precision[:-1], recall[:-1]
    candidates = [float(t) for r, t in zip(recall, thresholds) if r >= 0.90]
    review = max(candidates) if candidates else float(thresholds[int(np.argmax(recall))])
    block = None
    best_recall = -1.0
    for p, r, t in zip(precision, recall, thresholds):
        if p >= 0.90 and r > best_recall:
            best_recall, block = r, float(t)
    if block is None:
        f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
        block = float(thresholds[int(np.argmax(f1))])
    if review >= block:
        review = block * 0.5
    return {"review": review, "block": block}


def metrics_at(y_true, proba, thresholds):
    out = {}
    for name, t in thresholds.items():
        pred = (proba >= t).astype(int)
        out[name] = {
            "threshold": round(float(t), 4),
            "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
            "tp": int(((pred == 1) & (y_true == 1)).sum()),
            "fp": int(((pred == 1) & (y_true == 0)).sum()),
            "fn": int(((pred == 0) & (y_true == 1)).sum()),
            "tn": int(((pred == 0) & (y_true == 0)).sum()),
        }
    out["roc_auc"] = round(float(roc_auc_score(y_true, proba)), 4)
    out["average_precision"] = round(float(average_precision_score(y_true, proba)), 4)
    return out


def expected_loss(y_true, amounts, proba, thresholds):
    review, block = thresholds["review"], thresholds["block"]
    blocked = proba >= block
    reviewed = (proba >= review) & ~blocked
    missed = (~blocked) & (~reviewed) & (y_true == 1)
    false_blocked = blocked & (y_true == 0)
    loss = float(np.sum(amounts[missed]) + REVIEW_COST * int(reviewed.sum())
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


def build_model():
    try:
        import lightgbm as lgb

        print("model: LightGBM", lgb.__version__)
        return lgb.LGBMClassifier(
            objective="binary", n_estimators=800, learning_rate=0.05,
            num_leaves=63, min_child_samples=40, subsample=0.8,
            colsample_bytree=0.8, class_weight="balanced",
            n_jobs=4, random_state=SEED, verbose=-1), "lightgbm"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier

        print("model: HistGradientBoosting (LightGBM not available)")
        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
            min_samples_leaf=20, l2_regularization=0.1,
            class_weight="balanced", early_stopping=True,
            validation_fraction=0.15, random_state=SEED), "histgradientboosting"


def main():
    df = load_data()
    num_cols, cat_cols, features = build_feature_lists(df)
    print(f"features: {len(features)} ({len(num_cols)} numeric, {len(cat_cols)} categorical)")
    train, temp = train_test_split(df, test_size=0.30, stratify=df["isFraud"], random_state=SEED)
    cal, test = train_test_split(temp, test_size=0.50, stratify=temp["isFraud"], random_state=SEED)
    medians, cat_maps = fit_preprocess(train, num_cols, cat_cols)
    X_train = transform(train, num_cols, cat_cols, features, medians, cat_maps)
    y_train = train["isFraud"].to_numpy()
    X_cal = transform(cal, num_cols, cat_cols, features, medians, cat_maps)
    y_cal = cal["isFraud"].to_numpy()
    X_test = transform(test, num_cols, cat_cols, features, medians, cat_maps)
    y_test = test["isFraud"].to_numpy()
    amounts_test = test["TransactionAmt"].to_numpy(dtype=float)
    del train, cal, test, df

    model, model_type = build_model()
    print(f"training {model_type} on {len(y_train)} rows…")
    try:
        import lightgbm as lgb

        model.fit(X_train, y_train, eval_set=[(X_cal, y_cal)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    except Exception as exc:
        print(f"early-stopping fit failed ({exc}); fitting without it")
        model.fit(X_train, y_train)
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(model.predict_proba(X_cal)[:, 1], y_cal)
    proba_cal = calibrator.predict(model.predict_proba(X_cal)[:, 1])
    proba_test = calibrator.predict(model.predict_proba(X_test)[:, 1])
    thresholds = choose_thresholds(y_cal, proba_cal)
    metrics = metrics_at(y_test, proba_test, thresholds)
    metrics["test_n"] = len(y_test)
    metrics["test_fraud"] = int(y_test.sum())
    metrics["expected_loss"] = expected_loss(y_test, amounts_test, proba_test, thresholds)
    print(json.dumps(metrics, indent=2))

    artifact = {"model": model, "calibrator": calibrator, "medians": medians,
                "cat_maps": cat_maps, "thresholds": thresholds, "features": features,
                "metadata": {"model_type": model_type, "features": features,
                             "data": "Kaggle dataset lnasiri007/ieeecis-fraud-detection (full, no subsampling)",
                             "split": "70/15/15 stratified seed 42, thresholds on calibration only"}}
    with open(os.path.join(OUT_DIR, "model.pkl"), "wb") as f:
        pickle.dump(artifact, f)
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    import sklearn, pandas

    versions = {"python": __import__("sys").version, "sklearn": sklearn.__version__,
                "pandas": pandas.__version__}
    try:
        import lightgbm

        versions["lightgbm"] = lightgbm.__version__
    except ImportError:
        pass
    with open(os.path.join(OUT_DIR, "versions.json"), "w") as f:
        json.dump(versions, f, indent=2)
    with open(os.path.join(OUT_DIR, "model_card.md"), "w") as f:
        f.write(f"""# Model card — Gatekeeper risk model (Kaggle full-data run)

- Model: {model_type}
- Data: full IEEE-CIS train data via Kaggle dataset, no subsampling
- Features: {len(features)} — all numeric + categorical + TransactionAmt_log, identity left-joined
- Split: 70/15/15 stratified seed 42, thresholds on calibration only
- Test fraud rate: {metrics['test_fraud']}/{metrics['test_n']} = {metrics['test_fraud']/metrics['test_n']:.4f}
- Thresholds: REVIEW >= {thresholds['review']:.4f}, BLOCK >= {thresholds['block']:.4f}
- Test ROC AUC: {metrics['roc_auc']}, average precision: {metrics['average_precision']}
- Test at REVIEW: precision {metrics['review']['precision']}, recall {metrics['review']['recall']}
- Test at BLOCK: precision {metrics['block']['precision']}, recall {metrics['block']['recall']}
- Expected loss per 10K: gated ${metrics['expected_loss']['gated_loss_per_10k']} vs no-gate ${metrics['expected_loss']['no_gate_loss_per_10k']}
""")
    print("Wrote model.pkl, metrics.json, model_card.md, versions.json to", OUT_DIR)


if __name__ == "__main__":
    main()
