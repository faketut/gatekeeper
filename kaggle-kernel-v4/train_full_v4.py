"""Gatekeeper v4 training — runs on Kaggle (CPU), not locally.

Supersedes v3 (LightGBM, 432 features, test AUC 0.9647 / AP 0.7785).
Improvements attempted here, all measured against v3 on the SAME
70/15/15 stratified seed-42 split (same row order, same test set):

1. Feature engineering, fitted on TRAIN only, mapped to cal/test:
   - TransactionDT-derived: dt_day, dt_hour, dt_day_mod7
     (DT is seconds from an unknown origin, so mod-7 is NOT a weekday)
   - Frequency encoding for card1, card2, card4, P/R_emaildomain,
     DeviceInfo, id_30, id_31, ProductCD, addr1 (unseen -> 0,
     missing -> train missing-count)
   - Mean TransactionAmt per card1, card4, P_emaildomain, ProductCD
     (unseen/missing -> global train mean)
   - V_missing_count: number of missing V columns per row
2. Model candidates (all calibrated with isotonic on calibration):
   A: LightGBM with v3 params + FE, median-filled
   B: LightGBM tuned (leaves 127, lr 0.03, up to 3000 trees, early
      stopping on cal) + FE, median-filled
   C: LightGBM tuned + FE, NaN preserved for original numeric
      columns (LightGBM can split on missingness; median-filling
      destroys that signal)
   D: XGBoost (hist, depth 7, lr 0.03, scale_pos_weight) + FE, NaN
      preserved — skipped gracefully if xgboost is not installed
3. Selection on CALIBRATION raw AP only (never on test). The
   selected model's thresholds use the same rules as v3 (review =
   highest threshold with recall >= 0.90, block = highest-recall
   threshold with precision >= 0.90, on calibrated cal proba).
   Test metrics are reported for all candidates for transparency,
   but selection does not use them.

Outputs in /kaggle/working/: model.pkl (selected candidate only),
metrics.json, model_card.md, versions.json, feature_importance.json.
The model.pkl carries fe_maps + keep_nan so local inference in
gatekeeper/ieee_model.py can reproduce the FE transform exactly.
"""

import json
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

FREQ_GROUPS = ["card1", "card2", "card4", "P_emaildomain", "R_emaildomain",
               "DeviceInfo", "id_30", "id_31", "ProductCD", "addr1"]
AMT_MEAN_GROUPS = ["card1", "card4", "P_emaildomain", "ProductCD"]


def find_input_file(name):
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
    drop = {"TransactionID", "TransactionDT", "isFraud"}
    cat_cols = [c for c in df.columns
                if c not in drop and not pd.api.types.is_numeric_dtype(df[c])]
    num_cols = [c for c in df.columns
                if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
    return num_cols, cat_cols


def fit_preprocess(train, num_cols, cat_cols):
    medians = {}
    for col in num_cols:
        medians[col] = float(train[col].median()) if train[col].notna().any() else 0.0
    cat_maps = {}
    for col in cat_cols:
        vals = train[col].fillna("MISSING").astype(str).unique()
        cat_maps[col] = {v: i for i, v in enumerate(sorted(vals))}
    return medians, cat_maps


def fit_fe(train):
    """Fit FE maps on train only. Returns (fe_cols, fe_maps)."""
    fe_maps = {}
    fe_cols = ["dt_day", "dt_hour", "dt_day_mod7"]
    for group in FREQ_GROUPS:
        if group not in train.columns:
            continue
        counts = train[group].value_counts(dropna=False)
        missing = float(counts[counts.index.isna()].sum()) if counts.index.isna().any() else 0.0
        fe_maps[f"{group}_freq"] = {"group": group, "kind": "freq",
                                    "map": counts.to_dict(), "missing": missing,
                                    "default": 0.0}
        fe_cols.append(f"{group}_freq")
    global_mean = float(train["TransactionAmt"].mean())
    for group in AMT_MEAN_GROUPS:
        if group not in train.columns:
            continue
        fe_maps[f"amt_mean_{group}"] = {"group": group, "kind": "mean",
                                        "map": train.groupby(group)["TransactionAmt"].mean().to_dict(),
                                        "default": global_mean}
        fe_cols.append(f"amt_mean_{group}")
    v_cols = [c for c in train.columns if c.startswith("V")
              and pd.api.types.is_numeric_dtype(train[c])]
    fe_maps["V_missing_count"] = {"v_cols": v_cols}
    fe_cols.append("V_missing_count")
    return fe_cols, fe_maps


def _fe_series(df, name, spec):
    group, kind = spec["group"], spec["kind"]
    if kind == "freq":
        mapped = df[group].map(spec["map"])
        mapped = mapped.where(~df[group].isna(), spec["missing"]).fillna(spec["default"])
        return mapped.astype(np.float32).to_numpy()
    mapped = df[group].map(spec["map"]).fillna(spec["default"])
    return mapped.astype(np.float32).to_numpy()


def transform(df, num_cols, cat_cols, fe_cols, features, medians, cat_maps, fe_maps,
              keep_nan=False):
    cols = []
    for col in num_cols:
        s = df[col].astype(np.float32)
        if not keep_nan:
            s = s.fillna(medians[col])
        cols.append(s.to_numpy())
    dt = df["TransactionDT"].fillna(0).astype(np.int64)
    for col in fe_cols:
        if col == "dt_day":
            cols.append((dt // 86400).astype(np.float32).to_numpy())
        elif col == "dt_hour":
            cols.append(((dt // 3600) % 24).astype(np.float32).to_numpy())
        elif col == "dt_day_mod7":
            cols.append(((dt // 86400) % 7).astype(np.float32).to_numpy())
        elif col == "V_missing_count":
            v_cols = fe_maps["V_missing_count"]["v_cols"]
            present = [c for c in v_cols if c in df.columns]
            cols.append(df[present].isna().sum(axis=1).astype(np.float32).to_numpy())
        else:
            cols.append(_fe_series(df, col, fe_maps[col]))
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
    block, best_recall = None, -1.0
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


def lgbm_params(tuned):
    if tuned:
        return dict(n_estimators=3000, learning_rate=0.03, num_leaves=127,
                    min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
                    reg_lambda=10.0, class_weight="balanced", n_jobs=4,
                    random_state=SEED, verbose=-1)
    return dict(n_estimators=800, learning_rate=0.05, num_leaves=63,
                min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
                class_weight="balanced", n_jobs=4, random_state=SEED, verbose=-1)


def train_lgbm(X_train, y_train, X_cal, y_cal, tuned):
    import lightgbm as lgb

    model = lgb.LGBMClassifier(**lgbm_params(tuned))
    try:
        model.fit(X_train, y_train, eval_set=[(X_cal, y_cal)],
                  callbacks=[lgb.early_stopping(200 if tuned else 100, verbose=False)])
    except Exception as exc:
        print(f"early-stopping fit failed ({exc}); fitting without it")
        model.fit(X_train, y_train)
    return model


def train_xgb(X_train, y_train, X_cal, y_cal):
    from xgboost import XGBClassifier

    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    model = XGBClassifier(n_estimators=2000, learning_rate=0.03, max_depth=7,
                          subsample=0.8, colsample_bytree=0.8,
                          scale_pos_weight=neg / max(pos, 1), tree_method="hist",
                          n_jobs=4, random_state=SEED, early_stopping_rounds=100)
    try:
        model.fit(X_train, y_train, eval_set=[(X_cal, y_cal)], verbose=False)
    except Exception as exc:
        print(f"xgb early-stopping fit failed ({exc}); fitting without it")
        model = XGBClassifier(n_estimators=1000, learning_rate=0.03, max_depth=7,
                              subsample=0.8, colsample_bytree=0.8,
                              scale_pos_weight=neg / max(pos, 1), tree_method="hist",
                              n_jobs=4, random_state=SEED)
        model.fit(X_train, y_train)
    return model


def main():
    df = load_data()
    num_cols, cat_cols = build_feature_lists(df)
    train, temp = train_test_split(df, test_size=0.30, stratify=df["isFraud"], random_state=SEED)
    cal, test = train_test_split(temp, test_size=0.50, stratify=temp["isFraud"], random_state=SEED)
    medians, cat_maps = fit_preprocess(train, num_cols, cat_cols)
    fe_cols, fe_maps = fit_fe(train)
    features = num_cols + fe_cols + cat_cols + ["TransactionAmt_log"]
    print(f"features: {len(features)} ({len(num_cols)} numeric, {len(fe_cols)} FE, "
          f"{len(cat_cols)} categorical)")

    def make(split, keep_nan):
        return transform(split, num_cols, cat_cols, fe_cols, features,
                         medians, cat_maps, fe_maps, keep_nan=keep_nan)

    Xtr_med, Xcal_med, Xte_med = make(train, False), make(cal, False), make(test, False)
    Xtr_nan, Xcal_nan, Xte_nan = make(train, True), make(cal, True), make(test, True)
    y_train, y_cal, y_test = train["isFraud"].to_numpy(), cal["isFraud"].to_numpy(), test["isFraud"].to_numpy()
    amounts_test = test["TransactionAmt"].to_numpy(dtype=float)
    del train, cal, test, df

    candidates = {}
    print("training A: LGBM v3-params + FE (median fill)…")
    candidates["A_lgbm_v3_fe"] = train_lgbm(Xtr_med, y_train, Xcal_med, y_cal, tuned=False)
    print("training B: LGBM tuned + FE (median fill)…")
    candidates["B_lgbm_tuned_fe"] = train_lgbm(Xtr_med, y_train, Xcal_med, y_cal, tuned=True)
    print("training C: LGBM tuned + FE (NaN preserved)…")
    candidates["C_lgbm_tuned_fe_nan"] = train_lgbm(Xtr_nan, y_train, Xcal_nan, y_cal, tuned=True)
    try:
        print("training D: XGBoost + FE (NaN preserved)…")
        candidates["D_xgb_fe_nan"] = train_xgb(Xtr_nan, y_train, Xcal_nan, y_cal)
    except ImportError as exc:
        print(f"XGBoost unavailable, skipping D: {exc}")

    matrices = {"A_lgbm_v3_fe": (Xcal_med, Xte_med),
                "B_lgbm_tuned_fe": (Xcal_med, Xte_med),
                "C_lgbm_tuned_fe_nan": (Xcal_nan, Xte_nan),
                "D_xgb_fe_nan": (Xcal_nan, Xte_nan)}
    results = {}
    best, best_ap, best_auc = None, -1.0, -1.0
    for name, model in candidates.items():
        Xcal_m, Xte_m = matrices[name]
        raw_cal = model.predict_proba(Xcal_m)[:, 1]
        cal_ap = float(average_precision_score(y_cal, raw_cal))
        cal_auc = float(roc_auc_score(y_cal, raw_cal))
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw_cal, y_cal)
        proba_test = calibrator.predict(model.predict_proba(Xte_m)[:, 1])
        proba_cal = calibrator.predict(raw_cal)
        thresholds = choose_thresholds(y_cal, proba_cal)
        test_metrics = metrics_at(y_test, proba_test, thresholds)
        results[name] = {"cal_raw_ap": round(cal_ap, 4), "cal_raw_auc": round(cal_auc, 4),
                         "test": test_metrics}
        print(f"{name}: cal raw AP {cal_ap:.4f} AUC {cal_auc:.4f} | "
              f"test AUC {test_metrics['roc_auc']} AP {test_metrics['average_precision']}")
        if cal_ap > best_ap or (cal_ap == best_ap and cal_auc > best_auc):
            best, best_ap, best_auc = name, cal_ap, cal_auc
    print(f"SELECTED (cal raw AP): {best}")

    model = candidates[best]
    Xcal_m, Xte_m = matrices[best]
    keep_nan = best in ("C_lgbm_tuned_fe_nan", "D_xgb_fe_nan")
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(model.predict_proba(Xcal_m)[:, 1], y_cal)
    proba_cal = calibrator.predict(model.predict_proba(Xcal_m)[:, 1])
    proba_test = calibrator.predict(model.predict_proba(Xte_m)[:, 1])
    thresholds = choose_thresholds(y_cal, proba_cal)
    test_metrics = metrics_at(y_test, proba_test, thresholds)
    test_metrics["test_n"] = len(y_test)
    test_metrics["test_fraud"] = int(y_test.sum())
    test_metrics["expected_loss"] = expected_loss(y_test, amounts_test, proba_test, thresholds)
    results["selected"] = best
    results["selected_test"] = test_metrics
    print(json.dumps(test_metrics, indent=2))

    artifact = {"model": model, "calibrator": calibrator, "medians": medians,
                "cat_maps": cat_maps, "fe_maps": fe_maps, "keep_nan": keep_nan,
                "thresholds": thresholds, "features": features,
                "metadata": {"model_type": best, "features": features,
                             "data": "Kaggle dataset lnasiri007/ieeecis-fraud-detection (full, no subsampling)",
                             "split": "70/15/15 stratified seed 42, selection on calibration raw AP only, thresholds on calibration only"}}
    with open(os.path.join(OUT_DIR, "model.pkl"), "wb") as f:
        pickle.dump(artifact, f)
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    import sklearn, pandas

    versions = {"python": __import__("sys").version, "sklearn": sklearn.__version__,
                "pandas": pandas.__version__}
    try:
        import lightgbm

        versions["lightgbm"] = lightgbm.__version__
    except ImportError:
        pass
    try:
        import xgboost

        versions["xgboost"] = xgboost.__version__
    except ImportError:
        pass
    with open(os.path.join(OUT_DIR, "versions.json"), "w") as f:
        json.dump(versions, f, indent=2)
    try:
        imp = model.feature_importances_
        top = sorted(zip(features, [float(x) for x in imp]), key=lambda t: -t[1])[:30]
        with open(os.path.join(OUT_DIR, "feature_importance.json"), "w") as f:
            json.dump(top, f, indent=2)
    except Exception as exc:
        print(f"feature importance failed: {exc}")
    with open(os.path.join(OUT_DIR, "model_card.md"), "w") as f:
        f.write(f"""# Model card — Gatekeeper risk model (Kaggle v4 run)

- Selected model: {best} (selection: highest calibration raw AP, test not used for selection)
- Data: full IEEE-CIS train data via Kaggle dataset, no subsampling
- Features: {len(features)} — base numeric + FE (dt, frequency, amount-means, V-missing-count) + categorical + TransactionAmt_log, identity left-joined
- Keep-NaN for original numeric columns: {keep_nan}
- Split: 70/15/15 stratified seed 42, thresholds on calibration only
- Test fraud rate: {test_metrics['test_fraud']}/{test_metrics['test_n']} = {test_metrics['test_fraud']/test_metrics['test_n']:.4f}
- Thresholds: REVIEW >= {thresholds['review']:.4f}, BLOCK >= {thresholds['block']:.4f}
- Test ROC AUC: {test_metrics['roc_auc']}, average precision: {test_metrics['average_precision']}
- Test at REVIEW: precision {test_metrics['review']['precision']}, recall {test_metrics['review']['recall']}
- Test at BLOCK: precision {test_metrics['block']['precision']}, recall {test_metrics['block']['recall']}
- Expected loss per 10K: gated ${test_metrics['expected_loss']['gated_loss_per_10k']} vs no-gate ${test_metrics['expected_loss']['no_gate_loss_per_10k']}
- Candidates (cal raw AP / test AUC / test AP): see metrics.json
""")
    print("Wrote model.pkl, metrics.json, model_card.md, versions.json, feature_importance.json")


if __name__ == "__main__":
    main()
