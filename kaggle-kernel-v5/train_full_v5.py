"""Gatekeeper v5 training — runs on Kaggle (CPU), not locally.

Supersedes v4 (tuned LightGBM + FE, test AUC 0.9771 / AP 0.8711,
gated $6,587 per 10K). Measured against v4 on the SAME 70/15/15
stratified seed-42 split (same row order, same test set).

New in v5:
1. More features, fitted on TRAIN only:
   - amt_ratio_card1 / amt_diff_card1: amount vs the card's train
     mean amount (ratio and difference; the model can split on
     both, but an explicit ratio often helps trees)
   - Joint frequency: (card1, addr1) and (card1, ProductCD) pair
     counts in train (unseen -> 0, either group missing -> train
     missing-pair count)
   - OOF target encoding for card1, P_emaildomain, card4: within
     TRAIN, 5-fold out-of-fold smoothed fraud rates (m=50 shrink
     to the train fraud rate); cal/test use the full-train
     smoothed map. OOF is what keeps this honest — a plain
     train-mean encoding would leak each row's own label.
2. Ensemble candidates (all calibrated with isotonic on cal):
   - seed_avg: tuned LightGBM (v4 winner params) + v5 features,
     median-filled, seeds 42/2026/7, raw probas averaged
   - lgbm_nan: tuned LightGBM + v5 features, NaN preserved
   - xgb: XGBoost (hist, depth 7, lr 0.03) + v5 features, NaN
   - catboost: CatBoost (depth 7, lr 0.05, 1200 iters) + v5
     features, NaN, categorical indices passed — skipped
     gracefully if catboost is missing or rejects the matrix
   - ensemble: mean raw proba over ALL individual models above
3. Selection on CALIBRATION raw AP only (never on test).
   Primary thresholds use v4's rules (review = highest threshold
   with recall >= 0.90, block = highest-recall threshold with
   precision >= 0.90, on calibrated cal proba). A loss-optimal
   threshold pair (minimising expected loss on cal over a
   quantile grid) is reported as SECONDARY only — it does not
   change the primary operating point or the promotion rule.
Promotion rule (same discipline as v4): promote only if test AP
exceeds v4's 0.8711 with AUC not dropping below 0.9771 and gated
loss (primary thresholds) not rising above $6,587 per 10K.

Outputs in /kaggle/working/: model.pkl (selected candidate;
ensemble artifacts carry "members" with each member's keep_nan),
metrics.json, model_card.md, versions.json, feature_importance.json.
"""

import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (average_precision_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split

INPUT_DIR = "/kaggle/input/ieeecis-fraud-detection"
OUT_DIR = "/kaggle/working"
SEED = 42
REVIEW_COST = 2.0
FALSE_BLOCK_COST = 15.0
TE_SMOOTH_M = 50.0

FREQ_GROUPS = ["card1", "card2", "card4", "P_emaildomain", "R_emaildomain",
               "DeviceInfo", "id_30", "id_31", "ProductCD", "addr1"]
AMT_MEAN_GROUPS = ["card1", "card4", "P_emaildomain", "ProductCD"]
JOINT_GROUPS = [("card1", "addr1"), ("card1", "ProductCD")]
TE_GROUPS = ["card1", "P_emaildomain", "card4"]


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


def _smoothed_map(sums, counts, global_rate):
    return ((sums + TE_SMOOTH_M * global_rate) / (counts + TE_SMOOTH_M)).to_dict()


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
    # Ratio / difference vs the card's mean amount (derived at transform time).
    if "card1" in train.columns:
        fe_cols += ["amt_ratio_card1", "amt_diff_card1"]
    for g1, g2 in JOINT_GROUPS:
        if g1 not in train.columns or g2 not in train.columns:
            continue
        key = train[g1].astype(str) + "|" + train[g2].astype(str)
        counts = key.value_counts(dropna=False)
        missing = float(((train[g1].isna()) | (train[g2].isna())).sum())
        name = f"{g1}_{g2}_freq"
        fe_maps[name] = {"groups": [g1, g2], "kind": "joint_freq",
                         "map": counts.to_dict(), "missing": missing, "default": 0.0}
        fe_cols.append(name)
    global_rate = float(train["isFraud"].mean())
    for group in TE_GROUPS:
        if group not in train.columns:
            continue
        sums = train.groupby(group)["isFraud"].sum()
        counts = train.groupby(group)["isFraud"].count()
        fe_maps[f"te_{group}"] = {"group": group, "kind": "mean",
                                  "map": _smoothed_map(sums, counts, global_rate),
                                  "default": global_rate}
        fe_cols.append(f"te_{group}")
    return fe_cols, fe_maps


def fit_oof_te(train):
    """Out-of-fold target encoding for train rows only.

    Returns {te_feature: np array aligned with train's row order}.
    Each fold's values come from the other folds' smoothed means,
    so no row's own label leaks into its encoding.
    """
    y = train["isFraud"].to_numpy()
    global_rate = float(y.mean())
    out = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for group in TE_GROUPS:
        if group not in train.columns:
            continue
        arr = np.full(len(train), global_rate, dtype=np.float32)
        col = train[group].to_numpy()
        for tr_idx, va_idx in skf.split(np.zeros(len(train)), y):
            other = train.iloc[tr_idx]
            sums = other.groupby(group)["isFraud"].sum()
            counts = other.groupby(group)["isFraud"].count()
            smoothed = _smoothed_map(sums, counts, float(other["isFraud"].mean()))
            mapped = pd.Series(col[va_idx]).map(smoothed).fillna(float(other["isFraud"].mean()))
            arr[va_idx] = mapped.to_numpy(dtype=np.float32)
        out[f"te_{group}"] = arr
    return out


def _fe_series(df, name, spec):
    group, kind = spec["group"], spec["kind"]
    if kind == "freq":
        mapped = df[group].map(spec["map"])
        mapped = mapped.where(~df[group].isna(), spec["missing"]).fillna(spec["default"])
        return mapped.astype(np.float32).to_numpy()
    mapped = df[group].map(spec["map"]).fillna(spec["default"])
    return mapped.astype(np.float32).to_numpy()


def transform(df, num_cols, cat_cols, fe_cols, features, medians, cat_maps, fe_maps,
              keep_nan=False, te_oof=None):
    cols = []
    for col in num_cols:
        s = df[col].astype(np.float32)
        if not keep_nan:
            s = s.fillna(medians[col])
        cols.append(s.to_numpy())
    dt = df["TransactionDT"].fillna(0).astype(np.int64)
    amt = df["TransactionAmt"].fillna(0.0).astype(float).to_numpy()
    amt_mean_cache = {}

    def amt_mean_series(group):
        if group not in amt_mean_cache:
            amt_mean_cache[group] = _fe_series(df, f"amt_mean_{group}",
                                               fe_maps[f"amt_mean_{group}"])
        return amt_mean_cache[group]

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
        elif te_oof is not None and col in te_oof:
            cols.append(te_oof[col])
        elif col.startswith("amt_ratio_"):
            group = col[len("amt_ratio_"):]
            mean_s = amt_mean_series(group).astype(float)
            ratio = np.where(mean_s != 0, amt / np.where(mean_s != 0, mean_s, 1.0), 0.0)
            cols.append(ratio.astype(np.float32))
        elif col.startswith("amt_diff_"):
            group = col[len("amt_diff_"):]
            cols.append((amt - amt_mean_series(group).astype(float)).astype(np.float32))
        elif col in fe_maps and fe_maps[col].get("kind") == "joint_freq":
            spec = fe_maps[col]
            g1, g2 = spec["groups"]
            key = df[g1].astype(str) + "|" + df[g2].astype(str)
            mapped = key.map(spec["map"])
            missing_mask = df[g1].isna() | df[g2].isna()
            mapped = mapped.where(~missing_mask, spec["missing"]).fillna(spec["default"])
            cols.append(mapped.astype(np.float32).to_numpy())
        else:
            cols.append(_fe_series(df, col, fe_maps[col]))
    for col in cat_cols:
        mapped = df[col].fillna("MISSING").astype(str).map(cat_maps[col])
        cols.append(mapped.fillna(-1).astype(np.float32).to_numpy())
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


def loss_optimal_thresholds(y_cal, amounts_cal, proba_cal):
    """Grid-search the review/block pair minimising expected loss on cal.

    SECONDARY only: reported for comparison, never used for the
    primary operating point or the promotion decision.
    """
    grid = np.quantile(proba_cal, np.linspace(0.5, 0.999, 50))
    best, best_loss = None, float("inf")
    for tr in grid:
        for tb in grid:
            if tb <= tr:
                continue
            blocked = proba_cal >= tb
            reviewed = (proba_cal >= tr) & ~blocked
            missed = (~blocked) & (~reviewed) & (y_cal == 1)
            false_blocked = blocked & (y_cal == 0)
            loss = float(np.sum(amounts_cal[missed]) + REVIEW_COST * int(reviewed.sum())
                         + FALSE_BLOCK_COST * int(false_blocked.sum()))
            if loss < best_loss:
                best_loss, best = loss, {"review": float(tr), "block": float(tb)}
    return best


def lgbm_params(seed):
    return dict(n_estimators=3000, learning_rate=0.03, num_leaves=127,
                min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
                reg_lambda=10.0, class_weight="balanced", n_jobs=4,
                random_state=seed, verbose=-1)


def train_lgbm(X_train, y_train, X_cal, y_cal, seed):
    import lightgbm as lgb

    model = lgb.LGBMClassifier(**lgbm_params(seed))
    try:
        model.fit(X_train, y_train, eval_set=[(X_cal, y_cal)],
                  callbacks=[lgb.early_stopping(200, verbose=False)])
    except Exception as exc:
        print(f"lgbm early-stopping fit failed ({exc}); fitting without it")
        model.fit(X_train, y_train)
    return model


def train_xgb(X_train, y_train, X_cal, y_cal):
    from xgboost import XGBClassifier

    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    model = XGBClassifier(n_estimators=1500, learning_rate=0.03, max_depth=7,
                          subsample=0.8, colsample_bytree=0.8,
                          scale_pos_weight=neg / max(pos, 1), tree_method="hist",
                          n_jobs=4, random_state=SEED, early_stopping_rounds=100)
    try:
        model.fit(X_train, y_train, eval_set=[(X_cal, y_cal)], verbose=False)
    except Exception as exc:
        print(f"xgb early-stopping fit failed ({exc}); fitting without it")
        model = XGBClassifier(n_estimators=800, learning_rate=0.03, max_depth=7,
                              subsample=0.8, colsample_bytree=0.8,
                              scale_pos_weight=neg / max(pos, 1), tree_method="hist",
                              n_jobs=4, random_state=SEED)
        model.fit(X_train, y_train)
    return model


def train_catboost(X_train, y_train, X_cal, y_cal, cat_idx):
    from catboost import CatBoostClassifier

    model = CatBoostClassifier(iterations=1200, learning_rate=0.05, depth=7,
                               loss_function="Logloss", eval_metric="AUC",
                               random_seed=SEED, verbose=False,
                               od_type="Iter", od_wait=150, cat_features=cat_idx)
    model.fit(X_train, y_train, eval_set=(X_cal, y_cal), use_best_model=True)
    return model


def main():
    df = load_data()
    num_cols, cat_cols = build_feature_lists(df)
    train, temp = train_test_split(df, test_size=0.30, stratify=df["isFraud"], random_state=SEED)
    cal, test = train_test_split(temp, test_size=0.50, stratify=temp["isFraud"], random_state=SEED)
    medians, cat_maps = fit_preprocess(train, num_cols, cat_cols)
    fe_cols, fe_maps = fit_fe(train)
    oof_te = fit_oof_te(train)
    features = num_cols + fe_cols + cat_cols + ["TransactionAmt_log"]
    print(f"features: {len(features)} ({len(num_cols)} numeric, {len(fe_cols)} FE, "
          f"{len(cat_cols)} categorical)")

    def make(split, keep_nan, te_oof=None):
        return transform(split, num_cols, cat_cols, fe_cols, features,
                         medians, cat_maps, fe_maps, keep_nan=keep_nan, te_oof=te_oof)

    Xtr_med = make(train, False, te_oof=oof_te)
    Xcal_med, Xte_med = make(cal, False), make(test, False)
    Xtr_nan = make(train, True, te_oof=oof_te)
    Xcal_nan, Xte_nan = make(cal, True), make(test, True)
    y_train = train["isFraud"].to_numpy()
    y_cal = cal["isFraud"].to_numpy()
    y_test = test["isFraud"].to_numpy()
    amounts_cal = cal["TransactionAmt"].to_numpy(dtype=float)
    amounts_test = test["TransactionAmt"].to_numpy(dtype=float)
    del train, cal, test, df

    # Individual models: (name, model, Xcal, Xte, keep_nan)
    individuals = []
    seed_models = []
    for seed in (42, 2026, 7):
        print(f"training LGBM tuned seed {seed} (median fill)…")
        m = train_lgbm(Xtr_med, y_train, Xcal_med, y_cal, seed)
        seed_models.append(m)
        individuals.append((f"lgbm_seed_{seed}", m, Xcal_med, Xte_med, False))
    print("training LGBM tuned (NaN)…")
    lgbm_nan = train_lgbm(Xtr_nan, y_train, Xcal_nan, y_cal, SEED)
    individuals.append(("lgbm_nan", lgbm_nan, Xcal_nan, Xte_nan, True))
    try:
        print("training XGBoost (NaN)…")
        xgb = train_xgb(Xtr_nan, y_train, Xcal_nan, y_cal)
        individuals.append(("xgb", xgb, Xcal_nan, Xte_nan, True))
    except ImportError as exc:
        print(f"XGBoost unavailable, skipping: {exc}")
    cat_idx = [len(num_cols) + len(fe_cols) + i for i in range(len(cat_cols))]
    try:
        print("training CatBoost (NaN, categorical indices)…")
        cb = train_catboost(Xtr_nan, y_train, Xcal_nan, y_cal, cat_idx)
        individuals.append(("catboost", cb, Xcal_nan, Xte_nan, True))
    except Exception as exc:
        print(f"CatBoost unavailable/failed, skipping: {exc}")

    def raw_probas(model, X):
        return model.predict_proba(X)[:, 1]

    # Candidates: (name, raw_cal, raw_test, artifact_members)
    candidates = {}
    seed_raw_cal = np.mean([raw_probas(m, Xcal_med) for m in seed_models], axis=0)
    seed_raw_test = np.mean([raw_probas(m, Xte_med) for m in seed_models], axis=0)
    candidates["seed_avg"] = (seed_raw_cal, seed_raw_test,
                              [{"model": m, "keep_nan": False} for m in seed_models])
    for name, model, Xcal_m, Xte_m, keep_nan in individuals:
        if name.startswith("lgbm_seed_"):
            continue
        candidates[name] = (raw_probas(model, Xcal_m), raw_probas(model, Xte_m),
                            [{"model": model, "keep_nan": keep_nan}])
    # Ensemble raw = mean over ALL individual models (each on its own matrix).
    ens_cal = np.mean([raw_probas(m, Xm) for _, m, Xm, _, _ in individuals], axis=0)
    ens_test = np.mean([raw_probas(m, Xt) for _, m, _, Xt, _ in individuals], axis=0)
    candidates["ensemble"] = (ens_cal, ens_test,
                              [{"model": m, "keep_nan": kn} for _, m, _, _, kn in individuals])

    results = {}
    best, best_ap, best_auc = None, -1.0, -1.0
    for name, (raw_cal, raw_test, _members) in candidates.items():
        cal_ap = float(average_precision_score(y_cal, raw_cal))
        cal_auc = float(roc_auc_score(y_cal, raw_cal))
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw_cal, y_cal)
        proba_test = calibrator.predict(raw_test)
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

    raw_cal, raw_test, members = candidates[best]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_cal, y_cal)
    proba_cal = calibrator.predict(raw_cal)
    proba_test = calibrator.predict(raw_test)
    thresholds = choose_thresholds(y_cal, proba_cal)
    test_metrics = metrics_at(y_test, proba_test, thresholds)
    test_metrics["test_n"] = len(y_test)
    test_metrics["test_fraud"] = int(y_test.sum())
    test_metrics["expected_loss"] = expected_loss(y_test, amounts_test, proba_test, thresholds)
    loss_opt = loss_optimal_thresholds(y_cal, amounts_cal, proba_cal)
    test_metrics["loss_optimal"] = {
        "cal_thresholds": loss_opt,
        "test": metrics_at(y_test, proba_test, loss_opt),
        "test_expected_loss": expected_loss(y_test, amounts_test, proba_test, loss_opt),
    }
    results["selected"] = best
    results["selected_test"] = test_metrics
    print(json.dumps(test_metrics, indent=2))

    is_ensemble = len(members) > 1
    artifact = {"calibrator": calibrator, "medians": medians, "cat_maps": cat_maps,
                "fe_maps": fe_maps, "thresholds": thresholds, "features": features,
                "metadata": {"model_type": best, "features": features,
                             "data": "Kaggle dataset lnasiri007/ieeecis-fraud-detection (full, no subsampling)",
                             "split": "70/15/15 stratified seed 42, selection on calibration raw AP only, thresholds on calibration only"}}
    if is_ensemble:
        artifact["ensemble"] = True
        artifact["members"] = members
    else:
        artifact["model"] = members[0]["model"]
        artifact["keep_nan"] = members[0]["keep_nan"]
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
    try:
        import catboost

        versions["catboost"] = catboost.__version__
    except ImportError:
        pass
    with open(os.path.join(OUT_DIR, "versions.json"), "w") as f:
        json.dump(versions, f, indent=2)
    try:
        first = members[0]["model"]
        imp = first.feature_importances_
        top = sorted(zip(features, [float(x) for x in imp]), key=lambda t: -t[1])[:30]
        with open(os.path.join(OUT_DIR, "feature_importance.json"), "w") as f:
            json.dump(top, f, indent=2)
    except Exception as exc:
        print(f"feature importance failed: {exc}")
    with open(os.path.join(OUT_DIR, "model_card.md"), "w") as f:
        f.write(f"""# Model card — Gatekeeper risk model (Kaggle v5 run)

- Selected model: {best} (selection: highest calibration raw AP, test not used for selection)
- Data: full IEEE-CIS train data via Kaggle dataset, no subsampling
- Features: {len(features)} — base numeric + FE (v4 FE + amount ratio/diff, joint frequency, OOF target encoding) + categorical + TransactionAmt_log, identity left-joined
- Split: 70/15/15 stratified seed 42, thresholds on calibration only
- Test fraud rate: {test_metrics['test_fraud']}/{test_metrics['test_n']} = {test_metrics['test_fraud']/test_metrics['test_n']:.4f}
- Thresholds: REVIEW >= {thresholds['review']:.4f}, BLOCK >= {thresholds['block']:.4f}
- Test ROC AUC: {test_metrics['roc_auc']}, average precision: {test_metrics['average_precision']}
- Test at REVIEW: precision {test_metrics['review']['precision']}, recall {test_metrics['review']['recall']}
- Test at BLOCK: precision {test_metrics['block']['precision']}, recall {test_metrics['block']['recall']}
- Expected loss per 10K: gated ${test_metrics['expected_loss']['gated_loss_per_10k']} vs no-gate ${test_metrics['expected_loss']['no_gate_loss_per_10k']}
- Loss-optimal thresholds (secondary, cal-optimised): {loss_opt}
- Candidates (cal raw AP / test AUC / test AP): see metrics.json
""")
    print("Wrote model.pkl, metrics.json, model_card.md, versions.json, feature_importance.json")


if __name__ == "__main__":
    main()
