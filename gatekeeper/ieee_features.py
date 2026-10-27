"""Shared IEEE-CIS feature definition for training and inference.

Pure Python (no pandas) so the gateway can load a trained artifact
without the training stack. The training script fits the medians and
categorical maps; this module applies them.

Features: interpretable transaction fields + C1-C14 + D1-D15 + V1-V100.
V101-V339 are intentionally excluded in this first model: they add
memory and training time on a small VM for diminishing returns, and
the exclusion is recorded in the model card so it is not hidden.
"""

import math

BASE_NUMERIC = (["TransactionAmt", "card1", "card2", "card3", "card5",
                 "addr1", "addr2", "dist1", "dist2"]
                + [f"C{i}" for i in range(1, 15)]
                + [f"D{i}" for i in range(1, 16)])
CATEGORICAL = (["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain"]
               + [f"M{i}" for i in range(1, 10)])
V_FEATURES = [f"V{i}" for i in range(1, 101)]
FEATURES = BASE_NUMERIC + CATEGORICAL + V_FEATURES + ["TransactionAmt_log"]


def transform_row(row: dict, medians: dict, cat_maps: dict) -> list:
    """Map one raw transaction dict to a feature vector in FEATURES order."""
    out = []
    for name in BASE_NUMERIC + V_FEATURES:
        value = row.get(name)
        try:
            value = float(value) if value is not None else float("nan")
        except (TypeError, ValueError):
            value = float("nan")
        if value != value:  # NaN
            value = medians.get(name, 0.0)
        out.append(value)
    for name in CATEGORICAL:
        raw = row.get(name)
        key = "MISSING" if raw is None or (isinstance(raw, float) and raw != raw) else str(raw)
        out.append(cat_maps.get(name, {}).get(key, -1))
    amount = row.get("TransactionAmt")
    try:
        amount = float(amount) if amount is not None else 0.0
    except (TypeError, ValueError):
        amount = 0.0
    out.append(math.log1p(max(amount, 0.0)))
    return out


def _is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and value != value)


def _mean_lookup(row: dict, spec: dict) -> float:
    """Train mean amount for a group value (v5 amt_mean / TE maps)."""
    if spec is None:
        return 0.0
    group_val = row.get(spec["group"])
    if _is_missing(group_val):
        return float(spec["default"])
    return float(spec["map"].get(group_val, spec["default"]))


def transform_generic(row: dict, features: list, medians: dict, cat_maps: dict,
                      fe_maps: dict = None, keep_nan: bool = False) -> list:
    """Transform using an artifact's own feature order (Kaggle models).

    A feature is categorical iff it has an entry in cat_maps, numeric
    otherwise, except TransactionAmt_log which is always log1p(amount).
    This is used ONLY when the artifact carries a top-level "features"
    list (Kaggle format). Legacy local artifacts have no top-level
    list — their training order differs from FEATURES order, so they
    must keep using transform_row above, not this function.

    V4 artifacts add engineered features, reproduced here exactly:
    - dt_day / dt_hour / dt_day_mod7 from TransactionDT
    - V_missing_count: missing count over the artifact's V columns
    - fe_maps entries: {"group", "kind": "freq"|"mean", "map",
      "missing" (freq only), "default"} — freq maps a group value to
      its train count (missing group -> train missing-count, unseen
      -> 0); mean maps it to the train mean amount (missing/unseen
      -> global train mean). Python dict lookup treats int 1234 and
      float 1234.0 as the same key, matching the Kaggle transform.
    - keep_nan: when true, original numeric columns keep NaN instead
      of the train median (LightGBM/XGBoost split on missingness).
    """
    fe_maps = fe_maps or {}
    v_cols = [f for f in features if f.startswith("V") and f != "V_missing_count"]
    out = []
    for name in features:
        if name == "TransactionAmt_log":
            try:
                amount = float(row.get("TransactionAmt") or 0.0)
            except (TypeError, ValueError):
                amount = 0.0
            out.append(math.log1p(max(amount, 0.0)))
        elif name in ("dt_day", "dt_hour", "dt_day_mod7"):
            try:
                dt = int(row.get("TransactionDT") or 0)
            except (TypeError, ValueError):
                dt = 0
            if name == "dt_day":
                out.append(float(dt // 86400))
            elif name == "dt_hour":
                out.append(float((dt // 3600) % 24))
            else:
                out.append(float((dt // 86400) % 7))
        elif name == "V_missing_count":
            out.append(float(sum(1 for c in v_cols if _is_missing(row.get(c)))))
        elif name.startswith("amt_ratio_"):
            group = name[len("amt_ratio_"):]
            try:
                amt = float(row.get("TransactionAmt") or 0.0)
            except (TypeError, ValueError):
                amt = 0.0
            mean_val = _mean_lookup(row, fe_maps.get(f"amt_mean_{group}"))
            out.append(float(amt / mean_val) if mean_val else 0.0)
        elif name.startswith("amt_diff_"):
            group = name[len("amt_diff_"):]
            try:
                amt = float(row.get("TransactionAmt") or 0.0)
            except (TypeError, ValueError):
                amt = 0.0
            out.append(float(amt - _mean_lookup(row, fe_maps.get(f"amt_mean_{group}"))))
        elif name in fe_maps and fe_maps[name].get("kind") == "joint_freq":
            spec = fe_maps[name]
            g1, g2 = spec["groups"]
            v1, v2 = row.get(g1), row.get(g2)
            if _is_missing(v1) or _is_missing(v2):
                out.append(float(spec["missing"]))
            else:
                out.append(float(spec["map"].get(f"{v1}|{v2}", spec["default"])))
        elif name in fe_maps:
            spec = fe_maps[name]
            group_val = row.get(spec["group"])
            if _is_missing(group_val):
                out.append(float(spec["missing"]) if spec["kind"] == "freq"
                           else float(spec["default"]))
            else:
                out.append(float(spec["map"].get(group_val, spec["default"])))
        elif name in cat_maps:
            raw = row.get(name)
            key = "MISSING" if _is_missing(raw) else str(raw)
            out.append(cat_maps[name].get(key, -1))
        else:
            try:
                value = float(row.get(name)) if row.get(name) is not None else float("nan")
            except (TypeError, ValueError):
                value = float("nan")
            if value != value and not keep_nan:
                value = medians.get(name, 0.0)
            out.append(value)
    return out
