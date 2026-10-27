"""Generate a small IEEE-schema synthetic dataset — for PIPELINE TESTS ONLY.

This exists so risk/train_ieee.py can be run end-to-end and verified
before real Kaggle data arrives. Its fraud signal is deliberately
simple (high amount + high C-features + certain V-features) and its
numbers must NEVER be reported as IEEE-CIS results. Files are written
under data/synthetic-ieee/ and labelled synthetic in every output.

Usage:
    python3 risk/synthetic_ieee.py --out data/synthetic-ieee --n 20000 --seed 42
"""

import argparse

import numpy as np
import pandas as pd

COLS = (["TransactionAmt", "ProductCD", "card1", "card2", "card3", "card4",
         "card5", "card6", "addr1", "addr2", "dist1", "dist2",
         "P_emaildomain", "R_emaildomain"]
        + [f"M{i}" for i in range(1, 10)]
        + [f"C{i}" for i in range(1, 15)]
        + [f"D{i}" for i in range(1, 16)]
        + [f"V{i}" for i in range(1, 101)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/synthetic-ieee")
    parser.add_argument("--n", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    n = args.n
    # Fraud is rare (3.5%, matching IEEE-CIS) and correlated with amount
    # and a handful of features — enough to test the pipeline, not to
    # learn anything about real fraud.
    is_fraud = (rng.random(n) < 0.035).astype(int)
    df = pd.DataFrame({
        "TransactionID": np.arange(1, n + 1),
        "isFraud": is_fraud,
        "TransactionAmt": np.where(
            is_fraud == 1,
            rng.lognormal(5.0, 1.0, n), rng.lognormal(3.5, 0.8, n)),
        "ProductCD": rng.choice(["W", "C", "H", "R", "S"], n),
        "card1": rng.integers(1000, 20000, n),
        "card2": rng.integers(100, 999, n),
        "card3": rng.integers(100, 250, n),
        "card5": rng.integers(100, 250, n),
        "dist1": rng.normal(0, 100, n),
        "dist2": rng.normal(0, 100, n),
        "card4": rng.choice(["visa", "mastercard", "american express", "discover"], n),
        "card6": rng.choice(["credit", "debit"], n),
        "addr1": rng.integers(100, 500, n),
        "addr2": rng.integers(10, 100, n),
        "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", "hotmail.com"], n),
        "R_emaildomain": rng.choice(["gmail.com", "yahoo.com", None], n),
    })
    for i in range(1, 10):
        df[f"M{i}"] = rng.choice(["T", "F", None], n, p=[0.4, 0.4, 0.2])
    for i in range(1, 15):
        df[f"C{i}"] = rng.normal(0, 1, n) + is_fraud * rng.normal(0.8, 0.2)
    for i in range(1, 16):
        df[f"D{i}"] = rng.normal(0, 10, n)
    for i in range(1, 101):
        df[f"V{i}"] = rng.normal(0, 1, n) + (is_fraud * 0.5 if i <= 5 else 0)
    import os

    os.makedirs(args.out, exist_ok=True)
    df.to_csv(f"{args.out}/train_transaction.csv", index=False)
    print(f"Wrote {n} synthetic rows ({is_fraud.sum()} fraud) to {args.out}/train_transaction.csv")


if __name__ == "__main__":
    main()
