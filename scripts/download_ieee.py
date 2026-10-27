"""Download IEEE-CIS Fraud Detection data — Kaggle route only.

Why Kaggle only: no verified public mirror of train_transaction.csv
was found (searched Oct 26, 2026). An unverified third-party site
claiming the file was deliberately NOT used — training on tampered
data would poison every metric this project reports.

What you need (one-time, ~2 minutes):
  1. A Kaggle account that has accepted the IEEE-CIS competition rules
     at https://www.kaggle.com/c/ieee-fraud-detection/rules
  2. An API token: Kaggle -> Settings -> API -> Create New Token,
     which downloads kaggle.json.
  3. Place it at ~/.kaggle/kaggle.json (chmod 600), or set
     KAGGLE_USERNAME and KAGGLE_KEY in the environment.

Usage:
    pip install kaggle
    python3 scripts/download_ieee.py [--out data/ieee-cis]

The script first checks whether the files already exist, then tries
the Kaggle dataset mirror (lnasiri007/ieeecis-fraud-detection) and
falls back to the competition download (ieee-fraud-detection).
Exit code 2 with instructions if no Kaggle credentials are present —
it never fabricates data.
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path

REQUIRED = ["train_transaction.csv"]
OPTIONAL = ["train_identity.csv", "test_transaction.csv", "test_identity.csv"]


def already_present(out: Path) -> bool:
    return all((out / name).exists() for name in REQUIRED)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/ieee-cis")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if already_present(out):
        print(f"IEEE-CIS files already present in {out}")
        return 0
    has_token = (Path.home() / ".kaggle" / "kaggle.json").exists() or (
        os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    if not has_token:
        print(
            "No Kaggle credentials found.\n"
            "  1. Accept the competition rules: "
            "https://www.kaggle.com/c/ieee-fraud-detection/rules\n"
            "  2. Kaggle -> Settings -> API -> Create New Token -> kaggle.json\n"
            "  3. cp kaggle.json ~/.kaggle/kaggle.json && chmod 600 ~/.kaggle/kaggle.json\n"
            "  4. Re-run this script.\n"
            "No data was downloaded and no synthetic substitute was created.",
            file=sys.stderr)
        return 2
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("pip install kaggle, then re-run.", file=sys.stderr)
        return 2
    api = KaggleApi()
    api.authenticate()
    for source, is_competition in [
        ("lnasiri007/ieeecis-fraud-detection", False),
        ("ieee-fraud-detection", True),
    ]:
        try:
            if is_competition:
                api.competition_download_files(source, path=str(out), quiet=False)
            else:
                api.dataset_download_files(source, path=str(out), unzip=True, quiet=False)
            break
        except Exception as exc:  # try the next source, then fail loudly
            print(f"{source} failed: {exc}", file=sys.stderr)
    else:
        return 1
    for zf in out.glob("*.zip"):
        with zipfile.ZipFile(zf) as z:
            z.extractall(out)
        zf.unlink()
    if already_present(out):
        print(f"Downloaded IEEE-CIS to {out}")
        return 0
    print(f"Download finished but {REQUIRED} not found in {out}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
