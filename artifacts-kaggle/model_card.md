# Model card — Gatekeeper risk model (Kaggle full-data run)

- Model: lightgbm
- Data: full IEEE-CIS train data via Kaggle dataset, no subsampling
- Features: 432 — all numeric + categorical + TransactionAmt_log, identity left-joined
- Split: 70/15/15 stratified seed 42, thresholds on calibration only
- Test fraud rate: 3099/88581 = 0.0350
- Thresholds: REVIEW >= 0.0279, BLOCK >= 0.6279
- Test ROC AUC: 0.9647, average precision: 0.7785
- Test at REVIEW: precision 0.2358, recall 0.9077
- Test at BLOCK: precision 0.9206, recall 0.5502
- Expected loss per 10K: gated $7857.17 vs no-gate $51593.48
