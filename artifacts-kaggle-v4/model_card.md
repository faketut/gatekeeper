# Model card — Gatekeeper risk model (Kaggle v4 run)

- Selected model: B_lgbm_tuned_fe (selection: highest calibration raw AP, test not used for selection)
- Data: full IEEE-CIS train data via Kaggle dataset, no subsampling
- Features: 450 — base numeric + FE (dt, frequency, amount-means, V-missing-count) + categorical + TransactionAmt_log, identity left-joined
- Keep-NaN for original numeric columns: False
- Split: 70/15/15 stratified seed 42, thresholds on calibration only
- Test fraud rate: 3099/88581 = 0.0350
- Thresholds: REVIEW >= 0.0455, BLOCK >= 0.5750
- Test ROC AUC: 0.9771, average precision: 0.8711
- Test at REVIEW: precision 0.4437, recall 0.9051
- Test at BLOCK: precision 0.9287, recall 0.7477
- Expected loss per 10K: gated $6587.29 vs no-gate $51593.48
- Candidates (cal raw AP / test AUC / test AP): see metrics.json
