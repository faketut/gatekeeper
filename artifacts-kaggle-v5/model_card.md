# Model card — Gatekeeper risk model (Kaggle v5 run)

- Selected model: seed_avg (selection: highest calibration raw AP, test not used for selection)
- Data: full IEEE-CIS train data via Kaggle dataset, no subsampling
- Features: 457 — base numeric + FE (v4 FE + amount ratio/diff, joint frequency, OOF target encoding) + categorical + TransactionAmt_log, identity left-joined
- Split: 70/15/15 stratified seed 42, thresholds on calibration only
- Test fraud rate: 3099/88581 = 0.0350
- Thresholds: REVIEW >= 0.0397, BLOCK >= 0.3846
- Test ROC AUC: 0.9772, average precision: 0.8724
- Test at REVIEW: precision 0.45, recall 0.9038
- Test at BLOCK: precision 0.9075, recall 0.7725
- Expected loss per 10K: gated $6678.35 vs no-gate $51593.48
- Loss-optimal thresholds (secondary, cal-optimised): {'review': 0.009962640099626401, 'block': 0.8738461538461538}
- Candidates (cal raw AP / test AUC / test AP): see metrics.json
