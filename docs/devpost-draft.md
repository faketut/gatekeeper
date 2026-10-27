# Devpost draft — Gatekeeper (metrics below are MEASURED on real IEEE-CIS, 2026-10-27)

## Measured results — real IEEE-CIS (not synthetic)
- Data: IEEE-CIS train data, FULL 590,540 rows (fraud rate 3.50%, no
  subsampling); 70% train / 15% calibration / 15% test, stratified,
  seed 42. Model selection on calibration AP only (never on test);
  thresholds chosen on calibration only. Model: LightGBM tuned
  (leaves 127, lr 0.03, early stopping), balanced class weight,
  trained on Kaggle (private CPU kernel, dataset attached as input).
  Features: 450 — all numeric + categorical columns including
  V1-V339, identity left-joined (144,233 identity rows; string
  identity columns treated as categorical by dtype), PLUS
  train-only feature engineering: TransactionDT day/hour, frequency
  encoding for card/email/device groups, mean amount per card/email/
  product group, V-column missing count. Four candidates were run
  (v3-params LightGBM, tuned LightGBM median-filled, tuned LightGBM
  NaN-preserving, XGBoost); the tuned median-filled LightGBM won on
  calibration AP (0.8665) and is the model served.
- ROC AUC: 0.9771; average precision: 0.8711 (test n=88,581, 3,099 fraud)
- REVIEW threshold 0.0455: precision 0.4437, recall 0.9051
  (TP 2,805, FP 3,517, FN 294, TN 81,965) — the review queue catches
  ~91% of fraud while reviewing only 7.1% of all transactions.
  Review precision is still the honest weakness: most reviewed
  transactions are legitimate, which is the cost of catching 90%+
  of fraud at 3.5% prevalence — but it is roughly half the review
  burden of the previous model.
- BLOCK threshold 0.5750: precision 0.9287, recall 0.7477
  (TP 2,317, FP 178, FN 782, TN 85,304) — auto-block is precise and
  now catches three quarters of fraud outright.
- Expected loss per 10,000 transactions: gated $6,587 vs no-gate
  $51,593, under stated assumptions (review $2, false block $15,
  fraud loss = transaction amount, review catches fraud). Change the
  cost assumptions and the operating point changes.
- Superseded runs (for transparency, NOT the headline numbers):
  Kaggle v3, same data/split, no feature engineering — ROC AUC
  0.9647, AP 0.7785, review precision 0.2358 at 13.5% reviewed
  (model kept at artifacts/model-v3-kaggle-backup.pkl); the earlier
  local run, 150,228-row sample, HistGradientBoosting, V1-V100 only
  — ROC AUC 0.9141, AP 0.5874, review precision 0.0998 at 30.8%
  reviewed (model kept at artifacts/model-local-150k-backup.pkl).
- Earlier run note (for transparency, NOT for submission): a first
  training run inflated the sample fraud rate to 13.8% by keeping all
  fraud rows; its precision/loss numbers were discarded and the
  pipeline was fixed and rerun. Do not quote that run.
- PayPal sandbox: real order created and verified CREATED via Orders
  v2 (sandbox, 2026-10-27); capture requires payer approval via the
  approve link, per the Orders lifecycle.

## What it does
Gatekeeper is the trust layer for agentic commerce: an AI shopping agent
proposes a payment, a risk model scores it, and the payment is
auto-approved, held for human approval, or blocked before money moves —
via PayPal sandbox Orders v2, with Disputes v1 simulate endpoints for
the chargeback aftermath.

## Why it matters
Every agentic-commerce demo wires the agent to a wallet. Nobody ships
the brakes. Gatekeeper is the brakes, with the evaluation to back it:
precision-recall at the operating threshold and expected-loss math,
not accuracy on a 3.5%-fraud dataset.

## Demo (separate from the IEEE-CIS model)
- Behavioural demo scorer: normal ALLOW / $2,000 3am REVIEW /
  $12,000 burst BLOCK (verified in demo/run_demo.py, stdlib scorer —
  not the IEEE-CIS model). The IEEE-CIS model is served separately at
  POST /propose-ieee and verified against a real IEEE-CIS row.

## Prize categories
Best Use of PayPal+AI, Most Impactful, Most Creative.

## Links — FILL AT SUBMIT TIME
- Public repo: ___
- Demo URL / video (<3 min): ___
