# Implementation status — 2026-10-27

Autonomous build pass after the 2026-10-26 scaffold, at Jack's
"implement the whole project" instruction.

## Done and verified

- Full training pipeline (`risk/train_ieee.py`): chunked, memory-safe
  loading (all fraud kept, legit subsampled to `--max-rows`), 70/15/15
  stratified split, categorical maps + medians fitted on train only,
  isotonic calibration on the calibration split, thresholds chosen on
  calibration only (review = highest threshold with recall >= 0.90,
  block = highest-recall threshold with precision >= 0.90), test-only
  metrics + expected loss, `model.pkl` / `metrics.json` / `model_card.md`.
  XGBoost if installed, else sklearn HistGradientBoosting — the model
  card records which ran. In this VM xgboost's CUDA wheel download was
  interrupted by sandbox egress, so HistGradientBoosting ran.
- Pipeline verified end-to-end on synthetic IEEE-schema data
  (20,000 rows, 688 fraud): ROC AUC 0.989, review threshold 0.1028
  (precision 0.386, recall 0.952), block threshold 0.6667 (precision
  0.957, recall 0.641), gated loss $2,158 vs no-gate $83,063 per 10K
  under stated assumptions. THESE ARE SYNTHETIC NUMBERS. They prove
  the pipeline, nothing about real fraud.
- A real bug caught and fixed during that run: the first review
  threshold picked was 0.0 (reviewing 2,931/3,000 test rows) because
  the code took the first, not the highest, threshold with recall
  >= 0.90. Fixed and re-verified (185 reviews).
- PayPal Orders v2 client: OAuth token with caching, create/get/
  capture, error type, env-only credentials. Verified against a mock
  transport (token fetched once across create+capture).
- Disputes v1 client: list/show/provide-evidence/accept-claim/
  escalate/appeal + sandbox-only adjudicate; evidence packet is a
  merchant-review draft, never auto-submitted.
- Gateway: propose, propose-ieee, approve/reject, execute, capture,
  disputes list/draft. Agent: propose → execute → capture choke point
  (refuses unapproved/blocked payments).
- Tests: 8/8 pass (policy, PayPal mock, gateway flow, synthetic
  training end-to-end). Demos: stdlib demo (ALLOW 5.7% / REVIEW 45.0% /
  BLOCK 78.8%) and mock-PayPal E2E demo both run.
- PayPal Agent Toolkit v1.11.0 installed in `.venv` on 2026-10-27
  (submodules `openai`, `shared`); direct REST remains the primary
  path and the toolkit is an alternative agent adapter, not yet wired
  — it also needs sandbox credentials to do anything real.

## Blocked, and only on credentials (not code)

1. Real IEEE-CIS: `scripts/download_ieee.py` exits 2 with instructions
   — no Kaggle token exists in this VM and no verified mirror was
   found. An unverified site claiming the CSV was not used.
2. Real PayPal sandbox: no PayPal developer login or sandbox app
   credentials exist here. Both Kaggle (username+key) and PayPal
   (client_id+secret) are two-secret schemes, so the single-key secure
   API card does not fit them — the owner places kaggle.json at
   ~/.kaggle/kaggle.json and sandbox creds in .env (gitignored).

## Update 2026-10-27 — real IEEE-CIS run + real sandbox run (both unlocks cleared)

- Kaggle token supplied by Jack (file, moved to ~/.kaggle/kaggle.json,
  chmod 600, uploaded copy removed); download completed exit 0
  (train_transaction.csv 652M, total data/ieee-cis 1.3G).
- First training run finished but was DISCARDED: it kept all fraud
  rows and subsampled only legitimate rows, inflating sample fraud
  rate to 13.8% vs natural 3.5% (ROC AUC 0.9328, AP 0.806 on that
  distorted sample — do not quote). Pipeline fixed to
  prevalence-preserving sampling + balanced class weight, metrics now
  include TP/FP/FN/TN.
- Corrected retrain (150,228 rows, fraud 3.54%, HistGradientBoosting):
  ROC AUC 0.9141, AP 0.5874, test n=22,535 (797 fraud).
  REVIEW 0.0137: precision 0.0998, recall 0.9009 (TP 718, FP 6,479).
  BLOCK 0.7619: precision 0.8969, recall 0.2949 (TP 235, FP 27).
  Expected loss per 10K: gated $10,674 vs no-gate $52,519 (review $2,
  false block $15, fraud loss = amount, review catches fraud).
  Honest weakness: review precision 10% means 30.8% of transactions
  go to review; block is precise but low-recall.
- Gateway POST /propose-ieee verified against a real IEEE-CIS row
  (200, calibrated, REVIEW); 8/8 tests pass after the sampling fix.
- Kaggle full-data run (2026-10-27, private CPU script kernel
  jj00nnn/gatekeeper-ieee-cis-full-training, dataset
  lnasiri007/ieeecis-fraud-detection attached): v1 failed in 6s
  (wrong assumed input path), v2 failed at 42s after loading data
  (median() on an identity column holding 'NotFound' strings),
  v3 COMPLETE. LightGBM, full 590,540 rows, 432 features (all V +
  identity join, dtype-based categorical handling). Test n=88,581
  (3,099 fraud): ROC AUC 0.9647, AP 0.7785; REVIEW 0.0279 precision
  0.2358 recall 0.9077 (13.5% reviewed); BLOCK 0.6279 precision 0.9206
  recall 0.5502; gated $7,857 vs no-gate $51,593 per 10K. Model
  downloaded to artifacts-kaggle/, verified locally (loads with
  LightGBM 4.7.0, gateway /propose-ieee 200, 8/8 tests pass), then
  promoted to artifacts/model.pkl; the 150K local model is backed up
  at artifacts/model-local-150k-backup.pkl. Kernel script lives at
  kaggle-kernel/train_full.py; local inference supports Kaggle
  artifacts via a top-level "features" order in ieee_model.py.
- Kaggle v4 run (2026-10-27, same kernel, version 4, ~65 min — the
  watcher was again killed by a runtime restart; status checked
  directly = COMPLETE, output downloaded manually): feature
  engineering (dt day/hour, frequency encoding, amount means,
  V-missing count, train-only) + 4 candidates, selection on
  calibration raw AP only. Winner B (tuned LightGBM, median-filled):
  cal AP 0.8665; test n=88,581: ROC AUC 0.9771, AP 0.8711; REVIEW
  0.0455 precision 0.4437 recall 0.9051 (7.1% reviewed); BLOCK 0.5750
  precision 0.9287 recall 0.7477; gated $6,587 vs no-gate $51,593 per
  10K. Other candidates on test: A v3-params+FE 0.9701/0.8044,
  C tuned+NaN 0.9761/0.8696, D XGBoost 0.9722/0.8309. Top features:
  dt_day, TransactionAmt, card1, card1_freq, amt_mean_card1.
  Verified locally (450 features, gateway /propose-ieee 200, 11/11
  tests incl. 3 new FE-transform tests, 2,000-row mean prob fraud
  0.736 vs legit 0.003), then promoted to artifacts/model.pkl; v3
  model backed up at artifacts/model-v3-kaggle-backup.pkl. Local
  inference now handles FE features + keep_nan (ieee_features.py).
  Current IEEE thresholds are REVIEW 0.0455 / BLOCK 0.5750.
- Kaggle v5 run (2026-10-27, same kernel, version 5, ~60 min):
  ensemble + new features (amount ratio/diff vs card mean, joint
  (card1,addr1)/(card1,ProductCD) frequency, OOF target encoding
  for card1/P_emaildomain/card4 with m=50 shrinkage) on the same
  split. Candidates on test (cal raw AP / test AUC / test AP):
  seed_avg (3 tuned LGBM seeds averaged) 0.8744 / 0.9772 / 0.8724,
  lgbm_nan 0.8724 / 0.9770 / 0.8725, xgb 0.8180 / 0.9710 / 0.8189,
  ensemble (all models averaged) 0.8695 / 0.9768 / 0.8666.
  CatBoost did NOT run: it rejects a float32 matrix with
  cat_features specified ('data is numpy array of floating point
  numerical type'); a future CatBoost attempt would need the
  categorical columns cast to int first. Selected by cal AP:
  seed_avg — REVIEW 0.0397 precision 0.4500 recall 0.9038, BLOCK
  0.3846 precision 0.9075 recall 0.7725, gated $6,678 vs no-gate
  $51,593 per 10K. PROMOTION RULE FAILED: AP gain over v4 is only
  +0.0013 (0.8724 vs 0.8711), AUC is a tie (0.9772 vs 0.9771), and
  gated loss ROSE ($6,678 vs $6,587) because the lower block
  threshold admits more false blocks (244 vs 178). v5 was therefore
  NOT promoted; v4 stays live and v5's artifact is kept only at
  artifacts-kaggle-v5/model.pkl (123MB, 3-member seed ensemble,
  verified locally: loads, 457 features, 2,000-row fraud 0.732 vs
  legit 0.003). Secondary loss-optimal thresholds on cal (review
  0.0100 / block 0.8738) give test gated $6,410 but at 12.9%
  reviewed and review precision 0.2163 — cheaper under the stated
  cost assumptions, but double the review burden, so they were
  NOT adopted either. Lesson: the ensemble did not help because
  XGBoost (0.8189) dragged the average down; v4 is close to this
  pipeline's ceiling.
- PayPal sandbox: Jack supplied full Default Application credentials
  transiently in chat (first paste was truncated at the client ID and
  was not used/stored). Real sandbox order created and verified
  CREATED with approve link (order id recorded in the chat report, not
  in memory). Credentials were NOT stored in memory, repo, or .env.
  Capture not attempted — requires payer approval via the approve link.

## Still to do

- <3 min video, public repo push (owner's word required), and Devpost
  submit by Nov 12, 2026. Devpost draft now carries the real measured
  numbers; the demo-scorer thresholds (0.35/0.75) are separate from
  the IEEE-CIS thresholds (now REVIEW 0.0455 / BLOCK 0.5750 for the
  Kaggle v4 model; 0.0279 / 0.6279 were v3, 0.0137 / 0.7619 the
  superseded local model) and must not be conflated in the
  submission.
