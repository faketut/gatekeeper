# Gatekeeper — risk-scored payments for AI agents

PayPal AI Hackathon entry (due Nov 12, 2026). Scaffolded Oct 26, 2026; local only, not pushed.

## Thesis

The blocker for agentic commerce is trust: nobody wants an AI agent with a
wallet and no brakes. Gatekeeper sits **between agent intent and payment
execution**. An agent proposes a payment; a risk model scores it; the payment
is auto-approved, routed to a human for approval, or blocked — before any
money moves.

PayPal is load-bearing, not decorative: the demo path is PayPal sandbox
Orders v2 (create / capture / hold), with Disputes v1 simulate endpoints for
the chargeback aftermath. The risk model is trained on IEEE-CIS Fraud
Detection (590K real e-commerce transactions, 3.5% fraud).

## Architecture

```
shopping agent ──proposes──> Gatekeeper ──ALLOW──> PayPal Orders v2 (sandbox)
                   │              ├─REVIEW──> human approval queue (ui/)
                   │              └─BLOCK───> no order created
                   │
              risk model (risk/): IEEE-CIS + XGBoost, calibrated
              probabilities, thresholds set on precision-recall,
              not on vibes
```

| Component | Path | Status |
|---|---|---|
| Risk features + scorer (stdlib, runs anywhere) | `gatekeeper/` | Done, demo-verified |
| Approval policy ALLOW / REVIEW / BLOCK | `gatekeeper/policy.py` | Done; demo thresholds 0.35 / 0.75 |
| Training pipeline (IEEE-CIS, gradient boosting) | `risk/train_ieee.py` | Implemented + verified on synthetic IEEE-schema data; real IEEE-CIS run blocked on Kaggle credentials (see below) |
| IEEE model inference | `gatekeeper/ieee_model.py` | Done; thresholds chosen on calibration split only |
| PayPal sandbox client (Orders v2) | `gatekeeper/paypal_client.py` | Implemented (OAuth token caching, create/get/capture); verified against a mock transport; real sandbox run blocked on sandbox app credentials |
| Disputes v1 client + evidence draft | `gatekeeper/disputes.py` | Implemented (list/show/evidence/accept/escalate/appeal, sandbox-only adjudicate); draft is merchant-review only, never auto-submitted |
| Approval API + execute/capture | `gateway/app.py` | Done, gateway-flow tests pass |
| Approval UI | `ui/index.html` | Done (approve/reject/order/capture) |
| Honest evaluation | `eval/evaluate.py` | Done; training also writes `metrics.json` + `model_card.md` |
| Demo (no credentials, no installs) | `demo/run_demo.py` | 3 scenarios, verified |
| End-to-end demo (mock PayPal) | `demo/e2e_demo.py` | Done with venv deps: propose → ALLOW → order → capture → dispute draft |

## Quickstart — demo, no installs, no credentials

```bash
cd gatekeeper
python3 demo/run_demo.py
python3 -m unittest discover -s tests -v
```

Expected demo output: a normal daytime purchase is ALLOWed, a sudden
$2,000 electronics purchase at 3am from a new device is REVIEWed (human
approval, not auto-blocked), and an extreme $12,000 3am burst from a new
device is BLOCKed.

## Full stack

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in PayPal SANDBOX credentials only — never production
uvicorn gateway.app:app --reload
# open ui/index.html, or: curl localhost:8000/decisions
```

## Evaluation (the differentiator)

Do not report accuracy on a 3.5%-fraud dataset — it is meaningless there.
Report, at the operating threshold:

- precision / recall / F1 for the fraud class
- precision-recall curve, and the threshold chosen from it
- expected loss: `fraud_loss_avoided - review_cost - false_block_cost`

`eval/evaluate.py` computes all three from a predictions file. The numbers
in `docs/devpost-draft.md` stay blank until they are measured on real
IEEE-CIS holdout data — no invented metrics.

## Thresholds

Defaults in `gatekeeper/policy.py`: REVIEW at p >= 0.35, BLOCK at
p >= 0.75. These are starting points from the demo scorer, not tuned
values. Retune them on the IEEE-CIS precision-recall curve before the
video, and record the tuned values here.

## The two things only you can unlock (everything else is done)

1. **Real IEEE-CIS data.** No verified public mirror exists; an
   unverified third-party site claiming the file was deliberately not
   used. With a Kaggle token, one command finishes the job:
   ```bash
   pip install kaggle
   # accept rules at kaggle.com/c/ieee-fraud-detection/rules, then:
   cp kaggle.json ~/.kaggle/kaggle.json && chmod 600 ~/.kaggle/kaggle.json
   python3 scripts/download_ieee.py --out data/ieee-cis
   .venv/bin/python risk/train_ieee.py --data-dir data/ieee-cis --out-dir artifacts
   ```
   The pipeline is already proven on synthetic IEEE-schema data
   (20K rows, ROC AUC 0.989 synthetic — that number is synthetic and
   must never be quoted as an IEEE-CIS result; after the fix to the
   review-threshold bug, gated loss was $2,158 vs $83,063 no-gate per
   10K on synthetic test data, under the stated $2 review / $15 false
   block assumptions).
2. **Real PayPal sandbox.** Create a sandbox app at
   developer.paypal.com, put its client ID + secret in `.env`
   (sandbox only, gitignored), then the same gateway/agent flow runs
   against the real sandbox — the code path is already verified
   against a mock transport, and `demo/e2e_demo.py` shows the flow.

## Measured on real IEEE-CIS (2026-10-27, Kaggle v4 run)

Full 590,540 rows (fraud 3.50%, no subsampling), 70/15/15 split,
LightGBM tuned + feature engineering, trained on Kaggle (private
CPU kernel, dataset attached, 450 features: all V1-V339 + identity
left-joined + time/frequency/amount-mean features fitted on train
only). Model selected on calibration AP only, never on test.
Test n=88,581 (3,099 fraud): ROC AUC 0.9771, average precision
0.8711. REVIEW 0.0455: precision 0.4437, recall 0.9051 — only 7.1%
of transactions reviewed. BLOCK 0.5750: precision 0.9287, recall
0.7477. Expected loss per 10K: gated $6,587 vs no-gate $51,593
(review $2, false block $15, fraud loss = amount).
Full provenance, counts, and caveats: `artifacts/model_card-kaggle-v4.md`,
`docs/devpost-draft.md`. The Kaggle v3 model (ROC AUC 0.9647, AP
0.7785, no feature engineering) is superseded; it is kept at
`artifacts/model-v3-kaggle-backup.pkl`. The earlier local 150K run
(ROC AUC 0.9141, AP 0.5874) is also superseded; its model is kept
at `artifacts/model-local-150k-backup.pkl`. A first run with a
13.8% fraud-enriched sample was discarded — do not quote it.
Real sandbox order also verified CREATED (capture needs payer
approval via the approve link).

## What is NOT done yet

- [ ] <3 min video, Devpost submission (public repo + demo URL + video)
- [ ] Push to GitHub — local only until explicitly approved

## Prize fit

Best Use of PayPal+AI (payment rails + risk model are inseparable),
Most Impactful (the trust problem of agentic commerce), Most Creative.
