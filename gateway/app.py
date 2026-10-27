"""FastAPI approval gateway. Requires: pip install -r requirements.txt.

Endpoints:
    POST /propose            behavioural gate (agent payments)
    POST /propose-ieee       IEEE-CIS model score for a raw feature dict
                             (503 until artifacts/model.pkl exists)
    GET  /decisions, GET /decisions/{id}
    POST /decisions/{id}/approve   human approves a REVIEW -> approved
    POST /decisions/{id}/reject    human rejects a REVIEW -> rejected
    POST /decisions/{id}/execute   create a PayPal sandbox order for an
                                   ALLOW decision or an approved REVIEW
    POST /decisions/{id}/capture   capture an executed order
    GET  /disputes                 PayPal disputes (503 without creds)
    POST /disputes/draft           draft an evidence packet (merchant
                                   review only — never auto-submitted)

Approving does not move money. Executing creates an order; capturing
is a separate explicit step, matching PayPal's order lifecycle.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from gatekeeper.disputes import draft_evidence_packet
from gatekeeper.features import Transaction
from gatekeeper.policy import decide
from gatekeeper.scoring import score_transaction
from gatekeeper.store import DecisionStore

app = FastAPI(title="Gatekeeper")
store = DecisionStore()
_ieee_model = None


def get_paypal_client():
    """Overridable factory (tests inject a mock-transport client)."""
    from gatekeeper.paypal_client import from_env

    return from_env()


def _load_ieee_model():
    global _ieee_model
    if _ieee_model is None:
        from gatekeeper.ieee_model import IEEEModel

        _ieee_model = IEEEModel.load("artifacts/model.pkl")
    return _ieee_model


class Proposal(BaseModel):
    amount_usd: float
    hour_utc: int
    merchant_category: str
    new_device: bool = False
    new_merchant: bool = False
    buyer_median_amount_usd: float = 60.0
    purchases_last_hour: int = 1
    description: str = ""


@app.post("/propose")
def propose(p: Proposal):
    tx = Transaction(**p.model_dump())
    prob, contributions = score_transaction(tx)
    decision = decide(prob, contributions)
    return store.add(p.model_dump(), decision)


@app.post("/propose-ieee")
def propose_ieee(features: dict):
    try:
        model = _load_ieee_model()
    except Exception as exc:
        raise HTTPException(503, f"IEEE model not available: {exc}. "
                                 "Run risk/train_ieee.py first.")
    return model.score(features)


@app.get("/decisions")
def list_decisions():
    return store.all()


@app.get("/decisions/{decision_id}")
def get_decision(decision_id: str):
    record = store.get(decision_id)
    if not record:
        raise HTTPException(404, "decision not found")
    return record


@app.post("/decisions/{decision_id}/approve")
def approve(decision_id: str):
    record = store.get(decision_id)
    if not record or record["verdict"] != "REVIEW":
        raise HTTPException(400, "only a REVIEW decision can be approved")
    return store.set_status(decision_id, "approved")


@app.post("/decisions/{decision_id}/reject")
def reject(decision_id: str):
    record = store.get(decision_id)
    if not record or record["verdict"] != "REVIEW":
        raise HTTPException(400, "only a REVIEW decision can be rejected")
    return store.set_status(decision_id, "rejected")


def _executable(record: dict) -> bool:
    return record["verdict"] == "ALLOW" or record.get("status") == "approved"


@app.post("/decisions/{decision_id}/execute")
def execute(decision_id: str):
    record = store.get(decision_id)
    if not record:
        raise HTTPException(404, "decision not found")
    if not _executable(record):
        raise HTTPException(400, "decision is not approved for execution")
    try:
        client = get_paypal_client()
    except Exception as exc:
        raise HTTPException(503, str(exc))
    try:
        amount = float(record["transaction"].get("amount_usd", 0))
        order = client.create_order(amount, description=record["transaction"].get("description", ""))
        record["paypal_order_id"] = order.get("id")
        record["paypal_order_status"] = order.get("status")
        record["status"] = "order_created"
        return record
    finally:
        client.close()


@app.post("/decisions/{decision_id}/capture")
def capture(decision_id: str):
    record = store.get(decision_id)
    if not record or not record.get("paypal_order_id"):
        raise HTTPException(400, "no PayPal order to capture for this decision")
    try:
        client = get_paypal_client()
    except Exception as exc:
        raise HTTPException(503, str(exc))
    try:
        result = client.capture_order(record["paypal_order_id"])
        record["paypal_capture"] = result
        record["status"] = "captured"
        return record
    finally:
        client.close()


@app.get("/disputes")
def list_disputes():
    try:
        client = get_paypal_client()
    except Exception as exc:
        raise HTTPException(503, str(exc))
    try:
        from gatekeeper.disputes import DisputesClient

        return DisputesClient(client).list_disputes()
    finally:
        client.close()


@app.post("/disputes/draft")
def draft_dispute(order: dict, decision: dict):
    """Draft only. The merchant reviews and submits separately."""
    return draft_evidence_packet(order, decision)
