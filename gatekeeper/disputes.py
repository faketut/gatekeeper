"""PayPal Disputes v1 client + the merchant evidence-packet draft.

Endpoints verified against PayPal's Disputes v1 reference:
  GET  /v1/customer/disputes
  GET  /v1/customer/disputes/{id}
  POST /v1/customer/disputes/{id}/provide-evidence
  POST /v1/customer/disputes/{id}/accept-claim
  POST /v1/customer/disputes/{id}/escalate
  POST /v1/customer/disputes/{id}/appeal
  POST /v1/customer/disputes/{id}/adjudicate       (sandbox only)
  POST /v1/customer/disputes/{id}/require-evidence (sandbox only)

The sandbox-only adjudicate / require-evidence actions are the
"simulate" tools: in sandbox they let a test chargeback be driven to
resolution without a real issuer. They must never be presented as
production actions.

draft_evidence_packet() is a template draft from order + decision
data — it is NOT an LLM-generated legal argument and must be reviewed
by the merchant before anything is submitted.
"""

from .paypal_client import PayPalClient


class DisputesClient:
    def __init__(self, client: PayPalClient):
        self.client = client

    def list_disputes(self, **params) -> dict:
        query = "&".join(f"{k}={v}" for k, v in params.items())
        path = "/v1/customer/disputes" + (f"?{query}" if query else "")
        return self.client._request("GET", path)

    def get_dispute(self, dispute_id: str) -> dict:
        return self.client._request("GET", f"/v1/customer/disputes/{dispute_id}")

    def provide_evidence(self, dispute_id: str, evidences: list) -> dict:
        # Note: real provide-evidence is multipart with file uploads;
        # this JSON form covers evidence metadata (tracking, notes).
        # File upload is a documented follow-up, not silently faked.
        return self.client._request(
            "POST", f"/v1/customer/disputes/{dispute_id}/provide-evidence",
            {"evidences": evidences})

    def accept_claim(self, dispute_id: str, note: str = "") -> dict:
        return self.client._request(
            "POST", f"/v1/customer/disputes/{dispute_id}/accept-claim",
            {"note": note} if note else {})

    def escalate(self, dispute_id: str, note: str = "") -> dict:
        return self.client._request(
            "POST", f"/v1/customer/disputes/{dispute_id}/escalate",
            {"note": note} if note else {})

    def appeal(self, dispute_id: str, evidences: list | None = None) -> dict:
        return self.client._request(
            "POST", f"/v1/customer/disputes/{dispute_id}/appeal",
            {"evidences": evidences or []})

    def adjudicate_sandbox(self, dispute_id: str) -> dict:
        """Sandbox only: drive a test dispute to resolution."""
        return self.client._request(
            "POST", f"/v1/customer/disputes/{dispute_id}/adjudicate", {})


def draft_evidence_packet(order: dict, decision: dict) -> dict:
    """Template evidence packet for merchant review (not auto-submitted)."""
    amount = ""
    try:
        amount = order["purchase_units"][0]["amount"]["value"]
    except (KeyError, IndexError, TypeError):
        amount = ""
    return {
        "order_id": order.get("id", ""),
        "dispute_reason": "MERCHANDISE_OR_SERVICE_NOT_AS_DESCRIBED",
        "gatekeeper_verdict_at_purchase": decision.get("verdict", ""),
        "gatekeeper_risk_at_purchase": decision.get("probability", ""),
        "notes": (
            f"Order {order.get('id', '')} for ${amount} was scored by "
            "Gatekeeper at purchase time. This packet is a draft for "
            "merchant review — it is not submitted until the merchant "
            "approves it."
        ),
        "evidences": [{
            "evidence_type": "PROOF_OF_FULFILLMENT",
            "notes": "Attach tracking / delivery proof before submitting.",
        }],
        "status": "draft — awaiting merchant approval",
    }
