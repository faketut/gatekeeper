"""PayPal sandbox client — real Orders v2 wiring (not a stub anymore).

Flow (PayPal docs, sandbox):
  1. OAuth2 client-credentials token:
     POST {base}/v1/oauth2/token  (Basic auth client_id:client_secret,
     body grant_type=client_credentials)
  2. Create order: POST {base}/v2/checkout/orders, intent CAPTURE
  3. Capture order: POST {base}/v2/checkout/orders/{id}/capture
     (in a real checkout the payer approves the order first via the
     approval link; a card payment_source can capture in one step —
     see gateway docs. The gateway's approve step creates the order;
     capture is a separate, explicit step.)

Credentials come from the environment (sandbox only, never production):
    PAYPAL_CLIENT_ID, PAYPAL_CLIENT_SECRET,
    PAYPAL_BASE_URL (default https://api-m.sandbox.paypal.com)

For tests, pass a custom base_url (a local mock server) or an httpx
transport — no real credentials are needed and no real money moves.
"""

import os
import time

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None


class NotConfigured(RuntimeError):
    pass


class PayPalAPIError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"PayPal API error {status}: {body[:500]}")
        self.status = status
        self.body = body


class PayPalClient:
    def __init__(self, client_id: str, client_secret: str,
                 base_url: str = "https://api-m.sandbox.paypal.com",
                 transport=None):
        if httpx is None:
            raise ImportError("httpx is required: pip install -r requirements.txt")
        self.client_id = client_id
        self.client_secret = client_secret
        self.base_url = base_url.rstrip("/")
        self._http = httpx.Client(transport=transport, timeout=30.0)
        self._token = None
        self._token_expires_at = 0.0

    # -- auth ---------------------------------------------------------
    def _access_token(self) -> str:
        if self._token and time.time() < self._token_expires_at:
            return self._token
        resp = self._http.post(
            f"{self.base_url}/v1/oauth2/token",
            auth=(self.client_id, self.client_secret),
            data={"grant_type": "client_credentials"},
            headers={"Accept": "application/json"},
        )
        if resp.status_code != 200:
            raise PayPalAPIError(resp.status_code, resp.text)
        data = resp.json()
        self._token = data["access_token"]
        # Refresh a minute early so a token never expires mid-call.
        self._token_expires_at = time.time() + int(data.get("expires_in", 3600)) - 60
        return self._token

    def _request(self, method: str, path: str, json_body: dict | None = None) -> dict:
        token = self._access_token()
        resp = self._http.request(
            method, f"{self.base_url}{path}",
            headers={"Authorization": f"Bearer {token}",
                     "Content-Type": "application/json"},
            json=json_body,
        )
        if resp.status_code >= 400:
            raise PayPalAPIError(resp.status_code, resp.text)
        return resp.json() if resp.content else {}

    # -- orders -------------------------------------------------------
    def create_order(self, amount_usd: float, currency: str = "USD",
                     intent: str = "CAPTURE", description: str = "") -> dict:
        body = {
            "intent": intent,
            "purchase_units": [{
                "amount": {"currency_code": currency,
                           "value": f"{amount_usd:.2f}"},
                "description": description or "Gatekeeper-approved agent purchase",
            }],
        }
        return self._request("POST", "/v2/checkout/orders", body)

    def get_order(self, order_id: str) -> dict:
        return self._request("GET", f"/v2/checkout/orders/{order_id}")

    def capture_order(self, order_id: str) -> dict:
        return self._request("POST", f"/v2/checkout/orders/{order_id}/capture", {})

    def close(self) -> None:
        self._http.close()


def from_env(transport=None) -> PayPalClient:
    """Build a client from the environment, or raise NotConfigured.

    This is the only place credentials are read. Nothing here prints,
    logs, or stores them.
    """
    client_id = os.environ.get("PAYPAL_CLIENT_ID", "")
    secret = os.environ.get("PAYPAL_CLIENT_SECRET", "")
    base = os.environ.get("PAYPAL_BASE_URL", "https://api-m.sandbox.paypal.com")
    if not client_id or not secret:
        raise NotConfigured(
            "PayPal sandbox credentials not set. Create a sandbox app at "
            "developer.paypal.com, then set PAYPAL_CLIENT_ID and "
            "PAYPAL_CLIENT_SECRET in .env (sandbox only)."
        )
    return PayPalClient(client_id, secret, base, transport=transport)


# Backwards-compatible module functions (used by the original scaffold).
def create_order(amount_usd: float, currency: str = "USD") -> dict:
    client = from_env()
    try:
        return client.create_order(amount_usd, currency)
    finally:
        client.close()


def capture_order(order_id: str) -> dict:
    client = from_env()
    try:
        return client.capture_order(order_id)
    finally:
        client.close()
