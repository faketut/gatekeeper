"""Mock PayPal transport for tests/demos (httpx.MockTransport).

Mimics the sandbox surface Gatekeeper actually calls:
  POST /v1/oauth2/token -> access token (counted, to prove caching)
  POST /v2/checkout/orders -> CREATED order with an approval link
  POST /v2/checkout/orders/{id}/capture -> COMPLETED capture
  GET  /v1/customer/disputes -> one simulated dispute
  POST /v1/customer/disputes/{id}/accept-claim -> ok
Anything else -> 404. No real network, no real money.
"""

import json

import httpx


def mock_transport():
    state = {"token_calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/oauth2/token":
            state["token_calls"] += 1
            return httpx.Response(200, json={"access_token": "MOCK-TOKEN",
                                             "expires_in": 3600})
        if request.headers.get("Authorization") != "Bearer MOCK-TOKEN":
            return httpx.Response(401, json={"error": "unauthorized"})
        if path == "/v2/checkout/orders" and request.method == "POST":
            body = json.loads(request.content or b"{}")
            amount = body["purchase_units"][0]["amount"]["value"]
            return httpx.Response(201, json={
                "id": "MOCK-ORDER-1", "status": "CREATED",
                "purchase_units": [{"amount": {"currency_code": "USD", "value": amount}}],
                "links": [{"rel": "approve",
                           "href": "https://mock.paypal/approve/MOCK-ORDER-1"}]})
        if path.endswith("/capture") and request.method == "POST":
            order_id = path.split("/")[-2]
            return httpx.Response(201, json={
                "id": f"MOCK-CAPTURE-{order_id}", "status": "COMPLETED"})
        if path == "/v1/customer/disputes":
            return httpx.Response(200, json={"items": [{
                "dispute_id": "PP-D-MOCK1", "reason": "MERCHANDISE_OR_SERVICE_NOT_AS_DESCRIBED",
                "status": "WAITING_FOR_SELLER_RESPONSE"}]})
        if path.endswith("/accept-claim"):
            return httpx.Response(200, json={"links": [{"rel": "detail", "method": "GET"}]})
        return httpx.Response(404, json={"error": "not found"})

    transport = httpx.MockTransport(handler)
    transport.state = state  # type: ignore[attr-defined]
    return transport
