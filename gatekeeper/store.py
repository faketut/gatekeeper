"""In-memory decision store + JSONL audit log.

Deliberately simple for the hackathon: the gateway keeps decisions in
memory and appends every one to decisions.jsonl, so a held payment can
be approved later and the demo has an audit trail. A production version
would use a real database; this one must not be presented as one.
"""

import json
import os
import threading
import uuid

LOG_PATH = os.environ.get("GATEKEEPER_LOG", "decisions.jsonl")


class DecisionStore:
    def __init__(self, log_path: str = LOG_PATH):
        self._lock = threading.Lock()
        self._decisions: dict = {}
        self.log_path = log_path

    def add(self, tx_dict: dict, decision) -> dict:
        record = {
            "id": uuid.uuid4().hex[:12],
            "transaction": tx_dict,
            "verdict": decision.verdict,
            "probability": decision.probability,
            "reasons": decision.reasons,
            "contributions": decision.contributions,
            "status": "executed" if decision.verdict == "ALLOW" else (
                "pending" if decision.verdict == "REVIEW" else "blocked"
            ),
        }
        with self._lock:
            self._decisions[record["id"]] = record
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        return record

    def get(self, decision_id: str) -> dict | None:
        return self._decisions.get(decision_id)

    def all(self) -> list:
        return list(self._decisions.values())

    def set_status(self, decision_id: str, status: str) -> dict | None:
        with self._lock:
            record = self._decisions.get(decision_id)
            if record:
                record["status"] = status
        return record
