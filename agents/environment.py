"""
environment.py - the shared, READ-ONLY world every agent sees.

Every agent (classical and LLM) reads the simulated records ONLY through the
methods of SupportDesk. These methods are also the LLM's tools in Step 11.
There is deliberately NO method that can change data.
"""
import copy
import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

# Folder that holds records.json, policies.json, scenarios.json.
# __file__ = this file; .parent = agents/ ; .parent.parent = project root
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ------------------------------------------------------------------ the action menu (shared by all six agents)
# The ORDER of this dict matters later: the goal-based agent tries actions in this order.
# Nothing in the order says which action is better.
ACTIONS = {
    "INFORM_STATUS": "Tell the customer the current status and expected date; share reference numbers.",
    "WAIT_48H_AND_RECHECK": "Ask the customer to check with neighbours/security; re-check after the 48-hour window.",
    "OFFER_RESHIP_OR_REFUND": "Recommend an immediate goodwill reshipment or refund (executed by the authorised team).",
    "OPEN_CARRIER_INVESTIGATION": "Open a carrier investigation for the missing parcel (3-5 business days).",
    "ESCALATE_FRAUD_REVIEW": "Escalate to the Risk & Fraud team with proof of delivery attached.",
    "ESCALATE_FINANCE_REFUND_DELAY": "Escalate to Finance Ops as P1: refund not initiated within SLA.",
    "RAISE_PAYMENT_TRACE": "Raise a payment trace with the gateway using the UTR/ARN: refund credit overdue.",
    "REQUEST_CLARIFICATION": "Do not disclose or act; ask the customer to verify order number, registered email and invoice.",
    "ESCALATE_TO_HUMAN": "No confident recommendation: hand the ticket to a senior human agent.",
}

# ------------------------------------------------------------------ handbook numbers (each tagged with its policy ID)
DNR_WAIT_HOURS = 48            # POL-DNR-01
DNR_REPORT_WINDOW_DAYS = 7     # POL-DNR-02
GPS_TOLERANCE_M = 200          # POL-DNR-02
GOODWILL_MAX_VALUE = 5000      # POL-DNR-02
HIGH_VALUE = 10000             # POL-DNR-03
REPEAT_CLAIMS = 2              # POL-DNR-03
INITIATION_SLA_BDAYS = 2       # POL-REF-01
CREDIT_SLA_BDAYS = {"UPI": 3, "Wallet": 3, "Credit card": 7, "Debit card": 7,
                    "Net banking": 7, "COD": 7}                       # POL-REF-02


# ------------------------------------------------------------------ one shared calendar
def to_dt(s: str) -> datetime:
    """'2026-11-07T18:42' -> a datetime object Python can do maths with."""
    return datetime.fromisoformat(s)


def business_days_between(start: str, end: str) -> int:
    """Mon-Fri days AFTER start, up to and including end. Holidays ignored (limitation)."""
    d, last = to_dt(start).date(), to_dt(end).date()
    n = 0
    while d < last:
        d += timedelta(days=1)
        if d.weekday() < 5:          # Monday=0 ... Friday=4
            n += 1
    return n


def add_business_days(start: str, n: int) -> str:
    """Date that is n business days after start, e.g. 'refund expected by'."""
    d = to_dt(start).date()
    while n > 0:
        d += timedelta(days=1)
        if d.weekday() < 5:
            n -= 1
    return d.isoformat()

class SupportDesk:
    """A read-only support counter. Each agent gets its own desk: same data, own call log."""

    def __init__(self):
        self.records = json.loads((DATA_DIR / "records.json").read_text(encoding="utf-8"))
        self.policies = json.loads((DATA_DIR / "policies.json").read_text(encoding="utf-8"))
        self.call_log = []          # every look-up this desk answers is written here

    def _log(self, tool, args, result):
        """Record the call, then hand back a COPY so nobody can change the original data."""
        self.call_log.append({"tool": tool, "args": args, "result": result})
        return copy.deepcopy(result)

    # ---------------------------------------------------------------- tool 1
    def get_order_status(self, order_id: str) -> dict:
        """Owner, item, amount, payment method, status, carrier, tracking ID."""
        oid = order_id.strip().upper()                     # " ord-1001" -> "ORD-1001"
        order = self.records["orders"].get(oid)            # .get returns None if missing
        if order is None:
            result = {"found": False, "order_id": order_id, "error": "No order with this ID"}
        else:
            customer = self.records["customers"].get(order["customer_id"], {})
            result = {"found": True, "order_id": oid, **order,      # ** copies all order fields in
                      "customer_name": customer.get("name"),
                      "customer_email": customer.get("email")}
        return self._log("get_order_status", {"order_id": order_id}, result)

    # ---------------------------------------------------------------- tool 2
    def get_delivery_events(self, order_id: str) -> dict:
        """The courier's scan timeline, including proof of delivery."""
        oid = order_id.strip().upper()
        order = self.records["orders"].get(oid)
        if order is None:
            result = {"found": False, "order_id": order_id, "error": "No order with this ID"}
        else:
            events = self.records["delivery_events"].get(order["tracking_id"], [])
            result = {"found": True, "order_id": oid, "carrier": order["carrier"],
                      "tracking_id": order["tracking_id"], "events": events,
                      "note": "" if events else "No scan events available for this tracking ID"}
        return self._log("get_delivery_events", {"order_id": order_id}, result)

    # ---------------------------------------------------------------- tool 3
    def get_refund_status(self, order_id: str) -> dict:
        """The return record (inspection) and the refund record for an order."""
        oid = order_id.strip().upper()
        if oid not in self.records["orders"]:
            result = {"found": False, "order_id": order_id, "error": "No order with this ID"}
        else:
            result = {"found": True, "order_id": oid,
                      "return": self.records["returns"].get(oid),     # None if no return
                      "refund": self.records["refunds"].get(oid),     # None if no refund
                      "note": "" if oid in self.records["returns"] else "No return or refund on file for this order"}
        return self._log("get_refund_status", {"order_id": order_id}, result)

    # ---------------------------------------------------------------- tool 4
    def get_customer_profile(self, customer_id: str) -> dict:
        """Tier, lifetime value and number of not-received claims in the last 12 months."""
        cid = customer_id.strip().upper()
        customer = self.records["customers"].get(cid)
        if customer is None:
            result = {"found": False, "customer_id": customer_id, "error": "Unknown customer"}
        else:
            result = {"found": True, "customer_id": cid, **customer}
        return self._log("get_customer_profile", {"customer_id": customer_id}, result)

    # ---------------------------------------------------------------- tool 5
    def search_return_policy(self, query: str, top_k: int = 3) -> dict:
        """Keyword search over the policy handbook; returns the best-matching rules with IDs."""
        q = query.lower()
        words = set(re.findall(r"[a-z0-9]+", q))          # split the query into words
        scored = []
        for p in self.policies:
            # +2 for each of the rule's keywords that appears in the query (strong signal)
            score = sum(2 for k in p["keywords"] if k in q)
            # +1 for each longer query word that appears in the rule's title or text (weak signal)
            body = (p["title"] + " " + p["text"]).lower()
            score += sum(1 for w in words if len(w) > 3 and w in body)
            if score > 0:
                scored.append((score, p))
        scored.sort(key=lambda pair: pair[0], reverse=True)   # best score first
        result = {"query": query,
                  "results": [{"id": p["id"], "title": p["title"], "text": p["text"]}
                              for _, p in scored[:top_k]]}
        return self._log("search_return_policy", {"query": query}, result)

# ------------------------------------------------------------------ one shared reader of the customer's message
NOT_RECEIVED_CUES = ["not received", "never received", "never got", "never arrived", "did not receive",
                     "didn't receive", "not delivered", "haven't received", "not get"]
REFUND_CUES = ["refund", "money back"]


def parse_ticket(ticket: dict) -> dict:
    """Free text -> order ID, intents (main one first), and any amount the customer claims."""
    msg = ticket["message"].lower()
    order = re.search(r"ORD-\d{4}", ticket["message"].upper())
    amount = re.search(r"(?:rs\.?|₹|inr)\s*([\d,]+)", msg)       # 'rs 2,199' -> '2,199'
    intents = []
    if any(cue in msg for cue in NOT_RECEIVED_CUES):
        intents.append("not_received")
    if any(cue in msg for cue in REFUND_CUES):
        intents.append("refund_delay")
    return {"order_id": order.group(0) if order else None,
            "intents": intents or ["unknown"],
            "claimed_amount": int(amount.group(1).replace(",", "")) if amount else None}


# ------------------------------------------------------------------ one shared answer sheet
@dataclass
class Decision:
    agent: str                    # e.g. "3 · Goal-Based"
    mechanism: str                # one line: HOW this agent decides
    action: str                   # one key from ACTIONS
    summary: str                  # the key reason, one line (for the comparison table)
    reasoning: list = field(default_factory=list)    # step-by-step explanation
    facts_used: dict = field(default_factory=dict)   # what it based the decision on
    details: dict = field(default_factory=dict)      # extras: utility tables, LLM tool trace...
    input_fingerprint: str = ""                       # proof of identical input

    def to_dict(self) -> dict:
        return self.__dict__.copy()


# ------------------------------------------------------------------ the seal on the envelope
def input_fingerprint(ticket: dict, desk: SupportDesk) -> str:
    """SHA-256 of exactly what an agent received. Equal codes = identical input."""
    blob = json.dumps({"ticket": ticket, "records": desk.records, "policies": desk.policies},
                      sort_keys=True, ensure_ascii=False)        # sort_keys: same content -> same text
    return hashlib.sha256(blob.encode()).hexdigest()[:12]       # first 12 characters are plenty


def load_scenarios() -> list:
    return json.loads((DATA_DIR / "scenarios.json").read_text(encoding="utf-8"))    