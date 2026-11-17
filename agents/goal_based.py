"""
Agent 3 · GOAL-BASED

It becomes goal-based when it has an EXPLICIT GOAL and LOOKS AHEAD: for every candidate
action it predicts the outcome and keeps only actions whose outcome passes the goal test.

Goal test (pass/fail):
    the action addresses the customer's main issue
    AND every policy precondition for that action holds (cited by policy ID)
    AND we do not act on an order for an unverified / inconsistent request.

If several actions pass, the goal cannot rank them, so it takes the FIRST in the menu.
If none pass, it escalates and lists why each failed.
"""
from .environment import (ACTIONS, DNR_REPORT_WINDOW_DAYS, DNR_WAIT_HOURS, GOODWILL_MAX_VALUE,
                          HIGH_VALUE, REPEAT_CLAIMS, SupportDesk)
from .model_based_reflex import ModelBasedReflexAgent


# ---------------------------------------------------------------- small helper checks
def _dnr(s):
    """A not-received claim on a parcel the courier marked delivered."""
    return s["primary_intent"] == "not_received" and s.get("delivery_stage") == "delivered"


def _fraud_trigger(s):
    """POL-DNR-03: 2+ past claims, OR a signed delivery worth more than Rs 10,000."""
    return (s.get("dnr_claims_12m", 0) >= REPEAT_CLAIMS
            or (s.get("order_value", 0) > HIGH_VALUE and s.get("signature")))


def _dnr_window(s):
    """POL-DNR-02: at least 48 h since the Delivered scan, and within 7 days."""
    return _dnr(s) and DNR_WAIT_HOURS <= s["hours_since_delivered"] <= DNR_REPORT_WINDOW_DAYS * 24


# ---------------------------------------------------------------- the precondition table
# action -> list of (policy ID, plain-English condition, check(state) -> True/False)
PRECONDITIONS = {
    "INFORM_STATUS": [
        ("POL-REF-02", "a status update answers the main issue (refund on schedule / parcel in transit)",
         lambda s: (s["primary_intent"] == "refund_delay"
                    and s.get("refund_stage") in ("in_credit_window", "initiation_due", "credited"))
                   or (s["primary_intent"] == "not_received" and s.get("delivery_stage") == "in_transit")),
    ],
    "WAIT_48H_AND_RECHECK": [
        ("POL-DNR-01", "Delivered scan is less than 48 h old",
         lambda s: _dnr(s) and s["hours_since_delivered"] < DNR_WAIT_HOURS),
    ],
    "OFFER_RESHIP_OR_REFUND": [
        ("POL-DNR-02", "not-received claim, 48 h passed, within 7 days", _dnr_window),
        ("POL-DNR-02", "no proof of delivery or GPS > 200 m", lambda s: s.get("proof") in ("none", "weak")),
        ("POL-DNR-02", f"order value ≤ Rs {GOODWILL_MAX_VALUE:,}", lambda s: s.get("order_value", 1e9) <= GOODWILL_MAX_VALUE),
        ("POL-DNR-02", "no not-received claims in last 12 months", lambda s: s.get("dnr_claims_12m", 1) == 0),
        ("POL-DNR-03", "no repeat-claim / high-value-signed trigger", lambda s: not _fraud_trigger(s)),
    ],
    "OPEN_CARRIER_INVESTIGATION": [
        ("POL-DNR-02", "not-received claim, 48 h passed, within 7 days", _dnr_window),
        ("POL-DNR-02", "no proof of delivery or GPS > 200 m", lambda s: s.get("proof") in ("none", "weak")),
        ("POL-DNR-03", "no repeat-claim / high-value-signed trigger", lambda s: not _fraud_trigger(s)),
    ],
    "ESCALATE_FRAUD_REVIEW": [
        ("POL-DNR-03", "not-received claim with ≥2 past claims or a signed delivery > Rs 10,000",
         lambda s: _dnr(s) and _fraud_trigger(s)),
    ],
    "ESCALATE_FINANCE_REFUND_DELAY": [
        ("POL-REF-01", "inspection passed and refund not initiated within 2 business days",
         lambda s: s["primary_intent"] == "refund_delay" and s.get("refund_stage") == "initiation_overdue"),
    ],
    "RAISE_PAYMENT_TRACE": [
        ("POL-REF-02", "refund initiated and credit window passed",
         lambda s: s["primary_intent"] == "refund_delay" and s.get("refund_stage") == "credit_overdue"),
    ],
    "REQUEST_CLARIFICATION": [
        ("POL-VER-01", "identity, message details or records conflict / are missing",
         lambda s: bool(s["conflicts"])),
    ],
}
# Every action except asking for clarification acts on the order -> needs a verified, consistent request
ORDER_SPECIFIC = set(PRECONDITIONS) - {"REQUEST_CLARIFICATION"}


def check_policy(action: str, s: dict) -> list:
    """List of violated conditions for this action (empty list = compliant).
    Used by Agents 3-5 AND by the evaluator in Step 12."""
    if action == "ESCALATE_TO_HUMAN":
        return []
    failed = [f"{pid}: {desc}" for pid, desc, check in PRECONDITIONS.get(action, []) if not check(s)]
    if action in ORDER_SPECIFIC and s["conflicts"]:
        failed.append("POL-VER-01: request must be verified before acting (" + "; ".join(s["conflicts"]) + ")")
    return failed


class GoalBasedAgent(ModelBasedReflexAgent):
    name = "3 · Goal-Based"
    mechanism = "Look-ahead: keep actions whose predicted outcome passes an explicit goal test"

    # ------------------------------------------------------------ widen the state with what the GOAL needs
    def build_state(self, p: dict, desk: SupportDesk) -> dict:
        s = super().build_state(p, desk)               # everything Agent 2 knows ...
        s["primary_intent"] = p["intents"][0]          # ... plus the main issue
        conflicts = []                                 # ... plus identity & consistency checks
        if not s["order_found"]:
            conflicts.append(f"order {p['order_id']} not found")
        else:
            profile = desk.get_customer_profile(s["order_customer_id"])
            s["dnr_claims_12m"] = profile.get("dnr_claims_12m", 0)
            s["lifetime_value"] = profile.get("lifetime_value", 0)
            if s["order_customer_id"] != p["customer_id"]:
                conflicts.append(f"ticket customer {p['customer_id']} ≠ order customer {s['order_customer_id']}")
            if p["claimed_amount"] and p["claimed_amount"] != s["order_value"]:
                conflicts.append(f"message says Rs {p['claimed_amount']:,} but order is Rs {s['order_value']:,}")
            if s.get("delivery_stage") == "no_scan_data" and s["order_status"] in ("shipped", "delivered"):
                conflicts.append(f"order status '{s['order_status']}' but no tracking scans on file")
            if s["primary_intent"] == "refund_delay" and s.get("refund_stage") == "no_return_on_file":
                conflicts.append("refund asked for but no return on file")
        s["conflicts"] = conflicts
        return s

    # ------------------------------------------------------------ look-ahead + goal test
    def predict(self, action: str, s: dict) -> dict:
        """'What happens if I recommend this?' -> compliant or not, and why."""
        violations = check_policy(action, s)
        return {"policy_compliant": not violations, "violations": violations}

    def goal_test(self, outcome: dict) -> bool:
        """Pass or fail. Nothing in between."""
        return outcome["policy_compliant"]

    def evaluate_candidates(self, s: dict) -> list:
        rows = []
        for action in ACTIONS:
            if action == "ESCALATE_TO_HUMAN":          # the fallback, not a candidate
                continue
            outcome = self.predict(action, s)
            rows.append({"action": action, "passes_goal": self.goal_test(outcome),
                         "why_not": "; ".join(outcome["violations"])})
        return rows

    def decide(self, ticket: dict, desk: SupportDesk):
        p = self.perceive(ticket)
        s = self.build_state(p, desk)
        rows = self.evaluate_candidates(s)
        passing = [r["action"] for r in rows if r["passes_goal"]]
        steps = [f"Main issue: {s['primary_intent']}; conflicts: {s['conflicts'] or 'none'}",
                 f"Tested {len(rows)} actions against the goal; {len(passing)} pass: {passing or '—'}"]
        if not passing:
            steps.append("No action passes → escalate with the reasons above")
            return self._decision(ticket, desk, "ESCALATE_TO_HUMAN", "No action passes the goal test",
                                  steps, s, {"candidates": rows})
        choice = passing[0]
        if len(passing) > 1:
            steps.append(f"Several actions reach the goal; a yes/no goal can't rank them, so take the first: {choice}")
        cites = sorted({pid for pid, _, _ in PRECONDITIONS[choice]})
        return self._decision(ticket, desk, choice,
                              f"{len(passing)} of {len(rows)} actions pass the goal test ({', '.join(cites)}); took the first",
                              steps, s, {"candidates": rows})