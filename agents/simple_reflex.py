"""
Agent 1 · SIMPLE REFLEX

It is a simple reflex agent because it maps the CURRENT percept straight to an action
through fixed IF -> THEN rules. No memory, no sense of time, no goal, no look-ahead.

What it sees: the customer's intent, order.status, refund.status. Nothing else -
no dates, no proof of delivery, no customer history, no ownership check.
"""
from .base import BaseAgent
from .environment import SupportDesk


class SimpleReflexAgent(BaseAgent):
    name = "1 · Simple Reflex"
    mechanism = "Condition → action rules on the current percept"

    # (description, condition, action) - checked top to bottom, first match wins
    RULES = [
        ("intent=not_received AND order=delivered",
         lambda p: "not_received" in p["intents"] and p["order_status"] == "delivered",
         "OPEN_CARRIER_INVESTIGATION"),
        ("intent=refund_delay AND order=returned AND refund=not_initiated",
         lambda p: "refund_delay" in p["intents"] and p["order_status"] == "returned"
                   and p["refund_status"] == "not_initiated",
         "ESCALATE_FINANCE_REFUND_DELAY"),
        ("intent=refund_delay AND order=returned AND refund in (processing, completed)",
         lambda p: "refund_delay" in p["intents"] and p["order_status"] == "returned"
                   and p["refund_status"] in ("processing", "completed"),
         "INFORM_STATUS"),
    ]

    def decide(self, ticket: dict, desk: SupportDesk):
        # 1. PERCEIVE: the shared parse + one glance at two status fields
        p = self.perceive(ticket)
        order = desk.get_order_status(p["order_id"]) if p["order_id"] else {}
        refund = desk.get_refund_status(p["order_id"]) if p["order_id"] else {}
        p["order_status"] = order.get("status", "unknown")
        p["refund_status"] = (refund.get("refund") or {}).get("status", "none")

        facts = {"intents": p["intents"], "order.status": p["order_status"],
                 "refund.status": p["refund_status"]}
        steps = [f"Percept: {facts}"]

        # 2. MATCH a rule -> ACT
        for description, condition, action in self.RULES:
            if condition(p):
                steps.append(f"Rule fired: IF {description} THEN {action}")
                steps.append("Not checked: dates, proof of delivery, customer history, who owns the order.")
                return self._decision(ticket, desk, action, f"Rule fired: IF {description}", steps, facts)

        # 3. FALLBACK when nothing matches
        steps.append("No rule matched → fallback")
        return self._decision(ticket, desk, "ESCALATE_TO_HUMAN", "No rule matched the percept", steps, facts)