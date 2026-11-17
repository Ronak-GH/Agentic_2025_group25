"""
Agent 2 · MODEL-BASED REFLEX

It becomes model-based when it keeps an INTERNAL STATE about what it cannot see in a
single status field, and builds that state with a MODEL OF HOW THE WORLD WORKS:
  * delivery model: a 'Delivered' scan can come up to 48 h before hand-over;
    proof is strong only with photo/signature AND GPS near the address
  * refund model: inspection passed -> initiated (within 2 business days)
    -> credited (within the payment method's window)
It is still a REFLEX agent: IF -> THEN rules, but the rules read the inferred state.
It does NOT look at who is asking or at the customer's history.
"""
from .base import BaseAgent
from .environment import (CREDIT_SLA_BDAYS, DNR_WAIT_HOURS, GPS_TOLERANCE_M, INITIATION_SLA_BDAYS,
                          SupportDesk, add_business_days, business_days_between, to_dt)


class ModelBasedReflexAgent(BaseAgent):
    name = "2 · Model-Based Reflex"
    mechanism = "Rules over an internal state inferred from the event timeline"

    def __init__(self):
        self.memory = {}          # >>> BECOMES MODEL-BASED: state that persists across tickets

    # ------------------------------------------------------------ build the internal state
    def build_state(self, p: dict, desk: SupportDesk) -> dict:
        now = p["received_at"]
        s = {"intents": p["intents"], "order_id": p["order_id"], "now": now}

        order = desk.get_order_status(p["order_id"]) if p["order_id"] else {"found": False}
        s["order_found"] = order["found"]
        if not order["found"]:
            return s
        s.update(order_status=order["status"], order_value=order["amount"],
                 payment_method=order["payment_method"], carrier=order["carrier"],
                 order_customer_id=order["customer_id"], item=order["item"])

        # --- delivery model: where is the parcel REALLY, and how good is the proof?
        events = desk.get_delivery_events(p["order_id"])["events"]
        delivered = next((e for e in events if e["event"] == "Delivered"), None)   # first 'Delivered' scan, or None
        if not events:
            s["delivery_stage"] = "no_scan_data"
        elif delivered:
            hours = (to_dt(now) - to_dt(delivered["time"])).total_seconds() / 3600
            proof = delivered.get("proof", {})
            has_pod = proof.get("photo") or proof.get("signature")
            gps_ok = proof.get("gps_distance_m", 9999) <= GPS_TOLERANCE_M
            s.update(delivery_stage="delivered", delivered_at=delivered["time"],
                     hours_since_delivered=round(hours, 1),
                     proof="strong" if (has_pod and gps_ok) else "weak" if (has_pod or gps_ok) else "none",
                     signature=bool(proof.get("signature")), gps_distance_m=proof.get("gps_distance_m"))
        else:
            s["delivery_stage"] = "in_transit"

        # --- refund model: which stage of the lifecycle are we really in?
        rr = desk.get_refund_status(p["order_id"])
        ret, ref = rr.get("return"), rr.get("refund")
        if not ret:
            s["refund_stage"] = "no_return_on_file"
        elif ret["inspection_status"] != "passed":
            s["refund_stage"] = f"inspection_{ret['inspection_status']}"
        else:
            s["inspection_date"] = ret["inspection_date"]
            s["bdays_since_inspection"] = business_days_between(ret["inspection_date"], now)
            if not ref or ref["status"] == "not_initiated":
                s["refund_stage"] = ("initiation_overdue" if s["bdays_since_inspection"] > INITIATION_SLA_BDAYS
                                     else "initiation_due")
                s["initiate_by"] = add_business_days(ret["inspection_date"], INITIATION_SLA_BDAYS)
            elif ref["status"] == "completed":
                s["refund_stage"] = "credited"
            else:                                            # initiated / processing
                sla = CREDIT_SLA_BDAYS.get(ref["method"], 7)
                s["bdays_since_initiation"] = business_days_between(ref["initiated_date"], now)
                s["credit_sla_bdays"] = sla
                s["expected_credit_by"] = add_business_days(ref["initiated_date"], sla)
                s["refund_reference"] = ref.get("reference")
                s["refund_stage"] = "credit_overdue" if s["bdays_since_initiation"] > sla else "in_credit_window"
        return s

    # ------------------------------------------------------------ reflex rules over the STATE
    # (intent, condition on state, action, why) - first match wins
    RULES = [
        ("not_received", lambda s: s.get("delivery_stage") == "delivered" and s["hours_since_delivered"] < DNR_WAIT_HOURS,
         "WAIT_48H_AND_RECHECK", "Delivered scan is < 48 h old: carriers can scan early"),
        ("not_received", lambda s: s.get("delivery_stage") == "delivered" and s["proof"] in ("none", "weak"),
         "OPEN_CARRIER_INVESTIGATION", "Delivered > 48 h ago with weak/no proof of delivery"),
        ("not_received", lambda s: s.get("delivery_stage") == "delivered" and s["proof"] == "strong",
         "INFORM_STATUS", "Strong proof of delivery: share it with the customer"),
        ("not_received", lambda s: s.get("delivery_stage") == "in_transit",
         "INFORM_STATUS", "Parcel still in transit"),
        ("refund_delay", lambda s: s.get("refund_stage") in ("in_credit_window", "initiation_due", "credited"),
         "INFORM_STATUS", "Refund is on schedule"),
        ("refund_delay", lambda s: s.get("refund_stage") == "initiation_overdue",
         "ESCALATE_FINANCE_REFUND_DELAY", "Refund not initiated within 2 business days of inspection"),
        ("refund_delay", lambda s: s.get("refund_stage") == "credit_overdue",
         "RAISE_PAYMENT_TRACE", "Refund initiated but credit window has passed"),
    ]

    def decide(self, ticket: dict, desk: SupportDesk):
        p = self.perceive(ticket)
        s = self.build_state(p, desk)
        oid = p["order_id"] or "?"
        self.memory[oid] = self.memory.get(oid, 0) + 1          # update the persistent memory
        steps = ["Inferred state: " + ", ".join(f"{k}={v}" for k, v in s.items()
                                                 if k not in ("intents", "now", "order_id"))]
        if self.memory[oid] > 1:
            steps.append(f"Memory: contact #{self.memory[oid]} about {oid} this session.")
        if not s["order_found"]:
            return self._decision(ticket, desk, "ESCALATE_TO_HUMAN", "Order not found", steps, s)

        for intent in p["intents"]:                              # main intent first
            for rule_intent, condition, action, why in self.RULES:
                if rule_intent == intent and condition(s):
                    steps.append(f"Rule fired for '{intent}': {why} → {action}")
                    steps.append("Not checked (outside its model): who is asking, claim history, value limits.")
                    return self._decision(ticket, desk, action, why, steps, s)
            steps.append(f"No rule matched '{intent}' (delivery_stage={s.get('delivery_stage')}, "
                         f"refund_stage={s.get('refund_stage')})")
        return self._decision(ticket, desk, "ESCALATE_TO_HUMAN", "State matched no rule", steps, s)