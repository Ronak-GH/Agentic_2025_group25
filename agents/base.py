"""Common plumbing for the five classical agents: perceive -> decide -> Decision."""
from .environment import Decision, SupportDesk, input_fingerprint, parse_ticket


class BaseAgent:
    name = "Base"
    mechanism = ""

    def perceive(self, ticket: dict) -> dict:
        """What every agent starts from: the raw ticket fields + the shared parse."""
        return {**ticket, **parse_ticket(ticket)}

    def decide(self, ticket: dict, desk: SupportDesk) -> Decision:
        raise NotImplementedError          # each agent MUST write its own decide()

    def _decision(self, ticket, desk, action, summary, reasoning, facts=None, details=None) -> Decision:
        """Fill in the shared answer sheet, stamping the input fingerprint automatically."""
        return Decision(agent=self.name, mechanism=self.mechanism, action=action, summary=summary,
                        reasoning=reasoning, facts_used=facts or {}, details=details or {},
                        input_fingerprint=input_fingerprint(ticket, desk))