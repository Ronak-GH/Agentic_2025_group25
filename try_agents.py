from agents.environment import SupportDesk, load_scenarios
from agents.simple_reflex import SimpleReflexAgent
from agents.model_based_reflex import ModelBasedReflexAgent

agents = [SimpleReflexAgent(), ModelBasedReflexAgent()]
for s in load_scenarios():
    print(f"\n{s['id']}: {s['title']}")
    for agent in agents:
        d = agent.decide(s["ticket"], SupportDesk())
        ok = "✅" if d.action in s["gold"]["acceptable_actions"] else "❌"
        print(f"   {agent.name:<24} {d.action:<30} {ok}  {d.summary}")