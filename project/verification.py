from z3 import *
from guardrails import Filter, Rewriter, GuardrailChain

def verify_threat_blocked(chain: GuardrailChain, threat: str):
    solver = Solver()
    has_threat = Bool(f"has_{threat}")
    solver.add(has_threat)  
    current = has_threat
    
    for g in chain.guardrails:
        if isinstance(g, Filter):
            if threat in g.blocks:
                return "SAFE", None
        elif isinstance(g, Rewriter):
            if threat in g.transformations:
                outputs = g.transformations[threat]
                if threat in outputs:
                    current = Bool(f"still_{threat}")
                else:
                    return "SAFE", None
    
    solver.add(current)
    if solver.check() == sat:
        return "VULNERABLE", solver.model()
    return "SAFE", None