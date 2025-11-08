from typing import Set, Optional, Dict, List, Union
from dataclasses import dataclass

# Text is represented as a set of properties it exhibits
Property = str  # e.g., 'jailbreak', 'profanity', 'pii', 'violence'
Text = Set[Property]


@dataclass
class Filter:
    """
    A filter blocks text if it contains certain properties.
    """
    name: str
    blocks: Set[Property]
    
    def apply(self, text: Text) -> Optional[Text]:
        if text & self.blocks:  # intersection - does text have blocked properties?
            return None  # blocked
        return text
    
    def __repr__(self):
        return f"Filter({self.name}, blocks={self.blocks})"


@dataclass
class Rewriter:
    """
    A rewriter transforms text, potentially changing its properties.
    Example: PII redactor maps 'pii' -> {'benign'}
    """
    name: str
    transformations: Dict[Property, Set[Property]]
    
    def apply(self, text: Text) -> Text:
        result = set()
        for prop in text:
            if prop in self.transformations:
                # Property gets transformed
                result.update(self.transformations[prop])
            else:
                # Property passes through unchanged
                result.add(prop)
        return result
    
    def __repr__(self):
        return f"Rewriter({self.name})"


@dataclass
class Classifier:
    name: str
    detects: Set[Property]
    
    def apply(self, text: Text) -> tuple[Text, Set[Property]]:
        detected = text & self.detects
        return text, detected
    
    def __repr__(self):
        return f"Classifier({self.name}, detects={self.detects})"


Guardrail = Union[Filter, Rewriter, Classifier]


class GuardrailChain:
    """
    A sequence of guardrails applied in order.
    """
    def __init__(self, guardrails: List[Guardrail]):
        self.guardrails = guardrails
    
    def execute(self, text: Text, verbose=False) -> Optional[Text]:
        """
        Execute the chain on input text.
        Returns None if blocked, otherwise returns final text.
        """
        current = text
        
        for i, guardrail in enumerate(self.guardrails):
            if verbose:
                print(f"Step {i}: {guardrail}")
                print(f"  Input: {current}")
            
            result = guardrail.apply(current)
            
            # Handle different return types
            if result is None:
                if verbose:
                    print(f"  BLOCKED")
                return None
            elif isinstance(result, tuple):
                # Classifier returns (text, detected)
                current, detected = result
                if verbose:
                    print(f"  Detected: {detected}")
            else:
                current = result
            
            if verbose:
                print(f"  Output: {current}")
        
        return current
    
    def __repr__(self):
        chain_str = " -> ".join(g.name for g in self.guardrails)
        return f"GuardrailChain({chain_str})"