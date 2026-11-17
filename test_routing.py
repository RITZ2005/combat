"""Quick test script — runs 4 test cases through the router pipeline."""
import json
from graph import run_router_graph

tests = [
    ("Photosynthesis", "What is the Calvin cycle and where does it occur?"),
    ("Black Holes", "What happens at the event horizon of a black hole?"),
    ("Python", "How do you define a function in Python?"),
    ("Ambiguous", "Explain the general concept of systems and processes"),
]

for label, q in tests:
    print(f"\n{'='*60}")
    print(f"TEST: {label}")
    print(f"Q: {q}")
    print(f"{'='*60}")
    result = run_router_graph(q)
    print(json.dumps(result, indent=2))
