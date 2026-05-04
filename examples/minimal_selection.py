from meteora import HashingEncoder, MeteoraSelector, parse_rationales


chunks = [
    "The contract may not be assigned without prior written consent.",
    "The agreement binds successors and permitted assigns.",
    "Invoices are due within thirty days after receipt.",
    "Either party may terminate after a change of control event.",
    "The agreement is governed by New York law.",
]

response = """
Query: Is assignment restricted?
<rationale_1>[Assignment restriction] Look for language requiring consent before assigning rights or obligations.</rationale_1>
<rationale_2>[Successors and assigns] Search assignment, successors, assigns, or transfer limits.</rationale_2>
"""

selector = MeteoraSelector(HashingEncoder(), expansion_window=0)
result = selector.select(chunks, parse_rationales(response))

print(result.to_dict())
