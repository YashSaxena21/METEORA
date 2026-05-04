from meteora import HashingEncoder, MeteoraReranker


documents = [
    {"text": "The agreement may not be assigned without prior written consent.", "id": "a"},
    {"text": "Invoices are due within thirty days after receipt.", "id": "b"},
    {"text": "The agreement binds successors and permitted assigns.", "id": "c"},
]


def rationale_generator(query, docs):
    return """
<rationale_1>[Consent restriction] Search assigned written consent.</rationale_1>
<rationale_2>[Successors and assigns] Search successors permitted assigns.</rationale_2>
"""


reranker = MeteoraReranker(
    HashingEncoder(),
    rationale_generator=rationale_generator,
)

results = reranker.rerank(
    "Is assignment restricted?",
    documents,
    order="document",
)

for result in results:
    print(result.rank, result.index, round(result.score, 3), result.document["id"])
