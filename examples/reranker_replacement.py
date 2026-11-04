"""Use supplied rationales to filter original document objects in a RAG pipeline."""

import json
from pathlib import Path

from meteora_rag import HashingEncoder, MeteoraReranker


def main():
    data = Path(__file__).with_name("data")
    documents = json.loads((data / "chunks.json").read_text(encoding="utf-8"))
    rationales = (data / "rationales.txt").read_text(encoding="utf-8")
    reranker = MeteoraReranker(HashingEncoder())
    selected = reranker.filter(
        "Is assignment restricted?", documents, rationales=rationales, order="document"
    )
    print("Selected document IDs:", [document["id"] for document in selected])
    for document in selected:
        print(document["text"])


if __name__ == "__main__":
    main()
