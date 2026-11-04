"""Generate a search rationale and select evidence for a store return-policy question."""

from meteora_rag import HFRationaleGenerator, MeteoraReranker, SentenceTransformerEncoder


def main():
    # The question and the passages returned by a document search.
    query = "How long do I have to return an unopened item?"
    documents = [
        "Unopened items can be returned within 30 days of delivery.",
        "Orders normally ship within two business days.",
        "Customer support is available Monday to Friday.",
        "Gift cards cannot be exchanged for cash.",
        "Payments can be made by credit card or bank transfer.",
    ]

    # Show the language model the desired format using a different question.
    rationale_generator = HFRationaleGenerator(
        "Qwen/Qwen2.5-0.5B-Instruct",
        sample_shots=[
            {
                "query": "When will my order arrive?",
                "response": (
                    "<rationale_1>[Delivery time] Search for dispatch and delivery timelines. "
                    "Flag Instructions: Reject passages that contradict the stated delivery deadline.</rationale_1>"
                ),
            }
        ],
        domain="online store policies",
        num_rationales=1,
        include_flag_instructions=True,
        use_chat_template=True,
        max_new_tokens=160,
        do_sample=False,
    )

    # Use semantic embeddings to compare the generated rationale with the passages.
    reranker = MeteoraReranker(
        SentenceTransformerEncoder("sentence-transformers/all-MiniLM-L6-v2"),
        rationale_generator=rationale_generator,
    )

    # This call generates the rationale and then selects the supporting evidence.
    trace = reranker.trace(query, documents)

    for rationale in trace.rationales:
        print("Generated rationale:", rationale.text)
    for passage in trace.documents:
        print("Selected passage:", passage)


if __name__ == "__main__":
    main()
