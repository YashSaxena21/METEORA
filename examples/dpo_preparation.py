from meteora import DPODataConfig, build_dpo_preference_records, split_dpo_records


examples = [
    {
        "query": "Is assignment restricted?",
        "document_chunks": [
            {"text": "Invoices are due within thirty days."},
            {"text": "The agreement may not be assigned without prior written consent."},
        ],
        "correct_chunks": [1],
        "effective_rationales": [
            "Look for clauses that restrict assignment or transfer without consent."
        ],
        "ineffective_rationales": [
            "Look for payment timing and invoice due dates."
        ],
    }
]

sample_shots = [
    {
        "query": "Can the agreement be assigned without consent?",
        "response": "<rationale_1>[Assignment] Search for assignment, transfer, and consent restrictions.</rationale_1>",
    }
]

records = build_dpo_preference_records(
    examples,
    config=DPODataConfig(sample_shots=sample_shots, domain="commercial contracts"),
)
splits = split_dpo_records(records)

print(records[0].to_dict())
print({name: len(split) for name, split in splits.items()})
