import unittest

from meteora_rag import build_rationale_prompt, format_sample_shots

SAMPLE_SHOTS = [
    {
        "query": "Is there an anti-assignment clause?",
        "response": "<rationale_1>[Assignment] Search for assignment restrictions.</rationale_1>",
    }
]


class PromptTest(unittest.TestCase):
    def test_structured_sample_shots_preserve_flag_instructions(self):
        prompt = build_rationale_prompt(
            "Is assignment restricted?",
            sample_shots=[
                {
                    "query": "Can rights be transferred?",
                    "rationales": [
                        {
                            "text": "Search for consent.",
                            "flag_instructions": "Reject reversed permissions.",
                        }
                    ],
                }
            ],
            include_flag_instructions=True,
        )
        self.assertIn("Flag Instructions: Reject reversed permissions.</rationale_1>", prompt)
        self.assertIn("Inside each rationale block", prompt)

    def test_build_rationale_prompt_requires_sample_shots(self):
        with self.assertRaises(TypeError):
            build_rationale_prompt("Is assignment restricted?")

    def test_build_rationale_prompt_includes_sample_shots(self):
        prompt = build_rationale_prompt(
            "Is assignment restricted?",
            sample_shots=SAMPLE_SHOTS,
            domain="contracts",
            num_rationales=2,
        )

        self.assertIn("Sample shots:", prompt)
        self.assertIn("anti-assignment", prompt)
        self.assertIn("Generate 2 rationales", prompt)

    def test_format_sample_shots_rejects_empty_examples(self):
        with self.assertRaises(ValueError):
            format_sample_shots([])


if __name__ == "__main__":
    unittest.main()
