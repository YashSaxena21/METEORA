import unittest

from meteora import build_rationale_prompt, format_sample_shots


SAMPLE_SHOTS = [
    {
        "query": "Is there an anti-assignment clause?",
        "response": "<rationale_1>[Assignment] Search for assignment restrictions.</rationale_1>",
    }
]


class PromptTest(unittest.TestCase):
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
