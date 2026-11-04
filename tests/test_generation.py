import unittest

from meteora_rag import HFRationaleGenerator

SAMPLE_SHOTS = [
    {
        "query": "Is there an anti-assignment clause?",
        "response": "<rationale_1>[Assignment] Search for assignment restrictions.</rationale_1>",
    }
]


class GenerationTest(unittest.TestCase):
    def test_chat_template_formats_prompt_without_duplicate_special_tokens(self):
        class Tokenizer:
            chat_template = "example template"

            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                self.messages = messages
                self.template_options = (tokenize, add_generation_prompt)
                return "formatted chat prompt"

            def __call__(self, prompt, **kwargs):
                self.prompt = prompt
                self.tokenizer_options = kwargs
                return {}

            def decode(self, output, skip_special_tokens=True):
                return "<rationale_1>Search for return deadlines.</rationale_1>"

        class Model:
            def generate(self, **kwargs):
                return [[1, 2]]

        tokenizer = Tokenizer()
        generator = HFRationaleGenerator(
            "unused",
            sample_shots=SAMPLE_SHOTS,
            model=Model(),
            tokenizer=tokenizer,
            use_chat_template=True,
        )
        generator("How long can I return an item?")
        self.assertEqual(tokenizer.messages[0]["role"], "user")
        self.assertIn("How long can I return an item?", tokenizer.messages[0]["content"])
        self.assertEqual(tokenizer.template_options, (False, True))
        self.assertEqual(tokenizer.prompt, "formatted chat prompt")
        self.assertFalse(tokenizer.tokenizer_options["add_special_tokens"])

    def test_chat_template_requires_a_compatible_tokenizer(self):
        generator = HFRationaleGenerator(
            "unused",
            sample_shots=SAMPLE_SHOTS,
            model=object(),
            tokenizer=object(),
            use_chat_template=True,
        )
        with self.assertRaisesRegex(ValueError, "chat template"):
            generator("question")

    def test_generated_response_excludes_unrelated_continuation_and_keeps_flags(self):
        response = (
            "Query: Is assignment restricted?\n"
            "<rationale_1>[Consent] Search for consent. Flag Instructions: Reject reversed permissions.</rationale_1>\n"
            "Query: What is the governing law?\n"
            "<rationale_1>[Law] Search for jurisdiction.</rationale_1>"
        )
        generator = HFRationaleGenerator(
            "unused",
            sample_shots=SAMPLE_SHOTS,
            num_rationales=4,
            generate_fn=lambda prompt: response,
        )
        result = generator("Is assignment restricted?")
        self.assertIn("Reject reversed permissions.", result)
        self.assertNotIn("jurisdiction", result)
        self.assertNotIn("Query:", result)

    def test_generated_response_respects_requested_rationale_count(self):
        generator = HFRationaleGenerator(
            "unused",
            sample_shots=SAMPLE_SHOTS,
            num_rationales=1,
            generate_fn=lambda prompt: (
                "<rationale_1>first</rationale_1><rationale_2>second</rationale_2>"
            ),
        )
        self.assertEqual(generator("question"), "<rationale_1>first</rationale_1>")

    def test_invalid_prompt_settings_are_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            HFRationaleGenerator("unused", sample_shots=[], generate_fn=lambda prompt: "")
        with self.assertRaises(ValueError):
            HFRationaleGenerator("unused", sample_shots=SAMPLE_SHOTS, max_new_tokens=0)

    def test_hf_rationale_generator_uses_selected_model_path_and_sample_shots(self):
        prompts = []

        generator = HFRationaleGenerator(
            "models/meteora-rationale-dpo",
            sample_shots=SAMPLE_SHOTS,
            domain="contracts",
            generate_fn=lambda prompt: prompts.append(prompt) or "<rationale_1>ok</rationale_1>",
        )

        response = generator("Is assignment restricted?", ["doc"])

        self.assertEqual(response, "<rationale_1>ok</rationale_1>")
        self.assertEqual(generator.config.model_name_or_path, "models/meteora-rationale-dpo")
        self.assertIn("Sample shots:", prompts[0])
        self.assertIn("anti-assignment", prompts[0])

    def test_hf_generation_moves_inputs_to_model_device(self):
        class InputIds:
            shape = (1, 3)

        class Inputs(dict):
            def __init__(self):
                super().__init__({"input_ids": InputIds()})
                self.device = None

            def to(self, device):
                self.device = device
                return self

        class Tokenizer:
            def __call__(self, prompt, return_tensors):
                self.inputs = Inputs()
                return self.inputs

            def decode(self, output, skip_special_tokens=True):
                self.output = output
                return "decoded rationale"

        class Model:
            device = "mps"

            def generate(self, **kwargs):
                self.kwargs = kwargs
                return [[1, 2, 3, 4, 5]]

        tokenizer = Tokenizer()
        generator = HFRationaleGenerator(
            "models/meteora-rationale-dpo",
            sample_shots=SAMPLE_SHOTS,
            model=Model(),
            tokenizer=tokenizer,
            do_sample=False,
        )

        self.assertEqual(generator._generate_with_hf("prompt"), "decoded rationale")
        self.assertEqual(tokenizer.inputs.device, "mps")
        self.assertEqual(tokenizer.output, [4, 5])


if __name__ == "__main__":
    unittest.main()
