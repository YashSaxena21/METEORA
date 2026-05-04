import unittest

from meteora import HFRationaleGenerator


SAMPLE_SHOTS = [
    {
        "query": "Is there an anti-assignment clause?",
        "response": "<rationale_1>[Assignment] Search for assignment restrictions.</rationale_1>",
    }
]


class GenerationTest(unittest.TestCase):
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
