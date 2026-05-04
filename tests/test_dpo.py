import json
import tempfile
import unittest
from pathlib import Path

from meteora import (
    DPODataConfig,
    DPOSplitConfig,
    DPOTrainingConfig,
    build_dpo_preference_records,
    build_dpo_prompt,
    build_inference_prompt,
    format_rationale_completion,
    split_dpo_records,
)
from meteora.dpo import _create_dpo_config, _save_trained_model, read_preference_jsonl, write_preference_jsonl


SAMPLE_SHOTS = [
    {
        "query": "Is there an anti-assignment clause?",
        "response": "<rationale_1>[Assignment] Search for assignment restrictions.</rationale_1>",
    }
]


class DPOTest(unittest.TestCase):
    def test_build_prompt_conditions_on_oracle_evidence(self):
        prompt = build_dpo_prompt(
            "Is assignment restricted?",
            sample_shots=SAMPLE_SHOTS,
            evidence="The agreement may not be assigned without consent.",
            domain="contracts",
            num_rationales=2,
        )

        self.assertIn("Is assignment restricted?", prompt)
        self.assertIn("Oracle evidence", prompt)
        self.assertIn("without consent", prompt)
        self.assertIn("Generate 2 rationales", prompt)

    def test_inference_prompt_omits_oracle_evidence(self):
        prompt = build_inference_prompt("Is assignment restricted?", sample_shots=SAMPLE_SHOTS)

        self.assertNotIn("Oracle evidence", prompt)
        self.assertIn("Query:", prompt)

    def test_format_rationale_completion_outputs_xml_blocks(self):
        completion = format_rationale_completion(
            [
                {
                    "index": 2,
                    "label": "Consent",
                    "text": "Look for prior written consent language.",
                    "flag_instructions": "Reject contradictory permission language.",
                }
            ]
        )

        self.assertIn("<rationale_2>", completion)
        self.assertIn("[Consent]", completion)
        self.assertIn("Flag Instructions:", completion)

    def test_build_records_from_effective_and_ineffective_rationales(self):
        records = build_dpo_preference_records(
            [
                {
                    "query": "Is assignment restricted?",
                    "document_chunks": [
                        {"text": "Invoices are due in thirty days."},
                        {"text": "Assignment requires prior written consent."},
                    ],
                    "correct_chunks": [1],
                    "effective_rationales": ["Look for assignment consent restrictions."],
                    "ineffective_rationales": ["Look for invoice payment timing."],
                }
            ],
            config=DPODataConfig(sample_shots=SAMPLE_SHOTS, domain="contracts"),
        )

        self.assertEqual(len(records), 1)
        self.assertIn("Assignment requires", records[0].prompt)
        self.assertIn("assignment consent", records[0].chosen)
        self.assertIn("invoice payment", records[0].rejected)

    def test_build_records_from_candidate_rationale_selection_accuracy(self):
        records = build_dpo_preference_records(
            [
                {
                    "question": "Is assignment restricted?",
                    "chunks": [{"text": "Assignment requires consent."}, {"text": "Invoice terms."}],
                    "gold_chunks": [{"index": 0}],
                    "rationale_candidates": [
                        {"text": "Search for assignment consent.", "selected_chunks": [0]},
                        {"text": "Search for invoice due dates.", "selected_chunks": [1]},
                    ],
                }
            ],
            config=DPODataConfig(sample_shots=SAMPLE_SHOTS),
        )

        self.assertEqual(len(records), 1)
        self.assertIn("assignment consent", records[0].chosen)
        self.assertIn("invoice due", records[0].rejected)

    def test_split_records_is_deterministic(self):
        examples = [
            {
                "query": f"q{i}",
                "effective_rationales": [f"good {i}"],
                "ineffective_rationales": [f"bad {i}"],
            }
            for i in range(10)
        ]
        records = build_dpo_preference_records(examples, config=DPODataConfig(sample_shots=SAMPLE_SHOTS))

        first = split_dpo_records(records, DPOSplitConfig(seed=7))
        second = split_dpo_records(records, DPOSplitConfig(seed=7))

        self.assertEqual([record.query for record in first["train"]], [record.query for record in second["train"]])
        self.assertEqual((len(first["train"]), len(first["validation"]), len(first["test"])), (8, 1, 1))

    def test_single_record_split_keeps_training_row(self):
        records = build_dpo_preference_records(
            [{"query": "q", "effective_rationales": ["good"], "ineffective_rationales": ["bad"]}],
            config=DPODataConfig(sample_shots=SAMPLE_SHOTS),
        )

        splits = split_dpo_records(records)

        self.assertEqual((len(splits["train"]), len(splits["validation"]), len(splits["test"])), (1, 0, 0))

    def test_jsonl_round_trip(self):
        records = build_dpo_preference_records(
            [{"query": "q", "effective_rationales": ["good"], "ineffective_rationales": ["bad"]}],
            config=DPODataConfig(sample_shots=SAMPLE_SHOTS),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "train.jsonl"
            write_preference_jsonl(records, path)
            loaded = read_preference_jsonl(path)

        self.assertEqual(loaded[0]["prompt"], records[0].prompt)
        json.dumps(loaded[0])

    def test_save_trained_model_writes_model_tokenizer_and_metadata(self):
        class Trainer:
            def __init__(self):
                self.saved_model_dir = None
                self.saved_state = False

            def save_model(self, output_dir):
                self.saved_model_dir = output_dir
                Path(output_dir, "model.safetensors").write_text("model", encoding="utf-8")

            def save_state(self):
                self.saved_state = True

        class Tokenizer:
            def save_pretrained(self, output_dir):
                Path(output_dir, "tokenizer.json").write_text("tokenizer", encoding="utf-8")

        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = Trainer()
            _save_trained_model(
                trainer,
                Tokenizer(),
                config=DPOTrainingConfig(
                    model_name_or_path="base-model",
                    output_dir=temp_dir,
                ),
            )
            metadata = json.loads(Path(temp_dir, "meteora_dpo_config.json").read_text())

        self.assertTrue(trainer.saved_state)
        self.assertEqual(metadata["model_name_or_path"], "base-model")
        self.assertEqual(metadata["load_for_inference"], temp_dir)

    def test_dpo_config_saves_final_model_by_default(self):
        class TrainingArgs:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        args = _create_dpo_config(
            TrainingArgs,
            DPOTrainingConfig(model_name_or_path="base-model", output_dir="out"),
            do_eval=True,
        )

        self.assertFalse(args.kwargs["load_best_model_at_end"])
        self.assertNotIn("metric_for_best_model", args.kwargs)

    def test_dpo_config_can_opt_into_best_model_loading(self):
        class TrainingArgs:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        args = _create_dpo_config(
            TrainingArgs,
            DPOTrainingConfig(
                model_name_or_path="base-model",
                output_dir="out",
                load_best_model_at_end=True,
            ),
            do_eval=True,
        )

        self.assertTrue(args.kwargs["load_best_model_at_end"])
        self.assertEqual(args.kwargs["metric_for_best_model"], "eval_loss")
        self.assertFalse(args.kwargs["greater_is_better"])


if __name__ == "__main__":
    unittest.main()
