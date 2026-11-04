import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from meteora_rag import __version__


class CLITest(unittest.TestCase):
    def run_cli(self, *arguments):
        return subprocess.run(
            [sys.executable, "-m", "meteora_rag", *arguments],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_version(self):
        result = self.run_cli("--version")
        self.assertEqual(result.returncode, 0)
        self.assertIn(f"meteora-rag {__version__}", result.stdout)

    def test_missing_input_has_an_actionable_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            missing = str(Path(temporary) / "missing.txt")
            result = self.run_cli("chunk", missing)
        self.assertEqual(result.returncode, 2)
        self.assertIn("missing.txt", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_chunk_command_creates_output_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "document.txt"
            output = root / "nested/chunks.json"
            source.write_text("alpha beta gamma delta", encoding="utf-8")
            result = self.run_cli(
                "chunk", str(source), "--chunk-size", "2", "--overlap", "0", "--output", str(output)
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                [row["text"] for row in json.loads(output.read_text())],
                ["alpha beta", "gamma delta"],
            )

    def test_preparation_rejects_input_without_preferences(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            examples = root / "examples.json"
            shots = root / "shots.json"
            examples.write_text('[{"query": "question only"}]', encoding="utf-8")
            shots.write_text(
                '[{"query": "example", "response": "Search for evidence."}]', encoding="utf-8"
            )
            result = self.run_cli(
                "dpo-prepare",
                "--input",
                str(examples),
                "--sample-shots",
                str(shots),
                "--output-dir",
                str(root / "out"),
            )
        self.assertEqual(result.returncode, 2)
        self.assertIn("No preference pairs found", result.stderr)


if __name__ == "__main__":
    unittest.main()
