"""Verify the bundled evidence with a real Hugging Face chat model.

Requires the hf extra. Invalid model decisions are rejected by default.
"""

import argparse
import json
from pathlib import Path

from meteora_rag import MeteoraVerifier, parse_rationales


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, help="A Hugging Face chat-model ID or local checkpoint directory."
    )
    parser.add_argument("--device", help="Execution device, for example cpu, cuda, or mps.")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    args = parser.parse_args()
    try:
        import torch
        from transformers import pipeline
    except ImportError:
        parser.error('Install model support: pip install "meteora-rag[hf]"')
    device = args.device or (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    text_model = pipeline(
        "text-generation",
        model=args.model,
        device=device,
        torch_dtype=torch.float16 if device.startswith(("cuda", "mps")) else torch.float32,
    )

    def complete(prompt):
        output = text_model(
            [{"role": "user", "content": prompt}],
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            return_full_text=False,
        )[0]["generated_text"]
        if isinstance(output, list):
            return output[-1]["content"]
        return output

    data = Path(__file__).with_name("data")
    chunks = json.loads((data / "chunks.json").read_text(encoding="utf-8"))
    rationales = parse_rationales((data / "rationales.txt").read_text(encoding="utf-8"))
    verifier = MeteoraVerifier(complete)
    # One supporting and one unrelated passage demonstrate separate decisions.
    result = verifier.verify_chunks(
        "Does assignment require prior written consent?", chunks[:2], rationales
    )
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
