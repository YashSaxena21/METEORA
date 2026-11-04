"""Generate query-specific rationales with a Hugging Face language model.

Requires the hf extra. Weights are downloaded if they are not cached.
"""

import argparse
import json
from pathlib import Path

from meteora_rag import HFRationaleGenerator, load_sample_shots, parse_rationales


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "query", nargs="?", default="Does the agreement allow termination for non-payment?"
    )
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--device", help="Execution device, for example cpu, cuda, or mps.")
    parser.add_argument("--num-rationales", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        parser.error('Install generation support: pip install "meteora-rag[hf]"')
    device = args.device or (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    generator = HFRationaleGenerator(
        args.model,
        sample_shots=load_sample_shots(Path(__file__).with_name("data") / "sample_shots.json"),
        domain="commercial contracts",
        num_rationales=args.num_rationales,
        include_flag_instructions=True,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        torch_dtype="float16" if device.startswith(("cuda", "mps")) else "float32",
        device_map=None,
        device=device,
    )
    with torch.inference_mode():
        response = generator(args.query)
    print(
        json.dumps(
            {
                "query": args.query,
                "model": args.model,
                "rationales": [rationale.to_dict() for rationale in parse_rationales(response)],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
