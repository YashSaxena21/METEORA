"""Prepare preference splits from the bundled synthetic annotations."""

import argparse
import json
from pathlib import Path

from meteora_rag import (
    DPODataConfig,
    build_dpo_preference_records,
    load_preference_examples,
    load_sample_shots,
    split_dpo_records,
    write_preference_jsonl,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dpo"))
    args = parser.parse_args()
    data = Path(__file__).with_name("data")
    records = build_dpo_preference_records(
        load_preference_examples(data / "preference_examples.json"),
        config=DPODataConfig(
            sample_shots=load_sample_shots(data / "sample_shots.json"),
            domain="commercial contracts",
            skip_missing=False,
        ),
    )
    splits = split_dpo_records(records)
    for name, rows in splits.items():
        write_preference_jsonl(rows, args.output_dir / f"{name}.jsonl")
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                **{name: len(rows) for name, rows in splits.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
