"""Select evidence from supplied rationales without downloading model weights."""

import json
from pathlib import Path

from meteora_rag import HashingEncoder, MeteoraSelector, parse_rationales


def main():
    data = Path(__file__).with_name("data")
    chunks = json.loads((data / "chunks.json").read_text(encoding="utf-8"))
    rationales = parse_rationales((data / "rationales.txt").read_text(encoding="utf-8"))
    selector = MeteoraSelector(HashingEncoder(), expansion_window=0)
    result = selector.select(chunks, rationales)
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
