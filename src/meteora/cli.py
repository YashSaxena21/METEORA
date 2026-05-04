from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, List, Optional

from .chunking import chunk_text
from .dpo import (
    DPODataConfig,
    DPOSplitConfig,
    DPOTrainingConfig,
    build_dpo_preference_records,
    load_preference_examples,
    load_sample_shots,
    read_preference_jsonl,
    split_dpo_records,
    train_dpo_rationale_generator,
    write_preference_jsonl,
)
from .embeddings import HashingEncoder, SentenceTransformerEncoder
from .rationales import parse_rationales
from .selector import MeteoraSelector


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="meteora", description="METEORA evidence selection")
    subparsers = parser.add_subparsers(dest="command", required=True)

    select_parser = subparsers.add_parser("select", help="Select evidence chunks from rationales")
    select_parser.add_argument("--chunks", required=True, help="JSON or JSONL chunks file")
    select_parser.add_argument("--rationales", help="Text or JSON rationales file")
    select_parser.add_argument(
        "--rationale",
        action="append",
        default=[],
        help="Inline rationale. Can be passed multiple times.",
    )
    select_parser.add_argument(
        "--query",
        help="Optional query. Used only when --use-query-as-rationale is set.",
    )
    select_parser.add_argument(
        "--use-query-as-rationale",
        action="store_true",
        help="Fallback to the query as a single rationale when no rationales are supplied.",
    )
    select_parser.add_argument(
        "--encoder-model",
        help="SentenceTransformer model name. If omitted, a dependency-free hashing encoder is used.",
    )
    select_parser.add_argument("--device", help="Device for SentenceTransformer, e.g. cpu or cuda")
    select_parser.add_argument("--expansion-window", type=int, default=1)
    select_parser.add_argument("--output", help="Write JSON result to this path")
    select_parser.add_argument("--compact", action="store_true", help="Emit compact JSON")
    select_parser.set_defaults(func=_select_command)

    chunk_parser = subparsers.add_parser("chunk", help="Split a plain text file into chunks")
    chunk_parser.add_argument("input", help="Plain text input file")
    chunk_parser.add_argument("--chunk-size", type=int, default=256)
    chunk_parser.add_argument("--overlap", type=int)
    chunk_parser.add_argument("--output", help="Write chunks as JSON to this path")
    chunk_parser.add_argument("--compact", action="store_true", help="Emit compact JSON")
    chunk_parser.set_defaults(func=_chunk_command)

    dpo_prepare_parser = subparsers.add_parser(
        "dpo-prepare",
        help="Create paper-style DPO preference JSONL splits for rationale fine-tuning",
    )
    dpo_prepare_parser.add_argument("--input", required=True, help="JSON/JSONL QA annotations")
    dpo_prepare_parser.add_argument("--sample-shots", required=True, help="JSON/JSONL/text few-shot examples")
    dpo_prepare_parser.add_argument("--output-dir", required=True, help="Directory for split JSONL files")
    dpo_prepare_parser.add_argument("--domain", default="legal, financial, scientific, or policy")
    dpo_prepare_parser.add_argument("--train-ratio", type=float, default=0.8)
    dpo_prepare_parser.add_argument("--validation-ratio", type=float, default=0.1)
    dpo_prepare_parser.add_argument("--test-ratio", type=float, default=0.1)
    dpo_prepare_parser.add_argument("--seed", type=int, default=42)
    dpo_prepare_parser.add_argument(
        "--no-condition-on-evidence",
        action="store_true",
        help="Omit oracle evidence from the DPO prompt",
    )
    dpo_prepare_parser.set_defaults(func=_dpo_prepare_command)

    dpo_train_parser = subparsers.add_parser(
        "dpo-train",
        help="Fine-tune a rationale generator with TRL DPO",
    )
    dpo_train_parser.add_argument("--train", required=True, help="Train JSONL from dpo-prepare")
    dpo_train_parser.add_argument("--validation", help="Validation JSONL from dpo-prepare")
    dpo_train_parser.add_argument("--model", required=True, help="Base model name or local path")
    dpo_train_parser.add_argument("--output-dir", required=True, help="Directory to save the tuned model")
    dpo_train_parser.add_argument("--beta", type=float, default=0.05)
    dpo_train_parser.add_argument("--epochs", type=float, default=3)
    dpo_train_parser.add_argument("--learning-rate", type=float, default=3e-5)
    dpo_train_parser.add_argument("--train-batch-size", type=int, default=1)
    dpo_train_parser.add_argument("--eval-batch-size", type=int, default=1)
    dpo_train_parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    dpo_train_parser.add_argument("--torch-dtype", help="Torch dtype, e.g. float16, bfloat16, or auto")
    dpo_train_parser.add_argument("--device-map", default="auto")
    dpo_train_parser.add_argument("--trust-remote-code", action="store_true")
    dpo_train_parser.set_defaults(func=_dpo_train_command)

    args = parser.parse_args(argv)
    return args.func(args)


def _select_command(args: argparse.Namespace) -> int:
    chunks = _load_chunks(Path(args.chunks))
    rationales = list(args.rationale)
    if args.rationales:
        rationales.extend(_load_rationales(Path(args.rationales)))
    if not rationales and args.use_query_as_rationale and args.query:
        rationales = [args.query]
    if not rationales:
        raise SystemExit("No rationales supplied. Pass --rationales, --rationale, or --use-query-as-rationale.")

    encoder = (
        SentenceTransformerEncoder(args.encoder_model, device=args.device)
        if args.encoder_model
        else HashingEncoder()
    )
    selector = MeteoraSelector(encoder, expansion_window=args.expansion_window)
    result = selector.select(chunks, rationales)
    _emit_json(result.to_dict(), args.output, compact=args.compact)
    return 0


def _chunk_command(args: argparse.Namespace) -> int:
    text = Path(args.input).read_text(encoding="utf-8")
    chunks = [chunk.to_dict() for chunk in chunk_text(text, args.chunk_size, args.overlap)]
    _emit_json(chunks, args.output, compact=args.compact)
    return 0


def _dpo_prepare_command(args: argparse.Namespace) -> int:
    examples = load_preference_examples(args.input)
    records = build_dpo_preference_records(
        examples,
        config=DPODataConfig(
            sample_shots=load_sample_shots(args.sample_shots),
            domain=args.domain,
            condition_on_evidence=not args.no_condition_on_evidence,
        ),
    )
    splits = split_dpo_records(
        records,
        config=DPOSplitConfig(
            train_ratio=args.train_ratio,
            validation_ratio=args.validation_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        ),
    )
    output_dir = Path(args.output_dir)
    for split_name, split_records in splits.items():
        write_preference_jsonl(split_records, output_dir / f"{split_name}.jsonl")
    _emit_json(
        {
            "records": len(records),
            "train": len(splits["train"]),
            "validation": len(splits["validation"]),
            "test": len(splits["test"]),
        },
        output=None,
        compact=False,
    )
    return 0


def _dpo_train_command(args: argparse.Namespace) -> int:
    train_records = read_preference_jsonl(args.train)
    validation_records = read_preference_jsonl(args.validation) if args.validation else None
    train_dpo_rationale_generator(
        train_records,
        validation_records,
        config=DPOTrainingConfig(
            model_name_or_path=args.model,
            output_dir=args.output_dir,
            beta=args.beta,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
            trust_remote_code=args.trust_remote_code,
        ),
    )
    sys.stdout.write(f"Saved fine-tuned model to {args.output_dir}\n")
    return 0


def _load_chunks(path: Path) -> List[Any]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rows.append(json.loads(line))
        return rows

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if "chunks" in payload:
            return payload["chunks"]
        if "document_chunks" in payload:
            return payload["document_chunks"]
    if isinstance(payload, list):
        return payload
    raise ValueError("Chunks file must be a JSON list, JSONL file, or dict containing 'chunks'.")


def _load_rationales(path: Path) -> List[Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        if isinstance(payload, dict):
            if "rationales" in payload:
                return payload["rationales"]
            if "response" in payload:
                return list(parse_rationales(payload["response"]))
        if isinstance(payload, list):
            return payload
    parsed = parse_rationales(text, split_plain_text=True)
    return list(parsed) if parsed else [line.strip() for line in text.splitlines() if line.strip()]


def _emit_json(payload: Any, output: Optional[str], compact: bool = False) -> None:
    indent = None if compact else 2
    text = json.dumps(payload, indent=indent)
    if output:
        Path(output).write_text(text + "\n", encoding="utf-8")
    else:
        sys.stdout.write(text + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
