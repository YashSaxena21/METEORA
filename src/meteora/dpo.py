from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

from .prompts import SampleShot, format_sample_shots
from .rationales import coerce_rationales
from .types import RationaleLike


PREFERRED_RATIONALE_KEYS = (
    "effective_rationales",
    "preferred_rationales",
    "chosen_rationales",
    "positive_rationales",
    "winning_rationales",
)
REJECTED_RATIONALE_KEYS = (
    "ineffective_rationales",
    "rejected_rationales",
    "dispreferred_rationales",
    "negative_rationales",
    "losing_rationales",
)
QUERY_KEYS = ("query", "question", "input", "prompt")
EVIDENCE_KEYS = ("evidence", "oracle_evidence", "gold_evidence", "context", "answer_evidence")
IRRELEVANT_EVIDENCE_KEYS = ("irrelevant_evidence", "negative_evidence", "rejected_evidence")


@dataclass(frozen=True)
class DPODataConfig:
    """Preference-data settings for the METEORA DPO rationale generator."""

    sample_shots: Sequence[SampleShot]
    domain: str = "legal, financial, scientific, or policy"
    condition_on_evidence: bool = True
    include_flag_instructions: bool = True
    prompt_num_rationales: Optional[int] = None
    skip_missing: bool = True


@dataclass(frozen=True)
class DPOSplitConfig:
    """Deterministic split settings matching the paper's 80/10/10 setup by default."""

    train_ratio: float = 0.8
    validation_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42
    shuffle: bool = True


@dataclass(frozen=True)
class DPOTrainingConfig:
    """Paper-aligned defaults for training the METEORA rationale generator with DPO."""

    model_name_or_path: str
    output_dir: str
    beta: float = 0.05
    num_train_epochs: float = 3
    learning_rate: float = 3e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 2
    logging_steps: int = 10
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    save_total_limit: int = 2
    load_best_model_at_end: bool = False
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    remove_unused_columns: bool = False
    torch_dtype: Optional[str] = None
    device_map: Optional[Union[str, Mapping[str, Any]]] = "auto"
    trust_remote_code: bool = False
    max_prompt_length: Optional[int] = None
    max_length: Optional[int] = None
    peft_config: Any = None


@dataclass(frozen=True)
class DPOPreferenceRecord:
    """One DPO preference row in TRL's prompt/chosen/rejected format."""

    prompt: str
    chosen: str
    rejected: str
    query: str
    evidence: Optional[str] = None
    rejected_evidence: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_metadata: bool = True) -> Dict[str, Any]:
        payload = {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
        }
        if include_metadata:
            payload["query"] = self.query
            if self.evidence is not None:
                payload["evidence"] = self.evidence
            if self.rejected_evidence is not None:
                payload["rejected_evidence"] = self.rejected_evidence
            if self.metadata:
                payload["metadata"] = dict(self.metadata)
        return payload


def build_dpo_prompt(
    query: str,
    *,
    sample_shots: Sequence[SampleShot],
    evidence: Optional[str] = None,
    domain: str = "legal, financial, scientific, or policy",
    num_rationales: Optional[int] = None,
    include_flag_instructions: bool = True,
) -> str:
    """Build the query/evidence-conditioned prompt used during DPO training."""

    rationale_count = (
        f" Generate {num_rationales} rationales." if num_rationales is not None else ""
    )
    flag_line = (
        "\n- Include `Flag Instructions:` when a rationale implies a consistency or poisoning check."
        if include_flag_instructions
        else ""
    )
    evidence_block = ""
    if evidence:
        evidence_block = f"""

Oracle evidence available during training:
{evidence}
"""
    sample_block = format_sample_shots(sample_shots)

    return f"""You generate METEORA rationales for rank-free RAG evidence selection.
The rationales should describe semantic strategies that help recover evidence relevant to the query.
During training, use the oracle evidence to learn what good query-evidence alignment looks like.
At inference time, rationales will be generated from the query alone.

Domain: {domain}

Format requirements:
- Return XML-style blocks: <rationale_1>...</rationale_1>, <rationale_2>...</rationale_2>, etc.
- Start each block with a short label in square brackets.
- Do not answer the query directly.{rationale_count}{flag_line}

Sample shots:
{sample_block}

Query:
{query}{evidence_block}

Rationales:
"""


def build_inference_prompt(
    query: str,
    *,
    sample_shots: Sequence[SampleShot],
    domain: str = "legal, financial, scientific, or policy",
    num_rationales: int = 8,
    include_flag_instructions: bool = True,
) -> str:
    """Build the query-only rationale prompt used after DPO fine-tuning."""

    return build_dpo_prompt(
        query=query,
        sample_shots=sample_shots,
        evidence=None,
        domain=domain,
        num_rationales=num_rationales,
        include_flag_instructions=include_flag_instructions,
    )


def format_rationale_completion(
    rationales: Union[Iterable[RationaleLike], str],
    *,
    include_flag_instructions: bool = True,
) -> str:
    """Normalize rationale-like objects into XML blocks suitable for DPO completions."""

    normalized = tuple(coerce_rationales(rationales))
    blocks = []
    for fallback_index, rationale in enumerate(normalized, start=1):
        index = rationale.index or fallback_index
        label = f"[{rationale.label}]" if rationale.label else f"[Rationale {index}]"
        flags = ""
        if include_flag_instructions and rationale.flag_instructions:
            flags = f" Flag Instructions: {rationale.flag_instructions}"
        blocks.append(
            f"<rationale_{index}>{label} {rationale.text}{flags}</rationale_{index}>"
        )
    return "\n".join(blocks)


def build_dpo_preference_records(
    examples: Iterable[Mapping[str, Any]],
    config: DPODataConfig,
) -> Sequence[DPOPreferenceRecord]:
    """Construct DPO preference records from QA annotations or pre-labeled rationales."""

    data_config = config
    records = []
    for position, example in enumerate(examples):
        query = _first_text(example, QUERY_KEYS)
        preferred = _first_rationales(example, PREFERRED_RATIONALE_KEYS)
        rejected = _first_rationales(example, REJECTED_RATIONALE_KEYS)
        if not preferred or not rejected:
            candidate_preferred, candidate_rejected = _split_candidate_rationales(example)
            preferred = preferred or candidate_preferred
            rejected = rejected or candidate_rejected

        if not query or not preferred or not rejected:
            if data_config.skip_missing:
                continue
            raise ValueError(f"Example at position {position} is missing query or preference rationales.")

        evidence = _extract_oracle_evidence(example)
        rejected_evidence = _extract_rejected_evidence(example)
        prompt_evidence = evidence if data_config.condition_on_evidence else None
        prompt = build_dpo_prompt(
            query=query,
            sample_shots=data_config.sample_shots,
            evidence=prompt_evidence,
            domain=data_config.domain,
            num_rationales=data_config.prompt_num_rationales or len(preferred),
            include_flag_instructions=data_config.include_flag_instructions,
        )
        records.append(
            DPOPreferenceRecord(
                prompt=prompt,
                chosen=format_rationale_completion(
                    preferred,
                    include_flag_instructions=data_config.include_flag_instructions,
                ),
                rejected=format_rationale_completion(
                    rejected,
                    include_flag_instructions=data_config.include_flag_instructions,
                ),
                query=query,
                evidence=evidence,
                rejected_evidence=rejected_evidence,
                metadata={"source_position": position},
            )
        )
    return tuple(records)


def split_dpo_records(
    records: Sequence[DPOPreferenceRecord],
    config: Optional[DPOSplitConfig] = None,
) -> Mapping[str, Sequence[DPOPreferenceRecord]]:
    """Split preference records into train/validation/test partitions."""

    split_config = config or DPOSplitConfig()
    _validate_split_config(split_config)
    shuffled = list(records)
    if split_config.shuffle:
        random.Random(split_config.seed).shuffle(shuffled)

    total = len(shuffled)
    if total == 0:
        return {"train": (), "validation": (), "test": ()}
    if total == 1:
        return {"train": tuple(shuffled), "validation": (), "test": ()}
    if total == 2:
        return {"train": tuple(shuffled[:1]), "validation": tuple(shuffled[1:]), "test": ()}

    train_end = int(total * split_config.train_ratio)
    validation_end = train_end + int(total * split_config.validation_ratio)
    train_end = max(1, min(train_end, total - 2))
    validation_end = max(train_end + 1, min(validation_end, total - 1))

    return {
        "train": tuple(shuffled[:train_end]),
        "validation": tuple(shuffled[train_end:validation_end]),
        "test": tuple(shuffled[validation_end:]),
    }


def create_hf_dataset(records: Sequence[Union[DPOPreferenceRecord, Mapping[str, Any]]]):
    """Create a Hugging Face Dataset from DPO records."""

    try:
        from datasets import Dataset
    except ImportError as exc:
        raise ImportError("Install METEORA with the training extra: pip install -e '.[training]'") from exc
    return Dataset.from_list([_record_to_trl_dict(record) for record in records])


def create_hf_dataset_dict(
    splits: Mapping[str, Sequence[Union[DPOPreferenceRecord, Mapping[str, Any]]]]
):
    """Create a DatasetDict from split DPO records."""

    try:
        from datasets import DatasetDict
    except ImportError as exc:
        raise ImportError("Install METEORA with the training extra: pip install -e '.[training]'") from exc
    return DatasetDict({name: create_hf_dataset(records) for name, records in splits.items()})


def load_preference_examples(path: Union[str, Path]) -> Sequence[Mapping[str, Any]]:
    """Load raw examples from JSON, JSONL, or a wrapper object with `tests`/`examples`/`data`."""

    input_path = Path(path)
    text = input_path.read_text(encoding="utf-8")
    if input_path.suffix.lower() == ".jsonl":
        return tuple(json.loads(line) for line in text.splitlines() if line.strip())

    payload = json.loads(text)
    if isinstance(payload, list):
        return tuple(payload)
    if isinstance(payload, Mapping):
        for key in ("tests", "examples", "data", "records", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return tuple(value)
    raise ValueError("Preference input must be a JSON list, JSONL file, or object with examples.")


def load_sample_shots(path: Union[str, Path]) -> Sequence[SampleShot]:
    """Load required sample shots from JSON, JSONL, or plain text."""

    input_path = Path(path)
    text = input_path.read_text(encoding="utf-8")
    suffix = input_path.suffix.lower()
    if suffix == ".jsonl":
        return tuple(json.loads(line) for line in text.splitlines() if line.strip())
    if suffix == ".json":
        payload = json.loads(text)
        if isinstance(payload, list):
            return tuple(payload)
        if isinstance(payload, Mapping):
            for key in ("sample_shots", "examples", "shots"):
                value = payload.get(key)
                if isinstance(value, list):
                    return tuple(value)
        raise ValueError("Sample-shot JSON must be a list or object with sample_shots/examples.")
    shots = [block.strip() for block in text.split("\n\n") if block.strip()]
    return tuple(shots)


def write_preference_jsonl(
    records: Sequence[Union[DPOPreferenceRecord, Mapping[str, Any]]],
    path: Union[str, Path],
    *,
    include_metadata: bool = True,
) -> None:
    """Write DPO preference records as JSONL."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        json.dumps(_record_to_dict(record, include_metadata=include_metadata), ensure_ascii=False)
        for record in records
    ]
    output_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def read_preference_jsonl(path: Union[str, Path]) -> Sequence[Dict[str, Any]]:
    """Read TRL prompt/chosen/rejected records from JSONL."""

    input_path = Path(path)
    return tuple(
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def get_print_logs_callback():
    """Return a small TrainerCallback that prints log dictionaries during training."""

    try:
        from transformers import TrainerCallback
    except ImportError as exc:
        raise ImportError("Install METEORA with the hf extra: pip install -e '.[hf]'") from exc

    class PrintLogsCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: D401
            if logs:
                print(logs)

    return PrintLogsCallback()


def train_dpo_rationale_generator(
    train_records: Sequence[Union[DPOPreferenceRecord, Mapping[str, Any]]],
    validation_records: Optional[Sequence[Union[DPOPreferenceRecord, Mapping[str, Any]]]] = None,
    *,
    config: DPOTrainingConfig,
    model: Any = None,
    tokenizer: Any = None,
    ref_model: Any = None,
    callbacks: Optional[Sequence[Any]] = None,
):
    """Fine-tune a rationale generator with TRL DPO using METEORA preference records."""

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import DPOConfig, DPOTrainer
    except ImportError as exc:
        raise ImportError(
            "Install METEORA with Hugging Face training extras: pip install -e '.[hf,training]'"
        ) from exc

    tokenizer = tokenizer or AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=config.trust_remote_code,
    )
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model is None:
        model_kwargs = {
            "trust_remote_code": config.trust_remote_code,
        }
        if config.device_map is not None:
            model_kwargs["device_map"] = config.device_map
        dtype = _resolve_torch_dtype(torch, config.torch_dtype)
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, **model_kwargs)

    has_validation = bool(validation_records)
    train_dataset = create_hf_dataset(train_records)
    eval_dataset = create_hf_dataset(validation_records or []) if has_validation else None
    dpo_args = _create_dpo_config(DPOConfig, config, do_eval=has_validation)
    trainer_callbacks = list(callbacks or [])
    if not trainer_callbacks:
        trainer_callbacks.append(get_print_logs_callback())

    trainer = _create_dpo_trainer(
        DPOTrainer,
        model=model,
        ref_model=ref_model,
        args=dpo_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=config.peft_config,
        callbacks=trainer_callbacks,
    )
    trainer.train()
    _save_trained_model(trainer, tokenizer, config)
    return trainer


def prepare_and_train_dpo(
    examples: Iterable[Mapping[str, Any]],
    *,
    training_config: DPOTrainingConfig,
    data_config: DPODataConfig,
    split_config: Optional[DPOSplitConfig] = None,
):
    """Build preference data, split it, and run DPO fine-tuning."""

    records = build_dpo_preference_records(examples, config=data_config)
    splits = split_dpo_records(records, config=split_config)
    trainer = train_dpo_rationale_generator(
        splits["train"],
        splits["validation"],
        config=training_config,
    )
    return trainer, splits


def _create_dpo_config(
    config_cls: Callable[..., Any],
    config: DPOTrainingConfig,
    *,
    do_eval: bool = True,
) -> Any:
    load_best_model_at_end = config.load_best_model_at_end and do_eval
    kwargs = {
        "output_dir": config.output_dir,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "lr_scheduler_type": config.lr_scheduler_type,
        "num_train_epochs": config.num_train_epochs,
        "warmup_ratio": config.warmup_ratio,
        "logging_steps": config.logging_steps,
        "do_eval": do_eval,
        "eval_strategy": config.eval_strategy if do_eval else "no",
        "save_strategy": config.save_strategy,
        "save_total_limit": config.save_total_limit,
        "load_best_model_at_end": load_best_model_at_end,
        "remove_unused_columns": config.remove_unused_columns,
        "beta": config.beta,
    }
    if load_best_model_at_end:
        kwargs["metric_for_best_model"] = config.metric_for_best_model
        kwargs["greater_is_better"] = config.greater_is_better
    if config.max_prompt_length is not None:
        kwargs["max_prompt_length"] = config.max_prompt_length
    if config.max_length is not None:
        kwargs["max_length"] = config.max_length
    try:
        return config_cls(**kwargs)
    except TypeError as exc:
        if "eval_strategy" not in str(exc):
            raise
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
        return config_cls(**kwargs)


def _save_trained_model(trainer: Any, tokenizer: Any, config: DPOTrainingConfig) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(trainer, "save_model"):
        trainer.save_model(str(output_dir))
    elif hasattr(trainer, "model") and hasattr(trainer.model, "save_pretrained"):
        trainer.model.save_pretrained(str(output_dir))
    else:
        raise TypeError("Trainer does not expose save_model and model does not expose save_pretrained.")

    if hasattr(trainer, "save_state"):
        trainer.save_state()
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(str(output_dir))

    metadata = {
        "model_name_or_path": config.model_name_or_path,
        "output_dir": str(output_dir),
        "dpo_config": _json_safe(asdict(config)),
        "load_for_inference": str(output_dir),
    }
    (output_dir / "meteora_dpo_config.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _create_dpo_trainer(trainer_cls: Callable[..., Any], **kwargs) -> Any:
    try:
        return trainer_cls(**kwargs)
    except TypeError as exc:
        if "processing_class" not in str(exc):
            raise
        kwargs["tokenizer"] = kwargs.pop("processing_class")
        return trainer_cls(**kwargs)


def _resolve_torch_dtype(torch_module: Any, dtype_name: Optional[str]) -> Any:
    if dtype_name is None:
        return None
    if dtype_name == "auto":
        return "auto"
    dtype = getattr(torch_module, dtype_name, None)
    if dtype is None:
        raise ValueError(f"Unknown torch dtype: {dtype_name}")
    return dtype


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_safe(item) for item in value]
    return str(value)


def _record_to_trl_dict(record: Union[DPOPreferenceRecord, Mapping[str, Any]]) -> Dict[str, str]:
    payload = _record_to_dict(record, include_metadata=False)
    return {
        "prompt": str(payload["prompt"]),
        "chosen": str(payload["chosen"]),
        "rejected": str(payload["rejected"]),
    }


def _record_to_dict(
    record: Union[DPOPreferenceRecord, Mapping[str, Any]],
    *,
    include_metadata: bool,
) -> Dict[str, Any]:
    if isinstance(record, DPOPreferenceRecord):
        return record.to_dict(include_metadata=include_metadata)
    payload = dict(record)
    if "prompt" not in payload and "input" in payload:
        payload["prompt"] = payload["input"]
    missing = {"prompt", "chosen", "rejected"}.difference(payload)
    if missing:
        raise ValueError(f"DPO record is missing required fields: {sorted(missing)}")
    if not include_metadata:
        return {key: payload[key] for key in ("prompt", "chosen", "rejected")}
    return payload


def _validate_split_config(config: DPOSplitConfig) -> None:
    ratios = (config.train_ratio, config.validation_ratio, config.test_ratio)
    if any(ratio < 0 for ratio in ratios):
        raise ValueError("Split ratios must be non-negative.")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0.")


def _first_text(example: Mapping[str, Any], keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _first_rationales(example: Mapping[str, Any], keys: Sequence[str]) -> Sequence[Any]:
    for key in keys:
        value = example.get(key)
        if value:
            return _as_sequence(value)
    return ()


def _split_candidate_rationales(example: Mapping[str, Any]) -> Tuple[Sequence[Any], Sequence[Any]]:
    candidates = example.get("rationale_candidates") or example.get("candidate_rationales")
    if not candidates:
        return (), ()
    gold = set(_normal_indices(_first_present(example, ("correct_chunks", "gold_chunks"))))
    if not gold:
        return (), ()

    preferred = []
    rejected = []
    for candidate in _as_sequence(candidates):
        if not isinstance(candidate, Mapping):
            continue
        text = candidate.get("text") or candidate.get("rationale") or candidate.get("content")
        if not text:
            continue
        selected = _normal_indices(
            _first_present(
                candidate,
                ("selected_chunks", "selected_indices", "chunk_indices", "chunk_index"),
            )
        )
        target = preferred if gold.intersection(selected) else rejected
        target.append(str(text))
    return preferred, rejected


def _extract_oracle_evidence(example: Mapping[str, Any]) -> Optional[str]:
    explicit = _first_text(example, EVIDENCE_KEYS)
    if explicit:
        return explicit
    chunks = _first_present(example, ("document_chunks", "chunks"))
    gold_indices = _normal_indices(_first_present(example, ("correct_chunks", "gold_chunks")))
    evidence_texts = _texts_for_indices(chunks, gold_indices)
    return "\n\n".join(evidence_texts) if evidence_texts else None


def _extract_rejected_evidence(example: Mapping[str, Any]) -> Optional[str]:
    explicit = _first_text(example, IRRELEVANT_EVIDENCE_KEYS)
    if explicit:
        return explicit
    chunks = _first_present(example, ("document_chunks", "chunks"))
    rejected_indices = _normal_indices(
        _first_present(example, ("incorrect_chunks", "negative_chunks", "rejected_chunks"))
    )
    evidence_texts = _texts_for_indices(chunks, rejected_indices)
    return "\n\n".join(evidence_texts) if evidence_texts else None


def _texts_for_indices(chunks: Any, indices: Sequence[int]) -> Sequence[str]:
    if not chunks or not indices:
        return ()
    chunk_list = list(chunks)
    texts = []
    for index in indices:
        if index < 0 or index >= len(chunk_list):
            continue
        chunk = chunk_list[index]
        if isinstance(chunk, Mapping):
            text = chunk.get("text") or chunk.get("content") or chunk.get("page_content")
        else:
            text = getattr(chunk, "text", None) or getattr(chunk, "page_content", None) or str(chunk)
        if text:
            texts.append(str(text))
    return tuple(texts)


def _normal_indices(value: Any) -> Sequence[int]:
    if value is None:
        return ()
    if isinstance(value, int):
        return (value,)
    if isinstance(value, str):
        parts = [part.strip() for part in value.replace(";", ",").split(",")]
        return tuple(int(part) for part in parts if part)
    indices = []
    for item in _as_sequence(value):
        if isinstance(item, Mapping):
            found = False
            for key in ("index", "chunk_index", "id"):
                if key in item:
                    item = item[key]
                    found = True
                    break
            if not found:
                continue
        indices.append(int(item))
    return tuple(indices)


def _first_present(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _as_sequence(value: Any) -> Sequence[Any]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        return (value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value,)
    if isinstance(value, Sequence):
        return value
    return (value,)
