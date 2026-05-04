"""METEORA: rank-free, rationale-driven evidence selection for RAG."""

from .chunking import chunk_text, chunk_texts, find_spanning_chunks
from .dpo import (
    DPODataConfig,
    DPOPreferenceRecord,
    DPOSplitConfig,
    DPOTrainingConfig,
    build_dpo_preference_records,
    build_dpo_prompt,
    build_inference_prompt,
    create_hf_dataset,
    create_hf_dataset_dict,
    format_rationale_completion,
    get_print_logs_callback,
    load_preference_examples,
    load_sample_shots,
    prepare_and_train_dpo,
    read_preference_jsonl,
    split_dpo_records,
    train_dpo_rationale_generator,
    write_preference_jsonl,
)
from .embeddings import HashingEncoder, SentenceTransformerEncoder
from .evaluation import SelectionMetrics, precision_recall_f1
from .generation import HFRationaleGenerator, HFRationaleGeneratorConfig
from .prompts import SampleShot, build_rationale_prompt, build_verifier_prompt, format_sample_shots
from .rationales import extract_flag_instructions, parse_rationale_texts, parse_rationales
from .reranker import MeteoraRerankTrace, MeteoraReranker, RerankResult
from .selector import MeteoraConfig, MeteoraSelector, statistical_elbow
from .types import Chunk, ChunkScore, Rationale, RationaleMatch, SelectionDetails, SelectionResult
from .verifier import (
    MeteoraVerifier,
    VerificationResult,
    VerifiedSelection,
    parse_verifier_response,
)

__all__ = [
    "Chunk",
    "ChunkScore",
    "DPODataConfig",
    "DPOPreferenceRecord",
    "DPOSplitConfig",
    "DPOTrainingConfig",
    "HashingEncoder",
    "HFRationaleGenerator",
    "HFRationaleGeneratorConfig",
    "MeteoraConfig",
    "MeteoraRerankTrace",
    "MeteoraReranker",
    "MeteoraSelector",
    "MeteoraVerifier",
    "Rationale",
    "RationaleMatch",
    "RerankResult",
    "SelectionDetails",
    "SelectionMetrics",
    "SelectionResult",
    "SentenceTransformerEncoder",
    "SampleShot",
    "VerificationResult",
    "VerifiedSelection",
    "build_dpo_preference_records",
    "build_dpo_prompt",
    "build_inference_prompt",
    "build_rationale_prompt",
    "build_verifier_prompt",
    "chunk_text",
    "chunk_texts",
    "create_hf_dataset",
    "create_hf_dataset_dict",
    "extract_flag_instructions",
    "find_spanning_chunks",
    "format_rationale_completion",
    "format_sample_shots",
    "get_print_logs_callback",
    "load_preference_examples",
    "load_sample_shots",
    "parse_rationale_texts",
    "parse_rationales",
    "prepare_and_train_dpo",
    "precision_recall_f1",
    "parse_verifier_response",
    "read_preference_jsonl",
    "split_dpo_records",
    "statistical_elbow",
    "train_dpo_rationale_generator",
    "write_preference_jsonl",
]

__version__ = "0.1.0"
