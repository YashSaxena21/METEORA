# Python API guide

Install `meteora-rag` and import `meteora_rag`. This guide describes the public
interfaces; the package's `__all__` lists its exported names.

## Selection and reranker integration

`MeteoraReranker(encoder, ...)` accepts an encoder with an `encode(texts)` method,
or a callable that returns one embedding per input string. Embeddings must be a
two-dimensional numeric array. PyTorch tensors are also accepted and converted
to NumPy arrays for selection.

| Constructor option | Default | Meaning |
| --- | --- | --- |
| `rationale_generator` | `None` | A callable taking `(query, documents)` or just `(query)` and returning rationale text or rationale objects |
| `verifier` | `None` | A `MeteoraVerifier` or model callable accepting one prompt |
| `config` | `MeteoraConfig(expansion_window=0)` | Selection settings |
| `selector` | `None` | An existing `MeteoraSelector`; its settings take precedence over `config` |
| `fallback_to_query_rationale` | `True` | Use the query as one rationale when neither rationales nor a generator is supplied |
| `default_order` | `"score"` | Result ordering |

`filter()`, `rerank()`, and `trace()` all take `query` and `documents`, with these
optional keyword arguments:

- `rationales`: supplied rationales override a configured rationale generator.
- `top_k`: an upper limit on final results, applied after verification. `None`
  leaves the adaptive selection size unchanged; `0` returns no final results.
- `verify`: `None` runs a configured verifier; `False` skips it; `True` requires
  one to be configured.
- `order`: `"score"` sorts by descending score, then chunk index.
  `"document"`, `"input"`, and `"context"` sort by chunk index. For strings and
  ordinary document objects, indices correspond to their input positions.

`filter()` returns original document objects in a tuple. `rerank()` returns
`RerankResult` objects containing `index`, `text`, `score`, one-based `rank`,
`document`, `chunk`, and an optional `verification`. Scores are cosine-similarity
diagnostics, not calibrated relevance probabilities.

`trace()` returns a `MeteoraRerankTrace` with:

- `rationales`: the rationales actually used.
- `selection`: the result before verification and the final `top_k` limit.
- `verification`: decisions for all selected chunks, or `None` when disabled.
- `results`: final `RerankResult` objects.
- `indices` and `documents`: convenient views of the final results.

Call `to_dict()` on a trace, selection, or verification result to inspect its
fields. User-supplied metadata should contain JSON-compatible values if you plan
to serialize these dictionaries as JSON.

### Accepted documents and rationales

The reranker accepts strings, `Chunk` objects, dictionaries with a `text`,
`content`, or `page_content` field, and objects with a `page_content` or `text`
attribute. Dictionary fields such as IDs are retained as metadata. If using
custom `Chunk.index` values, give each chunk a unique index for clear diagnostics.

Rationales can be strings, `Rationale` objects, dictionaries with a `text` field,
or XML-style response text:

```python
from meteora_rag import Rationale

rationale = Rationale(
    text="Look for clauses requiring written consent before assignment.",
    index=1,
    label="Assignment restriction",
    flag_instructions="Reject passages that reverse the consent requirement.",
)
```

`parse_rationales(text)` extracts a list of `Rationale` objects from XML blocks,
numbered lists, or plain text. `parse_rationale_texts(text)` returns their text;
`extract_flag_instructions(text)` returns `(rationale_index, instructions)` pairs.
Keep `Flag Instructions:` inside the XML block so they remain attached to the
correct rationale. If passing a full prompt plus completion to the parser, it
uses the final `Query:` section; normally pass only the generated completion.

### Configure selection

```python
from meteora_rag import HashingEncoder, MeteoraConfig, MeteoraReranker

config = MeteoraConfig(
    expansion_window=0,
    elbow_z_threshold=1.0,
    enable_pairing=True,
    enable_pooling=True,
    enable_expansion=True,
)
reranker = MeteoraReranker(HashingEncoder(), config=config)
```

Pairing selects the best chunk for each rationale. Pooling compares chunks with
an average rationale embedding and applies `statistical_elbow()` to descending
scores. The cutoff uses the first score drop above the configured z-score
threshold, with a difference-based fallback when no drop exceeds it. It is a
heuristic cutoff, not a statistical significance guarantee.

Expansion includes neighboring input positions. Use `expansion_window=0` for
independent retrieved passages. Disable individual stages with the `enable_*`
settings. With one or two candidates, pooling keeps all candidates.

For direct access, call `MeteoraSelector(encoder, config=...).select(chunks,
rationales)`. Unlike the reranker, it accepts only strings, `Chunk` objects, or
mappings with `text`; it does not generate rationales or run verification. Its
default expansion window is `1`. A `SelectionResult` contains `selected_indices`,
`selected_chunks`, `rationales`, and `details` identifying each selection stage.

## Encoders

`HashingEncoder(n_features=384, lowercase=True)` uses deterministic signed word
hashes. It needs no downloaded weights and is useful for learning the API and
running tests. It does not provide learned semantic representations.

`SentenceTransformerEncoder(model_name, device=None, **model_kwargs)` loads a
SentenceTransformer model and returns normalized NumPy embeddings. It defaults
to `sentence-transformers/all-MiniLM-L6-v2`. Use the `sentence-transformers`
extra. Loading a model can download its weights; pass a local model directory
when working offline.

An offline model directory or cache must include the trained tokenizer files as
well as weights. The encoder rejects tokenizers that contain only special tokens
because they cannot distinguish ordinary words.

A custom encoder must return a consistent embedding dimension for both chunks
and rationales. Return one row per input string.

## Generation

The `hf` extra enables `HFRationaleGenerator`. This complete configuration uses
a base chat model with two examples from the bundled source tree:

```python
from meteora_rag import (
    HFRationaleGenerator,
    MeteoraReranker,
    SentenceTransformerEncoder,
    load_sample_shots,
)

generator = HFRationaleGenerator(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    sample_shots=load_sample_shots("examples/data/sample_shots.json"),
    domain="commercial contracts",
    num_rationales=2,
    include_flag_instructions=True,
    use_chat_template=True,
    max_new_tokens=384,
    do_sample=False,
)
reranker = MeteoraReranker(
    SentenceTransformerEncoder("sentence-transformers/all-MiniLM-L6-v2"),
    rationale_generator=generator,
)
result = reranker.trace(
    "Does the agreement allow termination for non-payment?",
    ["Either party may terminate if an invoice remains unpaid for sixty days."],
)
print([rationale.to_dict() for rationale in result.rationales])
```

The file path in this example is relative to the repository root. Installed
applications can supply their own list of `{query, response}` dictionaries
instead. `sample_shots` is the argument name for these example pairs; it is not
training data or a checkpoint.

`generator(query)` returns response text. `HFRationaleGeneratorConfig` records
its settings. The default is eight rationales, sampling enabled with temperature
`0.7`, up to `768` new tokens, and flag instructions enabled. Generation keeps
at most the requested number of complete rationale blocks and removes unrelated
example queries a model may append. A small token budget may truncate an output;
inspect the response and adjust the budget for your model and rationale count.

`device_map="auto"` delegates model placement to Hugging Face Accelerate. To
place a model explicitly, pass `device="cpu"`, `"cuda"`, or `"mps"` with
`device_map=None`. Leave `torch_dtype=None` to use the model's default dtype, or
choose a dtype supported by your hardware. Explicitly supplied `model` and
`tokenizer` objects avoid loading a second copy of their weights.

Set `use_chat_template=True` for a chat-trained model to format the prompt with
its tokenizer's chat template. The default is `False` for checkpoints trained
with plain completion prompts. This setting applies to local Hugging Face
generation; a `generate_fn` callback receives the unformatted prompt so its
backend can apply its own message format.

For an external backend, pass `generate_fn(prompt)` returning text. This callback
receives the fully constructed prompt and replaces local model loading. Use
`build_rationale_prompt()` and `format_sample_shots()` when constructing prompts
outside the generator. Prompt builders do not make network requests.

## Verification

`MeteoraVerifier(model, fail_closed=True)` calls a model with one text prompt per
chunk. The model can be callable or expose `generate(prompt)`, `complete(prompt)`,
or `invoke(prompt)`. Return either a mapping with `relevant` and `flagged`
booleans or a string containing that JSON object. Extract the text or decision
from a provider-specific response wrapper inside your adapter.

Attach a verifier with `MeteoraReranker(encoder, verifier=verifier)`. Alternatively,
call `verify_chunk(query, chunk, rationales)`, `verify_chunks(query, chunks,
rationales)`, or `verify_selection(query, selection)` directly. Chunks use the
same formats as `MeteoraSelector`.

The verifier prompt includes the query, current chunk, rationale flag instructions,
and previously accepted chunks. Chunks are checked sequentially in selection
order. An earlier accepted chunk is not re-evaluated if a later chunk contradicts
it, so input order can affect decisions. The package does not consult external
sources to determine factual truth.

The decision fields are:

| Field | Requirement | Meaning |
| --- | --- | --- |
| `relevant` | Required boolean | Whether the chunk supports the query |
| `flagged` | Required boolean | Whether the chunk violates a flag instruction or consistency check |
| `reason` | Optional string | Explanation for the decision |
| `flag_types` | Optional list of strings | Issue labels, for example `CONTRADICTION` |
| `confidence` | Optional number | Model-reported confidence, conventionally from 0 to 1 |

A `VerificationResult` is accepted only when relevant and unflagged.
`VerifiedSelection` exposes `accepted_chunks`, `flagged_chunks`,
`rejected_chunks`, and the corresponding index properties. An irrelevant chunk
is rejected even if it is not flagged.

`parse_verifier_response()` accepts JSON strings, fenced JSON, mappings, or text
with explicit `relevant:` and `flagged:` decisions. Conventional boolean strings
and `0`/`1` are accepted; unknown values or missing decisions are parse errors.
By default these reject the chunk with a `PARSE_ERROR` flag. Explicitly setting
`fail_closed=False` accepts malformed decisions. Model exceptions are not caught
or converted to accepted decisions.

`build_verifier_prompt()` exposes prompt construction without running a model.
See `examples/verify_evidence.py` for a real Hugging Face model adapter.

## Chunking and evaluation

- `chunk_text(text, chunk_size=256, overlap=None)` splits by whitespace-delimited
  words and preserves character offsets. Default overlap is 10 percent of the
  chunk size, rounded down.
- `chunk_texts(texts, ...)` assigns global chunk indices and records source document
  positions in metadata. Offsets remain relative to each original document.
- `find_spanning_chunks(span_start, span_end, chunks)` finds consecutive chunks
  covering a character span. End offsets use Python's exclusive-end convention.
- `precision_recall_f1(selected_indices, gold_indices)` compares sets of chunk
  indices and returns `SelectionMetrics`, including whether any gold chunk was
  found. Precision, recall, and F1 are zero when their denominators are empty.

## Training utilities

See [training.md](training.md) for data schemas and commands. The public training
API includes configuration dataclasses (`DPODataConfig`, `DPOSplitConfig`,
`DPOTrainingConfig`), the `DPOPreferenceRecord` data type, prompt and completion
formatters (`build_dpo_prompt`, `build_inference_prompt`,
`format_rationale_completion`), record construction and splitting
(`build_dpo_preference_records`, `split_dpo_records`), and JSON/JSONL loading and
writing helpers (`load_preference_examples`, `load_sample_shots`,
`read_preference_jsonl`, `write_preference_jsonl`).

`create_hf_dataset()` and `create_hf_dataset_dict()` convert records for the
Hugging Face ecosystem. `train_dpo_rationale_generator()` trains and saves a model;
`prepare_and_train_dpo()` combines preparation, splitting, and training.
`get_print_logs_callback()` creates an optional Transformers logging callback.
These model-training functions require the `training` extra.
