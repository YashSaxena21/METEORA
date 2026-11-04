# Train a rationale generator

Direct Preference Optimization (DPO) trains a model to prefer one completion over
another for the same prompt. METEORA uses a useful evidence-search rationale as
the preferred completion and a less useful rationale as the rejected completion.

Data preparation uses the base package. Training additionally requires:

```bash
python -m pip install ".[training]"
```

Run the commands below from the repository root. The bundled examples are small,
hand-authored examples of the data format, not the datasets or checkpoints used
for the paper's benchmark results.

## Input data

The simplest input is a JSON list of records with these fields:

```json
[
  {
    "query": "Is assignment restricted?",
    "document_chunks": [
      {"text": "Invoices are due within thirty days."},
      {"text": "Assignment requires prior written consent."}
    ],
    "correct_chunks": [1],
    "effective_rationales": ["Look for assignment restrictions and consent requirements."],
    "ineffective_rationales": ["Look for invoice payment dates."]
  }
]
```

`correct_chunks` contains zero-based positions in `document_chunks`. Supporting
passages are called *oracle evidence* in training prompts because their relevance
is known from annotations. By default they are included during training, but
rationale generation at inference uses only the query.

Both rationale lists must contain at least one entry. Entries can also be
rationale objects with `text`, `label`, and `flag_instructions` fields. The loader
accepts a JSON list, JSONL records, or a JSON object containing an `examples` list.

Alternatively, supply `rationale_candidates`: a list of objects with `text` and
`selected_indices`. Each candidate must have already been evaluated by an evidence
selector. Candidates retrieving any `correct_chunks` entry become preferred;
candidates with no overlap become rejected. A record still needs at least one
candidate in each group. The package does not generate or evaluate these
candidate rationales during data preparation.

By default, examples missing a query or a preference pair are skipped. Check the
reported record count. With the Python API, set `DPODataConfig(skip_missing=False)`
to raise an error instead. The CLI reports an error if no usable pairs remain.

## Format examples

`sample_shots` supplies example query/response pairs to demonstrate the output
format in prompts. See `examples/data/sample_shots.json`. These examples are
separate from the preferred/rejected training pairs.

Keep rejection criteria inside their rationale blocks:

```xml
<rationale_1>[Assignment] Search for consent requirements before assignment. Flag Instructions: Reject passages that reverse the stated consent requirement.</rationale_1>
```

## Prepare the bundled sample

```bash
meteora-rag dpo-prepare \
  --input examples/data/preference_examples.json \
  --sample-shots examples/data/sample_shots.json \
  --output-dir outputs/dpo \
  --domain "commercial contracts"
```

Output is three files: `train.jsonl`, `validation.jsonl`, and `test.jsonl`. Each row
contains `prompt`, `chosen`, and `rejected`, plus optional query/evidence metadata.
`chosen` and `rejected` contain formatted rationale completions.

The default split is 80/10/10 with seed `42`. Small inputs are adjusted to keep
usable partitions: one record goes to training only; two records produce one
training and one validation record; three or more produce nonempty partitions.
The bundled three-record example produces one record per split and demonstrates
preparation rather than a meaningful model training set.

## Train

After preparing an appropriately sized preference dataset, run:

```bash
meteora-rag dpo-train \
  --train outputs/dpo/train.jsonl \
  --validation outputs/dpo/validation.jsonl \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --output-dir outputs/rationale-model
```

`--model` accepts a Hugging Face causal-language-model ID or a local checkpoint.
Use a model and training settings that fit your available hardware. Default
training settings are three epochs, learning rate `3e-5`, cosine scheduling,
DPO beta `0.05`, batch size `1`, and gradient accumulation `2`. Use
`meteora-rag dpo-train --help` for the exposed CLI options, or `DPOTrainingConfig`
for the full Python configuration.

An empty or omitted validation split disables evaluation. The final checkpoint
is saved by default. To select the best checkpoint by evaluation loss instead,
configure `load_best_model_at_end=True` in the Python API and provide validation
records.

The output directory contains model weights, tokenizer files, and
`meteora_dpo_config.json` with training settings. Training a model is separate
from publishing this Python package; no model weights are included in the wheel.

## Load a trained checkpoint

```python
from meteora_rag import HFRationaleGenerator, load_sample_shots

generator = HFRationaleGenerator(
    "outputs/rationale-model",
    sample_shots=load_sample_shots("examples/data/sample_shots.json"),
    domain="commercial contracts",
    num_rationales=2,
    do_sample=False,
)
print(generator("Does the agreement restrict assignment?"))
```

For direct prompt control after DPO training, `build_inference_prompt()` exposes
the training-style prompt without supporting passages. `HFRationaleGenerator`
also accepts a custom generation callback through `generate_fn`.
