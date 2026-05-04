# METEORA

METEORA is a drop-in reranker replacement for RAG pipelines.

Instead of asking a reranker for a fixed top-k, METEORA asks an LLM to generate
search rationales, then uses those rationales to select evidence with a
rank-free selector.

The package code lives in `src/meteora`. The original paper experiments live in
`Experiments/`.

## Get Started

### Google Colab

Copy and paste this into one Colab code cell:

```python
!rm -rf /content/METEORA
!git clone https://github.com/YashSaxena21/METEORA.git /content/METEORA
%cd /content/METEORA
%pip install -q -e /content/METEORA

from meteora import HashingEncoder, MeteoraReranker
print("METEORA import works")

!python examples/reranker_replacement.py
```

Expected output:

```text
METEORA import works
Selected document ids: ['a', 'c']
```

If you already cloned the repo in the same Colab runtime, use this instead:

```python
%cd /content/METEORA
!git pull
%pip install -q -e /content/METEORA

from meteora import HashingEncoder, MeteoraReranker
print("METEORA import works")

!python examples/reranker_replacement.py
```

Do not create a virtual environment in Colab. Colab already runs inside a
managed Python environment, and `%cd /content/METEORA` is needed so pip installs
from the repository folder that contains `pyproject.toml`.

If `from meteora import ...` still fails, restart the Colab runtime and run the
single Colab cell again.

### Local Terminal

Copy and paste this from a terminal on your machine:

```bash
git clone https://github.com/YashSaxena21/METEORA.git
cd METEORA

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e .

python examples/reranker_replacement.py
```

The example should print the METEORA-selected document ids for a tiny contract
query. This quick path uses the built-in `HashingEncoder`, so it does not
download any model weights.

For a real embedding model, install the sentence-transformers extra:

```bash
python -m pip install -e ".[sentence-transformers]"
```

For Hugging Face rationale generation:

```bash
python -m pip install -e ".[hf]"
```

For DPO fine-tuning:

```bash
python -m pip install -e ".[hf,training]"
```

If you already cloned the repo and activated your environment, the minimal
install command is:

```bash
python -m pip install -e .
```

## Quick Start

This is the smallest copy-paste example. It uses a simple local rationale
function so you can test the METEORA interface before connecting an LLM.

```python
from meteora import HashingEncoder, MeteoraReranker

documents = [
    {"text": "The agreement may not be assigned without prior written consent.", "id": "a"},
    {"text": "Invoices are due within thirty days after receipt.", "id": "b"},
    {"text": "The agreement binds successors and permitted assigns.", "id": "c"},
]

def rationale_generator(query, docs):
    return """
<rationale_1>[Consent restriction] Search assigned written consent.</rationale_1>
<rationale_2>[Successors and assigns] Search successors permitted assigns.</rationale_2>
"""

reranker = MeteoraReranker(
    HashingEncoder(),
    rationale_generator=rationale_generator,
)

selected_documents = reranker.filter(
    "Is assignment restricted?",
    documents,
    order="document",
)

print([doc["id"] for doc in selected_documents])
```

That is the main intended use: replace your existing reranker with
`MeteoraReranker`.

## Using An LLM Rationale Generator

Every rationale prompt requires sample shots. Pass examples from your domain so
the model learns the style of rationales you want.

```python
from meteora import (
    HFRationaleGenerator,
    MeteoraReranker,
    SentenceTransformerEncoder,
)

sample_shots = [
    {
        "query": "Is there an anti-assignment clause?",
        "response": """
<rationale_1>[Assignment language] Search for assign, transfer, successors, assigns, and consent restrictions.</rationale_1>
<rationale_2>[Consent trigger] Look for clauses requiring prior written consent before assignment.</rationale_2>
""",
    }
]

# Choose either a normal model id or a fine-tuned model directory.
rationale_model = "meta-llama/Llama-3.1-8B-Instruct"
# rationale_model = "models/meteora-rationale-dpo"

rationale_generator = HFRationaleGenerator(
    rationale_model,
    sample_shots=sample_shots,
    domain="commercial contracts",
    torch_dtype="float16",
)

reranker = MeteoraReranker(
    SentenceTransformerEncoder("sentence-transformers/all-MiniLM-L6-v2"),
    rationale_generator=rationale_generator,
)

clean_documents = reranker.filter(
    query="Is assignment restricted?",
    documents=[
        "The agreement may not be assigned without prior written consent.",
        "Invoices are due within thirty days.",
        "This agreement binds successors and permitted assigns.",
    ],
)
```

To run your dataset with the normal model, set `rationale_model` to a Hugging
Face model id. To run with your fine-tuned model, set it to the DPO
`--output-dir` path, for example `models/meteora-rationale-dpo`.

## Inputs

`MeteoraReranker` accepts:

- plain strings
- dictionaries with `text`, `content`, or `page_content`
- `Chunk` objects
- LangChain-style documents with `page_content`

Use `rerank(...)` when you want scores and diagnostics:

```python
results = reranker.rerank(query, candidate_documents)
for result in results:
    print(result.rank, result.score, result.document)
```

Use `filter(...)` when your RAG pipeline expects documents back:

```python
documents = reranker.filter(query, candidate_documents)
```

## Sample Shots

Sample shots are required for prompt construction.

They can be structured:

```python
sample_shots = [
    {
        "query": "What licenses are granted?",
        "response": "<rationale_1>[License scope] Search for grant, license, use, sublicense, and restrictions.</rationale_1>",
    }
]
```

Or preformatted strings:

```python
sample_shots = [
    """Query: What happens after a change of control?
Rationales:
<rationale_1>[Control trigger] Search for change of control, merger, acquisition, and termination rights.</rationale_1>"""
]
```

## Optional Verifier

You can attach a verifier model to reject irrelevant, contradictory, or poisoned
evidence.

```python
from meteora import MeteoraReranker

def verifier_model(prompt: str) -> str:
    return '{"relevant": true, "flagged": false, "reason": "The chunk supports the query."}'

verified_reranker = MeteoraReranker(
    encoder,
    rationale_generator=rationale_generator,
    verifier=verifier_model,
)

clean_documents = verified_reranker.filter(query, candidate_documents)
```

## DPO Fine-Tuning

METEORA includes a DPO path for fine-tuning the rationale generator.

Create a `sample_shots.json` file:

```json
[
  {
    "query": "Is there an anti-assignment clause?",
    "response": "<rationale_1>[Assignment] Search for assignment, transfer, and consent restrictions.</rationale_1>"
  }
]
```

Prepare preference data:

```bash
meteora dpo-prepare \
  --input data/preference_examples.json \
  --sample-shots sample_shots.json \
  --output-dir data/dpo \
  --domain "commercial contracts"
```

Train:

```bash
meteora dpo-train \
  --train data/dpo/train.jsonl \
  --validation data/dpo/validation.jsonl \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --output-dir models/meteora-rationale-dpo \
  --torch-dtype float16
```

The fine-tuned model is saved to `--output-dir` with the tokenizer and a
`meteora_dpo_config.json` metadata file. Use that same directory as
`rationale_model` when running METEORA on your dataset:

```python
normal_model = "meta-llama/Llama-3.1-8B-Instruct"
fine_tuned_model = "models/meteora-rationale-dpo"

rationale_generator = HFRationaleGenerator(
    fine_tuned_model,  # change to normal_model if you do not want the tuned model
    sample_shots=sample_shots,
    domain="commercial contracts",
    torch_dtype="float16",
)
```

The DPO defaults match the paper setup: 80/10/10 split, 3 epochs, learning rate
`3e-5`, cosine scheduler, beta `0.05`, batch size `1`, and gradient
accumulation `2`. Training saves the final fine-tuned checkpoint by default;
set `load_best_model_at_end=True` in `DPOTrainingConfig` if you want to reload
the lowest validation-loss checkpoint before saving.

## CLI Selection

Chunk a document:

```bash
meteora chunk document.txt --chunk-size 256 --output chunks.json
```

Select evidence from saved chunks and rationales:

```bash
meteora select \
  --chunks chunks.json \
  --rationales rationales.txt \
  --encoder-model sentence-transformers/all-MiniLM-L6-v2 \
  --output selection.json
```

## Development

```bash
pip install -e ".[dev]"
python -m unittest discover -s tests
python -m build
```

## Citation

```bibtex
@inproceedings{
anonymous2026ranking,
title={Ranking Free {RAG}: Replacing Re-ranking with Selection in {RAG} for Sensitive Domains},
author={Anonymous},
booktitle={Forty-third International Conference on Machine Learning},
year={2026},
url={https://openreview.net/forum?id=O88FCPAPAj}
}
```
