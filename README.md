# METEORA

METEORA is a drop-in reranker replacement for retrieval-augmented generation
pipelines. Instead of choosing a fixed top-k with a reranker, METEORA asks a
language model to generate query-specific search rationales, then uses those
rationales to select the evidence chunks that should be sent to the final
generator.

The package code lives in `src/meteora`. The original research experiments live
in `Experiments/`.

## Quick Start In Colab

Use a GPU runtime if one is available. Copy this into one Colab cell and run it.
It installs METEORA, loads real Wikipedia text, generates rationales with a
1B-class model, selects evidence with a SentenceTransformer encoder, and
generates a final answer from the selected evidence.

```python
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request

os.chdir("/content")
shutil.rmtree("/content/METEORA", ignore_errors=True)
subprocess.run(
    ["git", "clone", "https://github.com/YashSaxena21/METEORA.git", "/content/METEORA"],
    check=True,
)
os.chdir("/content/METEORA")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "/content/METEORA[sentence-transformers,hf]"],
    check=True,
)
if "/content/METEORA/src" not in sys.path:
    sys.path.insert(0, "/content/METEORA/src")
importlib.invalidate_caches()

from meteora import HFRationaleGenerator, MeteoraReranker, SentenceTransformerEncoder

print("METEORA import works", flush=True)

def fetch_wikipedia_article(title):
    params = urllib.parse.urlencode(
        {
            "action": "query",
            "prop": "extracts",
            "explaintext": "1",
            "format": "json",
            "redirects": "1",
            "titles": title,
        }
    )
    url = "https://en.wikipedia.org/w/api.php?" + params
    request = urllib.request.Request(url, headers={"User-Agent": "METEORA example"})
    with urllib.request.urlopen(request, timeout=30) as response:
        data = json.load(response)
    page = next(iter(data["query"]["pages"].values()))
    return page["title"], page["extract"]

def chunk_words(text, max_words=140, overlap=30):
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    chunks = []
    step = max_words - overlap
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + max_words])
        if len(chunk.split()) > 30:
            chunks.append({"id": f"chunk-{len(chunks)}", "text": chunk})
    return chunks

title, article = fetch_wikipedia_article("Retrieval-augmented generation")
documents = chunk_words(article)
print("Loaded article:", title)
print("Chunks:", len(documents))

import torch

model_id = "meta-llama/Llama-3.2-1B-Instruct"
torch_dtype = "float16" if torch.cuda.is_available() else None

sample_shots = [
    {
        "query": "What is retrieval-augmented generation?",
        "response": """
<rationale_1>[Definition] Search for text that defines retrieval-augmented generation.</rationale_1>
<rationale_2>[Mechanism] Look for how retrieval is combined with model generation.</rationale_2>
<rationale_3>[Purpose] Find why external documents are used during generation.</rationale_3>
""",
    },
    {
        "query": "Why is external evidence useful for language models?",
        "response": """
<rationale_1>[External knowledge] Search for mentions of external data sources or documents.</rationale_1>
<rationale_2>[Accuracy] Look for evidence about improving factuality or reducing unsupported answers.</rationale_2>
<rationale_3>[Query grounding] Find text connecting a user query to retrieved context.</rationale_3>
""",
    },
]

rationale_generator = HFRationaleGenerator(
    model_id,
    sample_shots=sample_shots,
    domain="technical AI documentation",
    num_rationales=4,
    max_new_tokens=220,
    do_sample=False,
    temperature=1.0,
    torch_dtype=torch_dtype,
    device_map="auto",
)

encoder = SentenceTransformerEncoder("sentence-transformers/all-MiniLM-L6-v2")
reranker = MeteoraReranker(encoder, rationale_generator=rationale_generator)

query = "How does retrieval-augmented generation use external knowledge to answer questions?"

raw_rationales = rationale_generator(query)
raw_rationales = raw_rationales.split("Example 3:")[0].split("Flag Instructions:")[0].strip()

print("\nGenerated rationales:\n")
print(raw_rationales)

trace = reranker.trace(
    query=query,
    documents=documents,
    rationales=raw_rationales,
    order="document",
)
selected_documents = [result.document for result in trace.results]

print("\nSelected chunk ids:", [document["id"] for document in selected_documents])
print("\nSelected evidence:")
for document in selected_documents[:5]:
    print(f"\n[{document['id']}]\n{document['text'][:900]}")

def answer_with_model(query, selected_docs, generator, max_new_tokens=220):
    context = "\n\n".join(
        f"[{document['id']}]\n{document['text']}" for document in selected_docs[:6]
    )
    prompt = f"""Use only the evidence below to answer the question.

Evidence:
{context}

Question: {query}

Answer:"""

    tokenizer = generator.tokenizer
    model = generator.model
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2600)
    try:
        inputs = inputs.to(next(model.parameters()).device)
    except Exception:
        pass

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    input_length = inputs["input_ids"].shape[-1]
    answer_tokens = outputs[0][input_length:]
    return tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()

answer = answer_with_model(query, selected_documents, rationale_generator)

print("\nFinal answer:\n")
print(answer)
```

Expected output includes:

```text
METEORA import works
Loaded article: Retrieval-augmented generation
Chunks: ...
Generated rationales:
Selected chunk ids: [...]
Final answer:
```

Colab notes:

- Do not create a virtual environment in Colab.
- The setup cell uses a normal install instead of editable install because a
  running notebook kernel may not notice editable package paths immediately.
- If imports still fail, restart the Colab runtime and rerun the single cell.

## Local Install

For local development:

```bash
git clone https://github.com/YashSaxena21/METEORA.git
cd METEORA

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[sentence-transformers,hf]"
```

Run the packaged example:

```bash
python examples/reranker_replacement.py
```

For DPO fine-tuning support:

```bash
python -m pip install -e ".[sentence-transformers,hf,training]"
```

## Replace Your Reranker

Use `MeteoraReranker` where your RAG pipeline currently calls a reranker. The
encoder should be a SentenceTransformer encoder, and the rationale generator can
be a base Hugging Face model or your DPO fine-tuned model directory.

```python
from meteora import HFRationaleGenerator, MeteoraReranker, SentenceTransformerEncoder

sample_shots = [
    {
        "query": "Is there an anti-assignment clause?",
        "response": """
<rationale_1>[Assignment language] Search for assign, transfer, successors, assigns, and consent restrictions.</rationale_1>
<rationale_2>[Consent trigger] Look for clauses requiring prior written consent before assignment.</rationale_2>
""",
    }
]

rationale_model = "meta-llama/Llama-3.2-1B-Instruct"
# rationale_model = "models/meteora-rationale-dpo"

rationale_generator = HFRationaleGenerator(
    rationale_model,
    sample_shots=sample_shots,
    domain="commercial contracts",
    num_rationales=4,
    max_new_tokens=256,
    do_sample=False,
    torch_dtype="float16",
    device_map="auto",
)

reranker = MeteoraReranker(
    SentenceTransformerEncoder("sentence-transformers/all-MiniLM-L6-v2"),
    rationale_generator=rationale_generator,
)

selected_documents = reranker.filter(
    query="Is assignment restricted?",
    documents=[
        "The agreement may not be assigned without prior written consent.",
        "Invoices are due within thirty days.",
        "This agreement binds successors and permitted assigns.",
    ],
)
```

Use `trace(...)` when you want rationales, selected chunks, scores, and
verification diagnostics:

```python
trace = reranker.trace(query, candidate_documents)
print(trace.indices)
print([rationale.text for rationale in trace.rationales])
```

## Inputs

`MeteoraReranker` accepts:

- plain strings
- dictionaries with `text`, `content`, or `page_content`
- `Chunk` objects
- LangChain-style documents with `page_content`

The most common outputs are:

- `filter(query, documents)`: returns the selected original documents
- `rerank(query, documents)`: returns scored `RerankResult` objects
- `trace(query, documents)`: returns rationales, selection details, and results

## Sample Shots

Sample shots are required. They teach the rationale generator the format and
domain style you want.

```python
sample_shots = [
    {
        "query": "What licenses are granted?",
        "response": "<rationale_1>[License scope] Search for grant, license, use, sublicense, and restrictions.</rationale_1>",
    }
]
```

## Optional Verifier

You can attach a verifier model to reject irrelevant, contradictory, or poisoned
evidence after selection.

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

Create `sample_shots.json`:

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
  --model meta-llama/Llama-3.2-1B-Instruct \
  --output-dir models/meteora-rationale-dpo \
  --torch-dtype float16
```

The fine-tuned model is saved to `--output-dir` with the tokenizer and a
`meteora_dpo_config.json` metadata file. To use the fine-tuned model, set
`rationale_model` to that output directory:

```python
rationale_model = "models/meteora-rationale-dpo"
```

To use a normal model instead, set `rationale_model` to a Hugging Face model id:

```python
rationale_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

The DPO defaults match the paper setup: 80/10/10 split, 3 epochs, learning rate
`3e-5`, cosine scheduler, beta `0.05`, batch size `1`, and gradient
accumulation `2`. Training saves the final fine-tuned checkpoint by default.

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
python -m pip install -e ".[sentence-transformers,hf,training,dev]"
python -m unittest discover -s tests
python -m build
```

## Citation

```bibtex
@misc{saxena2026rankingfreeragreplacing,
      title={Ranking Free RAG: Replacing Re-ranking with Selection in RAG for Sensitive Domains}, 
      author={Yash Saxena and Ankur Padia and Mandar S Chaudhary and Kalpa Gunaratna and Srinivasan Parthasarathy and Manas Gaur},
      year={2026},
      eprint={2505.16014},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.16014}, 
}
```
