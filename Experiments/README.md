# Research experiment archive

These programs accompany the METEORA research experiments. They are separate
from the installable `meteora_rag` API and are not included in PyPI distributions.
For a runnable introduction, use the examples in [`examples/`](../examples/README.md).

| Directory | Purpose |
| --- | --- |
| `CP Task/` | Evidence-selection experiments and reranking baselines |
| `Ablation/` | Selection-stage ablations and verifier flag summaries |
| `Adverarial Defense/` | Poisoned-corpus generation and defense evaluations |

Directory and script names retain their original spelling so existing experiment
references continue to work.

## Environment and data

`requirements-experiments.txt` at the repository root records the original
research environment. It is a separate dependency snapshot, not the requirements
for installing the Python package. Create a separate environment if reproducing
these programs.

Datasets, document corpora, model weights, and experiment outputs are not bundled.
Programs use either command-line arguments or configuration constants near the
start of the script. Set dataset and corpus paths to files you have obtained
separately. Model IDs must identify a checkpoint you can load; some models require
provider access approval. Check each script's argument parser or configuration
section before running it.

Evidence annotations generally use document-relative character offsets or chunk
positions. Match the input format expected by the relevant loader; a raw dataset
download may require preprocessing before it matches these records. Output JSON
from a defense run is the input to the verifier flag distribution analysis.

These archived programs can load large models, run training, and write evaluation
results. They are not imported or executed when `meteora_rag` is installed or
imported. The supported package workflow for preference-data preparation and
training is documented in [`docs/training.md`](../docs/training.md).
