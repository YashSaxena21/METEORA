# Use METEORA in Google Colab

Install into the notebook's Python environment. A separate virtual environment
is not needed. Until the package is published on PyPI, run this setup cell:

```python
from pathlib import Path
import subprocess
import sys

checkout = Path("/content/METEORA")
if not checkout.exists():
    subprocess.run(
        ["git", "clone", "https://github.com/YashSaxena21/METEORA.git", str(checkout)],
        check=True,
    )
subprocess.run(
    [sys.executable, "-m", "pip", "install", f"{checkout}[sentence-transformers,hf]"],
    check=True,
)
```

An existing checkout is reused. If you already have a different checkout, set
`checkout` to its directory. After installing into a running notebook, restart
the runtime if Python has already imported an older package version.

The [base selection example](../examples/minimal_selection.py) runs without model
downloads. The [README quick start](../README.md#quick-start-find-the-return-deadline)
shows model-generated rationales and semantic evidence selection. To generate a rationale with a real model, run:

```python
subprocess.run(
    [sys.executable, str(checkout / "examples/generate_rationales.py"),
     "Does the agreement allow termination for non-payment?"],
    check=True,
)
```

Model-backed examples download weights if they are not cached. A GPU runtime can
speed up inference. Use `--device cpu` to request CPU execution explicitly, or
`--model` to select your own checkpoint.
