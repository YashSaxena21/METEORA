# Contributing

## Development environment

From a source checkout, create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

On Windows, activate with `.venv\Scripts\activate`. Add the optional `hf`,
`sentence-transformers`, or `training` extras when working on those integrations.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/meteora_rag/` | Installable Python package |
| `tests/` | Unit and CLI regression tests |
| `examples/` | Runnable examples and small sample inputs |
| `docs/api.md` | Public API and configuration guide |
| `docs/training.md` | Preference-data format and training workflow |
| `docs/index.html` | Static research project website |
| `scripts/` | Release validation utilities |
| `Experiments/` | Archived research programs with separate dependencies |

## Checks

```bash
python -m unittest discover -s tests
ruff check src/meteora_rag tests examples scripts
ruff format --check src/meteora_rag tests examples scripts
python -m build
python -m twine check --strict dist/*
```

The unit suite runs without downloading model weights or installing optional
model dependencies. Use deterministic test doubles for backend contracts and
focused regression cases for parsing or selection changes. Real model examples
can be run separately when changing model integrations; their generated wording
is not a stable unit-test assertion.

Verify built wheels in a clean virtual environment with
`scripts/smoke_test_package.py`, as described in [PUBLISHING.md](PUBLISHING.md).
The release check exercises installed imports, metadata, flag instructions,
selection, a deterministic verifier callback, and CLI entry points.

Keep the README, CLI help, examples, and website snippets consistent when a
public API changes. Include sample inputs for examples and explain which steps
require optional dependencies or downloaded weights. Describe configuration
placeholders explicitly instead of relying on a local machine's paths.

## Releases

Update the version in `pyproject.toml` and `src/meteora_rag/__init__.py` together,
then add a changelog entry. Build and validate both the wheel and source archive.
See [PUBLISHING.md](PUBLISHING.md) for publishing credentials and release tags.
