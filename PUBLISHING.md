# Publishing METEORA

The distribution name is `meteora-rag`; users import `meteora_rag` and run the
`meteora-rag` CLI. The initial version is `0.1.0`, licensed under CC BY 4.0.

Python imports use an underscore because hyphens are not valid in module names.
Wheel and source archive filenames also normalize the distribution's hyphen to
an underscore.

## Build and validate locally

From the repository root, with a virtual environment activated:

```bash
python -m pip install -e ".[dev]"
python -m unittest discover -s tests
python -m build
python -m twine check --strict dist/*
```

The build creates:

```text
dist/meteora_rag-0.1.0-py3-none-any.whl
dist/meteora_rag-0.1.0.tar.gz
```

`python -m build` builds the wheel from the source distribution. Check that the
wheel works independently of the source checkout using a fresh environment:

```bash
METEORA_SMOKE_DIR="$(mktemp -d)/meteora-smoke"
python -m venv "$METEORA_SMOKE_DIR"
"$METEORA_SMOKE_DIR/bin/python" -m pip install dist/meteora_rag-0.1.0-py3-none-any.whl
"$METEORA_SMOKE_DIR/bin/python" -I scripts/smoke_test_package.py
```

The smoke check covers public imports, version and license metadata, rationale
flag instructions, selection, a verifier callable, and both CLI entry points.
It uses the hashing encoder and a deterministic verifier response, so it does not
download or evaluate an LLM.

## Configure trusted publishing once

The publishing workflow runs when a Git tag matching the package version is
pushed. It uses PyPI Trusted Publishing to authenticate through GitHub Actions.

Before the first upload, create a pending publisher on
[PyPI's publishing settings page](https://pypi.org/manage/account/publishing/)
using these values:

| Field | Value |
| --- | --- |
| PyPI project name | `meteora-rag` |
| GitHub owner | `YashSaxena21` |
| Repository | `METEORA` |
| Workflow filename | `publish.yml` |
| Environment | `pypi` |

Create the matching `pypi` environment in the GitHub repository's settings.
PyPI creates the project when the workflow successfully publishes the first
release; a pending publisher does not reserve the project name. See
[PyPI's setup documentation](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).

## Publish a release

Ensure the intended changes are committed and pushed to GitHub, then create and
push the version tag:

```bash
git tag -a v0.1.0 -m "Release meteora-rag 0.1.0"
git push origin v0.1.0
```

The `Publish to PyPI` workflow checks that the tag matches the package version,
runs tests, builds and checks both distributions, and installs the wheel in a
fresh environment. A separate job publishes those artifacts using PyPI Trusted
Publishing, without a stored PyPI API token.

After publication, verify installation from PyPI in a fresh environment:

```bash
python -m pip install "meteora-rag==0.1.0"
python -c "import meteora_rag; print(meteora_rag.__version__)"
meteora-rag --help
```

For later releases, update the version in both `pyproject.toml` and
`src/meteora_rag/__init__.py`, update `CHANGELOG.md`, and create the matching tag.
Build in a fresh checkout so that `dist/` contains only that release's files.

## Manual upload

If you prefer an API token instead of the GitHub workflow, upload only the
validated files for this version:

```bash
python -m twine upload dist/meteora_rag-0.1.0-py3-none-any.whl dist/meteora_rag-0.1.0.tar.gz
```

Twine prompts for PyPI credentials. Use `__token__` as the username and your PyPI
API token as the password. This command publishes the release to PyPI.
