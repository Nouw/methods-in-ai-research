# methods-in-ai-research

## Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you haven't already:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

From the project root, install the project and its dependencies:

```bash
uv sync
```

This downloads all the dependencies and creates a python virtual environment.
Active the venv by running:
```bash
source .venv/bin/activate
```

## Running the notebooks

From the project root, run:

```bash
uv run --with jupyter jupyter notebook
```

Open the URL printed in the terminal, then select a notebook in `notebooks/`. You can import project functions directly,
for example:

```python
from methods_in_ai_research.processing import preprocess
```

# Notes

The grouped split uses StratifiedGroupKFold with normalized utterances as groups, ensuring that duplicate utterances never occur in both partitions. Because groups cannot be divided, the grouped split results in an approximately 84/16 division. The reqmore label is absent from the grouped test set because all five examples share the same utterance (more) and must remain together.
