# methods-in-ai-research

## Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you haven't already:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```cmd
# Windows
pip install uv
```


From the project root, install the project and its dependencies:

```bash
uv sync
```

This downloads all the dependencies and creates a python virtual environment.
Active the venv by running:
```bash
# macOS / Linux
source .venv/bin/activate
```

```cmd
# Windows
.\.venv\Scripts\Activate
```

To see the specific commands used by the classification and dialog system.
Take a look at their respective markdown files:
- [Classification](./docs/classification.md)
- [Dialoag](./docs/dialog.md)