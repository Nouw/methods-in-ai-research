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

## Running the program

```bash
uv run main.py --help

usage: main.py [-h] {experiment,train,evaluate,interactive} ...

Methods in AI Research dialog-act classification pipeline.

positional arguments:
  {experiment,train,evaluate,interactive}
    experiment          Train and evaluate models on generated splits
    train               Train and save final models
    evaluate            Evaluate saved models on labeled data
    interactive         Classify utterances using saved models

options:
  -h, --help            show this help message and exit
```

## Running experiments

Train and evaluate every classifier on the original and grouped splits:

```bash
uv run python main.py experiment
```

Splits are created in memory. Add `--save-splits` to also write them under `artifacts/splits/`. Results are saved under `results/<classifier>/<split>/`.

## Training final models

Train every classifier on the complete provided dataset and save them under `artifacts/models/`:

```bash
uv run python main.py train
```

Use `--classifier <name>` to train only one classifier. 

## Evaluating a held-out dataset

Evaluate every saved model without retraining:

```bash
uv run python main.py evaluate /path/to/dialog_acts_test.dat
```

The command prints accuracy, balanced accuracy, and macro F1. Detailed metrics, the confusion matrix, and individual predictions are saved under `results/<classifier>/provided/`.

## Interactive classification

Load every saved model and compare their predictions for each utterance:

```bash
uv run python main.py interactive
```

Use `--classifier bow-linear-svm` to load only one model. Enter `/exit` to stop the prompt.

## Custom paths

The commands use `data/dailog_acts.dat`, `artifacts/models/`, and `results/` by default. Override them when needed:

```bash
uv run python main.py train --data custom.dat --models-dir custom/models
uv run python main.py evaluate held-out.dat --models-dir custom/models --results-dir custom/results
```

# Notes

The grouped split uses StratifiedGroupKFold with normalized utterances as groups, ensuring that duplicate utterances never occur in both partitions. Because groups cannot be divided, the grouped split results in an approximately 84/16 division. The reqmore label is absent from the grouped test set because all five examples share the same utterance (more) and must remain together.

Classification reports and confusion matrices always include all 15 dialog-act labels. Labels absent from an evaluation set are shown with zero support and counts.

The bag-of-words vocabulary is fitted exclusively on the training partition. During evaluation, tokens and n-grams absent from this vocabulary are ignored by the TF-IDF vectorizer.
