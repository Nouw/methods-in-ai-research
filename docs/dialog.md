# Dialog system

## Running the dialog

Talk to the restaurant recommender in the terminal (type `/exit` to stop):

```bash
uv run main.py dialog
```

Options:
- `--classifier <name>`: saved dialog-act classifier to use (default `bow-linear-svm`, see `uv run main.py dialog --help`).
- `--matcher levenshtein|distilbert`: fallback for preferences that keyword matching misses (default `levenshtein`).
- `--restaurants <path>`: restaurant data (default `data/restaurant_info_extended.csv`).

Turn-by-turn details are written to `app.log`.

## Tests

To run an example test run the following command:
```shell
uv run pytest tests/dialog
```

This runs pytest and tests basic functionality of the dialog system.