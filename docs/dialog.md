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
- `--transparency on|off`: show or hide the reasoning behind a recommendation (default `on`), see below.
- `--log-dir <path>`: directory for interaction logs (default `logs/dialog`).

## Reasoning transparency

The configurable feature for the Part 2 user study. With `--transparency on` a recommendation explains
each additional requirement, e.g. "It is romantic because it allows a long stay." With `--transparency off`
the same restaurant is recommended without that explanation. Everything else, including filtering by
requirements, is identical in both conditions.

The explanation only appears when the user states additional requirements, so study tasks should ask for one
(e.g. "find a romantic restaurant"), or the two conditions behave the same.

```bash
uv run main.py dialog --transparency off
```

## Interaction logs

Every conversation is written to `logs/dialog/<session_id>.jsonl`, one JSON record per line:

- `start`: the configuration (transparency, classifier, matcher, restaurants), the initial state, the opening system action and response.
- `turn`: turn number, normalized utterance, predicted dialog act, state before and after, system action, system response and latency in milliseconds.
- `end`: why the conversation stopped (`finished`, `exit` or `interrupted`), the number of turns and the final state.

The system action has a `type` (e.g. `ask_area`, `confirm_value`, `recommend`), its `parameters` (for a recommendation: name, food, area and price range) and an `explanation`. The explanation is the recommendation evidence: for each requested requirement, the derived value, the rule that fired and any rules it overruled. The action is `null` when the user asked to repeat and the previous response was reused.

All records carry `session_id` and a UTC `timestamp`. 

Debug output is also written to `app.log`.

## Comparing the fallback matchers

Slot extraction always tries keyword matching first. Words it cannot match go to the fallback chosen with
`--matcher`: Levenshtein distance for typos, or DistilBERT similarity for paraphrases.

The test cases in `data/matcher_cases.csv` come from the DSTC2 dialogs: user turns where keyword matching
misses a value that the annotated speech act contains (e.g. "chineese" annotated as `food=chinese`), and
turns without any value, where the fallback should propose nothing. Rebuild them with:

```bash
uv run python scripts/build_matcher_cases.py
```

Compare both matchers on these cases with:

```bash
uv run python scripts/compare_matchers.py
```

This prints correct, wrong and missed matches per matcher and kind of case, trying DistilBERT at the
similarity thresholds 0.85, 0.9 and 0.95. Per-case results are saved to
`results/matchers/predictions.csv` and the summary to `results/matchers/summary.csv`.
