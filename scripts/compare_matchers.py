"""Compare the Levenshtein and DistilBERT fallbacks on the cases from build_matcher_cases.py.

Each case goes through the whole slot extractor, so keyword matching still runs first.
A case is correct if the expected value is found, or if nothing is found when no value was meant.

Usage: uv run python scripts/compare_matchers.py
"""
from pathlib import Path

import pandas as pd

from methods_in_ai_research.dialog.restaurants import load_restaurants
from methods_in_ai_research.dialog.slots import KeywordSlotExtractor, LevenshteinMatcher, SemanticMatcher, \
    values_from_restaurants
from methods_in_ai_research.dialog.state import ClassifiedInput, DialogState, Slot, SlotProposal
from methods_in_ai_research.embeddings import DistilBertEncoder

cases = pd.read_csv("data/matcher_cases.csv", keep_default_na=False)
values = values_from_restaurants(load_restaurants("data/restaurant_info_extended.csv"))
encoder = DistilBertEncoder()

# DistilBERT is tried with a few similarity thresholds, to see how much the threshold matters.
matchers = {
    "levenshtein": LevenshteinMatcher(),
    "distilbert@0.85": SemanticMatcher(encoder, threshold=0.85),
    "distilbert@0.9": SemanticMatcher(encoder, threshold=0.9),
    "distilbert@0.95": SemanticMatcher(encoder, threshold=0.95),
}

results = []

for name, matcher in matchers.items():
    extractor = KeywordSlotExtractor(values, matcher)

    for case in cases.itertuples():
        asked = Slot(case.requested) if case.requested else None
        proposals = extractor.extract(ClassifiedInput(act="inform", text=case.utterance),
                                      DialogState(node=1, last_requested_slot=asked))
        # Only real values count, not "don't care".
        found = {p.slot.value: p.value for p in proposals if isinstance(p, SlotProposal) and isinstance(p.value, str)}

        if not case.slot:
            outcome = "correct" if not found else "wrong"
        elif case.slot not in found:
            outcome = "missed"
        else:
            outcome = "correct" if found[case.slot] == case.expected else "wrong"

        results.append([name, case.kind, case.utterance, case.expected, found, outcome])

results = pd.DataFrame(results, columns=["matcher", "kind", "utterance", "expected", "found", "outcome"])
summary = pd.crosstab([results["matcher"], results["kind"]], results["outcome"])
summary["accuracy"] = (summary["correct"] / summary.sum(axis=1)).round(2)

Path("results/matchers").mkdir(parents=True, exist_ok=True)
results.to_csv("results/matchers/predictions.csv", index=False)
summary.to_csv("results/matchers/summary.csv")
print(summary)
