"""Build test cases for the fallback matchers from the DSTC2 dialogs.

We keep the user turns where keyword matching does not find the value from the annotation
(e.g. "chineese food" annotated as inform(food=chinese)), because those reach the fallback.
Turns without any value are kept as negative cases: there the fallback should find nothing.

Usage: uv run python scripts/build_matcher_cases.py
"""
import re

import pandas as pd

from methods_in_ai_research.classification.processing import normalize_utterance
from methods_in_ai_research.dialog.restaurants import load_restaurants
from methods_in_ai_research.dialog.slots import KeywordSlotExtractor, LevenshteinMatcher, values_from_restaurants
from methods_in_ai_research.dialog.state import ClassifiedInput, DialogState, Slot, SlotProposal

SLOT_NAMES = {"food": Slot.FOOD, "area": Slot.AREA, "pricerange": Slot.PRICE}
QUESTIONS = {"kind of food": Slot.FOOD, "part of town": Slot.AREA, "price range": Slot.PRICE}

values = values_from_restaurants(load_restaurants("data/restaurant_info_extended.csv"))
# A maximum distance of 0 only accepts exact matches, so this extractor does keyword matching only.
keyword_extractor = KeywordSlotExtractor(values, LevenshteinMatcher(max_distance=0))

cases = []
system = ""
user = ""

for line in open("data/all_dialogs.txt", encoding="utf-8"):
    if line.startswith("system: "):
        system = line[len("system: "):].lower()
    elif line.startswith("user: "):
        user = normalize_utterance(line[len("user: "):])
    elif line.startswith("speech act: ") and "inform(" in line and user:
        # The slot the system just asked for, so short answers are handled like in the dialog.
        asked = next((slot for question, slot in QUESTIONS.items() if question in system), None)

        # The values the user meant, e.g. inform(food=chinese) gives {Slot.FOOD: "chinese"}.
        expected = {}

        for inform in re.findall(r"inform\((.*?)\)", line):
            for name, value in re.findall(r"(\w+)=([\w ]+)", inform):
                if name in SLOT_NAMES and value in values[SLOT_NAMES[name]]:
                    expected[SLOT_NAMES[name]] = value

        state = DialogState(node=1, last_requested_slot=asked)
        proposals = keyword_extractor.extract(ClassifiedInput(act="inform", text=user), state)
        found = {p.slot: p.value for p in proposals if isinstance(p, SlotProposal)}
        asked_name = asked.value if asked else ""

        for slot, value in expected.items():
            if found.get(slot) != value:
                cases.append(["missed by keywords", user, asked_name, slot.value, value])

        # "dontcare" is not a value, so a turn like "i dont care" is a negative case.
        has_value = "=" in line.replace("=dontcare", "")

        if not expected and not found and not has_value:
            cases.append(["no value", user, asked_name, "", ""])

table = pd.DataFrame(cases, columns=["kind", "utterance", "requested", "slot", "expected"]).drop_duplicates()
table.to_csv("data/matcher_cases.csv", index=False)
print(table["kind"].value_counts())
