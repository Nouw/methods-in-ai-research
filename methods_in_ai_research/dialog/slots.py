"""Extract preferences from user utterances.

Matching runs in three steps:
1. Keyword matching finds known values and synonyms anywhere, e.g. "italian" or "center".
2. Patterns such as "{word} food" capture unknown words, which a fallback matcher maps to
   the closest known value. Fallback matches are flagged for confirmation.
3. "Don't care" phrases set a slot to ANY: the slot they mention, or the slot just asked for.

TODO:
- Handle negation ("not chinese") and extract additional requirements.
"""
import re
from collections.abc import Collection, Mapping
from typing import Protocol

import numpy as np
import pandas as pd

from methods_in_ai_research.dialog.restaurants import PREFERENCE_COLUMNS
from methods_in_ai_research.dialog.state import AnyValue, ClassifiedInput, DialogState, RequirementProposal, Slot, SlotProposal

# Spelling variants mapped to their value in the restaurant data.
SYNONYMS = {
    Slot.AREA: {"center": "centre", "central": "centre", "downtown": "centre"},
    Slot.PRICE: {"moderately": "moderate"},
    Slot.FOOD: {"asian": "asian oriental", "oriental": "asian oriental", "american": "north american"},
}

# Each pattern captures one word that may hold a value for the slot. Order decides which
# slot gets a word that several patterns capture.
PATTERNS = (
    (Slot.FOOD, r"(\w+) (?:food|cuisine)"),
    (Slot.FOOD, r"serv(?:es|ing) (\w+)"),
    (Slot.AREA, r"in the (\w+)"),
    (Slot.AREA, r"(\w+) (?:part|side) of town"),
    (Slot.AREA, r"(\w+) area"),
    (Slot.PRICE, r"(\w+) (?:price|priced)"),
    (Slot.PRICE, r"(\w+) (?:restaurant|place)"),
    (Slot.FOOD, r"(\w+) (?:restaurant|place)"),
)

DONTCARE_PHRASES = ("any", "anything", "anywhere", "whatever", "don't care", "dont care", "doesn't matter", "does not matter", "no preference")

# Words that tie a "don't care" phrase to a slot, e.g. "any kind of food".
SLOT_CUES = {
    Slot.FOOD: ("food", "cuisine", "kind", "type"),
    Slot.AREA: ("area", "part", "where", "anywhere", "location"),
    Slot.PRICE: ("price", "prices", "priced", "cost"),
}

# Words that patterns capture but that never name a value.
STOPWORDS = {"a", "an", "the", "some", "that", "this", "what", "which", "good", "nice", "other", "another", "restaurant", "place"}
STOPWORDS |= {word for phrase in DONTCARE_PHRASES for word in phrase.split()}
STOPWORDS |= {cue for cues in SLOT_CUES.values() for cue in cues}


class SlotExtractor(Protocol):
    def extract(self, user_input: ClassifiedInput, state: DialogState) -> tuple[SlotProposal | RequirementProposal, ...]:
        """Extract preferences and additional requirements using the utterance and conversation context.

        Flag uncertain matches for confirmation. Do not modify state.
        """
        pass


class Matcher(Protocol):
    def closest(self, word: str, values: Collection[str]) -> str | None:
        """Return the value closest to the word, or None when nothing is close enough."""
        pass


def levenshtein(a: str, b: str) -> int:
    """Count the insertions, deletions and substitutions that turn a into b."""
    previous = list(range(len(b) + 1))

    for i, char_a in enumerate(a, 1):
        current = [i]

        for j, char_b in enumerate(b, 1):
            current.append(min(current[j - 1] + 1, previous[j] + 1, previous[j - 1] + (char_a != char_b)))

        previous = current

    return previous[-1]


class LevenshteinMatcher:
    """Map typos to the value with the smallest edit distance."""

    def __init__(self, max_distance: int = 2) -> None:
        self.max_distance = max_distance

    def closest(self, word: str, values: Collection[str]) -> str | None:
        # Sorting makes ties resolve the same way every run.
        distance, value = min((levenshtein(word, value), value) for value in sorted(values))

        return value if distance <= self.max_distance else None


class SemanticMatcher:
    """Map paraphrases to the value with the most similar DistilBERT embedding."""

    def __init__(self, encoder, threshold: float = 0.9) -> None:
        """Take a DistilBertEncoder and the minimum cosine similarity to accept a value."""
        self.encoder = encoder
        self.threshold = threshold
        self.cache: dict[str, np.ndarray] = {}

    def closest(self, word: str, values: Collection[str]) -> str | None:
        values = sorted(values)
        vectors = self._vectors([word, *values])
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        similarities = vectors[1:] @ vectors[0]
        best = int(np.argmax(similarities))

        return values[best] if similarities[best] >= self.threshold else None

    def _vectors(self, texts: list[str]) -> np.ndarray:
        """Encode texts once and reuse the cached embeddings afterwards."""
        missing = [text for text in dict.fromkeys(texts) if text not in self.cache]

        if missing:
            self.cache.update(zip(missing, self.encoder.encode(missing)))

        return np.stack([self.cache[text] for text in texts])


def values_from_restaurants(data: pd.DataFrame) -> dict[Slot, frozenset[str]]:
    """Collect the known values of each preference slot from the restaurant data."""
    return {slot: frozenset(value for value in data[column] if value) for slot, column in PREFERENCE_COLUMNS.items()}


class KeywordSlotExtractor:
    def __init__(self, values: Mapping[Slot, Collection[str]], matcher: Matcher) -> None:
        """Take the known values per slot and the fallback matcher for unknown words."""
        self.values = values
        self.matcher = matcher
        self.keywords = self._keywords()

    def extract(self, user_input: ClassifiedInput, state: DialogState) -> tuple[SlotProposal, ...]:
        """Propose at most one value per slot; fallback matches need confirmation."""
        text = user_input.text.lower().replace("’", "'")
        found, used = self._match_keywords(text)

        for slot, word in self._captured_words(text):
            if slot in found or not self._is_candidate(word, used):
                continue

            value = self.matcher.closest(word, self.values[slot])

            if value is not None:
                found[slot] = SlotProposal(slot, value, needs_confirmation=True, given=word)
                used.add(word)

        dontcare = [phrase for phrase in DONTCARE_PHRASES if re.search(rf"\b{phrase}\b", text)]

        if dontcare:
            for slot, cues in SLOT_CUES.items():
                if slot not in found and any(re.search(rf"\b{cue}\b", text) for cue in cues):
                    found[slot] = SlotProposal(slot, AnyValue.ANY)

        requested = state.last_requested_slot

        if requested is not None and requested not in found and user_input.act == "inform":
            # A short reply answers the question the system just asked.
            if dontcare:
                found[requested] = SlotProposal(requested, AnyValue.ANY)
            else:
                proposal = self._match_reply(text, requested, used)

                if proposal is not None:
                    found[requested] = proposal

        return tuple(found.values())

    def _keywords(self) -> list[tuple[str, Slot, str]]:
        """List (phrase, slot, value) for values and synonyms, longest phrase first."""
        keywords = [(value, slot, value) for slot, values in self.values.items() for value in values]
        keywords += [(phrase, slot, value) for slot, synonyms in SYNONYMS.items() for phrase, value in synonyms.items()]

        # Longest first, so "north american" is food rather than "north" being an area.
        return sorted(keywords, key=lambda keyword: len(keyword[0]), reverse=True)

    def _match_keywords(self, text: str) -> tuple[dict[Slot, SlotProposal], set[str]]:
        """Find known phrases and blank them out so shorter phrases cannot reuse them."""
        found = {}
        used = set()

        for phrase, slot, value in self.keywords:
            pattern = rf"\b{re.escape(phrase)}\b"

            if slot in found or not re.search(pattern, text):
                continue

            found[slot] = SlotProposal(slot, value)
            used.update(phrase.split())
            text = re.sub(pattern, " ", text)

        return found, used

    @staticmethod
    def _is_candidate(word: str, used: set[str]) -> bool:
        """Only send unused content words of three or more letters to the fallback matcher."""
        return len(word) >= 3 and word not in used and word not in STOPWORDS

    @staticmethod
    def _captured_words(text: str) -> list[tuple[Slot, str]]:
        """Return the words that patterns capture, in pattern order."""
        return [(slot, match.group(1)) for slot, pattern in PATTERNS for match in re.finditer(pattern, text)]

    def _match_reply(self, text: str, slot: Slot, used: set[str]) -> SlotProposal | None:
        """Map the first unused word that is close to a value of the requested slot."""
        for word in re.findall(r"[\w']+", text):
            if not self._is_candidate(word, used):
                continue

            value = self.matcher.closest(word, self.values[slot])

            if value is not None:
                return SlotProposal(slot, value, needs_confirmation=True, given=word)

        return None
