"""Extract preferences from user utterances.

Matching runs in three steps:
1. Keyword matching finds known values and synonyms anywhere, e.g. "italian" or "center".
2. Patterns such as "{word} food" capture unknown words, which a fallback matcher maps to
   the closest known value. Fallback matches are flagged for confirmation.
3. "Don't care" phrases set a slot to ANY: the slot they mention, or the slot just asked for.

Additional requirements (romantic, touristic, ...) are found by keyword as well.

A negation word flips the meaning of the first value after it: "not chinese" excludes chinese
food, and "not touristic" asks for the property to be false.
"""
import re
from collections.abc import Collection, Mapping
from typing import Protocol

import numpy as np
import pandas as pd

from methods_in_ai_research.dialog.reasoning import RestaurantProperty
from methods_in_ai_research.dialog.restaurants import PREFERENCE_COLUMNS
from methods_in_ai_research.dialog.state import AnyValue, ClassifiedInput, DialogState, ExclusionProposal, \
    RequirementProposal, Slot, SlotProposal, UnknownProposal

Proposal = SlotProposal | ExclusionProposal | RequirementProposal | UnknownProposal

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

# Phrases that name an additional requirement.
REQUIREMENT_PHRASES = {
    RestaurantProperty.TOURISTIC: ("touristic", "touristy", "tourists", "tourist"),
    RestaurantProperty.ASSIGNED_SEATS: ("assigned seats", "assigned seating", "assigned seat"),
    RestaurantProperty.CHILDREN: ("child friendly", "children", "child", "kids", "kid"),
    RestaurantProperty.ROMANTIC: ("romantic", "romance"),
}

# Words that negate the first value after them. "no" is left out: in DSTC2, "no indian food" corrects
# the system rather than excluding indian food.
NEGATIONS = {"not", "never", "without", "dont", "doesnt", "isnt"}

DONTCARE_PHRASES = ("any", "anything", "anywhere", "whatever", "dont care", "doesnt matter", "does not matter", "no preference")

# Words that tie a "don't care" phrase to a slot, e.g. "any kind of food".
SLOT_CUES = {
    Slot.FOOD: ("food", "cuisine", "kind", "type"),
    Slot.AREA: ("area", "part", "where", "anywhere", "location"),
    Slot.PRICE: ("price", "prices", "priced", "cost"),
}

# Words that patterns capture but that never name a value.
STOPWORDS = {"a", "an", "the", "some", "that", "this", "what", "which", "good", "nice", "other", "another", "restaurant", "place",
             "for", "eat"}
STOPWORDS |= {word for phrase in DONTCARE_PHRASES for word in phrase.split()}
STOPWORDS |= {cue for cues in SLOT_CUES.values() for cue in cues}


def _negated_starts(text: str, value_starts: Collection[int]) -> set[int]:
    """Return the positions of the values that are negated.

    A negation word negates the first value after it, so "dont want chinese i want thai" excludes
    chinese but asks for thai. "dont care" and "doesnt matter" are not negations.
    """
    negated = set()

    for match in re.finditer(r"\w+", text):
        is_negation = match.group() in NEGATIONS and not text[match.end():].startswith((" care", " matter"))
        later = [start for start in value_starts if start > match.start()]

        if is_negation and later:
            negated.add(min(later))

    return negated


class SlotExtractor(Protocol):
    def extract(self, user_input: ClassifiedInput, state: DialogState) -> tuple[Proposal, ...]:
        """Extract preferences, exclusions and additional requirements using the utterance and conversation context.

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

    def extract(self, user_input: ClassifiedInput, state: DialogState) -> tuple[Proposal, ...]:
        """Propose at most one value per slot and per requirement; fallback matches need confirmation."""
        text = user_input.text.lower().replace("'", "").replace("’", "")
        negated = _negated_starts(text, self._value_starts(text))
        requirements, requirement_words = self._match_requirements(text, negated)
        found, exclusions, used = self._match_keywords(text, negated)
        used |= requirement_words

        unknown = []

        for slot, word, start in self._captured_words(text):
            if slot in found or not self._is_candidate(word, used) or start in negated:
                continue

            value = self.matcher.closest(word, self.values[slot])

            if value is not None:
                found[slot] = SlotProposal(slot, value, needs_confirmation=True, given=word)
                used.add(word)
            elif text[start + len(word):].startswith((" food", " cuisine")):
                # Only "{word} food" surely names a food, so only then is an unmatched word reported.
                unknown.append(UnknownProposal(slot, word))

        dontcare = [phrase for phrase in DONTCARE_PHRASES if re.search(rf"\b{phrase}\b", text)]

        if dontcare:
            for slot, cues in SLOT_CUES.items():
                if slot not in found and any(re.search(rf"\b{cue}\b", text) for cue in cues):
                    found[slot] = SlotProposal(slot, AnyValue.ANY)

        requested = state.last_requested_slot

        if requested is not None and requested not in found and user_input.act == "inform":
            if dontcare:
                found[requested] = SlotProposal(requested, AnyValue.ANY)
            else:
                proposal = self._match_reply(text, requested, used, negated)

                if proposal is not None:
                    found[requested] = proposal

        unknown = [proposal for proposal in unknown if proposal.slot not in found]

        return tuple(found.values()) + exclusions + requirements + tuple(unknown)

    def _value_starts(self, text: str) -> list[int]:
        """Return the positions of all words and phrases that may name a value or requirement."""
        phrases = [phrase for phrase, _, _ in self.keywords]
        phrases += [phrase for phrases_ in REQUIREMENT_PHRASES.values() for phrase in phrases_]
        starts = [match.start() for phrase in phrases for match in re.finditer(rf"\b{re.escape(phrase)}\b", text)]

        return starts + [start for _, word, start in self._captured_words(text) if word not in STOPWORDS]

    @staticmethod
    def _match_requirements(text: str, negated: set[int]) -> tuple[tuple[RequirementProposal, ...], set[str]]:
        """Find requirement phrases and whether each is negated, at most one per property."""
        proposals = []
        used = set()

        for property_, phrases in REQUIREMENT_PHRASES.items():
            for phrase in phrases:
                match = re.search(rf"\b{re.escape(phrase)}\b", text)

                if match is None:
                    continue

                proposals.append(RequirementProposal(property_, match.start() not in negated))
                used.update(phrase.split())
                break

        return tuple(proposals), used

    def _keywords(self) -> list[tuple[str, Slot, str]]:
        """List (phrase, slot, value) for values and synonyms, longest phrase first."""
        keywords = [(value, slot, value) for slot, values in self.values.items() for value in values]
        keywords += [(phrase, slot, value) for slot, synonyms in SYNONYMS.items() for phrase, value in synonyms.items()]

        return sorted(keywords, key=lambda keyword: len(keyword[0]), reverse=True)

    def _match_keywords(self, text: str,
                        negated: set[int]) -> tuple[dict[Slot, SlotProposal], tuple[ExclusionProposal, ...], set[str]]:
        """Find known phrases and blank them out so shorter phrases cannot reuse them.

        Negated phrases become exclusions, so "not chinese but italian" gives both.
        """
        found = {}
        exclusions = []
        used = set()

        for phrase, slot, value in self.keywords:
            pattern = rf"\b{re.escape(phrase)}\b"

            for match in re.finditer(pattern, text):
                if match.start() in negated:
                    exclusions.append(ExclusionProposal(slot, value))
                elif slot not in found:
                    found[slot] = SlotProposal(slot, value)
                else:
                    continue

                used.update(phrase.split())

            text = re.sub(pattern, lambda match: " " * len(match.group()), text)

        return found, tuple(dict.fromkeys(exclusions)), used

    @staticmethod
    def _is_candidate(word: str, used: set[str]) -> bool:
        """Only send unused content words of three or more letters to the fallback matcher."""
        return len(word) >= 3 and word not in used and word not in STOPWORDS

    @staticmethod
    def _captured_words(text: str) -> list[tuple[Slot, str, int]]:
        """Return the words that patterns capture with their position, in pattern order."""
        return [(slot, match.group(1), match.start(1)) for slot, pattern in PATTERNS for match in re.finditer(pattern, text)]

    def _match_reply(self, text: str, slot: Slot, used: set[str], negated: set[int]) -> SlotProposal | None:
        """Map the first unused word that is close to a value of the requested slot."""
        for match in re.finditer(r"[\w']+", text):
            word = match.group()

            if not self._is_candidate(word, used) or match.start() in negated:
                continue

            value = self.matcher.closest(word, self.values[slot])

            if value is not None:
                return SlotProposal(slot, value, needs_confirmation=True, given=word)

        return None
