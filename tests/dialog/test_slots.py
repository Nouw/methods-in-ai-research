import pytest

from methods_in_ai_research.dialog.slots import KeywordSlotExtractor, LevenshteinMatcher, levenshtein
from methods_in_ai_research.dialog.state import AnyValue, ClassifiedInput, DialogState, Slot, SlotProposal

VALUES = {
    Slot.FOOD: frozenset({"italian", "spanish", "seafood", "north american", "swedish", "danish", "polish"}),
    Slot.AREA: frozenset({"centre", "north", "south", "east", "west"}),
    Slot.PRICE: frozenset({"cheap", "moderate", "expensive"}),
}


@pytest.fixture
def extractor() -> KeywordSlotExtractor:
    return KeywordSlotExtractor(VALUES, LevenshteinMatcher())


def extract(extractor: KeywordSlotExtractor, text: str, requested: Slot | None = None) -> set[SlotProposal]:
    state = DialogState(node=1, last_requested_slot=requested)
    return set(extractor.extract(ClassifiedInput(act="inform", text=text), state))


def test_levenshtein_distance():
    assert levenshtein("spenish", "spanish") == 1
    assert levenshtein("spenish", "swedish") == 2
    assert levenshtein("", "abc") == 3


def test_keywords_match_several_slots(extractor: KeywordSlotExtractor):
    assert extract(extractor, "I want a cheap restaurant in the south part of town") == {
        SlotProposal(Slot.PRICE, "cheap"),
        SlotProposal(Slot.AREA, "south"),
    }


def test_longest_keyword_wins(extractor: KeywordSlotExtractor):
    assert extract(extractor, "north american food in the center") == {
        SlotProposal(Slot.FOOD, "north american"),
        SlotProposal(Slot.AREA, "centre"),
    }


def test_typo_needs_confirmation(extractor: KeywordSlotExtractor):
    assert extract(extractor, "I want a restaurant serving Spenish food") == {
        SlotProposal(Slot.FOOD, "spanish", needs_confirmation=True),
    }


def test_dontcare_binds_to_mentioned_slot(extractor: KeywordSlotExtractor):
    assert extract(extractor, "any food in the west") == {
        SlotProposal(Slot.FOOD, AnyValue.ANY),
        SlotProposal(Slot.AREA, "west"),
    }
    assert extract(extractor, "I don't care about the price") == {SlotProposal(Slot.PRICE, AnyValue.ANY)}


def test_short_reply_answers_requested_slot(extractor: KeywordSlotExtractor):
    assert extract(extractor, "any", requested=Slot.AREA) == {SlotProposal(Slot.AREA, AnyValue.ANY)}
    assert extract(extractor, "spenish please", requested=Slot.FOOD) == {
        SlotProposal(Slot.FOOD, "spanish", needs_confirmation=True),
    }


def test_unrelated_words_are_ignored(extractor: KeywordSlotExtractor):
    assert extract(extractor, "what is the phone number") == set()
    assert extract(extractor, "it is fine", requested=Slot.AREA) == set()
