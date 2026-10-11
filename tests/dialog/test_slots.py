import pytest

from methods_in_ai_research.dialog.slots import KeywordSlotExtractor, LevenshteinMatcher, levenshtein
from methods_in_ai_research.dialog.reasoning import RestaurantProperty
from methods_in_ai_research.dialog.state import AnyValue, ClassifiedInput, DialogState, ExclusionProposal, RequirementProposal, Slot, \
    SlotProposal, UnknownProposal

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


@pytest.mark.parametrize(("text", "expected"), [
    ("romantic", {RequirementProposal(RestaurantProperty.ROMANTIC, True)}),
    ("I want a romantic place that is good for kids", {
        RequirementProposal(RestaurantProperty.ROMANTIC, True),
        RequirementProposal(RestaurantProperty.CHILDREN, True),
    }),
    ("it should not be too touristic", {RequirementProposal(RestaurantProperty.TOURISTIC, False)}),
    ("a place without assigned seats", {RequirementProposal(RestaurantProperty.ASSIGNED_SEATS, False)}),
    ("no, romantic please", {RequirementProposal(RestaurantProperty.ROMANTIC, True)}),
    ("no", set()),
])
def test_requirements(extractor: KeywordSlotExtractor, text: str, expected: set[RequirementProposal]):
    assert extract(extractor, text) == expected


def test_requirement_word_is_not_mistaken_for_food(extractor: KeywordSlotExtractor):
    # "romantic" is close to "romanian", but must not become a food preference.
    extractor = KeywordSlotExtractor({**VALUES, Slot.FOOD: VALUES[Slot.FOOD] | {"romanian"}}, LevenshteinMatcher())

    assert extract(extractor, "a romantic restaurant") == {RequirementProposal(RestaurantProperty.ROMANTIC, True)}


@pytest.mark.parametrize(("text", "expected"), [
    ("not italian", {ExclusionProposal(Slot.FOOD, "italian")}),
    ("i dont want spanish i want italian food", {ExclusionProposal(Slot.FOOD, "spanish"), SlotProposal(Slot.FOOD, "italian")}),
    ("i do not care about the price i want something in the north", {SlotProposal(Slot.AREA, "north")}),
    ("I don't want expensive food", {ExclusionProposal(Slot.PRICE, "expensive")}),
    ("not italian but spanish", {ExclusionProposal(Slot.FOOD, "italian"), SlotProposal(Slot.FOOD, "spanish")}),
    ("not in the north, somewhere cheap", {ExclusionProposal(Slot.AREA, "north"), SlotProposal(Slot.PRICE, "cheap")}),
    # In DSTC2, "no" corrects the system, so these still state a preference.
    ("no italian", {SlotProposal(Slot.FOOD, "italian")}),
    ("no i want cheap", {SlotProposal(Slot.PRICE, "cheap")}),
    ("no, swedish food", {SlotProposal(Slot.FOOD, "swedish")}),
])
def test_negated_values_are_excluded(extractor: KeywordSlotExtractor, text: str, expected: set):
    assert extract(extractor, text) == expected


def test_negated_typo_is_ignored(extractor: KeywordSlotExtractor):
    assert extract(extractor, "not spenish food") == set()


def test_negation_without_apostrophe(extractor: KeywordSlotExtractor):
    # The runner drops apostrophes to match the DSTC2 spelling the classifier was trained on.
    assert extract(extractor, "i dont want expensive food") == {ExclusionProposal(Slot.PRICE, "expensive")}
    assert extract(extractor, "i want expensive food") == {SlotProposal(Slot.PRICE, "expensive")}


def test_unknown_food_is_reported(extractor: KeywordSlotExtractor):
    assert extract(extractor, "i would like some japanese food") == {UnknownProposal(Slot.FOOD, "japanese")}
    # A typo close to a known food is corrected instead.
    assert extract(extractor, "i would like some spenish food") == {
        SlotProposal(Slot.FOOD, "spanish", needs_confirmation=True, given="spenish"),
    }
