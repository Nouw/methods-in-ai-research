"""Flow entry actions that combine lookup, reasoning and dialog state.

Each action returns the updated state and the system action to render.
Actions that need the restaurant data take it as a second argument, so the
flow binds it when registering them, for example:

    "recommend": lambda state: recommend(state, data)
"""
import random
from dataclasses import replace

import pandas as pd
import difflib
from methods_in_ai_research.dialog.reasoning import infer, satisfies
from methods_in_ai_research.dialog.restaurants import PREFERENCE_COLUMNS, find_restaurants, restaurant_details
from methods_in_ai_research.dialog.state import DialogState, Preference, Slot, SystemAction, SystemActionType
import re


DETAILS = (
    ("phone", "phone number", ("phone", "telephone", "number")),
    ("addr", "address", ("address", "street", "located", "where")),
    ("postcode", "post code", ("post code", "postcode", "postal code", "zip")),
)




def welcome(state: DialogState) -> tuple[DialogState, SystemAction]:
    """Start a fresh conversation, which also makes restarting forget all preferences."""
    return DialogState(node=state.node), SystemAction(SystemActionType.WELCOME)

def bye(state: DialogState) -> tuple[DialogState, SystemAction]:
    return state, SystemAction(SystemActionType.BYE)

def ask_food(state: DialogState) -> tuple[DialogState, SystemAction]:
    return replace(state, last_requested_slot=Slot.FOOD), SystemAction(SystemActionType.ASK_FOOD)

def ask_area(state: DialogState) -> tuple[DialogState, SystemAction]:
    return replace(state, last_requested_slot=Slot.AREA), SystemAction(SystemActionType.ASK_AREA)

def ask_price(state: DialogState) -> tuple[DialogState, SystemAction]:
    return replace(state, last_requested_slot=Slot.PRICE), SystemAction(SystemActionType.ASK_PRICE_RANGE)

def ask_requirements(state: DialogState) -> tuple[DialogState, SystemAction]:
    return replace(state, requirements_asked=True, last_requested_slot=None), SystemAction(SystemActionType.ASK_REQUIREMENT)


def describe(price: Preference, food: Preference, area: Preference) -> str:
    """Describe a restaurant or the wanted one, e.g. "cheap italian restaurant in the centre".

    Unknown and "don't care" values are left out, as are missing areas in the data.
    """
    words = [value for value in (price, food) if _is_known(value)] + ["restaurant"]
    description = " ".join(words)

    return f"{description} in the {area}" if _is_known(area) else description

def _is_known(value: Preference) -> bool:
    return isinstance(value, str) and value != ""

def has_matches(state: DialogState, data: pd.DataFrame) -> bool:
    """Check whether any restaurant fits the preferences, before asking for requirements."""
    preferences = {Slot.FOOD: state.food, Slot.AREA: state.area, Slot.PRICE: state.price}
    return not find_restaurants(data, preferences, state.excluded).empty


def recommend(state: DialogState, data: pd.DataFrame) -> tuple[DialogState, SystemAction]:
    """Find all matching restaurants, shuffle them, and recommend the first one."""
    preferences = {Slot.FOOD: state.food, Slot.AREA: state.area, Slot.PRICE: state.price}
    requirements = dict(state.requirements)

    names = []

    for restaurant in find_restaurants(data, preferences, state.excluded).to_dict(orient="records"):
        if satisfies(infer(restaurant), requirements):
            names.append(restaurant["restaurantname"])

    if not names:
        # Tell apart "nothing fits the preferences" from "nothing fits the additional requirements".
        no_match = SystemActionType.NO_MATCH_REQUIREMENTS if has_matches(state, data) else SystemActionType.NO_MATCH
        description = describe(state.price, state.food, state.area)
        article = "an" if description[0] in "aeiou" else "a"
        action = SystemAction(no_match, parameters=(("description", f"{article} {description}"),))

        return replace(state, current_restaurant_id=None, alternative_ids=()), action

    random.shuffle(names)

    return _recommend_next(replace(state, alternative_ids=tuple(names)), data)

def alternative(state: DialogState, data: pd.DataFrame) -> tuple[DialogState, SystemAction]:
    """Recommend the next stored match for the same preferences."""
    if not state.alternative_ids:
        return state, SystemAction(SystemActionType.NO_ALTERNATIVE)

    return _recommend_next(state, data)

def _recommend_next(state: DialogState, data: pd.DataFrame) -> tuple[DialogState, SystemAction]:
    """Recommend the first stored match and explain the requested properties."""
    name = state.alternative_ids[0]
    restaurant = restaurant_details(data, name)
    inferences = infer(restaurant)

    action = SystemAction(
        SystemActionType.RECOMMEND,
        parameters=(
            ("restaurantname", restaurant["restaurantname"]),
            ("description", describe(restaurant["pricerange"], restaurant["food"], restaurant["area"])),
        ),
        explanation=tuple(inferences[property_] for property_, _ in state.requirements),
    )

    return replace(state, current_restaurant_id=name, alternative_ids=state.alternative_ids[1:]), action

def check_fact(state: DialogState, data: pd.DataFrame) -> tuple[DialogState, SystemAction]:
    """Answer a yes/no question about the current restaurant, e.g. "is it in the centre?"."""
    if state.current_restaurant_id is None or not state.checked:
        return state, SystemAction(SystemActionType.CLARIFY)

    restaurant = restaurant_details(data, state.current_restaurant_id)
    labels = {Slot.FOOD: "food", Slot.AREA: "area", Slot.PRICE: "price range"}

    for slot, value in state.checked:
        actual = restaurant[PREFERENCE_COLUMNS[slot]]

        if actual != value:
            return state, SystemAction(SystemActionType.FACT_FALSE, parameters=(("field", labels[slot]), ("value", actual)))

    return state, SystemAction(SystemActionType.FACT_TRUE)

def requested_detail(text: str) -> str | None:
    """Look for which piece of information the user asked for"""
    for column, _, phrases in DETAILS:
        if any(re.search(rf"\b{phrase}\b", text) for phrase in phrases):
            return column

    # People misspell things like "adress", so also allow close matches on single words.
    words = re.findall(r"\w+", text)
    for column, _, phrases in DETAILS:
        single = [phrase for phrase in phrases if " " not in phrase]
        if any(difflib.get_close_matches(word, single, n=1, cutoff=0.85) for word in words):
            return column
    return None


def detail(state: DialogState, data: pd.DataFrame) -> tuple[DialogState, SystemAction]:
    labels = {column: label for column, label, _ in DETAILS}
    if state.current_restaurant_id is None or state.requested_detail not in labels:
        return state, SystemAction(SystemActionType.CLARIFY)

    restaurant = restaurant_details(data, state.current_restaurant_id)
    parameters = (
        ("detail", labels[state.requested_detail]),
        ("restaurantname", restaurant["restaurantname"]),
    )
    value = restaurant[state.requested_detail]
    if not value:
        return state, SystemAction(SystemActionType.DETAIL_UNKNOWN, parameters=parameters)
    return state, SystemAction(SystemActionType.DETAIL, parameters=parameters + (("value", value),))