"""Flow entry actions that combine lookup, reasoning and dialog state.

Each action returns the updated state and the system action to render.
Actions that need the restaurant data take it as a second argument, so the
flow binds it when registering them, for example:

    "recommend": lambda state: recommend(state, data)
"""
import random
from dataclasses import replace

import pandas as pd

from methods_in_ai_research.dialog.reasoning import infer, satisfies
from methods_in_ai_research.dialog.restaurants import find_restaurants, restaurant_details
from methods_in_ai_research.dialog.state import DialogState, Slot, SystemAction, SystemActionType


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


def recommend(state: DialogState, data: pd.DataFrame) -> tuple[DialogState, SystemAction]:
    """Find all matching restaurants, shuffle them, and recommend the first one."""
    preferences = {Slot.FOOD: state.food, Slot.AREA: state.area, Slot.PRICE: state.price}
    requirements = dict(state.requirements)

    names = []

    for restaurant in find_restaurants(data, preferences).to_dict(orient="records"):
        if satisfies(infer(restaurant), requirements):
            names.append(restaurant["restaurantname"])

    if not names:
        return replace(state, current_restaurant_id=None, alternative_ids=()), SystemAction(SystemActionType.NO_MATCH)

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
        parameters=tuple((field, restaurant[field]) for field in ("restaurantname", "food", "area", "pricerange")),
        explanation=tuple(inferences[property_] for property_, _ in state.requirements),
    )

    return replace(state, current_restaurant_id=name, alternative_ids=state.alternative_ids[1:]), action
