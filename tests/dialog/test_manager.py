from dataclasses import replace

import pytest

from methods_in_ai_research.dialog.engine import FlowEngine
from methods_in_ai_research.dialog.flow import Flow, Node, Transition
from methods_in_ai_research.dialog.manager import DialogManager
from methods_in_ai_research.dialog.responses import ResponseRenderer, TEMPLATES
from methods_in_ai_research.dialog.state import DialogState, SystemAction, Slot, ClassifiedInput, SlotProposal, \
    SystemActionType


def welcome(state: DialogState) -> tuple[DialogState, SystemAction]:
    return state, SystemAction(SystemActionType.WELCOME)

def ask_food(state: DialogState) -> tuple[DialogState, SystemAction]:
    return replace(state, last_requested_slot=Slot.FOOD), SystemAction(SystemActionType.ASK_FOOD)

def ask_area(state: DialogState) -> tuple[DialogState, SystemAction]:
    return replace(state, last_requested_slot=Slot.AREA), SystemAction(SystemActionType.ASK_AREA)

def complete(state: DialogState) -> tuple[DialogState, SystemAction]:
    return replace(state, last_requested_slot=None), SystemAction(SystemActionType.BYE)

# Example from assignment description
flow = Flow(
    initial=1,
    nodes=(
        Node(1, "Welcome", action="welcome"),
        Node(2, "Check missing preferences"),
        Node(3, "Ask food", action="ask_food"),
        Node(4, "Ask area", action="ask_area"),
        Node(5, "Example complete", action="complete"),
    ),
    transitions=(
        Transition(1, 2, event="inform"),
        Transition(3, 2, event="inform"),
        Transition(4, 2, event="inform"),
        Transition(5, 2, event="inform"),
        Transition(2, 3, guard="food_unknown"),
        Transition(2, 4, guard="area_unknown"),
        Transition(2, 5),
    ),
)

ASK_FOOD = "What kind of food would you like?"
ASK_AREA = "What part of town do you have in mind?"
BYE = "Thank you for using our system. Goodbye!"
CONFIRM_SPANISH = "I did not recognise spenish. Did you mean spanish?"

class FakeExtractor:
    def extract(self, user_input: ClassifiedInput, state: DialogState) -> tuple[SlotProposal, ...]:
        examples = {
            "Chinese food": (SlotProposal(Slot.FOOD, "chinese"),),
            "north": (SlotProposal(Slot.AREA, "north"),),
            "Spenish food": (SlotProposal(Slot.FOOD, "spanish", needs_confirmation=True, given="spenish"),),
            "No, Swedish food": (SlotProposal(Slot.FOOD, "swedish"),),
        }
        return examples.get(user_input.text, ())

@pytest.fixture
def manager() -> DialogManager:
    """Build a manager using the example flow and fake components"""
    engine = FlowEngine(
        flow=flow,
        guards={
            "always": lambda state: True,
            "food_unknown": lambda state: state.food is None,
            "area_unknown": lambda state: state.area is None,
        },
        actions={
            "welcome": welcome,
            "ask_food": ask_food,
            "ask_area": ask_area,
            "complete": complete,
        }
    )

    return DialogManager(engine, FakeExtractor(), ResponseRenderer())

def test_supplied_food_skips_food_question(manager: DialogManager):
    initial_state, _ = manager.start()

    state, response = manager.transition(
        initial_state,
        ClassifiedInput(act="inform", text="Chinese food"),
    )

    assert state.food == "chinese"
    assert state.last_requested_slot == Slot.AREA
    assert response == ASK_AREA
    assert initial_state == DialogState(node=1, last_response=TEMPLATES[SystemActionType.WELCOME])


def test_area_completes_example_without_losing_food(manager: DialogManager):
    state, _ = manager.start()
    state, _ = manager.transition(
        state,
        ClassifiedInput(act="inform", text="Chinese food"),
    )

    state, response = manager.transition(
        state,
        ClassifiedInput(act="inform", text="north"),
    )

    assert state.food == "chinese"
    assert state.area == "north"
    assert state.last_requested_slot is None
    assert response == BYE


def test_repeat_preserves_current_question(manager: DialogManager):
    state, _ = manager.start()
    state, previous_response = manager.transition(
        state,
        ClassifiedInput(act="inform", text="Chinese food"),
    )

    next_state, response = manager.transition(
        state,
        ClassifiedInput(act="repeat", text="Please repeat"),
    )

    assert next_state == state
    assert response == previous_response


def test_uncertain_value_is_confirmed_before_use(manager: DialogManager):
    state, _ = manager.start()

    state, response = manager.transition(state, ClassifiedInput(act="inform", text="Spenish food"))

    assert state.food is None
    assert state.pending == (SlotProposal(Slot.FOOD, "spanish", needs_confirmation=True),)
    assert response == CONFIRM_SPANISH


def test_affirm_commits_value_and_continues(manager: DialogManager):
    state, _ = manager.start()
    state, _ = manager.transition(state, ClassifiedInput(act="inform", text="Spenish food"))

    state, response = manager.transition(state, ClassifiedInput(act="affirm", text="yes"))

    assert state.food == "spanish"
    assert state.pending == ()
    assert response == ASK_AREA


def test_negate_drops_value_and_asks_again(manager: DialogManager):
    state, _ = manager.start()
    state, _ = manager.transition(state, ClassifiedInput(act="inform", text="Spenish food"))

    state, response = manager.transition(state, ClassifiedInput(act="negate", text="no"))

    assert state.food is None
    assert state.pending == ()
    assert response == ASK_FOOD


def test_negate_with_correction_uses_new_value(manager: DialogManager):
    state, _ = manager.start()
    state, _ = manager.transition(state, ClassifiedInput(act="inform", text="Spenish food"))

    state, response = manager.transition(state, ClassifiedInput(act="negate", text="No, Swedish food"))

    assert state.food == "swedish"
    assert response == ASK_AREA


def test_repeat_during_confirmation_repeats_question(manager: DialogManager):
    state, _ = manager.start()
    state, _ = manager.transition(state, ClassifiedInput(act="inform", text="Spenish food"))

    next_state, response = manager.transition(state, ClassifiedInput(act="repeat", text="what?"))

    assert next_state == state
    assert response == CONFIRM_SPANISH
