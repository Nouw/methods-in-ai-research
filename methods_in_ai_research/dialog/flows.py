"""Define the restaurant recommendation flow and build an engine for it.

Events are the dialog acts predicted by the classifier. Nodes with an action wait
for the user; nodes without one decide where to go next.

TODO:
- Answer information requests (phone, address, postcode) once the state tracks them.
"""
import pandas as pd

from methods_in_ai_research.dialog import actions
from methods_in_ai_research.dialog.engine import FlowEngine
from methods_in_ai_research.dialog.flow import Flow, Node, Transition

WELCOME = 1
CHECK_PREFERENCES = 2
ASK_FOOD = 3
ASK_AREA = 4
ASK_PRICE = 5
ASK_REQUIREMENTS = 6
RECOMMEND = 7
ALTERNATIVE = 8
BYE = 9

NODES = (
    Node(WELCOME, "Welcome", action="welcome"),
    Node(CHECK_PREFERENCES, "Check missing preferences"),
    Node(ASK_FOOD, "Ask food", action="ask_food"),
    Node(ASK_AREA, "Ask area", action="ask_area"),
    Node(ASK_PRICE, "Ask price range", action="ask_price"),
    Node(ASK_REQUIREMENTS, "Ask additional requirements", action="ask_requirements"),
    Node(RECOMMEND, "Recommend restaurant", action="recommend"),
    Node(ALTERNATIVE, "Recommend alternative", action="alternative"),
    Node(BYE, "Goodbye", action="bye"),
)

# Nodes that wait for the user.
WAITING = tuple(node.id for node in NODES if node.action is not None and node.id != BYE)

TRANSITIONS = (
    # Restarting or leaving works from anywhere.
    *(Transition(node, WELCOME, event="restart") for node in WAITING),
    *(Transition(node, BYE, event=event) for node in WAITING for event in ("bye", "thankyou")),

    # Preferences can be given at any question, and after a recommendation to change it.
    *(Transition(node, CHECK_PREFERENCES, event="inform")
      for node in (WELCOME, ASK_FOOD, ASK_AREA, ASK_PRICE, RECOMMEND, ALTERNATIVE)),

    Transition(CHECK_PREFERENCES, ASK_FOOD, guard="food_unknown"),
    Transition(CHECK_PREFERENCES, ASK_AREA, guard="area_unknown"),
    Transition(CHECK_PREFERENCES, ASK_PRICE, guard="price_unknown"),
    Transition(CHECK_PREFERENCES, ASK_REQUIREMENTS, guard="requirements_not_asked"),
    Transition(CHECK_PREFERENCES, RECOMMEND),

    # Any answer to the requirements question moves on to a recommendation.
    *(Transition(ASK_REQUIREMENTS, RECOMMEND, event=event) for event in ("inform", "affirm", "negate", "deny")),

    Transition(RECOMMEND, ALTERNATIVE, event="reqalts"),
    Transition(ALTERNATIVE, ALTERNATIVE, event="reqalts"),
)

FLOW = Flow(initial=WELCOME, nodes=NODES, transitions=TRANSITIONS)

GUARDS = {
    "always": lambda state: True,
    "food_unknown": lambda state: state.food is None,
    "area_unknown": lambda state: state.area is None,
    "price_unknown": lambda state: state.price is None,
    "requirements_not_asked": lambda state: not state.requirements_asked,
}


def build_engine(data: pd.DataFrame) -> FlowEngine:
    """Create the flow engine with actions bound to the restaurant data."""
    return FlowEngine(
        flow=FLOW,
        guards=GUARDS,
        actions={
            "welcome": actions.welcome,
            "ask_food": actions.ask_food,
            "ask_area": actions.ask_area,
            "ask_price": actions.ask_price,
            "ask_requirements": actions.ask_requirements,
            "recommend": lambda state: actions.recommend(state, data),
            "alternative": lambda state: actions.alternative(state, data),
            "bye": actions.bye,
        },
    )
