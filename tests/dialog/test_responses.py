import pytest

from methods_in_ai_research.dialog.reasoning import RULES, Inference, RestaurantProperty
from methods_in_ai_research.dialog.responses import ResponseRenderer
from methods_in_ai_research.dialog.state import DialogState, SystemAction, SystemActionType

RECOMMENDATION = SystemAction(
    SystemActionType.RECOMMEND,
    parameters=(("restaurantname", "zizzi cambridge"), ("description", "cheap italian restaurant in the centre")),
    explanation=(Inference(RestaurantProperty.ROMANTIC, True, rule=RULES[5]),),
)


@pytest.mark.parametrize(("transparent", "expected"), [
    (True, "zizzi cambridge is a nice cheap italian restaurant in the centre. "
           "It is romantic because it allows a long stay."),
    (False, "zizzi cambridge is a nice cheap italian restaurant in the centre."),
])
def test_transparency_shows_or_hides_reasoning(transparent: bool, expected: str):
    assert ResponseRenderer(transparent).render(RECOMMENDATION, DialogState(node=1)) == expected
