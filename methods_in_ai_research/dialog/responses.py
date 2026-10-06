"""Turn system actions into sentences, using one template per action."""

from methods_in_ai_research.dialog.state import SystemAction, DialogState, SystemActionType


"""Turn system actions into sentences, using one template per action."""
TEMPLATES = dict({
    SystemActionType.WELCOME: "Hello, welcome to the Cambridge restaurant system. You can ask for restaurants by area, price range or food type. How may I help you?",
    SystemActionType.ASK_AREA: "What part of town do you have in mind?",
    SystemActionType.ASK_PRICE_RANGE: "Would you like something in the cheap, moderate, or expensive price range?",
    SystemActionType.ASK_FOOD: "What kind of food would you like?",
    SystemActionType.CONFIRM_VALUE: "I did not recognise {given}. Did you mean {corrected}?",
    SystemActionType.ASK_REQUIREMENT: "Do you have any additional requirements, for example romantic or touristic?",
    SystemActionType.RECOMMEND: "{restaurantname} is a nice place in the {area} of town and the prices are {pricerange}.",
    SystemActionType.RECOMMEND_REASON: "I recommend {restaurantname}. It is {requirement} because it is {reason}.",
    SystemActionType.NO_MATCH: "I'm sorry but there is no restaurant serving {food} food.",
    SystemActionType.NO_ALTERNATIVE: "I'm sorry but there are no other restaurants matching your preferences.",
    SystemActionType.DETAIL: "The {detail} of {restaurantname} is {value}.",
    SystemActionType.DETAIL_UNKNOWN: "I'm sorry but I do not have the {detail} of {restaurantname}.",
    SystemActionType.REPEAT: "Could you please repeat that?",
    SystemActionType.CLARIFY: "Sorry, I did not understand that. Could you say it in a different way?",
    SystemActionType.ANYTHING_ELSE: "Can I help you with anything else?",
    SystemActionType.BYE: "Thank you for using our system. Goodbye!",
})

class ResponseRenderer:
    def render(self, action: SystemAction, state: DialogState) -> str:
        """Fill in the template of the given action with the given values."""
        if action not in TEMPLATES:
           raise ValueError(f"unknown action: {action.type}")

        return TEMPLATES[action.type].format(**dict(action.parameters))


if __name__ == "__main__":
    from methods_in_ai_research.dialog.state import (
        DialogState,
        SystemAction,
        SystemActionType,
    )

    renderer = ResponseRenderer()
    state = DialogState(node=1)

    actions = [
        SystemAction(SystemActionType.WELCOME),
        SystemAction(SystemActionType.ASK_FOOD),
        SystemAction(
            SystemActionType.CONFIRM_VALUE,
            parameters=(
                ("given", "spenish"),
                ("corrected", "spanish"),
            ),
        ),
        SystemAction(
            SystemActionType.RECOMMEND,
            parameters=(
                ("restaurantname", "the oak bistro"),
                ("area", "centre"),
                ("pricerange", "moderate"),
            ),
        ),
        SystemAction(
            SystemActionType.DETAIL,
            parameters=(
                ("detail", "phone number"),
                ("restaurantname", "the oak bistro"),
                ("value", "01223 323361"),
            ),
        ),
        SystemAction(
            SystemActionType.DETAIL_UNKNOWN,
            parameters=(
                ("detail", "postcode"),
                ("restaurantname", "the oak bistro"),
            ),
        ),
        SystemAction(SystemActionType.BYE),
    ]

    for action in actions:
        print(renderer.render(action, state))