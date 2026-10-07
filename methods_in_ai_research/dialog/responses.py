"""Turn system actions into sentences, using one template per action."""
from methods_in_ai_research.dialog.reasoning import RestaurantProperty, Inference
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
    SystemActionType.NO_MATCH: "I'm sorry, but no restaurant matches your preferences.",
    SystemActionType.NO_ALTERNATIVE: "I'm sorry but there are no other restaurants matching your preferences.",
    SystemActionType.DETAIL: "The {detail} of {restaurantname} is {value}.",
    SystemActionType.DETAIL_UNKNOWN: "I'm sorry but I do not have the {detail} of {restaurantname}.",
    SystemActionType.REPEAT: "Could you please repeat that?",
    SystemActionType.CLARIFY: "Sorry, I did not understand that. Could you say it in a different way?",
    SystemActionType.ANYTHING_ELSE: "Can I help you with anything else?",
    SystemActionType.BYE: "Thank you for using our system. Goodbye!",
})

PROPERTY_CLAUSES = {
    RestaurantProperty.TOURISTIC: {
        True: 'it is touristic',
        False: 'it is not touristic',
    },
    RestaurantProperty.ASSIGNED_SEATS: {
        True: 'it has assigned seats',
        False: 'it does not have assigned seats',
    },
    RestaurantProperty.CHILDREN: {
        True: 'It is suitable for children',
        False: 'It is not suitable for children',
    },
    RestaurantProperty.ROMANTIC: {
        True: 'it is romantic',
        False: 'it is not romantic',
    }
}

RULE_REASONS = {
    1: 'it is cheap and has good food',
    2: 'it serves Romanian food',
    3: 'it is busy',
    4: 'it allows a long stay',
    5: 'it is busy',
    6: 'it allows a long stay',
}

class ResponseRenderer:
    def render(self, action: SystemAction, state: DialogState) -> str:
        if action.type not in TEMPLATES:
            raise ValueError(f"unknown action: {action.type}")

        text = TEMPLATES[action.type].format(**dict(action.parameters))
        explanations = " ".join(self._explain(inference) for inference in action.explanation)

        return f"{text} {explanations}" if explanations else text

    def _explain(self, inference: Inference) -> str:
        clause = PROPERTY_CLAUSES[inference.property][inference.value].capitalize()

        if inference.rule is None:
            return f"{clause}, as nothing suggests otherwise."

        sentence = f"{clause} because {RULE_REASONS[inference.rule.id]}"

        if inference.overruled:
            sentence += f", even though {' and '.join(RULE_REASONS[rule.id] for rule in inference.overruled)}"

        return f"{sentence}."



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