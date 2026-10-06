"""Render abstract system actions as template-based utterances.

TODO:
- Define templates for greetings, questions, confirmations, and clarification.
- Define templates for recommendations, details, alternatives, and no matches.
- Define templates for reasoning explanations and contradiction resolutions.
- Render actions using supplied slot values, restaurant details, and evidence.
- Support showing or hiding reasoning without changing recommendation logic.
- Keep state transitions, lookup, and inference outside this module.
"""


"""Turn system actions into sentences, using one template per action."""

TEMPLATES = {
    "welcome": "Hello, welcome to the Cambridge restaurant system. You can ask for restaurants by area, price range or food type. How may I help you?",
    "askarea": "What part of town do you have in mind?",
    "askpricerange": "Would you like something in the cheap, moderate, or expensive price range?",
    "askfood": "What kind of food would you like?",
    "confirmvalue": "I did not recognise {given}. Did you mean {corrected}?",
    "askrequirement": "Do you have any additional requirements, for example romantic or touristic?",
    "recommend": "{restaurantname} is a nice place in the {area} of town and the prices are {pricerange}.",
    "recommendreason": "I recommend {restaurantname}. It is {requirement} because it is {reason}.",
    "nomatch": "I'm sorry but there is no restaurant serving {food} food.",
    "noalternative": "I'm sorry but there are no other restaurants matching your preferences.",
    "detail": "The {detail} of {restaurantname} is {value}.",
    "detailunknown": "I'm sorry but I do not have the {detail} of {restaurantname}.",
    "repeat": "Could you please repeat that?",
    "unclear": "Sorry, I did not understand that. Could you say it in a different way?",
    "anythingelse": "Can I help you with anything else?",
    "bye": "Thank you for using our system. Goodbye!",
}


def render(action: str, **values) -> str:
    """Fill in the template of the given action with the given values."""
    if action not in TEMPLATES:
        raise ValueError(f"unknown action: {action}")

    return TEMPLATES[action].format(**values)


if __name__ == "__main__":
    restaurant = {
        "restaurantname": "the oak bistro",
        "pricerange": "moderate",
        "area": "centre",
        "food": "british",
        "phone": "01223 323361",
        "postcode": "",
    }

    print(render("welcome"))
    print(render("askfood"))
    print(render("confirmvalue", given="spenish", corrected="spanish"))
    print(render("recommend", **restaurant))
    print(render("recommendreason", restaurantname="zizzi cambridge", requirement="romantic", reason="a place where you can stay for a long time"))
    print(render("detail", detail="phone number", restaurantname=restaurant["restaurantname"], value=restaurant["phone"]))
    print(render("detailunknown", detail="postcode", restaurantname=restaurant["restaurantname"]))
    print(render("bye"))