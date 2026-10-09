import pandas as pd

from methods_in_ai_research.dialog import flows
from methods_in_ai_research.dialog.manager import DialogManager
from methods_in_ai_research.dialog.responses import ResponseRenderer
from methods_in_ai_research.dialog.runner import run_conversation
from methods_in_ai_research.dialog.slots import KeywordSlotExtractor, LevenshteinMatcher, values_from_restaurants

RESTAURANTS = pd.DataFrame([
    {"restaurantname": "la tasca", "pricerange": "cheap", "area": "centre", "food": "spanish",
     "phone": "01223 000000", "addr": "1 main street", "postcode": "c.b 1",
     "food quality": "good", "crowdedness": "quiet", "length of stay": "short"},
])

# Dialog acts for the scripted utterances, standing in for a trained classifier.
ACTS = {
    "i want spenish food": "inform",
    "yes": "affirm",
    "in the centre": "inform",
    "i don't care": "inform",
    "no": "negate",
    "thank you, goodbye": "thankyou",
}


class ScriptedClassifier:
    def predict_one(self, utterance: str) -> str:
        return ACTS[utterance]


def test_scripted_conversation_reaches_recommendation():
    manager = DialogManager(
        flows.build_engine(RESTAURANTS),
        KeywordSlotExtractor(values_from_restaurants(RESTAURANTS), LevenshteinMatcher()),
        ResponseRenderer(),
    )
    inputs = iter(ACTS)
    outputs = []

    state = run_conversation(
        manager,
        ScriptedClassifier(),
        is_finished=lambda state: state.node == flows.BYE,
        input_function=lambda prompt: next(inputs),
        output_function=outputs.append,
    )

    assert outputs == [
        "system: Hello, welcome to the Cambridge restaurant system. You can ask for restaurants by area, price range or food type. How may I help you?",
        "system: I did not recognise spenish. Did you mean spanish?",
        "system: What part of town do you have in mind?",
        "system: Would you like something in the cheap, moderate, or expensive price range?",
        "system: Do you have any additional requirements, for example romantic or touristic?",
        "system: la tasca is a nice place in the centre of town and the prices are cheap.",
        "system: Thank you for using our system. Goodbye!",
    ]
    assert state.node == flows.BYE
