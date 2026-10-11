import json

import pandas as pd

from methods_in_ai_research.classification.processing import normalize_utterance

from methods_in_ai_research.dialog import flows
from methods_in_ai_research.dialog.manager import DialogManager
from methods_in_ai_research.dialog.responses import ResponseRenderer
from methods_in_ai_research.dialog.runner import run_conversation
from methods_in_ai_research.dialog.reasoning import RULES, Inference, RestaurantProperty
from methods_in_ai_research.dialog.session_log import SessionLog
from methods_in_ai_research.dialog.state import DialogState, Slot, SystemAction, SystemActionType
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
        # The runner normalizes input first, which drops the apostrophe in "don't".
        return {normalize_utterance(text): act for text, act in ACTS.items()}[utterance]


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
        "system: la tasca is a nice cheap spanish restaurant in the centre.",
        "system: Thank you for using our system. Goodbye!",
    ]
    assert state.node == flows.BYE


def test_session_log_records_every_turn(tmp_path):
    manager = DialogManager(
        flows.build_engine(RESTAURANTS),
        KeywordSlotExtractor(values_from_restaurants(RESTAURANTS), LevenshteinMatcher()),
        ResponseRenderer(),
    )
    inputs = iter(ACTS)
    session_log = SessionLog.create(tmp_path)

    run_conversation(
        manager,
        ScriptedClassifier(),
        is_finished=lambda state: state.node == flows.BYE,
        input_function=lambda prompt: next(inputs),
        output_function=lambda text: None,
        session_log=session_log,
        config={"matcher": "levenshtein"},
    )

    records = [json.loads(line) for line in session_log.path.read_text().splitlines()]
    start, turns, end = records[0], records[1:-1], records[-1]

    assert {record["session_id"] for record in records} == {session_log.session_id}
    assert start["event"] == "start" and start["config"] == {"matcher": "levenshtein"}
    assert [turn["utterance"] for turn in turns] == [normalize_utterance(text) for text in ACTS]
    assert [turn["act"] for turn in turns] == list(ACTS.values())
    assert turns[0]["state_after"]["pending"][0]["value"] == "spanish"
    assert turns[0]["action"]["type"] == "confirm_value"
    assert ["restaurantname", "la tasca"] in turns[4]["action"]["parameters"]
    assert end == end | {"event": "end", "reason": "finished", "turns": len(ACTS)}


def test_session_log_records_fired_rules(tmp_path):
    session_log = SessionLog.create(tmp_path)
    explanation = (Inference(RestaurantProperty.ROMANTIC, True, rule=RULES[5], overruled=(RULES[4],)),)

    session_log.turn(
        turn=1,
        utterance="romantic",
        act="inform",
        state_before=DialogState(node=1),
        state_after=DialogState(node=1),
        action=SystemAction(SystemActionType.RECOMMEND, explanation=explanation),
        response="",
        latency_ms=1.0,
    )

    inference = json.loads(session_log.path.read_text())["action"]["explanation"][0]
    assert inference["property"] == "romantic" and inference["value"] is True
    assert inference["rule"]["id"] == 6
    assert [rule["id"] for rule in inference["overruled"]] == [5]


def test_requirement_answer_filters_and_explains_recommendation():
    restaurants = pd.DataFrame([
        {**RESTAURANTS.iloc[0], "restaurantname": "short stay", "length of stay": "short"},
        {**RESTAURANTS.iloc[0], "restaurantname": "long stay", "length of stay": "long"},
    ])
    manager = DialogManager(
        flows.build_engine(restaurants),
        KeywordSlotExtractor(values_from_restaurants(restaurants), LevenshteinMatcher()),
        ResponseRenderer(),
    )
    # The classifier labels the unfamiliar requirement answer as null.
    acts = {"cheap spanish food in the centre": "inform", "somewhere for a romantic dinner": "null"}
    inputs = iter(acts)
    outputs = []

    class Classifier:
        def predict_one(self, utterance: str) -> str:
            return {normalize_utterance(text): act for text, act in acts.items()}[utterance]

    run_conversation(
        manager,
        Classifier(),
        is_finished=lambda state: state.node == flows.RECOMMEND,
        input_function=lambda prompt: next(inputs),
        output_function=outputs.append,
    )

    assert outputs[-1] == (
        "system: long stay is a nice cheap spanish restaurant in the centre. "
        "It is romantic because it allows a long stay."
    )


def test_excluded_food_is_never_recommended():
    restaurants = pd.DataFrame([
        {**RESTAURANTS.iloc[0], "restaurantname": "spanish place", "food": "spanish"},
        {**RESTAURANTS.iloc[0], "restaurantname": "italian place", "food": "italian"},
    ])
    manager = DialogManager(
        flows.build_engine(restaurants),
        KeywordSlotExtractor(values_from_restaurants(restaurants), LevenshteinMatcher()),
        ResponseRenderer(),
    )
    acts = {"cheap food in the centre": "inform", "not spanish": "negate", "i don't care": "inform", "no": "negate"}
    inputs = iter(acts)
    outputs = []

    class Classifier:
        def predict_one(self, utterance: str) -> str:
            return {normalize_utterance(text): act for text, act in acts.items()}[utterance]

    state = run_conversation(
        manager,
        Classifier(),
        is_finished=lambda state: state.node == flows.RECOMMEND,
        input_function=lambda prompt: next(inputs),
        output_function=outputs.append,
    )

    assert outputs[2] == "system: What kind of food would you like?"
    assert state.excluded == ((Slot.FOOD, "spanish"),)
    assert outputs[-1] == "system: italian place is a nice cheap italian restaurant in the centre."


REFERENCE_RESTAURANTS = pd.DataFrame([
    {**RESTAURANTS.iloc[0], "restaurantname": "busy long", "food": "italian", "pricerange": "expensive",
     "crowdedness": "busy", "length of stay": "long"},
    {**RESTAURANTS.iloc[0], "restaurantname": "quiet short", "food": "seafood", "pricerange": "expensive"},
    {**RESTAURANTS.iloc[0], "restaurantname": "no area", "food": "turkish", "pricerange": "moderate", "area": ""},
    {**RESTAURANTS.iloc[0], "restaurantname": "cheap north", "area": "north"},
])


def talk(script: list[tuple[str, str]], restaurants: pd.DataFrame = REFERENCE_RESTAURANTS) -> list[str]:
    """Run a scripted conversation of (utterance, act) pairs and return the system responses."""
    manager = DialogManager(
        flows.build_engine(restaurants),
        KeywordSlotExtractor(values_from_restaurants(restaurants), LevenshteinMatcher()),
        ResponseRenderer(),
    )
    acts = dict(script)
    inputs = iter(acts)
    outputs = []

    class Classifier:
        def predict_one(self, utterance: str) -> str:
            return acts[utterance]

    def read(prompt: str) -> str:
        # Ending the script works like ending terminal input.
        utterance = next(inputs, None)

        if utterance is None:
            raise EOFError

        return utterance

    run_conversation(manager, Classifier(), is_finished=lambda state: state.node == flows.BYE,
                     input_function=read, output_function=outputs.append)

    return [output.removeprefix("system: ") for output in outputs]


def test_greeting_and_none_move_the_dialog_on():
    outputs = talk([("hello", "hello"), ("italian", "inform"), ("centre", "inform"), ("expensive", "inform"),
                    ("none", "null")])

    assert outputs[1] == "What kind of food would you like?"
    assert outputs[-1] == "busy long is a nice expensive italian restaurant in the centre."


def test_long_stay_outweighs_busy_for_romantic():
    # Reference dialog 17.
    outputs = talk([("expensive italian food in the centre", "inform"), ("romantic", "inform")])

    assert outputs[-1] == ("busy long is a nice expensive italian restaurant in the centre. "
                           "It is romantic because it allows a long stay, even though it is busy.")


def test_requirement_without_a_rule_is_not_met():
    # Reference dialog 18: no rule says the restaurant suits children, so it does not count.
    outputs = talk([("expensive seafood in the centre", "inform"), ("suitable for children", "inform")])

    assert outputs[-1] == ("I'm sorry, but I could not find an expensive seafood restaurant in the centre "
                           "that meets your additional requirements.")


def test_no_match_is_reported_before_requirements_and_new_preference_is_used():
    # Reference dialog 10.
    outputs = talk([("expensive italian food in the north", "inform"), ("what about the centre", "reqalts"),
                    ("no", "negate")])

    assert outputs[1] == ("I'm sorry, but I could not find an expensive italian restaurant in the north. "
                          "Would you like to change a preference?")
    assert outputs[2] == "Do you have any additional requirements, for example romantic or touristic?"
    assert outputs[3] == "busy long is a nice expensive italian restaurant in the centre."


def test_fact_questions_about_the_recommendation():
    # Reference dialog 8.
    outputs = talk([("expensive italian food in the centre", "inform"), ("no", "negate"),
                    ("is it in the centre", "confirm"), ("is it cheap", "confirm")])

    assert outputs[-2:] == ["Yes, that's correct.", "No, the price range is actually expensive."]


def test_unknown_food_and_missing_area():
    # Reference dialogs 6 and 14.
    outputs = talk([("swedish food", "inform"), ("how about turkish food", "reqalts"), ("any area", "inform"),
                    ("moderate", "inform"), ("no", "negate")])

    assert outputs[1] == "I'm sorry, I don't know any swedish restaurants. Would you like something else?"
    assert outputs[-1] == "no area is a nice moderate turkish restaurant."
