"""Run the terminal conversation and record interaction logs.

Conversation decisions stay inside the dialog manager; this module only reads input,
classifies it, and shows the responses.

TODO:
- Log timing and recommendation evidence per turn.
"""
import logging
from collections.abc import Callable

from methods_in_ai_research.classification.models.classifier import Classifier
from methods_in_ai_research.classification.processing import normalize_utterance
from methods_in_ai_research.dialog.manager import DialogManager
from methods_in_ai_research.dialog.state import ClassifiedInput, DialogState

logger = logging.getLogger(__name__)


def run_conversation(
    manager: DialogManager,
    classifier: Classifier,
    is_finished: Callable[[DialogState], bool],
    input_function: Callable[[str], str] = input,
    output_function: Callable[[str], None] = print,
) -> DialogState:
    """Talk with the user until the flow finishes, they type /exit, or input ends.

    Input and output functions can be replaced for scripted conversations.
    Return the final dialog state.
    """
    state, response = manager.start()
    output_function(f"system: {response}")

    while not is_finished(state):
        try:
            utterance = normalize_utterance(input_function("user: "))
        except (EOFError, KeyboardInterrupt):
            break

        if utterance == "/exit":
            break

        if not utterance:
            continue

        classified_input = ClassifiedInput(act=classifier.predict_one(utterance), text=utterance)
        next_state, response = manager.transition(state, classified_input)

        logger.debug("Turn: %r classified as %s; %s -> %s", utterance, classified_input.act, state, next_state)

        state = next_state
        output_function(f"system: {response}")

    return state
