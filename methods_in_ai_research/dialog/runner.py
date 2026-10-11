"""Run the terminal conversation and record interaction logs.

Conversation decisions stay inside the dialog manager; this module only reads input,
classifies it, and shows the responses.
"""
import logging
import time
from collections.abc import Callable

from methods_in_ai_research.classification.models.classifier import Classifier
from methods_in_ai_research.classification.processing import normalize_utterance
from methods_in_ai_research.dialog.manager import DialogManager
from methods_in_ai_research.dialog.session_log import SessionLog
from methods_in_ai_research.dialog.state import ClassifiedInput, DialogState

logger = logging.getLogger(__name__)


def run_conversation(manager: DialogManager, classifier: Classifier, is_finished: Callable[[DialogState], bool], input_function: Callable[[str], str] = input, output_function: Callable[[str], None] = print, session_log: SessionLog | None = None,config: dict | None = None) -> DialogState:
    """Talk with the user until the flow finishes, they type /exit, or input ends.

    Input and output functions can be replaced for scripted conversations.
    When a session log is given, every turn is written to it together with the config.
    Return the final dialog state.
    """
    state, action, response = manager.start_with_action()
    output_function(f"system: {response}")

    if session_log is not None:
        session_log.start(config or {}, state, action, response)

    turns = 0
    reason = "finished"

    while not is_finished(state):
        try:
            utterance = normalize_utterance(input_function("user: "))
        except (EOFError, KeyboardInterrupt):
            reason = "interrupted"
            break

        if utterance == "/exit":
            reason = "exit"
            break

        if not utterance:
            continue

        started = time.perf_counter()
        classified_input = ClassifiedInput(act=classifier.predict_one(utterance), text=utterance)
        next_state, action, response = manager.transition_with_action(state, classified_input)
        latency_ms = (time.perf_counter() - started) * 1000

        turns += 1
        logger.debug("Turn: %r classified as %s; %s -> %s", utterance, classified_input.act, state, next_state)

        if session_log is not None:
            session_log.turn(
                turn=turns,
                utterance=utterance,
                act=classified_input.act,
                state_before=state,
                state_after=next_state,
                action=action,
                response=response,
                latency_ms=latency_ms,
            )

        state = next_state
        output_function(f"system: {response}")

    if session_log is not None:
        session_log.end(reason=reason, turns=turns, state=state)

    return state
