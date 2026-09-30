"""Provide an interactive prompt for comparing classifier predictions."""

from collections.abc import Callable, Mapping

from methods_in_ai_research.classification.models.classifier import Classifier
from methods_in_ai_research.classification.processing import normalize_utterance


def run_interactive(
    classifiers: Mapping[str, Classifier],
    input_function: Callable[[str], str] = input,
    output_function: Callable[[str], None] = print,
) -> None:
    """Read utterances and display a prediction from each supplied classifier.

    Normalize input, ignore blank utterances, and stop on '/exit' or end of input. Input and output functions can be replaced
    for testing.

    Raises:
         ValueError: if no classifiers are supplied
    """
    if not classifiers:
        raise ValueError("At least one classifier is required")

    output_function("Enter an utterance to classify. Use /exit to stop.")

    while True:
        try:
            utterance = input_function("Utterance: ")
        except EOFError:
            return

        utterance = normalize_utterance(utterance)

        if utterance == "/exit":
            return

        if not utterance:
            continue

        output_function("Predictions:")

        for name, classifier in classifiers.items():
            output_function(f"  {name}: {classifier.predict_one(utterance)}")
