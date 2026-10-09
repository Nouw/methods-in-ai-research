"""Provide the command-line command for talking to the dialog system."""

import argparse
import logging

from methods_in_ai_research.classification.persistence import load_classifier
from methods_in_ai_research.classification.registry import classifier_names
from methods_in_ai_research.dialog import flows
from methods_in_ai_research.dialog.manager import DialogManager
from methods_in_ai_research.dialog.responses import ResponseRenderer
from methods_in_ai_research.dialog.restaurants import load_restaurants
from methods_in_ai_research.dialog.runner import run_conversation
from methods_in_ai_research.dialog.slots import KeywordSlotExtractor, LevenshteinMatcher, Matcher, SemanticMatcher, \
    values_from_restaurants

logger = logging.getLogger(__name__)

DEFAULT_CLASSIFIER = "bow-linear-svm"
DEFAULT_MODELS_DIRECTORY = "artifacts/models"
DEFAULT_RESTAURANTS_PATH = "data/restaurant_info_extended.csv"


def add_commands(commands: argparse._SubParsersAction) -> None:
    """Register the dialog command on the shared command-line parser."""
    dialog_parser = commands.add_parser("dialog", help="Have a dialog with a classifier and state machine")
    dialog_parser.add_argument("--classifier", choices=classifier_names, default=DEFAULT_CLASSIFIER)
    dialog_parser.add_argument("--models-dir", default=DEFAULT_MODELS_DIRECTORY)
    dialog_parser.add_argument("--restaurants", default=DEFAULT_RESTAURANTS_PATH)
    dialog_parser.add_argument("--matcher", choices=("levenshtein", "distilbert"), default="levenshtein",
                               help="Fallback for preferences that keyword matching does not recognise")
    dialog_parser.set_defaults(handler=lambda args: run_dialog(
        classifier_name=args.classifier,
        models_directory=args.models_dir,
        restaurants_path=args.restaurants,
        matcher_name=args.matcher,
    ))


def create_matcher(name: str) -> Matcher:
    """Create the fallback matcher selected on the command line."""
    if name == "levenshtein":
        return LevenshteinMatcher()

    if name == "distilbert":
        from methods_in_ai_research.embeddings import DistilBertEncoder
        return SemanticMatcher(DistilBertEncoder())

    raise ValueError(f"Unknown matcher: {name}")


def run_dialog(*, classifier_name: str, models_directory: str, restaurants_path: str, matcher_name: str) -> None:
    """Load the classifier and restaurant data, then hold a conversation in the terminal."""
    logger.debug("Dialog configuration: classifier=%s matcher=%s restaurants=%s",
                 classifier_name, matcher_name, restaurants_path)

    classifier = load_classifier(classifier_name, f"{models_directory}/{classifier_name}.pkl")
    data = load_restaurants(restaurants_path)
    extractor = KeywordSlotExtractor(values_from_restaurants(data), create_matcher(matcher_name))
    manager = DialogManager(flows.build_engine(data), extractor, ResponseRenderer())

    run_conversation(manager, classifier, is_finished=lambda state: state.node == flows.BYE)
