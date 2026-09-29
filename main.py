import argparse
import logging
from pathlib import Path

from methods_in_ai_research.classification.interactive import run_interactive
from methods_in_ai_research.classification.persistence import load_interactive_models
from methods_in_ai_research.classification.processing import load_dataset
from methods_in_ai_research.classification.registry import classifier_names, create_classifier
from transformers.utils import logging as hf_logging

from methods_in_ai_research.classification.splitting import create_splits
from methods_in_ai_research.classification.workflows import run_pipeline

logger = logging.getLogger(__name__)


DEFAULT_DATA_PATH = "data/dailog_acts.dat"
DEFAULT_MODELS_DIRECTORY = "artifacts/models"
DEFAULT_RESULTS_DIRECTORY = "results"
DEFAULT_SPLITS_DIRECTORY = "artifacts/splits"

def add_classifier_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--classifier", choices=(*classifier_names, "all"), default="all")

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Methods in AI Research dialog-act classification pipeline.")
    commands = parser.add_subparsers(dest="command", required=True)

    experiment_parser = commands.add_parser("experiment", help="Train and evaluate models on generated splits")
    experiment_parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    experiment_parser.add_argument("--split-strategy", choices=("original", "grouped", "both"), default="both")
    experiment_parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIRECTORY)
    experiment_parser.add_argument("--save-splits", action="store_true")
    experiment_parser.add_argument("--splits-dir", default=DEFAULT_SPLITS_DIRECTORY)
    add_classifier_argument(experiment_parser)

    train_parser = commands.add_parser("train", help="Train and save final models")
    train_parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    train_parser.add_argument("--models-dir", default=DEFAULT_MODELS_DIRECTORY)
    add_classifier_argument(train_parser)

    evaluate_parser = commands.add_parser("evaluate", help="Evaluate saved models on labeled data")
    evaluate_parser.add_argument("data")
    evaluate_parser.add_argument("--models-dir", default=DEFAULT_MODELS_DIRECTORY)
    evaluate_parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIRECTORY)
    add_classifier_argument(evaluate_parser)

    interactive_parser = commands.add_parser("interactive", help="Classify utterances using saved models")
    interactive_parser.add_argument("--models-dir", default=DEFAULT_MODELS_DIRECTORY)
    add_classifier_argument(interactive_parser)

    dialog_parser = commands.add_parser("dialog", help="Have a dialog with a classifier and state machine")
    add_classifier_argument(dialog_parser)

    return parser.parse_args()

def run_experiment(*, data_path: str | Path, classifier_name: str, split_strategy: str, results_directory: str | Path, save_splits: bool = False, splits_directory: str | Path = "artifacts/splits") -> None:
    source_data = load_dataset(data_path)

    datasets = create_splits(
        data=source_data,
        strategy=split_strategy,
        output_directory=splits_directory,
        save_splits=save_splits,
    )

    for split_name, (train_data, test_data) in datasets.items():
        run_pipeline(
            classifier_name=classifier_name,
            split_name=split_name,
            train_data=train_data,
            test_data=test_data,
            train_enabled=True,
            evaluate_enabled=True,
            results_directory=results_directory,
        )


def train_models(*, data_path: str | Path, classifier_name: str, models_directory: str | Path) -> None:
    train_data = load_dataset(data_path)

    run_pipeline(
        classifier_name=classifier_name,
        split_name="provided",
        train_data=train_data,
        test_data=None,
        train_enabled=True,
        evaluate_enabled=False,
        results_directory="results",
        models_directory=models_directory,
    )


def evaluate_models(*, data_path: str | Path, classifier_name: str, models_directory: str | Path, results_directory: str | Path) -> None:
    test_data = load_dataset(data_path)

    run_pipeline(
        classifier_name=classifier_name,
        split_name="provided",
        train_data=None,
        test_data=test_data,
        train_enabled=False,
        evaluate_enabled=True,
        results_directory=results_directory,
        models_directory=models_directory,
    )

def run_dialog(*, classifier_name: str) -> None:
    pass


def main():
    args = parse_arguments()

    if args.command == "experiment":
        run_experiment(
            data_path=args.data,
            classifier_name=args.classifier,
            split_strategy=args.split_strategy,
            results_directory=args.results_dir,
            save_splits=args.save_splits,
            splits_directory=args.splits_dir,
        )

    elif args.command == "train":
        train_models(
            data_path=args.data,
            classifier_name=args.classifier,
            models_directory=args.models_dir,
        )

    elif args.command == "evaluate":
        evaluate_models(
            data_path=args.data,
            classifier_name=args.classifier,
            models_directory=args.models_dir,
            results_directory=args.results_dir,
        )

    elif args.command == "interactive":
        classifiers = load_interactive_models(
            args.classifier,
            args.models_dir,
        )
        run_interactive(classifiers)

    elif args.command == "dialog":
        run_dialog(classifier_name="")


    
if __name__ == "__main__":
    # Disable huggingface logs
    hf_logging.set_verbosity_error()
    hf_logging.disable_default_handler()
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

    logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s: %(message)s",
            handlers=[
                logging.FileHandler("app.log"),
                logging.StreamHandler()
            ],
        )

    main()
