"""Provide command-line commands for classifier experiments, training, evaluation, and interactive use."""

import argparse
import logging
from pathlib import Path

from methods_in_ai_research.classification.interactive import run_interactive
from methods_in_ai_research.classification.persistence import load_interactive_models
from methods_in_ai_research.classification.processing import load_dataset
from methods_in_ai_research.classification.registry import classifier_names
from methods_in_ai_research.classification.splitting import create_splits
from methods_in_ai_research.classification.workflows import run_pipeline, print_summary, save_summary

logger = logging.getLogger(__name__)


DEFAULT_DATA_PATH = "data/dailog_acts.dat"
DEFAULT_MODELS_DIRECTORY = "artifacts/models"
DEFAULT_RESULTS_DIRECTORY = "results"
DEFAULT_SPLITS_DIRECTORY = "artifacts/splits"

def add_classifier_argument(parser: argparse.ArgumentParser) -> None:
    """Add a command-line option for selecting one classifier or all classifiers."""
    parser.add_argument("--classifier", choices=(*classifier_names, "all"), default="all")

def add_commands(commands: argparse._SubParsersAction) -> None:
    """Register the classification commands on the shared command-line parser."""
    experiment_parser = commands.add_parser("experiment", help="Train and evaluate models on generated splits")
    experiment_parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    experiment_parser.add_argument("--split-strategy", choices=("original", "grouped", "both"), default="both")
    experiment_parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIRECTORY)
    experiment_parser.add_argument("--save-splits", action="store_true")
    experiment_parser.add_argument("--splits-dir", default=DEFAULT_SPLITS_DIRECTORY)
    add_classifier_argument(experiment_parser)
    experiment_parser.set_defaults(handler=lambda args: run_experiment(
        data_path=args.data,
        classifier_name=args.classifier,
        split_strategy=args.split_strategy,
        results_directory=args.results_dir,
        save_splits=args.save_splits,
        splits_directory=args.splits_dir,
    ))

    train_parser = commands.add_parser("train", help="Train and save final models")
    train_parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    train_parser.add_argument("--models-dir", default=DEFAULT_MODELS_DIRECTORY)
    add_classifier_argument(train_parser)
    train_parser.set_defaults(handler=lambda args: train_models(
        data_path=args.data,
        classifier_name=args.classifier,
        models_directory=args.models_dir,
    ))

    evaluate_parser = commands.add_parser("evaluate", help="Evaluate saved models on labeled data")
    evaluate_parser.add_argument("data")
    evaluate_parser.add_argument("--models-dir", default=DEFAULT_MODELS_DIRECTORY)
    evaluate_parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIRECTORY)
    add_classifier_argument(evaluate_parser)
    evaluate_parser.set_defaults(handler=lambda args: evaluate_models(
        data_path=args.data,
        classifier_name=args.classifier,
        models_directory=args.models_dir,
        results_directory=args.results_dir,
    ))

    interactive_parser = commands.add_parser("interactive", help="Classify utterances using saved models")
    interactive_parser.add_argument("--models-dir", default=DEFAULT_MODELS_DIRECTORY)
    add_classifier_argument(interactive_parser)
    interactive_parser.set_defaults(handler=lambda args: run_interactive(
        load_interactive_models(args.classifier, args.models_dir),
    ))

def run_experiment(*, data_path: str | Path, classifier_name: str, split_strategy: str, results_directory: str | Path, save_splits: bool = False, splits_directory: str | Path = "artifacts/splits") -> None:
    """Train and evaluate the selected classifiers on generated splits, then print and save comparison tables."""
    source_data = load_dataset(data_path)

    datasets = create_splits(
        data=source_data,
        strategy=split_strategy,
        output_directory=splits_directory,
        save_splits=save_splits,
    )

    grouped_summaries = []
    original_summaries = []

    for split_name, (train_data, test_data) in datasets.items():
        summaries = run_pipeline(
            classifier_name=classifier_name,
            split_name=split_name,
            train_data=train_data,
            test_data=test_data,
            train_enabled=True,
            evaluate_enabled=True,
            results_directory=results_directory,
        )

        if split_name == "original":
            original_summaries = summaries
        elif split_name == "grouped":
            grouped_summaries = summaries

    original_path = Path(results_directory) / "original.csv"
    print_summary(original_summaries)
    save_summary(original_summaries, original_path)
    logger.info(f"Saved original summary to {original_path}")

    grouped_path = Path(results_directory) / "grouped.csv"
    print_summary(grouped_summaries)
    save_summary(grouped_summaries, grouped_path)
    logger.info(f"Saved grouped summary to {grouped_path}")


def train_models(*, data_path: str | Path, classifier_name: str, models_directory: str | Path) -> None:
    """Fit the selected classifiers on the complete dataset and save their state."""
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
    """Evaluate saved classifiers on a labeled dataset without retraining, then print and save the results."""
    test_data = load_dataset(data_path)

    summaries = run_pipeline(
        classifier_name=classifier_name,
        split_name="provided",
        train_data=None,
        test_data=test_data,
        train_enabled=False,
        evaluate_enabled=True,
        results_directory=results_directory,
        models_directory=models_directory,
    )

    output_path = Path(results_directory) / "evaluate_results.csv"

    print_summary(summaries)
    save_summary(summaries, output_path)

    logger.info(f"Saved summary to {output_path}")
