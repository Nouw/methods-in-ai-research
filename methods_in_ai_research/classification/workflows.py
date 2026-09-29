import logging
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

from methods_in_ai_research.classification.evaluation import evaluate_classifier, save_evaluation, EvaluationResult
from methods_in_ai_research.classification.models.classifier import BagOfWordsClassifier
from methods_in_ai_research.classification.persistence import load_classifier, save_classifier
from methods_in_ai_research.classification.registry import classifier_names, create_classifier

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class EvaluationSummary:
    classifier: str
    split: str
    result: EvaluationResult

def resolve_model_path(
    classifier_name: str,
    split_name: str,
    models_directory: str | Path | None,
) -> Path | None:
    if not models_directory:
        return None

    directory = Path(models_directory)

    if split_name != "provided":
        directory /= split_name

    return directory / f"{classifier_name}.pkl"

def run_pipeline(classifier_name: str, split_name: str, train_data: pd.DataFrame | None, test_data: pd.DataFrame | None, *, train_enabled: bool, evaluate_enabled: bool, results_directory: str | Path, models_directory: str | Path | None = None) -> list[EvaluationSummary]:
    if classifier_name == "all":
        summaries = []

        for name in classifier_names:
            summaries.extend(run_pipeline(classifier_name=name, split_name=split_name, train_data=train_data, test_data=test_data, train_enabled=train_enabled, evaluate_enabled=evaluate_enabled, results_directory=results_directory, models_directory=models_directory))

        return summaries
    else:
        logger.info("")
        logger.info(f"=== {classifier_name} | {split_name} ===")

        if train_data is not None:
            logger.info(f"Training examples: {len(train_data)}")

        if test_data is not None:
            logger.info(f"Testing examples: {len(test_data)}")

        resolved_model_path = resolve_model_path(classifier_name, split_name, models_directory)

        if resolved_model_path and not train_enabled:
            logger.info(f"Loading {classifier_name} from {resolved_model_path}")
            classifier = load_classifier(classifier_name, resolved_model_path)
        else:
            classifier = create_classifier(classifier_name)

        if train_enabled:
            if train_data is None:
                raise ValueError("Training data is required")

            logger.info("Preparing classifier...")

            classifier.fit(train_data["utterance"], train_data["label"])

            if resolved_model_path:
                save_classifier(classifier_name, classifier, resolved_model_path)
                logger.info(f"Saved {classifier_name} to {resolved_model_path}")

        if evaluate_enabled:
            if test_data is None:
                raise ValueError("Test data is required")

            logger.info("Evaluating...")

            result = evaluate_classifier(classifier, test_data)

            results_directory = (Path(results_directory) / classifier_name / split_name)

            save_evaluation(result, results_directory)
            logger.info(
                "Accuracy: %.4f | Balanced accuracy: %.4f | Macro F1: %.4f",
                result.summary["accuracy"],
                result.summary["balanced_accuracy"],
                result.summary["macro_f1"],
            )
            logger.info("Results saved to %s", results_directory)

            if isinstance(classifier, BagOfWordsClassifier):
                oov = classifier.calculate_oov_statistics(
                    test_data["utterance"]
                )

                logger.info("Test tokens: %d", oov.total_tokens)
                logger.info("OOV tokens: %d", oov.oov_tokens)
                logger.info(
                    "OOV token rate: %.2f%%",
                    oov.oov_token_rate * 100,
                )
                logger.info(
                    "All-OOV utterances: %d",
                    oov.zero_vector_utterances,
                )

            return [
                EvaluationSummary(
                    classifier=classifier_name,
                    split=split_name,
                    result=result,
                )
            ]

    return []

def print_summary(summaries: list[EvaluationSummary]) -> None:
    """Print comparable metrics for every classifier and split."""
    if not summaries:
        return

    headers = ("Classifier", "Split", "Accuracy", "Balanced accuracy", "Macro F1", "Weighted F1", "Macro Precision", "Weighted Precision", "Macro Recall", "Weighted Recall")
    rows = [
        (
            result.classifier,
            result.split,
            f"{result.result.summary['accuracy']:.4f}",
            f"{result.result.summary['balanced_accuracy']:.4f}",
            f"{result.result.summary['macro_f1']:.4f}",
            f"{result.result.summary['weighted_f1']:.4f}",
            f"{result.result.summary['precision_macro']:.4f}",
            f"{result.result.summary['precision_weighted']:.4f}",
            f"{result.result.summary['recall_macro']:.4f}",
            f"{result.result.summary['recall_weighted']:.4f}",
        )
        for result in summaries
    ]

    widths = [
        max(len(row[column]) for row in [headers, *rows])
        for column in range(len(headers))
    ]

    def format_row(row: tuple[str, ...]) -> str:
        """Align names left and numeric columns right."""
        return "  ".join(
            value.ljust(width) if index < 2 else value.rjust(width)
            for index, (value, width) in enumerate(zip(row, widths))
        )

    lines = [
        "",
        "Experiment summary",
        format_row(headers),
        "  ".join("-" * width for width in widths),
        *(format_row(row) for row in rows),
    ]
    logger.info("\n".join(lines))


def save_summary(results: list[EvaluationSummary], output_path: str | Path) -> None:
    """Save evaluation results to a csv file per group"""

    headers = ("Classifier", "Split", "Accuracy", "Balanced accuracy", "Macro F1", "Weighted F1", "Macro Precision",
               "Weighted Precision", "Macro Recall", "Weighted Recall")
    rows = []

    for result in results:
        rows.append((
            result.classifier,
            result.split,
            f"{result.result.summary['accuracy']:.4f}",
            f"{result.result.summary['balanced_accuracy']:.4f}",
            f"{result.result.summary['macro_f1']:.4f}",
            f"{result.result.summary['weighted_f1']:.4f}",
            f"{result.result.summary['precision_macro']:.4f}",
            f"{result.result.summary['precision_weighted']:.4f}",
            f"{result.result.summary['recall_macro']:.4f}",
            f"{result.result.summary['recall_weighted']:.4f}",
        ))

    pd.DataFrame(rows, columns=headers).to_csv(output_path, index=False)
