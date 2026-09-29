import logging
from pathlib import Path

from methods_in_ai_research.classification.evaluation import evaluate_classifier, save_evaluation
from methods_in_ai_research.classification.models.classifier import BagOfWordsClassifier
from methods_in_ai_research.classification.persistence import load_classifier, save_classifier
from methods_in_ai_research.classification.registry import classifier_names, create_classifier

logger = logging.getLogger(__name__)

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

def run_pipeline(classifier_name: str, split_name: str, train_data: pd.DataFrame | None, test_data: pd.DataFrame | None, *, train_enabled: bool, evaluate_enabled: bool, results_directory: str | Path, models_directory: str | Path | None = None) -> None:
    if classifier_name == "all":
        for name in classifier_names:
            run_pipeline(classifier_name=name, split_name=split_name, train_data=train_data, test_data=test_data, train_enabled=train_enabled, evaluate_enabled=evaluate_enabled, results_directory=results_directory, models_directory=models_directory)
    else:
        resolved_model_path = resolve_model_path(classifier_name, split_name, models_directory)

        if resolved_model_path and not train_enabled:
            logger.info(f"Loading {classifier_name} from {resolved_model_path}")
            classifier = load_classifier(classifier_name, resolved_model_path)
        else:
            classifier = create_classifier(classifier_name)

        if train_enabled:
            if train_data is None:
                raise ValueError("Training data is required")

            logger.info(f"Training {classifier_name} on the {split_name}")

            classifier.fit(train_data["utterance"], train_data["label"])

            if resolved_model_path:
                save_classifier(classifier_name, classifier, resolved_model_path)
                logger.info(f"Saved {classifier_name} to {resolved_model_path}")

        if evaluate_enabled:
            if test_data is None:
                raise ValueError("Test data is required")

            logger.info(f"Evaluating {classifier_name} on the {split_name} split")

            result = evaluate_classifier(classifier, test_data)

            results_directory = (Path(results_directory) / classifier_name / split_name)

            save_evaluation(result, results_directory)
            logger.info("===[EVALUATION]===")
            logger.info(f"Accuracy: {result.summary['accuracy']}")
            logger.info(f"Balanced accuracy: {result.summary['balanced_accuracy']}")
            logger.info(f"Macro F1: {result.summary['macro_f1']}")
            logger.info(f"Results saved to {results_directory}")

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