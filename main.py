import argparse
import logging
import pickle
from methods_in_ai_research.evaluation import evaluate_classifier, save_evaluation
import pandas as pd
from pathlib import Path
from methods_in_ai_research.interactive import run_interactive
from methods_in_ai_research.models.classifier import BagOfWordsClassifier, Classifier, EmbeddingClassifier
from methods_in_ai_research.models.decision_tree import DecisionTreeBagOfWordsClassifier
from methods_in_ai_research.models.linear_svm import LinearSVMBagOfWordsClassifier, LinearSVMEmbeddingClassifier
from methods_in_ai_research.models.logistic_regression import LogisticRegressionBagOfWordsClassifier, LogisticRegressionEmbeddingClassifier
from methods_in_ai_research.models.naive_bayes import NaiveBayesBagOfWordsClassifier
from methods_in_ai_research.models.rule_based import RuleBasedClassifier
from methods_in_ai_research.processing import load_dialog_acts
from methods_in_ai_research.splitting import create_original_split, create_grouped_split, log_split_summary, validate_split, save_split
from transformers.utils import logging as hf_logging

logger = logging.getLogger(__name__)

classifier_names = ("rule-based", "bow-logistic-regression", "bow-linear-svm", "embedding-logistic-regression", "embedding-linear-svm", "naive-bayes", "decision-tree")

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

    return parser.parse_args()

def create_classifier(name: str) -> Classifier:
    if name == "rule-based":
        return RuleBasedClassifier()

    if name == "bow-logistic-regression":
        return LogisticRegressionBagOfWordsClassifier()
    
    if name == "bow-linear-svm":
        return LinearSVMBagOfWordsClassifier()

    if name == "embedding-logistic-regression":
        return LogisticRegressionEmbeddingClassifier()

    if name == "embedding-linear-svm":
        return LinearSVMEmbeddingClassifier()

    if name == "naive-bayes":
        return NaiveBayesBagOfWordsClassifier()

    if name == "decision-tree":
        return DecisionTreeBagOfWordsClassifier()

    raise ValueError(f"Unknown classifier: {name}")

def load_dataset(file_path: str | Path) -> pd.DataFrame:
    path = Path(file_path)

    if path.suffix.lower() == ".dat":
        return load_dialog_acts(path)

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, index_col="row_id", keep_default_na=False)

    raise ValueError(f"Unsupported dataset format: {path.suffix or '<none>'}. Expected .csv or .dat")

def save_classifier(classifier_name: str, classifier: Classifier, file_path: str | Path) -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(classifier, BagOfWordsClassifier):
        state = {"pipeline": classifier.pipeline}
    elif isinstance(classifier, EmbeddingClassifier):
        state = {"estimator": classifier.estimator}
    elif isinstance(classifier, RuleBasedClassifier):
        state = {}
    else:
        raise TypeError(f"Unsupported classifier type: {type(classifier).__name__}")

    artifact = {
        "format_version": 1,
        "classifier_name": classifier_name,
        "state": state,
    }

    with path.open("wb") as model_file:
        pickle.dump(artifact, model_file)

def load_classifier(classifier_name: str, file_path: str | Path) -> Classifier:
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"Saved model not found: {path}")

    with path.open("rb") as model_file:
        artifact = pickle.load(model_file)

    if not isinstance(artifact, dict) or artifact.get("format_version") != 1:
        raise ValueError(f"Unsupported saved model format: {path}")

    saved_classifier_name = artifact.get("classifier_name")

    if saved_classifier_name != classifier_name:
        raise ValueError(
            f"Saved model contains {saved_classifier_name!r}, not {classifier_name!r}"
        )

    classifier = create_classifier(classifier_name)
    state = artifact.get("state")

    if not isinstance(state, dict):
        raise ValueError(f"Saved model has invalid state: {path}")

    if isinstance(classifier, BagOfWordsClassifier):
        classifier.pipeline = state["pipeline"]
    elif isinstance(classifier, EmbeddingClassifier):
        classifier.estimator = state["estimator"]
    elif not isinstance(classifier, RuleBasedClassifier):
        raise TypeError(f"Unsupported classifier type: {type(classifier).__name__}")

    return classifier

def load_interactive_models(classifier_name: str, models_directory: str | Path) -> dict[str, Classifier]:
    classifiers = {}
    names = classifier_names if classifier_name == "all" else (classifier_name,)

    for classifier_name in names:
        model_path = Path(models_directory) / f"{classifier_name}.pkl"
        logger.info(f"Loading {classifier_name} from {model_path}")
        classifiers[classifier_name] = load_classifier(classifier_name, model_path)

    return classifiers

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

def create_splits(data: pd.DataFrame, strategy: str, output_directory: str | Path, *, save_splits: bool = False) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    splits = {}
    
    if strategy in {"original", "both"}:
        train_data, test_data = create_original_split(data)

        validate_split(data, train_data, test_data)

        log_split_summary("Original", data, train_data, test_data)

        if save_splits:
            save_split(train_data, test_data, Path(output_directory) / "original")

        splits["original"] = (train_data, test_data)

    if strategy in {"grouped", "both"}:
        train_data, test_data = create_grouped_split(data)

        validate_split(data, train_data, test_data)

        log_split_summary("Grouped", data, train_data, test_data)

        if save_splits:
            save_split(train_data, test_data, Path(output_directory) / "grouped")

        splits["grouped"] = (train_data, test_data)

    return splits

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

def main():
    args = parse_arguments()

    if args.command == "interactive":
        classifiers = load_interactive_models(args.classifier, args.models_dir)
        run_interactive(classifiers)
        return

    if args.command == "experiment":
        source_data = load_dataset(args.data)
        datasets = create_splits(
            source_data,
            args.split_strategy,
            args.splits_dir,
            save_splits=args.save_splits,
        )

        for split_name, (train_data, test_data) in datasets.items():
            run_pipeline(
                classifier_name=args.classifier,
                split_name=split_name,
                train_data=train_data,
                test_data=test_data,
                train_enabled=True,
                evaluate_enabled=True,
                results_directory=args.results_dir,
            )

        return

    if args.command == "train":
        train_data = load_dataset(args.data)
        run_pipeline(
            classifier_name=args.classifier,
            split_name="provided",
            train_data=train_data,
            test_data=None,
            train_enabled=True,
            evaluate_enabled=False,
            results_directory=DEFAULT_RESULTS_DIRECTORY,
            models_directory=args.models_dir,
        )
        return

    test_data = load_dataset(args.data)
    run_pipeline(
        classifier_name=args.classifier,
        split_name="provided",
        train_data=None,
        test_data=test_data,
        train_enabled=False,
        evaluate_enabled=True,
        results_directory=args.results_dir,
        models_directory=args.models_dir,
    )


    
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
