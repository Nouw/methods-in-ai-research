import argparse
import logging
from methods_in_ai_research.evaluation import evaluate_classifier, save_evaluation
import pandas as pd
from pathlib import Path
from methods_in_ai_research.models.classifier import Classifier
from methods_in_ai_research.models.rule_based import RuleBasedClassifier
from methods_in_ai_research.processing import preprocess, load_dialog_acts
from methods_in_ai_research.splitting import create_original_split, create_grouped_split, log_split_summary, validate_split, save_split

logger = logging.getLogger(__name__)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Methods in AI Research dialog-act classification pipeline.")
    parser.add_argument("--split", action="store_true", help="Create new train/test splits from a raw .dat file")
    parser.add_argument("--train", action="store_true", help="Train the selected classifier")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the selected classifier")
    parser.add_argument("--data-path", help="Raw .dat file used when --split is enabled")
    parser.add_argument("--train-path", help="Existing training CSV used without --split")
    parser.add_argument("--test-path", help="Existing test CSV used without --split")
    parser.add_argument("--split-strategy", choices=("original", "grouped", "both"), default="both")
    parser.add_argument("--classifier", choices=("rule-based",), default="rule-based")
    parser.add_argument("--split-output-dir", default="artifacts/splits")
    parser.add_argument("--results-dir", default="results")

    args = parser.parse_args()

    if not any((args.split, args.train, args.evaluate)):
        parser.error("Enable at least one of --split --train or --evaluate")

    if args.split and not args.data_path:
        parser.error("--split requires --data-path")

    if not args.split and args.train and not args.train_path:
        parser.error("--train without --split requires --train_path")

    if not args.split and args.evaluate and not args.test_path:
        parser.error("--evaluate without --split requires --test_path")

    return args

def create_classifier(name: str) -> Classifier:
    if name == "rule-based":
        return RuleBasedClassifier()

    raise ValueError(f"Unknown classifier: {name}")

def load_split(file_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(file_path, index_col="row_id", keep_default_na=False)

def create_splits(data: pd.DataFrame, strategy: str, output_directory: str | Path) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    splits = {}
    
    if strategy in {"original", "both"}:
        train_data, test_data = create_original_split(data)

        validate_split(data, train_data, test_data)

        log_split_summary("Original", data, train_data, test_data)

        save_split(train_data, test_data, Path(output_directory) / "original")

        splits["original"] = (train_data, test_data)

    if strategy in {"grouped", "both"}:
        train_data, test_data = create_grouped_split(data)

        validate_split(data, train_data, test_data)

        log_split_summary("Grouped", data, train_data, test_data)

        save_split(train_data, test_data, Path(output_directory) / "grouped")

        splits["grouped"] = (train_data, test_data)

    return splits

def run_pipeline(classifier_name: str, split_name: str, train_data: pd.DataFrame, test_data: pd.DataFrame, *, train_enabled: bool, evaluate_enabled: bool, results_directory: str | Path) -> None:
    classifier = create_classifier(classifier_name)

    if train_enabled:
        if train_data is None:
            raise ValueError("Training data is required")

        logger.info(f"Training {classifier_name} on the {split_name}")

        classifier.fit(train_data["utterance"], train_data["label"])

    if evaluate_enabled:
        if test_data is None:
            raise ValueError("Test data is required")

        logger.info(f"Evaluating {classifier_name} on the {split_name} split")

        result = evaluate_classifier(classifier, test_data)

        results_directory = (Path(results_directory) / classifier_name / split_name)

        save_evaluation(result, results_directory)

        logger.info(f"Accuracy: {result.summary['accuracy']}")
        logger.info(f"Balanced accuracy: {result.summary['balanced_accuracy']}")
        logger.info(f"Macro F1: {result.summary['macro_f1']}")
        logger.info(f"Results saved to {results_directory}")
 




def main():
    args = parse_arguments() 

    if args.split:
        source_data = load_dialog_acts(args.data_path)
        datasets = create_splits(source_data, args.split_strategy, args.split_output_dir)
    else:
        train_data = (load_split(args.train_path) if args.train_path else None)
        test_data = (load_split(args.test_path) if args.test_path else None)
        datasets = {"provided": (train_data, test_data)}

    for split_name, (train_data, test_data) in datasets.items():
        if args.train or args.evaluate:
            run_pipeline(classifier_name=args.classifier, split_name=split_name, train_data=train_data, test_data=test_data, train_enabled=args.train, evaluate_enabled=args.evaluate, results_directory=args.results_dir)


    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    main()
