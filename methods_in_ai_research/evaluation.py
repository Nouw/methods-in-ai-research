from pathlib import Path
import json
from methods_in_ai_research.processing import VALID_LABELS
import pandas as pd
from dataclasses import dataclass
from typing import Any

from methods_in_ai_research.models.classifier import Classifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

@dataclass(frozen=True)
class EvaluationResult:
    summary: dict[str, float | int]
    report: dict[str, Any]
    confusion_matrix: pd.DataFrame
    predictions: pd.DataFrame

def evaluate_classifier(classifier: Classifier, data: pd.DataFrame):
    required_columns = { "label", "utterance" }
    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(f"Evaluation is missing columns: {sorted(missing_columns)}")

    expected = data["label"].tolist()
    predicted = classifier.predict(data["utterance"])

    if len(predicted) != len(expected):
        raise ValueError("Classifier returned a different number of predictions")

    unknown_labels = set(predicted) - VALID_LABELS

    if unknown_labels:
        raise ValueError(f"Classifier returned unknown labels: {sorted(unknown_labels)}")

    labels = sorted(VALID_LABELS)

    summary = {
            "examples": len(expected),
            "precision_macro": float(precision_score(expected, predicted, average="macro", zero_division=0)),
            "precision_weighted": float(precision_score(expected, predicted, average="weighted", zero_division=0)),
            "recall_macro": float(recall_score(expected, predicted, average="macro", zero_division=0)),
            "recall_weighted": float(recall_score(expected, predicted, average="weighted", zero_division=0)),
            "accuracy": float(accuracy_score(expected, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(expected, predicted)),
            "macro_f1": float(f1_score(expected, predicted, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(expected, predicted, average="weighted", zero_division=0))
        }

    report = classification_report(expected, predicted, labels=labels, output_dict=True, zero_division=0)
    matrix = pd.DataFrame(
        confusion_matrix(
            expected,
            predicted,
            labels=labels,
        ),
        index=labels,
        columns=labels,
    )
    matrix.index.name = "true_label"
    matrix.columns.name = "predicted_label"

    predictions = pd.DataFrame({
        "utterance": data["utterance"].tolist(),
        "expected_label": expected,
        "predicted_label": predicted,
    })
    predictions["correct"] = predictions["expected_label"] == predictions["predicted_label"]
    
    return EvaluationResult(
        summary=summary,
        report=report,
        confusion_matrix=matrix,
        predictions=predictions,
    )

def save_evaluation(result: EvaluationResult, output_directory: str | Path) -> None:
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    metrics = {
            "summary": result.summary,
            "classification_report": result.report
        }

    metrics_path = output_path / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    result.confusion_matrix.to_csv(output_path / "confusion_matrix.csv")
    result.predictions.to_csv(output_path / "predictions.csv", index=False)
