import logging
import pickle
from pathlib import Path

from methods_in_ai_research.classification.models.classifier import Classifier, BagOfWordsClassifier, \
    EmbeddingClassifier
from methods_in_ai_research.classification.models.rule_based import RuleBasedClassifier
from methods_in_ai_research.classification.registry import create_classifier, classifier_names


logger = logging.getLogger(__name__)

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

