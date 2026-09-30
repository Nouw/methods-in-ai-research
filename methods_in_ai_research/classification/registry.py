"""Define supported classifier names and construct their implementations."""

from methods_in_ai_research.classification.models.classifier import Classifier
from methods_in_ai_research.classification.models.decision_tree import DecisionTreeBagOfWordsClassifier
from methods_in_ai_research.classification.models.linear_svm import LinearSVMBagOfWordsClassifier, \
    LinearSVMEmbeddingClassifier
from methods_in_ai_research.classification.models.logistic_regression import LogisticRegressionBagOfWordsClassifier, \
    LogisticRegressionEmbeddingClassifier
from methods_in_ai_research.classification.models.naive_bayes import NaiveBayesBagOfWordsClassifier
from methods_in_ai_research.classification.models.rule_based import RuleBasedClassifier

classifier_names = ("rule-based", "bow-logistic-regression", "bow-linear-svm", "embedding-logistic-regression", "embedding-linear-svm", "naive-bayes", "decision-tree")


def create_classifier(name: str) -> Classifier:
    """Create an unfitted classifier for the given registered name.

    Raises:
         ValueError: if the classifier name is unknown.
    """
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