"""Provide a decision tree classifier using TF-IDF features."""

from methods_in_ai_research.classification.models.classifier import BagOfWordsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator

from methods_in_ai_research.classification.splitting import RANDOM_STATE

class DecisionTreeBagOfWordsClassifier(BagOfWordsClassifier):
    """Classify dialog acts with a decision tree trained on TF-IDF features."""

    def create_estimator(self) -> BaseEstimator:
        """Create a decision tree using the configured random seed."""
        return DecisionTreeClassifier(random_state=RANDOM_STATE)
    
