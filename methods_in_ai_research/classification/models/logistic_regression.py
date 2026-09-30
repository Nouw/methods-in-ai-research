"""Provide logistic regression classifiers using TF-IDF features or DistilBERT embeddings."""

from methods_in_ai_research.classification.models.classifier import BagOfWordsClassifier, EmbeddingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
from methods_in_ai_research.classification.splitting import RANDOM_STATE

class LogisticRegressionBagOfWordsClassifier(BagOfWordsClassifier):
    """Classify dialog acts with logistic regression trained on TF-IDF features."""

    def create_estimator(self) -> BaseEstimator:
        """Create logistic regression with a 1.000 iteration limit and the configured seed."""
        return LogisticRegression(max_iter=1000, random_state=RANDOM_STATE) 

class LogisticRegressionEmbeddingClassifier(EmbeddingClassifier):
    """Classify dialog acts with logistic regression trained on DistilBERT embeddings."""

    def create_estimator(self) -> BaseEstimator:
        """Provide logistic regression classifiers using TF-IDF features or DistilBERT embeddings."""
        return LogisticRegression(max_iter=1000, random_state=RANDOM_STATE) 
