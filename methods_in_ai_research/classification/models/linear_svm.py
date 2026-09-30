"""Provide linear SVM classifiers using TF-IDF features or DistilBERT embeddings."""

from methods_in_ai_research.classification.models.classifier import BagOfWordsClassifier, EmbeddingClassifier
from methods_in_ai_research.classification.splitting import RANDOM_STATE
from sklearn.base import BaseEstimator
from sklearn.svm import LinearSVC


class LinearSVMBagOfWordsClassifier(BagOfWordsClassifier):
    """Classify dialog acts with a linear SVM trained on TF-IDF features."""

    def create_estimator(self) -> BaseEstimator:
        """Create a linear SVM classifier."""
        return LinearSVC(random_state=RANDOM_STATE) 

class LinearSVMEmbeddingClassifier(EmbeddingClassifier):
    """Classify dialog acts with a linear SVM trained on DistilBERT embeddings."""

    def create_estimator(self) -> BaseEstimator:
        """Create a linear SVM classifier."""
        return LinearSVC(random_state=RANDOM_STATE) 
