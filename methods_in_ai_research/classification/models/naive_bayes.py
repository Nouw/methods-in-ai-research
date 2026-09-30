"""Provide a multinomial naive Bayes classifier using TF-IDF features."""

from methods_in_ai_research.classification.models.classifier import BagOfWordsClassifier
from sklearn.base import BaseEstimator
from sklearn.naive_bayes import MultinomialNB

class NaiveBayesBagOfWordsClassifier(BagOfWordsClassifier):
    """Classify dialog acts with multinomial naive Bayes trained on TF-IDF features."""

    def create_estimator(self) -> BaseEstimator:
        """Create a multinomial naive Bayes estimator with default settings."""
        return MultinomialNB() 
