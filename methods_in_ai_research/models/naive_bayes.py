from methods_in_ai_research.models.classifier import BagOfWordsClassifier
from sklearn.base import BaseEstimator
from sklearn.naive_bayes import MultinomialNB

class NaiveBayesBagOfWordsClassifier(BagOfWordsClassifier):
    def create_estimator(self) -> BaseEstimator:
        return MultinomialNB() 
