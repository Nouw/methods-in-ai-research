from methods_in_ai_research.models.classifier import BagOfWordsClassifier, EmbeddingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
from methods_in_ai_research.splitting import RANDOM_STATE

class LogisticRegressionBagOfWordsClassifier(BagOfWordsClassifier):
    def create_estimator(self) -> BaseEstimator:
        return LogisticRegression(max_iter=1000, random_state=RANDOM_STATE) 

class LogisticRegressionEmbeddingClassifier(EmbeddingClassifier):
    def create_estimator(self) -> BaseEstimator:
        return LogisticRegression(max_iter=1000, random_state=RANDOM_STATE) 
