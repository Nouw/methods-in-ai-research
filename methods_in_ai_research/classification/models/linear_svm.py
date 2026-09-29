from methods_in_ai_research.classification.models.classifier import BagOfWordsClassifier, EmbeddingClassifier
from methods_in_ai_research.classification.splitting import RANDOM_STATE
from sklearn.base import BaseEstimator
from sklearn.svm import LinearSVC


class LinearSVMBagOfWordsClassifier(BagOfWordsClassifier):
    def create_estimator(self) -> BaseEstimator:
        return LinearSVC(random_state=RANDOM_STATE) 

class LinearSVMEmbeddingClassifier(EmbeddingClassifier):
    def create_estimator(self) -> BaseEstimator:
        return LinearSVC(random_state=RANDOM_STATE) 
