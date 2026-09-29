from methods_in_ai_research.classification.models.classifier import BagOfWordsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator

from methods_in_ai_research.classification.splitting import RANDOM_STATE

class DecisionTreeBagOfWordsClassifier(BagOfWordsClassifier):
    def create_estimator(self) -> BaseEstimator:
        return DecisionTreeClassifier(random_state=RANDOM_STATE)
    
