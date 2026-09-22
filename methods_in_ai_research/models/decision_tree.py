from methods_in_ai_research.models.classifier import BagOfWordsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator

from methods_in_ai_research.splitting import RANDOM_STATE

class DecisionTreeBagOfWordsClassifier(BagOfWordsClassifier):
    def create_estimator(self) -> BaseEstimator:
        return DecisionTreeClassifier(random_state=RANDOM_STATE)
    
