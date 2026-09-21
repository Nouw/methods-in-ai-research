from abc import ABC, abstractmethod
from typing import Iterable

class Classifier(ABC):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, utterances: Iterable[str], labels: Iterable[str]) -> "Classifier":
        return self

    def predict(self, utterances: Iterable[str]) -> list[str]:
        return [self.predict_one(utterance) for utterance in utterances]

    @abstractmethod
    def predict_one(self, utterance: str) -> str:
        raise NotImplementedError()
    
