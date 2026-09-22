from abc import ABC, abstractmethod
from typing import Iterable
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator
from dataclasses import dataclass
from methods_in_ai_research.embeddings import DistilBertEncoder
from methods_in_ai_research.processing import normalize_utterance

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
    
@dataclass(frozen=True)
class OOVStatistics:
    total_tokens: int
    oov_tokens: int
    oov_token_rate: float
    zero_vector_utterances: int

class BagOfWordsClassifier(Classifier, ABC):
    def __init__(self) -> None:
        super().__init__()

        self.pipeline = Pipeline([
            # Bigram are important e.g. "phone number"
            ("vectorizer", TfidfVectorizer(lowercase=False, ngram_range=(1, 2))),
            ("classifier", self.create_estimator())
        ])

    @abstractmethod
    def create_estimator(self) -> BaseEstimator:
        raise NotImplementedError

    def fit(self, utterances: Iterable[str], labels: Iterable[str]) -> "BagOfWordsClassifier":
        normalized = [normalize_utterance(utterance) for utterance in utterances]

        self.pipeline.fit(normalized, list(labels))
        return self
    
    # Faster way since sklearn can predict multiple rows at once
    def predict(self, utterances: Iterable[str]) -> list[str]:
        normalized = [normalize_utterance(utterance) for utterance in utterances]
        predictions = self.pipeline.predict(normalized)

        return predictions.tolist()

    def predict_one(self, utterance: str) -> str:
        normalized = normalize_utterance(utterance) 
        prediction = self.pipeline.predict([normalized])

        return str(prediction[0])
        
    def calculate_oov_statistics(self, utterances: Iterable[str]):
        normalized = [normalize_utterance(utterance) for utterance in utterances]

        vectorizer = (self.pipeline.named_steps["vectorizer"])
        vocabulary = vectorizer.vocabulary_
        preprocessor = vectorizer.build_preprocessor()
        tokenizer = vectorizer.build_tokenizer()

        total_tokens = 0
        oov_tokens = 0

        for utterance in normalized:
            tokens = tokenizer(preprocessor(utterance)) 

            total_tokens += len(tokens)
            oov_tokens += sum(token not in vocabulary for token in tokens)

        vectors = vectorizer.transform(normalized)
        zero_vector_utterances = int((vectors.getnnz(axis=1) == 0).sum())

        oov_rate = (oov_tokens / total_tokens if total_tokens else 0.0) 

        return OOVStatistics(
            total_tokens=total_tokens,
            oov_tokens=oov_tokens,
            oov_token_rate=oov_rate,
            zero_vector_utterances=zero_vector_utterances,
        )

class EmbeddingClassifier(Classifier, ABC):
    def __init__(self) -> None:
        super().__init__() 

        self.encoder = DistilBertEncoder()
        self.estimator = self.create_estimator()

    @abstractmethod
    def create_estimator(self) -> BaseEstimator:
        raise NotImplementedError

    def fit(self, utterances: Iterable[str], labels: Iterable[str]) -> "EmbeddingClassifier":
        utterances = list(utterances)
        embeddings = self.encoder.encode(utterances)
        # TODO: Fix typing
        # Ignore error because they do implement this
        self.estimator.fit(embeddings, list(labels))

        return self

    def predict(self, utterances: Iterable[str]) -> list[str]:
        utterances = list(utterances)
        embeddings = self.encoder.encode(utterances)
        # TODO: Fix typing
        # Ignore error because they do implement this
        return self.estimator.predict(embeddings).tolist()

    def predict_one(self, utterance: str) -> str:
        return self.predict([utterance])[0]
