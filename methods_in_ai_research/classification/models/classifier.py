"""Define the classifier interface and shared TF-IDF and embedding implementations."""

from abc import ABC, abstractmethod
from typing import Iterable
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator
from dataclasses import dataclass
from methods_in_ai_research.embeddings import DistilBertEncoder
from methods_in_ai_research.classification.processing import normalize_utterance

class Classifier(ABC):
    """Define a shared interface for fitting classifiers and predicting dialog acts."""
    def __init__(self) -> None:
        """Initialize the base classifier."""
        super().__init__()

    def fit(self, utterances: Iterable[str], labels: Iterable[str]) -> "Classifier":
        """Return te classifier unchanged. Subclasses override this method when training is required."""
        return self

    def predict(self, utterances: Iterable[str]) -> list[str]:
        """Predict dialog acts based on utterances."""
        return [self.predict_one(utterance) for utterance in utterances]

    @abstractmethod
    def predict_one(self, utterance: str) -> str:
        """Return the predicted dialog-act label for one utterance."""
        raise NotImplementedError()
    
@dataclass(frozen=True)
class OOVStatistics:
    """Store token counts, the out-of-vocabulary rate, and the number of zero-vector utterances."""
    total_tokens: int
    oov_tokens: int
    oov_token_rate: float
    zero_vector_utterances: int

class BagOfWordsClassifier(Classifier, ABC):
    """Classify normalized utterances using TF-IDF unigram and bigram features."""
    def __init__(self) -> None:
        """Create a pipeline containing a TF-IDF vectorizer and the subclass's estimator."""
        super().__init__()

        self.pipeline = Pipeline([
            # Bigram are important e.g. "phone number"
            ("vectorizer", TfidfVectorizer(lowercase=False, ngram_range=(1, 2))),
            ("classifier", self.create_estimator())
        ])

    @abstractmethod
    def create_estimator(self) -> BaseEstimator:
        """Return an unfitted estimator for classifying TF-IDF feature vectors."""
        raise NotImplementedError

    def fit(self, utterances: Iterable[str], labels: Iterable[str]) -> "BagOfWordsClassifier":
        """Fit the TF-IDF vectorizer and estimator on normalized training utterances, then return this classifier."""
        normalized = [normalize_utterance(utterance) for utterance in utterances]

        self.pipeline.fit(normalized, list(labels))
        return self
    
    # Faster way since sklearn can predict multiple rows at once
    def predict(self, utterances: Iterable[str]) -> list[str]:
        """Normalize utterances and predict their labels in a batch using the fitted pipeline."""
        normalized = [normalize_utterance(utterance) for utterance in utterances]
        predictions = self.pipeline.predict(normalized)

        return predictions.tolist()

    def predict_one(self, utterance: str) -> str:
        """Normalize an utterance and terun its label from the fitted pipeline."""
        normalized = normalize_utterance(utterance) 
        prediction = self.pipeline.predict([normalized])

        return str(prediction[0])
        
    def calculate_oov_statistics(self, utterances: Iterable[str]):
        """Measure vocabulary coverage for utterances using the fitted vectorizer.

        Count individual tokens absent from the learned vocabulary and utterances whose TF-IDF vectors contain nonzero features.
        Return a zero OOV rate when there are no tokens.
        """
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
    """Classify utterances using pretrained DistilBERT embeddings and a separate estimator."""

    def __init__(self) -> None:
        """Load the DistilBERT encoder and create the subclass's estimator."""
        super().__init__() 

        self.encoder = DistilBertEncoder()
        self.estimator = self.create_estimator()

    @abstractmethod
    def create_estimator(self) -> BaseEstimator:
        """Return an unfitted estimator for classifying utterance embeddings."""
        raise NotImplementedError

    def fit(self, utterances: Iterable[str], labels: Iterable[str]) -> "EmbeddingClassifier":
        """Encode training utterances, fit the estimator without fine-tuning DistilBERT, and return this classifier."""
        utterances = list(utterances)
        embeddings = self.encoder.encode(utterances)
        # TODO: Fix typing
        # Ignore error because they do implement this
        self.estimator.fit(embeddings, list(labels))

        return self

    def predict(self, utterances: Iterable[str]) -> list[str]:
        """Encode utterances and return the fitted estimator's predictions in input order."""
        utterances = list(utterances)
        embeddings = self.encoder.encode(utterances)
        # TODO: Fix typing
        # Ignore error because they do implement this
        return self.estimator.predict(embeddings).tolist()

    def predict_one(self, utterance: str) -> str:
        """Encode a single utterance and return its predicted label."""
        return self.predict([utterance])[0]
