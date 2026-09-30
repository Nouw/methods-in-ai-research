"""Encode utterances as mean-pooled vectors from pretrained DistilBERT."""

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

class DistilBertEncoder:
    """Produce utterance embeddings with DistilBERT without fine-tuning its weights."""

    def __init__(self) -> None:
        """Load the uncased DistilBERT tokenizer and model and enable evaluation mode."""
        model_name = "distilbert/distilbert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name) 
        self.model = AutoModel.from_pretrained(model_name)
        # Freeze model
        self.model.eval()

    def encode(self, utterances: list[str], batch_size: int = 32) -> np.ndarray:
        """Return one mean-pooled DistilBERT embedding per utterance in input order.

        Process a nonempty list in batches, padding and truncating tokenized input.
        Average token vectors using the attention mask to exclude padding.
        Run inference without gradient tracking.

        Return a NumPy array with shape (number of utterances, hidden size).
        """
        embeddings = []

        for start in range(0, len(utterances), batch_size):
            batch = utterances[start : start + batch_size]

            tokens = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")

            with torch.no_grad():
                output = self.model(**tokens)

            mask = tokens["attention_mask"].unsqueeze(-1)
            token_vectors = output.last_hidden_state

            summed = (token_vectors * mask).sum(dim=1)
            counts = mask.sum(dim=1)
            batch_embeddings = summed / counts

            embeddings.append(batch_embeddings.numpy())
        
        return np.concatenate(embeddings)
