import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

class DistilBertEncoder:
    def __init__(self) -> None:
        model_name = "distilbert/distilbert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name) 
        self.model = AutoModel.from_pretrained(model_name)
        # Freeze model
        self.model.eval()

    def encode(self, utterances: list[str], batch_size: int = 32) -> np.ndarray:
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
