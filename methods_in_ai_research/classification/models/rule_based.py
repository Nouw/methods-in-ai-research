"""Classify dialog acts using ordered prhase-matching rules."""

import re
from collections.abc import Iterable

from methods_in_ai_research.classification.models.classifier import Classifier
from methods_in_ai_research.classification.processing import normalize_utterance

def contains_phrase(
    utterance: str,
    phrases: str | Iterable[str],
) -> bool:
    """Return whether any supplied phrase appears without adjacent word characters.

    Treat a single string as one phrase. Matching is case-sensitive and phrases are interpreted literally rather than as
    regular expressions.
    """
    if isinstance(phrases, str):
        phrases = (phrases,)

    for phrase in phrases:
        pattern = rf"(?<!\w){re.escape(phrase)}(?!\w)"

        if re.search(pattern, utterance):
            return True

    return False

class RuleBasedClassifier(Classifier):
    """Predict dialog acts using the first matching rule, defaulting to 'inform'."""

    def predict_one(self, utterance: str) -> str:
        """Normalize the utterance and return the first matching rule's label or 'inform'."""
        text = normalize_utterance(utterance)

        if contains_phrase(text, ("kay", "okay")):
            return "ack"

        if contains_phrase(text, ("yes",)):
            return "affirm"

        if contains_phrase(text, ("thank you", "thanks")):
            return "thankyou"

        if contains_phrase(text, ("bye",)):
            return "bye"
        
        if contains_phrase(text, ("does it", "is it", "is that")):
            return "confirm"

        if contains_phrase(text, ("wrong", "dont", "change")):
            return "deny"

        if contains_phrase(text, ("hello", "hi")):
            return "hello"

        if contains_phrase(text, ("no",)):
            return "negate"
        
        if contains_phrase(text, ("again", "repeat")):
            return "repeat"

        if contains_phrase(text, ("how about", "is there", "what about")):
            return "reqalts"
        
        if contains_phrase(text, ("more", "more please")):
            return "reqmore"

        if contains_phrase(text, ("whats", "what is", "may i", "cost", "address", "post code", "phone", "area")):
            return "request"

        if contains_phrase(text, ("start over",)):
            return "restart"
        
        if contains_phrase(text, ("noise", "tv_noise", "sil", "cough", "unintelligible", "code", "um", "um hm")):
            return "null"


        return "inform" 

