import re
from collections.abc import Iterable

from methods_in_ai_research.models.classifier import Classifier
from methods_in_ai_research.processing import normalize_utterance

def contains_phrase(
    utterance: str,
    phrases: str | Iterable[str],
) -> bool:
    if isinstance(phrases, str):
        phrases = (phrases,)

    for phrase in phrases:
        pattern = rf"(?<!\w){re.escape(phrase)}(?!\w)"

        if re.search(pattern, utterance):
            return True

    return False

class RuleBasedClassifier(Classifier): 
    def predict_one(self, utterance: str) -> str:
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

