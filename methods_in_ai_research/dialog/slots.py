"""Extract structured preferences and requests from user utterances.

TODO:
- Match keywords and patterns first, including multiword values.
- Extract multiple slots, negation, "don't care", and requested information.
- Use dialog context to interpret short replies such as "any" or "yes".
- Support runtime selection of Levenshtein or DistilBERT similarity fallback.
- Reject weak or ambiguous matches.
- Return uncertain matches for confirmation, without changing dialog state.
- Reuse the shared DistilBERT encoder and cache ontology embeddings.
"""
from typing import Protocol

from methods_in_ai_research.dialog.state import ClassifiedInput, DialogState, SlotProposal


class SlotExtractor(Protocol):
    def extract(self, user_input: ClassifiedInput, state: DialogState) -> tuple[SlotProposal, ...]:
        """Extract preferences using the utterance and conversation context.

        Flag uncertain matches for confirmation. Do not modify state.
        """
        pass