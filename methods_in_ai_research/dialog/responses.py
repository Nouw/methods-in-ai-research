"""Render abstract system actions as template-based utterances.

TODO:
- Define templates for greetings, questions, confirmations, and clarification.
- Define templates for recommendations, details, alternatives, and no matches.
- Define templates for reasoning explanations and contradiction resolutions.
- Render actions using supplied slot values, restaurant details, and evidence.
- Support showing or hiding reasoning without changing recommendation logic.
- Keep state transitions, lookup, and inference outside this module.
"""
from typing import Protocol

from methods_in_ai_research.dialog.state import SystemAction, DialogState


class ResponseRenderer(Protocol):
    def render(self, action: SystemAction, state: DialogState) -> str:
        """Render an abstract action as text using templates and the current state."""
        pass