"""Choose the next dialog state and system response.

TODO:
- Implement transition(state, classified_input) -> (next_state, response).
- Accept configured extraction, lookup, reasoning, and randomness dependencies.
- Update preferences and handle confirmations, corrections, and negation.
- Ask for missing preferences, then additional requirements.
- Handle recommendations, information requests, alternatives, and no matches.
- Handle repeat, restart, unclear input, and goodbye.
- Generate responses through templates; keep terminal I/O and logging outside.
"""
from dataclasses import replace
from typing import Protocol

from methods_in_ai_research.dialog.engine import FlowEngine
from methods_in_ai_research.dialog.responses import ResponseRenderer
from methods_in_ai_research.dialog.slots import SlotExtractor
from methods_in_ai_research.dialog.state import DialogState, ClassifiedInput, SystemAction


class DialogManager:
    def __init__(self, engine: FlowEngine, extractor: SlotExtractor, renderer: ResponseRenderer) -> None:
        """Connect the flow engine, extractor, and response renderer."""
        self.engine = engine
        self.extractor = extractor
        self.renderer = renderer

    def start(self) -> tuple[DialogState, str]:
        """Create a fresh conversation state and render the opening response."""
        state, action = self.engine.start()

        return self._respond(state, action)

    def transition(self, state: DialogState, classified_input: ClassifiedInput) -> tuple[DialogState, str]:
        """Extract preferences, update the state, execute the flow, and render.

        Return a new state and response text without modifying the input state.
        """
        if classified_input.act == "repeat" and state.last_response is not None:
            return replace(state), state.last_response

        proposals = self.extractor.extract(classified_input, state)

        if state.pending or any(p.needs_confirmation for p in proposals):
            raise NotImplementedError("Confirmation handling is not implemented yet.")

        updates = {}

        for proposal in proposals:
            field = proposal.slot.value

            if field in updates:
                raise ValueError(f"Multiple proposals for slot: {field}")

            updates[field] = proposal.value

        changed = any(getattr(state, field) != value for field, value in updates.items())

        next_state = replace(state, **updates)

        if changed:
            next_state = replace(next_state, current_restaurant_id=None, alternative_ids=())

        next_state, action = self.engine.advance(next_state, classified_input.act)

        return self._respond(next_state, action)

    def _respond(self, state: DialogState, action: SystemAction) -> tuple[DialogState, str]:
        """Render an action and remember the response for repeat requests."""
        response = self.renderer.render(action, state)
        return replace(state, last_response=response), response
