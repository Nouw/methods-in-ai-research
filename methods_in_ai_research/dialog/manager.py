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

from methods_in_ai_research.dialog.engine import FlowEngine
from methods_in_ai_research.dialog.responses import ResponseRenderer
from methods_in_ai_research.dialog.slots import SlotExtractor
from methods_in_ai_research.dialog.state import DialogState, ClassifiedInput, SystemAction, SlotProposal, RequirementProposal, \
    SystemActionType

AFFIRM_ACTS = {"affirm"}
DENY_ACTS = {"negate", "deny"}
ANSWER_ACTS = AFFIRM_ACTS | DENY_ACTS


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
        slot_proposals = [p for p in proposals if isinstance(p, SlotProposal)]
        requirement_proposals = [p for p in proposals if isinstance(p, RequirementProposal)]

        confirmed, remaining = self._answer_pending(state.pending, classified_input.act)
        certain = [p for p in slot_proposals if not p.needs_confirmation]
        certain_slots = {p.slot for p in certain}
        pending = remaining + tuple(p for p in slot_proposals if p.needs_confirmation and p.slot not in certain_slots)

        # Confirmed values go first, so a value the user states in the same reply wins.
        updates = {p.slot.value: p.value for p in confirmed}
        stated = set()

        for proposal in certain:
            field = proposal.slot.value

            if field in stated:
                raise ValueError(f"Multiple proposals for slot: {field}")

            stated.add(field)
            updates[field] = proposal.value

        # Dicts keep insertion order, so requirements stay in the order the user mentioned them.
        requirements = dict(state.requirements)

        for proposal in requirement_proposals:
            requirements[proposal.property] = proposal.value

        updates["requirements"] = tuple(requirements.items())

        changed = any(getattr(state, field) != value for field, value in updates.items())

        next_state = replace(state, **updates)

        if changed:
            next_state = replace(next_state, current_restaurant_id=None, alternative_ids=())

        if pending:
            return self._respond(replace(next_state, pending=pending), self._confirm(pending[0]))

        # Answering a confirmation completes the earlier inform, so the flow continues from there.
        event = "inform" if state.pending and classified_input.act in ANSWER_ACTS else classified_input.act
        next_state, action = self.engine.advance(replace(next_state, pending=()), event)

        return self._respond(next_state, action)

    @staticmethod
    def _answer_pending(pending: tuple[SlotProposal, ...], act: str) -> tuple[tuple[SlotProposal, ...], tuple[SlotProposal, ...]]:
        """Split pending proposals into confirmed ones and ones still to ask about.

        Affirming accepts the first, denying drops it, and any other act drops them all
        because the user moved on.
        """
        if not pending or act not in ANSWER_ACTS:
            return (), ()

        first, rest = pending[0], pending[1:]

        if act in AFFIRM_ACTS:
            return (replace(first, needs_confirmation=False),), rest

        return (), rest

    @staticmethod
    def _confirm(proposal: SlotProposal) -> SystemAction:
        """Ask whether the user meant the corrected value."""
        return SystemAction(
            SystemActionType.CONFIRM_VALUE,
            parameters=(("given", proposal.given), ("corrected", str(proposal.value))),
        )

    def _respond(self, state: DialogState, action: SystemAction) -> tuple[DialogState, str]:
        """Render an action and remember the response for repeat requests."""
        response = self.renderer.render(action, state)
        return replace(state, last_response=response), response