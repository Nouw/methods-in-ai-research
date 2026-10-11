"""Choose the next dialog state and system response."""
from dataclasses import replace

from methods_in_ai_research.dialog.engine import FlowEngine
from methods_in_ai_research.dialog.responses import ResponseRenderer
from methods_in_ai_research.dialog.slots import SlotExtractor
from methods_in_ai_research.dialog.state import DialogState, ClassifiedInput, SystemAction, SlotProposal, RequirementProposal, \
    SystemActionType, ExclusionProposal, UnknownProposal
from methods_in_ai_research.dialog.actions import requested_detail

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
        state, _, response = self.start_with_action()

        return state, response

    def transition(self, state: DialogState, classified_input: ClassifiedInput) -> tuple[DialogState, str]:
        """Extract preferences, update the state, execute the flow, and render.

        Return a new state and response text without modifying the input state.
        """
        next_state, _, response = self.transition_with_action(state, classified_input)

        return next_state, response

    def start_with_action(self) -> tuple[DialogState, SystemAction, str]:
        """Like start, but also return the chosen system action for logging."""
        state, action = self.engine.start()

        return self._respond(state, action)

    def transition_with_action(self, state: DialogState, classified_input: ClassifiedInput) -> tuple[DialogState, SystemAction | None, str]:
        """Like transition, but also return the chosen system action for logging.

        The action is None when the user asks to repeat, because the previous response is reused.
        """
        if classified_input.act == "repeat" and state.last_response is not None:
            return replace(state), None, state.last_response
        

        proposals = self.extractor.extract(classified_input, state)

        # "is it in the centre?" asks about the recommendation, so it must not change the preferences.
        if classified_input.act == "confirm" and state.current_restaurant_id is not None:
            checked = tuple((p.slot, p.value) for p in proposals if isinstance(p, SlotProposal) and isinstance(p.value, str))
            next_state, action = self.engine.advance(replace(state, checked=checked, pending=()), "confirm")
            return self._respond(next_state, action)
        slot_proposals = [p for p in proposals if isinstance(p, SlotProposal)]
        requirement_proposals = [p for p in proposals if isinstance(p, RequirementProposal)]
        exclusion_proposals = [p for p in proposals if isinstance(p, ExclusionProposal)]
        unknown_proposals = [p for p in proposals if isinstance(p, UnknownProposal)]
        newly_excluded = {(p.slot, p.value) for p in exclusion_proposals}

        confirmed, remaining = self._answer_pending(state.pending, classified_input.act)
        certain = [p for p in slot_proposals if not p.needs_confirmation]
        certain_slots = {p.slot for p in certain}
        pending = remaining + tuple(p for p in slot_proposals if p.needs_confirmation and p.slot not in certain_slots)
        pending = tuple(p for p in pending if (p.slot, p.value) not in newly_excluded)

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

        # Asking for a value again lifts its exclusion; excluding the current value forgets it.
        excluded = dict.fromkeys(state.excluded)

        for proposal in (*confirmed, *certain):
            excluded.pop((proposal.slot, proposal.value), None)

        for proposal in exclusion_proposals:
            excluded[(proposal.slot, proposal.value)] = None
            field = proposal.slot.value

            if updates.get(field, getattr(state, field)) == proposal.value:
                updates[field] = None

        updates["excluded"] = tuple(excluded)

        changed = any(getattr(state, field) != value for field, value in updates.items())

        next_state = replace(state, **updates)

        if changed:
            next_state = replace(next_state, current_restaurant_id=None, alternative_ids=())

        next_state = replace(next_state, requested_detail=requested_detail(classified_input.text))
        
        if pending:
            return self._respond(replace(next_state, pending=pending), self._confirm(pending[0]))

        # Stay at the same question, so the user can name something else.
        if unknown_proposals:
            action = SystemAction(SystemActionType.UNKNOWN_VALUE, parameters=(("given", unknown_proposals[0].given),))
            return self._respond(replace(next_state, pending=()), action)

        event = "inform" if state.pending and classified_input.act in ANSWER_ACTS else classified_input.act

        # The classifier often labels requirements it never saw in training ("a place for kids") as null,
        # but a recognised requirement is still information.
        if event == "null" and requirement_proposals:
            event = "inform"

        # "not chinese" is often labelled negate or deny, but it still tells which food the user wants.
        if event in {"null", *DENY_ACTS} and exclusion_proposals:
            event = "inform"

        # "how about chinese food" asks for something else, but with a new preference it changes the search.
        if event in {"reqalts", "reqmore"} and certain:
            event = "inform"

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

    def _respond(self, state: DialogState, action: SystemAction) -> tuple[DialogState, SystemAction, str]:
        """Render an action and remember the response for repeat requests."""
        response = self.renderer.render(action, state)
        return replace(state, last_response=response), action, response