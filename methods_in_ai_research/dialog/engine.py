from typing import Protocol, Mapping
from dataclasses import replace

from methods_in_ai_research.dialog.flow import Flow
from methods_in_ai_research.dialog.state import DialogState, SystemAction


class Guard(Protocol):
    def __call__(self, state: DialogState) -> bool:
        """Check whether a transition is eligible without modifying the state."""
        pass

class Action(Protocol):
    def __call__(self, state: DialogState) -> tuple[DialogState, SystemAction]:
        """Return updated state and an abstract response when entering a node."""
        pass

class FlowEngine:
    def __init__(self, flow: Flow, guards: Mapping[str, Guard], actions: Mapping[str, Action], max_steps: int = 32):
        self.flow = flow
        self.guards = dict(guards)
        self.actions = dict(actions)
        self.nodes = { node.id: node for node in flow.nodes }
        self.max_steps = max_steps

        if max_steps < 1:
            raise ValueError("max_steps bust be positive")

        if len(self.nodes) != len(flow.nodes):
            raise ValueError("Node IDs must be unique")

        if flow.initial not in self.nodes:
            raise ValueError("The initial node does not exist")

        for node in flow.nodes:
            if node.action is not None and node.action not in self.actions:
                raise ValueError(f"Unknown action: {node.action}")

        for transition in flow.transitions:
            if transition.source not in self.nodes:
                raise ValueError(f"Unknown source: {transition.source}")
            if transition.target not in self.nodes:
                raise ValueError(f"Unknown target: {transition.target}")
            if transition.guard not in self.guards:
                raise ValueError(f"Unknown guard: {transition.guard}")

    def start(self) -> tuple[DialogState, SystemAction]:
        """Enter the initial node and follow decisions to an action."""
        state = DialogState(node=self.flow.initial)
        return self._enter(state)

    def advance(self, state: DialogState, event: str) -> tuple[DialogState, SystemAction]:
        """Consume one event and follow internal transitions to a response."""
        if state.node not in self.nodes:
            raise ValueError(f"Unknown current node: {state.node}")

        target = self._target(state, event)

        if target is None:
            return replace(state), SystemAction("clarify")

        return self._enter(replace(state, node=target))


    def _target(self, state: DialogState, event: str | None) -> int | None:
        """Find the first matching transition from the current node."""
        for transition in self.flow.transitions:
            if transition.source != state.node:
                continue
            if transition.event != event:
                continue
            if self.guards[transition.guard](state):
                return transition.target

        return None

    def _enter(self, state: DialogState) -> tuple[DialogState, SystemAction]:
        """Follow decision nodes until an entry action produces a response."""
        for _ in range(self.max_steps):
            node = self.nodes[state.node]

            if node.action is not None:
                return self.actions[node.action](state)

            target = self._target(state, event=None)

            if target is None:
                raise ValueError(f"No internal transition matched at {state.node}")

            state = replace(state, node=target)

        raise RuntimeError("Flow exceeded its internal step limit")