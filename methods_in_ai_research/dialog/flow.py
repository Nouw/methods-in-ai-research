"""

TODO: Change ints to pointers in transition.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Node:
    id: int
    name: str
    action: str | None = None

@dataclass(frozen=True)
class Transition:
    source: int
    target: int
    event: str | None = None
    guard: str = "always"

@dataclass(frozen=True)
class Flow:
    initial: int
    nodes: tuple[Node, ...]
    transitions: tuple[Transition, ...]