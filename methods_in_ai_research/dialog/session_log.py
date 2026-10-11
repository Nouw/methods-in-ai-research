"""Write one JSON Lines file per conversation for the Part 2 analysis.

Every record carries the session id and a UTC timestamp. A session holds one start
record with the configuration, one record per turn, and one end record.
"""
import json
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from methods_in_ai_research.dialog.state import DialogState, SystemAction


def _to_json(value: Any) -> Any:
    """Convert enums that json cannot encode on its own."""
    if isinstance(value, Enum):
        return value.value

    raise TypeError(f"Cannot serialise {type(value).__name__}")


class SessionLog:
    def __init__(self, path: Path, session_id: str) -> None:
        self.path = path
        self.session_id = session_id

    @classmethod
    def create(cls, directory: str | Path) -> "SessionLog":
        """Start a new session with a fresh id in the given directory."""
        session_id = uuid.uuid4().hex
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        return cls(directory / f"{session_id}.jsonl", session_id)

    def start(self, config: dict[str, Any], state: DialogState, action: SystemAction, response: str) -> None:
        """Record the configuration and the opening response."""
        self._write("start", config=config, state=asdict(state), action=asdict(action), response=response)

    def turn(self, *, turn: int, utterance: str, act: str, state_before: DialogState, state_after: DialogState, action: SystemAction | None, response: str, latency_ms: float) -> None:
        """Record one user utterance and the system's reaction to it."""
        self._write(
            "turn",
            turn=turn,
            utterance=utterance,
            act=act,
            state_before=asdict(state_before),
            state_after=asdict(state_after),
            action=asdict(action) if action is not None else None,
            response=response,
            latency_ms=round(latency_ms, 3),
        )

    def end(self, *, reason: str, turns: int, state: DialogState) -> None:
        """Record why the conversation ended and the final state."""
        self._write("end", reason=reason, turns=turns, state=asdict(state))

    def _write(self, event: str, **fields: Any) -> None:
        # Append and close every time, so a crashed session still leaves its earlier turns.
        record = {
            "session_id": self.session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **fields,
        }

        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, default=_to_json) + "\n")
