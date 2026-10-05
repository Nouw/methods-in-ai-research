"""Define the data carried between dialog turns.

TODO:
- Define dialog phases using an enum.
- Define dataclasses for dialog state and classified user input (act + text).
- Distinguish unknown preferences, explicit "don't care", and specific values.
- Track excluded values, pending confirmations, and the last requested slot.
- Track additional requirements and whether they have been collected.
- Track the current restaurant, remaining alternatives, and previous response.
"""
from dataclasses import dataclass
from enum import Enum

class Slot(Enum):
    """Define the possible slots for dialog state."""
    FOOD = "food"
    AREA = "area"
    PRICE = "price"

class AnyValue(Enum):
    """When no preference is defined use this type."""
    ANY = "any"

# None means unknown; ANY means explicitly no preference.
Preference = str | AnyValue | None

@dataclass(frozen=True)
class ClassifiedInput:
    act: str
    text: str


@dataclass(frozen=True)
class SlotProposal:
    slot: Slot
    value: str | AnyValue
    needs_confirmation: bool = False

@dataclass(frozen=True)
class SystemAction:
    name: str
    parameters: tuple[tuple[str, str], ...] = ()

@dataclass(frozen=True)
class DialogState:
    node: int
    food: Preference = None
    area: Preference = None
    price: Preference = None
    pending: tuple[SlotProposal, ...] = ()
    last_requested_slot: Slot | None = None
    last_response: str | None = None
    current_restaurant_id: str | None = None
    alternative_ids: tuple[str, ...] = ()