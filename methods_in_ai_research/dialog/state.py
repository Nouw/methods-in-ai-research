"""Define the data carried between dialog turns.
"""
from dataclasses import dataclass, field
from enum import Enum
from methods_in_ai_research.dialog.reasoning import RestaurantProperty, Inference


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
    # The words the user said, shown when asking to confirm a corrected value.
    given: str = field(default="", compare=False)

@dataclass(frozen=True)
class ExclusionProposal:
    """A value the user does not want, e.g. "not chinese"."""
    slot: Slot
    value: str

@dataclass(frozen=True)
class UnknownProposal:
    """A food the user asked for that is not in the data, e.g. "swedish food"."""
    slot: Slot
    given: str

@dataclass(frozen=True)
class RequirementProposal:
    property: RestaurantProperty
    value: bool

class SystemActionType(str, Enum):
    WELCOME = "welcome"
    ASK_AREA = "ask_area"
    ASK_PRICE_RANGE = "ask_price_range"
    ASK_FOOD = "ask_food"
    CONFIRM_VALUE = "confirm_value"
    UNKNOWN_VALUE = "unknown_value"
    ASK_REQUIREMENT = "ask_requirement"
    RECOMMEND = "recommend"
    NO_MATCH = "no_match"
    NO_MATCH_REQUIREMENTS = "no_match_requirements"
    NO_ALTERNATIVE = "no_alternative"
    DETAIL = "detail"
    DETAIL_UNKNOWN = "detail_unknown"
    FACT_TRUE = "fact_true"
    FACT_FALSE = "fact_false"
    REPEAT = "repeat"
    CLARIFY = "clarify"
    ANYTHING_ELSE = "anything_else"
    BYE = "bye"

@dataclass(frozen=True)
class SystemAction:
    type: SystemActionType
    parameters: tuple[tuple[str, str], ...] = ()
    explanation: tuple[Inference, ...] = ()



@dataclass(frozen=True)
class DialogState:
    node: int
    food: Preference = None
    area: Preference = None
    price: Preference = None
    pending: tuple[SlotProposal, ...] = ()
    excluded: tuple[tuple[Slot, str], ...] = ()
    last_requested_slot: Slot | None = None
    requirements: tuple[tuple[RestaurantProperty, bool], ...] = ()
    requirements_asked: bool = False
    last_response: str | None = None
    current_restaurant_id: str | None = None
    alternative_ids: tuple[str, ...] = ()
    requested_detail: str | None = None
    # Values the user asks about, e.g. "is it in the centre?", checked against the current restaurant.
    checked: tuple[tuple[Slot, str], ...] = ()