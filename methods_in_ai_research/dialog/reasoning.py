"""Derive restaurant properties using the six assignment rules."""
from dataclasses import dataclass
from enum import Enum
from collections.abc import Mapping


class RestaurantProperty(str, Enum):
    TOURISTIC = "touristic"
    ASSIGNED_SEATS = "assigned seats"
    CHILDREN = "children"
    ROMANTIC = "romantic"

@dataclass(frozen=True)
class Rule:
    id: int
    antecedent: tuple[tuple[str, str], ...]
    consequent: RestaurantProperty
    value: bool
    priority: int = 0

RULES = (
    Rule(1, (("pricerange", "cheap"), ("food quality", "good")), RestaurantProperty.TOURISTIC, True),
    Rule(2, (("food", "romanian"),), RestaurantProperty.TOURISTIC, False, priority=1),
    Rule(3, (("crowdedness", "busy"),), RestaurantProperty.ASSIGNED_SEATS, True),
    Rule(4, (("length of stay", "long"),), RestaurantProperty.CHILDREN, False),
    Rule(5, (("crowdedness", "busy"),), RestaurantProperty.ROMANTIC, False, priority=1),
    Rule(6, (("length of stay", "long"),), RestaurantProperty.ROMANTIC, True),
)

class RestaurantField(str, Enum):
    NAME = "restaurantname"
    PRICE_RANGE = "pricerange"
    AREA = "area"
    FOOD = "food"
    PHONE = "phone"
    ADDRESS = "addr"
    POSTCODE = "postcode"
    FOOD_QUALITY = "food quality"
    CROWDEDNESS = "crowdedness"
    LENGTH_OF_STAY = "length of stay"

# Assumed value when no rule fires for a property.
DEFAULTS = {
    RestaurantProperty.TOURISTIC: False,
    RestaurantProperty.ASSIGNED_SEATS: False,
    RestaurantProperty.CHILDREN: True,
    RestaurantProperty.ROMANTIC: False,
}

@dataclass(frozen=True)
class Inference:
    property: RestaurantProperty
    value: bool
    rule: Rule | None = None
    overruled: tuple[Rule, ...] = ()

def infer(restaurant: Mapping[str, str]) -> dict[RestaurantProperty, Inference]:
    """Apply every rule that fires and resolve condtradictions by priority."""
    fired = [rule for rule in RULES if all(restaurant[column] == value for column, value in rule.antecedent)]
    inferences = {}

    for property_ in RestaurantProperty:
        rules = sorted((rule for rule in fired if rule.consequent is property_), key=lambda rule: rule.priority, reverse=True)

        if not rules:
            inferences[property_] = Inference(property_, DEFAULTS[property_])
            continue

        winner = rules[0]
        overruled = tuple(rule for rule in rules[1:] if rule.value != winner.value)
        inferences[property_] = Inference(property_, winner.value, winner, overruled)

    return inferences

def satisfies(inferences: Mapping[RestaurantProperty, Inference], requirements: Mapping[RestaurantProperty, bool]) -> bool:
    return all(inferences[property_].value == wanted for property_, wanted in requirements.items())

