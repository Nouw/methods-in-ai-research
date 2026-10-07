"""Load restaurant data and retrieve matching restaurants.

TODO:
- Load and validate the basic or extended restaurant CSV.
- Preserve phone numbers and postcodes as strings; handle missing values.
- Match committed preferences and exclusions, respecting "don't care".
- Return matching restaurants with stable identifiers.
- Provide restaurant details for follow-up questions.
- Leave conversation state and alternative tracking to the dialog manager.
"""
from collections.abc import Mapping

from methods_in_ai_research.dialog.state import AnyValue, Slot, Preference

"""Load the restaurant data and find restaurants that match the preferences."""

import pandas as pd

PREFERENCE_COLUMNS = {Slot.FOOD: "food", Slot.AREA: "area", Slot.PRICE: "pricerange"}

def find_restaurants(data: pd.DataFrame, preferences: Mapping[Slot, Preference]) -> pd.DataFrame:
    """Return rows matching every concrete preference: None and ANY do not filter."""
    matches = data

    for slot, value in preferences.items():
        if value is None or value is AnyValue.ANY:
            continue

        matches = matches[matches[PREFERENCE_COLUMNS[slot]] == value]

    return matches

def load_restaurants(file_path: str) -> pd.DataFrame:
    """Load the restaurant csv as text and lowercase everything"""
    data = pd.read_csv(file_path, dtype=str, keep_default_na=False)

    for column in data.columns:
        data[column] = data[column].str.strip().str.lower()

    return data


def restaurant_details(data: pd.DataFrame, name: str) -> dict | None:
    """Return one restaurant as a dict, or None if it is not in the data"""
    rows = data[data["restaurantname"] == name]

    if rows.empty:
        return None

    return rows.iloc[0].to_dict()

if __name__ == "__main__":
    data = load_restaurants("data/restaurant_info.csv")
    print(len(data))
    print(find_restaurants(data, {Slot.PRICE: "cheap", Slot.AREA: "centre"}))
    print(restaurant_details(data, "the oak bistro"))