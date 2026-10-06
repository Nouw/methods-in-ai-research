"""Load restaurant data and retrieve matching restaurants.

TODO:
- Load and validate the basic or extended restaurant CSV.
- Preserve phone numbers and postcodes as strings; handle missing values.
- Match committed preferences and exclusions, respecting "don't care".
- Return matching restaurants with stable identifiers.
- Provide restaurant details for follow-up questions.
- Leave conversation state and alternative tracking to the dialog manager.
"""


"""Load the restaurant data and find restaurants that match the preferences."""

import pandas as pd


def load_restaurants(file_path: str) -> pd.DataFrame:
    """Load the restaurant csv as text and lowercase everything"""
    data = pd.read_csv(file_path, dtype=str, keep_default_na=False)

    for column in data.columns:
        data[column] = data[column].str.strip().str.lower()

    return data


def find_restaurants(data: pd.DataFrame, pricerange=None, area=None, food=None, excluded=None) -> pd.DataFrame:
    """Return the restaurants matching the preferences.

    A preference of None means the user has no preference (or said "any"),
    so that column is not filtered on. Excluded is a dict like
    {"food": ["vietnamese"]} with values the user has rejected.
    """
    matches = data

    for column, value in [("pricerange", pricerange), ("area", area), ("food", food)]:
        if value is not None:
            matches = matches[matches[column] == value]

    if excluded:
        for column, values in excluded.items():
            matches = matches[~matches[column].isin(values)]

    return matches


def restaurant_details(data: pd.DataFrame, name: str) -> dict | None:
    """Return one restaurant as a dict, or None if it is not in the data"""
    rows = data[data["restaurantname"] == name]

    if rows.empty:
        return None

    return rows.iloc[0].to_dict()

if __name__ == "__main__":
    data = load_restaurants("data/restaurant_info.csv")
    print(len(data))
    print(find_restaurants(data, pricerange="cheap", area="centre")[["restaurantname", "food"]])
    print(restaurant_details(data, "the oak bistro"))