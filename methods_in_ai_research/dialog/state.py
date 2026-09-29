"""Define the data carried between dialog turns.

TODO:
- Define dialog phases using an enum.
- Define dataclasses for dialog state and classified user input (act + text).
- Distinguish unknown preferences, explicit "don't care", and specific values.
- Track excluded values, pending confirmations, and the last requested slot.
- Track additional requirements and whether they have been collected.
- Track the current restaurant, remaining alternatives, and previous response.
"""