"""Choose the next dialog state and system response.

TODO:
- Implement transition(state, classified_input) -> (next_state, response).
- Accept configured extraction, lookup, reasoning, and randomness dependencies.
- Update preferences and handle confirmations, corrections, and negation.
- Ask for missing preferences, then additional requirements.
- Handle recommendations, information requests, alternatives, and no matches.
- Handle repeat, restart, unclear input, and goodbye.
- Generate responses through templates; keep terminal I/O and logging outside.
"""