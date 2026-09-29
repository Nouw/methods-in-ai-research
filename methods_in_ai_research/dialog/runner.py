"""Run the terminal conversation and record interaction logs.

TODO:
- Accept a loaded classifier and configured dialog manager.
- Initialize the conversation and display the opening response.
- Read input, classify it, call the transition function, and display its response.
- Support injectable input/output functions for scripted conversation tests.
- Handle /exit, EOF, and keyboard interruption gracefully.
- Log session configuration and structured turn data, including predictions,
  state changes, responses, timing, and recommendation evidence.
- Keep conversation decisions inside the dialog manager.
"""