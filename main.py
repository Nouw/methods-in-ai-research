"""Provide command-line commands for classifier experiments, training, dialog, and evaluation."""

import argparse
import logging

from transformers.utils import logging as hf_logging

from methods_in_ai_research.classification import cli as classification_cli
from methods_in_ai_research.dialog import cli as dialog_cli


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the commands each package registers."""
    parser = argparse.ArgumentParser(description="Methods in AI Research dialog-act classification pipeline.")
    commands = parser.add_subparsers(dest="command", required=True)

    classification_cli.add_commands(commands)
    dialog_cli.add_commands(commands)

    return parser.parse_args()


def main():
    """Parse commandline args and run the selected command."""
    args = parse_arguments()
    args.handler(args)


if __name__ == "__main__":
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    file_handler = logging.FileHandler("app.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )

    # Disable huggingface logs
    hf_logging.set_verbosity_error()
    hf_logging.disable_default_handler()
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

    logging.basicConfig(
            level=logging.INFO,
            handlers=[
                console_handler,
                file_handler,
            ],
        )
    # Turn-by-turn dialog details only go to the log file, whose handler accepts DEBUG.
    logging.getLogger("methods_in_ai_research").setLevel(logging.DEBUG)

    main()
