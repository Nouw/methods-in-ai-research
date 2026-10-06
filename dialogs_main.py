"""Interactive driver for SlotExtractor. Install PyYAML: pip install pyyaml."""
from __future__ import annotations
import argparse
from pathlib import Path
import yaml
from slot_extractor import SlotExtractor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/slot_extractor_config.yaml"))
    args = parser.parse_args()

    config_path = args.config.resolve()
    with config_path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    extractor = SlotExtractor(config, base_dir=config_path.parent)
    interaction = config["interaction"]
    exit_commands = {str(x).casefold() for x in interaction["exit_commands"]}

    while True:
        try:
            user_utterance = input(interaction["prompt"])
        except EOFError:
            break
        if user_utterance.strip().casefold() in exit_commands:
            break
        if not user_utterance.strip():
            continue
        slot_string = extractor.extract_slots(user_utterance)
        print(f"{interaction['system_prefix']}{slot_string}", flush=True)


if __name__ == "__main__":
    main()
