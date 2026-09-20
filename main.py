import argparse
import logging
from methods_in_ai_research.processing import preprocess, load_dialog_acts
from methods_in_ai_research.splitting import create_original_split, create_grouped_split, log_split_summary, validate_split, save_split

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True, help="Path to a dialog-act .dat file")
    args = parser.parse_args()

    data = load_dialog_acts(args.data_path)

    logger.info(f"Loaded {len(data)} records")
    logger.info(f"Found {data['label'].nunique()} labels")
    logger.info(f"Columns: {list(data.columns)}")
    logger.info(f"Missing values: {data.isna().sum().sum()}")

    original_train, original_test = create_original_split(data)

    validate_split(data, original_train, original_test)
    log_split_summary("Original", data, original_train, original_test)

    grouped_train, grouped_test = create_grouped_split(data)

    validate_split(data, grouped_train, grouped_test)
    log_split_summary("Grouped", data, grouped_train, grouped_test)
    # TODO: Make path environment variable 
    save_split(original_train, original_test, "artifacts/splits/original")
    save_split(grouped_train, grouped_test, "artifacts/splits/grouped")

    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    main()
