import pandas as pd
import time
import logging

from pathlib import Path
from sklearn.model_selection import (
        StratifiedGroupKFold,
        train_test_split,
    )


# TODO: Move to .env
RANDOM_STATE = 42
TEST_SIZE = 0.15

logger = logging.getLogger(__name__)


def create_original_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_data, test_data = train_test_split(data, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True, stratify=data["label"])

    return train_data.copy(), test_data.copy()

def create_grouped_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    full_disctribution = data["label"].value_counts(normalize=True)
    all_labels = set(data["label"])

    best_split = None
    best_score = float("inf")

    candidates = splitter.split(X=data, y=data["label"], groups=data["utterance"])

    for train_indices, test_indices in candidates:
        train_candidate = data.iloc[train_indices]
        test_candidate = data.iloc[test_indices]

        groups_per_label = data.groupby("label")[
            "utterance"
        ].nunique()

        required_test_labels = set(
            groups_per_label[groups_per_label > 1].index
        )

        # Every label should exist
        if set(train_candidate["label"]) != all_labels:
            continue
        
        if not required_test_labels.issubset(
            set(test_candidate["label"])
        ):
            continue

        test_fraction = len(test_candidate) / len(data)
        size_error = abs(test_fraction - TEST_SIZE)

        test_distribution = (test_candidate["label"].value_counts(normalize=True).reindex(full_disctribution.index, fill_value=0))

        distribution_error = (test_distribution - full_disctribution).abs().mean()

        score = size_error + distribution_error

        if score < best_score:
            best_score = score
            best_split = (train_indices, test_indices)

    if best_split is None:
        raise ValueError("Could not construct a valid grouped split")

    train_indices, test_indices = best_split

    return (data.iloc[train_indices].copy(), data.iloc[test_indices].copy())


def validate_split(source: pd.DataFrame, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    train_indices = set(train_data.index)
    test_indices = set(test_data.index)
    source_indices = set(source.index)

    # Check for leakage
    if train_indices & test_indices:
        raise ValueError("Train and test contain overlapping rows")
    # Train and test should be created from source
    if train_indices | test_indices != source_indices:
        raise ValueError("Train and test do not cover every source row")

def save_split(train_data: pd.DataFrame, test_data: pd.DataFrame, output_directory: str | Path):
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    
    train_data.to_csv(output_path / f"train.csv", index=True, index_label="row_id")
    test_data.to_csv(output_path / f"test.csv", index=True, index_label="row_id")

def log_split_summary(
    name: str,
    source: pd.DataFrame,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> None:
    total = len(source)
    train_percentage = 100 * len(train_data) / total
    test_percentage = 100 * len(test_data) / total
    
    logger.info("===[SPLIT SUMMARY]===")
    logger.info(f"{name} split:")
    logger.info(f"Training {len(train_data)} records {train_percentage}")
    logger.info(f"Testing {len(test_data)} records {test_percentage}") 
    logger.info(f"Training labels: {train_data['label'].nunique()}") 
    logger.info(f"Testing labels: {test_data['label'].nunique()}") 
 
