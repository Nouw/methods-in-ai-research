"""Load dialog-act datasets and normalize utterances for classification."""

from pathlib import Path

import pandas as pd

VALID_LABELS = set({
    "ack",
    "affirm",
    "bye",
    "confirm",
    "deny",
    "hello",
    "inform",
    "negate",
    "null",
    "repeat",
    "reqalts",
    "reqmore",
    "request",
    "restart",
    "thankyou",
})

def load_dialog_acts(file_path: str | Path) -> pd.DataFrame:
    """Read and validate a dataset containing one label and utterance per line.

    Skip blank lines and normalize utterances. Return a DataFrame with 'label' and 'utterance' columns.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If the dataset contains no valid records.
    """
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    records: list[dict[str, str]] = []
    errors: list[str] = []

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        # Ignore empty rows
        if not line:
            continue

        parts = line.split(maxsplit=1)

        if len(parts) != 2:
            errors.append(f"Line {line_number}: missing utterance")
            continue

        label, utterance = parts
        utterance = normalize_utterance(utterance) 

        if label not in VALID_LABELS:
            errors.append(f"Line {line_number}: unknown label {label}")
            continue

        if not utterance:
            errors.append(f"Line {line_number}: empty utterance")
            continue

        records.append({ "label": label, "utterance": utterance })

    if errors:
        preview = "\n".join(errors[:10])
        remaining = len(errors) - 10

        if remaining > 0:
            preview += f"\n... and {remaining} more errors"
            
        raise ValueError(f"Invalid dialog-act data: \n{preview}")

    if not records:
        raise ValueError(f"Dataset contains no valid records: {path}")

    return pd.DataFrame(records, columns=["label", "utterance"])

def normalize_utterance(utterance: str) -> str:
    """Remove leading and trailing whitespace and convert utterance to lowercase."""
    return utterance.strip().lower()

def preprocess(file_path: str) -> pd.DataFrame:
    """Read and print a single-column table, then add label and sentence columns by splitting at the first
    space."""
    df = pd.read_table(file_path, header=None, names=["Column"])
    print(df)
    df[["label", "sentence"]] = df["Column"].str.split(" ", n=1, expand=True)
    
    return df

def load_dataset(file_path: str | Path) -> pd.DataFrame:
    """Load a labeled .dat file or a split CSV file.

    Validate and normalize .dat records through load_dialog_acts.
    Read CSV files using 'row_id' as the index without normalizing their contents.

    Raises:
         ValueError: If the file type is not supported.
    """
    path = Path(file_path)

    if path.suffix.lower() == ".dat":
        return load_dialog_acts(path)

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, index_col="row_id", keep_default_na=False)

    raise ValueError(f"Unsupported dataset format: {path.suffix or '<none>'}. Expected .csv or .dat")


    
