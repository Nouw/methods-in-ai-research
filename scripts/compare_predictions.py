#Merge per model predictions into one comparison table

import sys
from pathlib import Path
import pandas as pd

split = sys.argv[1] if len(sys.argv) > 1 else "provided"
base = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results")
frames = []

for path in sorted(base.glob(f"*/{split}/predictions.csv")):
    model = path.parts[-3]
    df = pd.read_csv(path, keep_default_na=False)
    frames.append(df.set_index(["utterance", "expected_label"])["predicted_label"].rename(model))

table = pd.concat(frames, axis=1).reset_index()
table["n_wrong"] = (table.iloc[:, 2:].ne(table["expected_label"], axis=0)).sum(axis=1)
table = table.sort_values("n_wrong", ascending=False)
table.to_csv(base / f"comparison_{split}.csv", index=False)
print(table.to_string(index=False))