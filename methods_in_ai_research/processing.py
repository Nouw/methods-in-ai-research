import pandas as pd

def preprocess(file_path: str) -> pd.DataFrame:
    df = pd.read_table(file_path, header=None, names=["Column"])
    print(df)
    df[["label", "sentence"]] = df["Column"].str.split(" ", n=1, expand=True)
    
    return df 
    
