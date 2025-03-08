import pandas as pd

def process_csv(filepath):
    df = pd.read_csv(filepath)
    return df
