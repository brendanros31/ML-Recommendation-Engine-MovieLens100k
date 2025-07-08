import pandas as pd


# Loading data
def load_data(path, usecols):
    df = pd.read_csv(path, usecols=usecols)
    return df
