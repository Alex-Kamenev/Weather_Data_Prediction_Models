import pandas as pd

_df = None

def load_df(location):
    _df = pd.read_csv(location)
    return _df