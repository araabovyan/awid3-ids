from tqdm import tqdm
import pandas as pd

def read_csvs(path_list):
    """Creates DataFrame from list of CSV paths

    Args:
        path_list (List): list of paths

    Returns:
        DataFrame: dataframe constructed by concatinating
    """
    dfs = [pd.read_csv(f, low_memory=False) for f in tqdm(path_list)]
    df = pd.concat(dfs, ignore_index=True)
    return df