import os

import pandas as pd
import numpy as np
from tqdm import tqdm


DATA_DIR="/data/security/CSV/"

files_dict = {}
for folder in os.listdir(DATA_DIR):
    subroot = os.path.join(DATA_DIR, folder)
    files_dict[folder] = [os.path.join(subroot, file) for file in os.listdir(subroot)]


print(files_dict.keys())

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

# attack_list = ['1.Deauth', '7.SSH', '10.SQL_Injection', '12.Evil_Twin', '13.Website_spoofing']
dfs = [read_csvs(files_dict[attack]) for attack in tqdm(files_dict.keys())]
data = pd.concat(dfs, ignore_index=True)

print(data['Label'].value_counts())

data.to_csv('awid.csv', index=False)
