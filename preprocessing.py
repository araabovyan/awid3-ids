import os

import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import read_csvs


DATA_DIR="/data/ara/CSV/"

files_dict = {}
for folder in os.listdir(DATA_DIR):
    subroot = os.path.join(DATA_DIR, folder)
    files_dict[folder] = [os.path.join(subroot, file) for file in os.listdir(subroot)]


print(files_dict.keys())


# attack_list = ['1.Deauth', '7.SSH', '10.SQL_Injection', '12.Evil_Twin', '13.Website_spoofing']

for attack in files_dict.keys():
    print(attack)
    data = read_csvs(files_dict[attack])

    print('Dropingn columns with unique values: ')
    drop_cols = []
    for col in data.columns:
        if len(data[col].unique()) == 1:
            drop_cols.append(col)

    data.drop(drop_cols, axis=1, inplace=True)
    print(", ".join(drop_cols))

    print('Dropingn columns with more than 1 million missing values: ')
    cols_without_missing_values = [c for c in data if data[c].isnull().sum() < 1000000]
    data = data[cols_without_missing_values]
    print(', '.join(cols_without_missing_values))
    print(data.isna().sum())

    data.drop(['frame.time'], axis=1, inplace=True)
    print('Object columns: ')
    print(data.describe(include='object'))

    print(data['Label'].value_counts())

    data.to_csv('/data/ara/' + attack + '.csv', index=False)
