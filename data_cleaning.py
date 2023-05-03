import os

import pandas as pd
from sklearn.model_selection import train_test_split
from utils import read_csvs

DATA_DIR="/data/ara/processed/"
SAVE_DIR="/data/ara/cleaned/"
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)


df = read_csvs([os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR)])


null_col_to_drop = [c for c in df if df[c].isnull().sum() > 1000000]
objects_col_to_drop = ['radiotap.dbm_antsignal',
                    'wlan.ra',
                    'wlan.fc.subtype']
df.drop(null_col_to_drop, axis=1, inplace=True)
df.drop(objects_col_to_drop, axis=1, inplace=True)

print(df['Label'].value_counts())
df.dropna(inplace=True)

print(df.isna().sum())
print(df.describe(include='object'))
print(df.shape)

labels_dict = {
    'Normal': 0,
    'Deauth': 1,
    'Disas': 2,
    '(Re)Assoc': 3,
    'RogueAP': 4,
    'Krack': 5,
    'Kr00k': 6,
    'Kr00K': 6,
    'SSH': 7,
    'Botnet': 8,
    'Malware': 9,
    'SQL_Injection': 10,
    'SSDP': 11,
    'SDDP': 11,
    'Evil_Twin': 12,
    'Website_spoofing': 13}

df["label"]=df['Label'].map(lambda a: labels_dict[a])
df.drop(['Label'], axis=1, inplace=True)

df.to_csv('/data/ara/processed/awid.csv', index=False)
