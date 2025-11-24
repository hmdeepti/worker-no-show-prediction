# exploration.py
# Quick checks on the sample dataset

import pandas as pd

def load(path='../data/sample_noshow.csv'):
    return pd.read_csv(path)

def basic_summary(df):
    print("Rows:", len(df))
    print("No-show rate:", df['no_show'].mean())
    print("\nSample head:")
    print(df.head())

if __name__ == "__main__":
    df = load()
    basic_summary(df)
