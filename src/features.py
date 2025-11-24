# features.py
# Feature engineering matching DailyGo rules:
# - shift category (morning/afternoon/night)
# - commute flags (long commute)
# - facilities flag is used directly (0/1)
# - previous no-show indicators

import pandas as pd
import numpy as np

def shift_category(hour):
    if hour < 12:
        return 'morning'
    if 12 <= hour < 17:
        return 'afternoon'
    return 'night'

def prepare_features(df):
    df = df.copy()
    # shift category
    df['shift_cat'] = df['shift_hour'].apply(shift_category)
    # binary features from shift
    df['is_morning'] = (df['shift_cat'] == 'morning').astype(int)
    df['is_afternoon'] = (df['shift_cat'] == 'afternoon').astype(int)
    df['is_night'] = (df['shift_cat'] == 'night').astype(int)
    # commute flags
    df['commute_long'] = (df['commute_km'] > 5.0).astype(int)  # your rule: >5 km increases risk
    # previous no-show indicator
    df['high_prev_no_show'] = (df['prev_no_shows'] >= 2).astype(int)
    # pay per hour proxy (simple)
    df['pay_bucket'] = pd.cut(df['pay'], bins=[0,200,300,400,10000], labels=[0,1,2,3]).astype(int)
    # final feature set
    features = ['pay','pay_bucket','worker_rating','commute_km','commute_long',
                'facilities_provided','prev_no_shows','high_prev_no_show',
                'is_morning','is_afternoon','is_night']
    X = df[features]
    y = df['no_show']
    return X, y

if __name__ == "__main__":
    df = pd.read_csv('../data/sample_noshow.csv')
    X, y = prepare_features(df)
    print("Feature sample:")
    print(X.head())
