import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
import math

from datetime import timedelta, datetime

def generate_corr(region: int) -> float:
    """
    Generates the PCC to determine how correlated a region's average temperature
        is to its illness trend.

    Args:
        region: The region number.

    Returns:
        A float representing the desired PCC value.
    """
    df_260 = pd.read_csv('260_weeks_data.csv')
    df_260.replace(r'^\s*$', np.nan, regex=True)

    df_260.loc[((1909 >= df_260['row']) & (df_260['row'] >= 1380)), 'Week of'] = np.nan

    df_260['week number'] = ""

    start_week = datetime(2018, 11, 10, 0, 0)
    for idx, row in df_260.iterrows():
        if not pd.isna(row['Week of']):
            week_date = datetime.strptime(row['Week of'], '%m/%d/%Y')
            df_260.at[idx, 'week number'] = (week_date - start_week) // timedelta(7)

    df = df_260[df_260['Region'] == region]

    avg_t, ill = [], []

    for idx, row in df.iterrows():
        if row['week number'] != '':
            avg_temp = row['Avg. temperature']
            illnesses = row['Regional Illnesses']

            if not math.isnan(avg_temp) and not math.isnan(illnesses):
                avg_t.append(avg_temp)
                ill.append(illnesses)

    res = scipy.stats.pearsonr(avg_t, ill)
    return res.statistic


if __name__ == '__main__':
    for i in range(1, 11):
        corr = generate_corr(i)
        print(f'Correlation for region {i} is {corr}')