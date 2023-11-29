import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from datetime import timedelta, datetime

def plot_region(region: int) -> NotImplemented:
    """
    Plots the average temperature and regional illness count vs. week number
        for the inputted region
    """
    df_260 = pd.read_csv('260_weeks_data.csv')
    df_260.replace(r'^\s*$', np.nan, regex=True)

    # removing COVID weeks
    df_260.loc[((1909 >= df_260['row']) & (df_260['row'] >= 1380)), 'Week of'] = np.nan

    df_260['week number'] = ""

    start_week = datetime(2018, 11, 10, 0, 0)
    for idx, row in df_260.iterrows():
        if not pd.isna(row['Week of']):
            week_date = datetime.strptime(row['Week of'], '%m/%d/%Y')
            df_260.at[idx, 'week number'] = (week_date - start_week) // timedelta(7)

    df = df_260[df_260['Region'] == region]

    week_no1, temp1, ill1 = [], [], []
    week_no2, temp2, ill2 = [], [], []
    for idx, row in df.iterrows():
        if idx <= 1379:
            week_no1.append(row['week number'])
            temp1.append(row['Avg. temperature'])
            ill1.append(row['Regional Illnesses'])
        if idx >= 1919:
            week_no2.append(row['week number'])
            temp2.append(row['Avg. temperature'])
            ill2.append(row['Regional Illnesses'])

    fig, ax1 = plt.subplots()

    ax1.plot(week_no1, temp1, color='b')
    ax1.plot(week_no2, temp2, color='b')
    ax1.set_xlabel('Week Number')
    ax1.set_ylabel('Average Temperature', color='b')
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    ax2.plot(week_no1, ill1, color='r')
    ax2.plot(week_no2, ill2, color='r')
    ax2.set_ylabel('Regional Illness Count', color='r')
    ax2.tick_params('y', colors='r')

    fig.tight_layout()
    plt.savefig(f'region{region}_plot.png')

if __name__ == '__main__':
    plot_region(4)