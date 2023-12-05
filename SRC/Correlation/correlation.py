import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('output.csv')

df['Regional Illnesses'] = df['Regional Illnesses'].astype(int)

# Aggregates average temperature for each region
df_temp = df.groupby('Region')['Avg. temperature'].mean()

# Aggregates average number of HRIs for each region
df_hri = df.groupby('Region')['Regional Illnesses'].mean()

# Aggregates the average temperatures from each county into average temperatures for each region for each week
df = df.groupby(['Region', 'Week of', 'Regional Illnesses'], as_index=False, sort=False)['Avg. temperature'].mean()

# Determines correlation between regional average temperature and HRIs
df = df.groupby('Region')[['Avg. temperature', 'Regional Illnesses']].corr().iloc[0::2, -1]

# Merges the correlation values with average temperatures and average HRIs for each region
df_merge = pd.merge(df_temp, df, on='Region', how='outer')

df_merge = pd.merge(df_merge, df_hri, on='Region', how='outer')

print(df_merge)
