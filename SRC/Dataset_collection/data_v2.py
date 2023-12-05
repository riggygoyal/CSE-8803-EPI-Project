import requests
import pandas as pd

from datetime import date, timedelta
from tqdm import tqdm

region_1 = ['ME', 'NH', 'VT', 'CT', 'MA', 'RI']
region_2 = ['NY', 'NJ']
region_3 = ['PA', 'MD', 'DE', 'VA', 'WV', 'DC']
region_4 = ['KY', 'NC', 'SC', 'TN', 'GA', 'FL', 'AL', 'MS']
region_5 = ['MN', 'WI', 'MI', 'IL', 'IN', 'OH']
region_6 = ['AR', 'LA', 'OK', 'TX', 'NM']
region_7 = ['IA', 'MO', 'KS', 'NE']
region_8 = ['ND', 'SD', 'MT', 'WY', 'CO', 'UT']
region_9 = ['AZ', 'CA', 'NV']
region_10 = ['WA', 'OR', 'ID']

true_regions, true_avg_temps, true_regional_illnesses, true_dates = [], [], [], []
all_regions = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']

weeks = 260 # Last 260 weeks of data

for i in tqdm(range(weeks)):
    request_date = date(2023, 10, 28) - timedelta(i * 7)

    illness_api = 'https://ephtracking.cdc.gov/DataExplorer/getCoreHolder/1237/2241/ALL/ALL/9/{0}/0/1'.format(request_date.strftime('%Y%m%d'))
    temperature_api = 'https://ephtracking.cdc.gov/DataExplorer/getCoreHolder/1025/2/ALL/ALL/9/{0}/0/1'.format(request_date.strftime('%Y%m%d'))

    illnesses_res = requests.get(illness_api)
    temperatures_res = requests.get(temperature_api)

    region_temps = {} # stores region : list of temperatures

    if illnesses_res.status_code == 200 and temperatures_res.status_code == 200:
        illnesses = illnesses_res.json()['regionPMTableResult']

        for result in temperatures_res.json()['tableResult']:
            curr_region = None
            if result['parentGeoAbbreviation'] in region_1:
                curr_region = '01'
            elif result['parentGeoAbbreviation'] in region_2:
                curr_region = '02'
            elif result['parentGeoAbbreviation'] in region_3:
                curr_region = '03'
            elif result['parentGeoAbbreviation'] in region_4:
                curr_region = '04'
            elif result['parentGeoAbbreviation'] in region_5:
                curr_region = '05'
            elif result['parentGeoAbbreviation'] in region_6:
                curr_region = '06'
            elif result['parentGeoAbbreviation'] in region_7:
                curr_region = '07'
            elif result['parentGeoAbbreviation'] in region_8:
                curr_region = '08'
            elif result['parentGeoAbbreviation'] in region_9:
                curr_region = '09'
            else:
                curr_region = '10'

            curr_temp = float(result['dataValue'])

            if curr_region in region_temps:
                region_temps[curr_region].append(curr_temp)
            else:
                region_temps[curr_region] = [curr_temp]

        
        true_regions.extend(all_regions)
        for region in all_regions:
            if region in region_temps:
                true_avg_temps.append(sum(region_temps[region]) / len(region_temps[region]))
            else:
                true_avg_temps.append(float('nan'))

            true_regional_illnesses.append(illnesses[int(region) - 1]['dataValue'])
            true_dates.append(request_date.strftime('%m/%d/%Y'))

data = {'Region': true_regions, 'Avg. temperature': true_avg_temps, 'Regional Illnesses': true_regional_illnesses, 'Week of': true_dates}
df = pd.DataFrame(data)

df.to_csv(f'{weeks}_weeks_data.csv')
