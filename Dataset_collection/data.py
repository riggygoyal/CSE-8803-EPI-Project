import requests
import pandas as pd
from datetime import date, timedelta

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

titles, temperatures, regions, regional_illnesses, dates = [], [], [], [], []

weeks = 260 # Last n weeks of data

for i in range(weeks):
    request_date = date(2023, 10, 28) - timedelta(i * 7)

    illness_api = 'https://ephtracking.cdc.gov/DataExplorer/getCoreHolder/1237/2241/ALL/ALL/9/{0}/0/1'.format(request_date.strftime('%Y%m%d'))
    temperature_api = 'https://ephtracking.cdc.gov/DataExplorer/getCoreHolder/1025/2/ALL/ALL/9/{0}/0/1'.format(request_date.strftime('%Y%m%d'))

    illnesses_res = requests.get(illness_api)
    temperatures_res = requests.get(temperature_api)

    if illnesses_res.status_code == 200 and temperatures_res.status_code == 200:
        illnesses = illnesses_res.json()['regionPMTableResult']
        for result in temperatures_res.json()['tableResult']:
            titles.append(result['title'])
            temperatures.append(result['dataValue'])
        
            if result['parentGeoAbbreviation'] in region_1:
                regions.append('01')
            elif result['parentGeoAbbreviation'] in region_2:
                regions.append('02')
            elif result['parentGeoAbbreviation'] in region_3:
                regions.append('03')
            elif result['parentGeoAbbreviation'] in region_4:
                regions.append('04')
            elif result['parentGeoAbbreviation'] in region_5:
                regions.append('05')
            elif result['parentGeoAbbreviation'] in region_6:
                regions.append('06')
            elif result['parentGeoAbbreviation'] in region_7:
                regions.append('07')
            elif result['parentGeoAbbreviation'] in region_8:
                regions.append('08')
            elif result['parentGeoAbbreviation'] in region_9:
                regions.append('09')
            else:
                regions.append('10')
            
            regional_illnesses.append(illnesses[int(regions[-1]) - 1]['dataValue'])

        dates += len(temperatures_res.json()['tableResult']) * [request_date.strftime('%m/%d/%Y')]

data = {'County': titles, 'Avg. temperature': temperatures, 'Region': regions, 'Regional Illnesses': regional_illnesses, 'Week of': dates}
df = pd.DataFrame(data)

df.to_csv('output.csv')
