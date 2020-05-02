#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 18:06:57 2020

@author: tauro
"""

import os
import pandas as pd
import numpy as np
from airtable_getter import get_districts, write_to_airtable





from helperfuncs import prepare_cases_district, get_posteriors,\
    highest_density_interval, get_latest_file



# Setting up the filepaths - change this if you get an error
try:
    os.chdir('/home/tauro/projects/covid/data')
    cwd = os.getcwd()
except:
    cwd = os.getcwd()

list_of_dirs = sorted([x[1] for x in os.walk(cwd)][0])


# Each directory can have multiple files. Fetching only the latest ones
to_open = {}

for _dir in list_of_dirs:
    list_of_files = [x[2] for x in os.walk(_dir)][0]
    paths = [os.path.join(cwd+"/"+_dir, file) for file in list_of_files]
    
    latest_file_index = get_latest_file(list_of_files)
    latest_file = paths[latest_file_index]
    to_open[_dir] = latest_file

data = {}

for date, path in to_open.items():
    df = pd.read_csv(path)
    data[date] = df



"""
Extracting "usable" districts - i.e. those that have a data record
every day since the beginning. 

To do - add a time window.
"""
prev_list = []

for i, sub_df in enumerate(data.values()):
    districts = [row['district'] for _, row in sub_df.iterrows()]
    
    if i == 0:
        prev_list = districts
        continue
    else:
        usable_districts = list(set(districts) & set(prev_list))
        
        # Update
        prev_list = usable_districts





# """
# Specify you district as it appears in the districts list.

# To do - create a class or method for this. 
# """
# district = 'Bengaluru'

# dates = list(data.keys())
# deltas = []

# for date, sub_df in data.items():
#     delta = int(sub_df[sub_df['district']==district]['delta'])
#     deltas.append(delta)


# # Combine the extracted data in a data frame
# df = pd.DataFrame(list(zip(dates, deltas)),
#                   columns=['date', 'delta']).set_index(['date'])




all_districts = get_districts()

outputs = []

skipped = []

for i, district in enumerate(all_districts):
    
    district_api = district['district_api']
    state =  district['state']
    
    try:
        if district_api in usable_districts:
            output = prepare_cases_district(district, data)
            
            if output['status'] == 'use':
                
                smoothed = output['smoothed_data']
                
                posteriors, log_likelihood = get_posteriors(smoothed, sigma=.25)
                
                
                # Note that this takes a while to execute - it's not the most efficient algorithm
                hdis = highest_density_interval(posteriors, p=.9)
                
                most_likely = posteriors.idxmax().rename('ML')
    
                result = pd.concat([most_likely, hdis], axis=1)
    
                latest_rt = result.iloc[-1]['ML']
                
                output['all_rt'] = result
                output['latest_rt'] = latest_rt
                
                outputs.append(output)
                
                write_to_airtable(output)
    
    except:
        print(i)
        skipped.append(i)
        continue


# district = red_districts[54]

# aa = prepare_cases_district(district, data)



# smoothed = aa["smoothed_data"]

# posteriors, log_likelihood = get_posteriors(smoothed, sigma=.25)

# ax = posteriors.plot(title=f'{state} - Daily Posterior for $R_t$',
#            legend=False, 
#            lw=1,
#            c='k',
#            alpha=.3,
#            xlim=(0.4,6))

# ax.set_xlabel('$R_t$');


# # Note that this takes a while to execute - it's not the most efficient algorithm
# hdis = highest_density_interval(posteriors, p=.9)

# most_likely = posteriors.idxmax().rename('ML')

# # Look into why you shift -1
# result = pd.concat([most_likely, hdis], axis=1)

# latest_rt = result.iloc[-1]['ML']

