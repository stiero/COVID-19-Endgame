#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 18:06:57 2020

@author: tauro
"""

import os
import pandas as pd
import numpy as np
from airtable_getter import red_districts

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
    latest = max(paths, key = os.path.getctime)
    to_open[_dir] = latest

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





"""
Specify you district as it appears in the districts list.

To do - create a class or method for this. 
"""
district = 'Bengaluru'

dates = list(data.keys())
deltas = []

for date, sub_df in data.items():
    delta = int(sub_df[sub_df['district']==district]['delta'])
    deltas.append(delta)


# Combine the extracted data in a data frame
df = pd.DataFrame(list(zip(dates, deltas)),
                  columns=['date', 'delta']).set_index(['date'])












R_T_MAX = 12
r_t_range = np.linspace(0, R_T_MAX, R_T_MAX*100+1)

# Gamma is 1/serial interval
GAMMA = 1/7

cutoff = 7



def prepare_cases(district):

    dates = list(data.keys())
    deltas = []
    
    for date, sub_df in data.items():
        delta = int(sub_df[sub_df['district']==district]['delta'])
        deltas.append(delta)
    
    
    # Combine the extracted data in a data frame
    df = pd.DataFrame(list(zip(dates, deltas)),
                      columns=['date', 'delta']).set_index(['date'])
    
    district_data = df['delta']
    
    smoothed = district_data.rolling(window = 10, 
                                  win_type = 'gaussian',
                                  min_periods = 1,
                                  center = True).mean(std=3).round()
    
    idx_start = np.searchsorted(smoothed.iloc[:], cutoff)
    
    if idx_start == len(district_data):
        print("Ignored: ", district)
        return "Cannot use this data as it contains zero as its latest value"
    
    smoothed = smoothed.iloc[idx_start:]
    
    district_data = district_data.loc[smoothed.index]
    
    print("Added :", district)
    
    return smoothed




red_districts = red_districts()

for district in red_districts:
    if district in usable_districts:
        prepare_cases(district)









