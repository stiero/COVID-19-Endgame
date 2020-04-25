#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 18:06:57 2020

@author: tauro
"""

import os
import pandas as pd


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


# Write the extracted data to a data frame
df = pd.DataFrame(list(zip(dates, deltas)),
                  columns=['date', 'delta'])
    