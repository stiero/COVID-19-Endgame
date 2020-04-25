#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 21:27:40 2020

@author: tauro
"""


import pandas as pd
import time
import requests
import os

# Setting up the filepaths - change this if you get an error
try:
    os.chdir('/home/tauro/projects/covid/data')
    cwd = os.getcwd()
except:
    cwd = os.getcwd()


list_of_dirs = [x[1] for x in os.walk(cwd)][0]

url = "https://api.covid19india.org/v2/state_district_wise.json"

response = requests.get(url).json()

# Initialising empty lists as containers
states = []
districts = []
deltas = []
confirms = []
lastupdatedtimes = []

# Looping through every record, first by state
for statewise_record in response:

    state = statewise_record['state']
    level2 = statewise_record['districtData']
    
    # And then by district
    for district_data in level2:
        
        states.append(state)
        district = district_data['district']
        districts.append(district)
        
        delta = district_data['delta']['confirmed']
        deltas.append(delta)
        
        confirmed = district_data['confirmed']
        confirms.append(confirmed)
        
        # Write the current time if time field is absent
        try:
            lastupdatedtime = district_data['lastupdatedtime']
        except:
            lastupdatedtime = None
        
        if lastupdatedtime:
            lastupdatedtimes.append(lastupdatedtime)
        else:
            lastupdatedtimes.append(time.ctime())
        
    
# Write everything to a data frame
df = pd.DataFrame(list(zip(states, districts, confirms, deltas, lastupdatedtimes)),
                  columns = ["state", "district", "confirmed", "delta", "date"])    


# Writing to disk in the following format -> data/current_date/current_time.csv
current_date = time.strftime("%d-%m-%y")
current_time = time.strftime("%H:%M:%S") 

if current_date not in list_of_dirs:
    os.mkdir(current_date)    
    

write_path = cwd+"/"+current_date
print("Writing file to path: {}".format(write_path))    
os.chdir(write_path)       
df.to_csv(current_time+".csv")



