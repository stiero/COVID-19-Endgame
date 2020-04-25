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

os.chdir('/home/tauro/projects/covid/data')

cwd = os.getcwd()

list_of_dirs = [x[1] for x in os.walk(cwd)][0]

url = "https://api.covid19india.org/v2/state_district_wise.json"

response = requests.get(url).json()




states = []

districts = []

deltas = []

confirms = []

lastupdatedtimes = []

for statewise_record in response:

    state = statewise_record['state']
    
    level2 = statewise_record['districtData']
    
    for district_data in level2:
        
        states.append(state)
    
        district = district_data['district']
        districts.append(district)
        
        delta = district_data['delta']['confirmed']
        deltas.append(delta)
        
        confirmed = district_data['confirmed']
        confirms.append(confirmed)
        
        try:
            lastupdatedtime = district_data['lastupdatedtime']
        except:
            lastupdatedtime = None
        
        if lastupdatedtime:
            lastupdatedtimes.append(lastupdatedtime)
        else:
            lastupdatedtimes.append(time.ctime())
        
    
    
df = pd.DataFrame(list(zip(states, districts, confirms, deltas, lastupdatedtimes)),
                  columns = ["state", "district", "confirmed", "delta", "date"])    


current_date = time.strftime("%d-%m-%y")
current_time = time.strftime("%H:%M:%S") 

if current_date not in list_of_dirs:
    os.mkdir(current_date)    
    
    
os.chdir(cwd+"/"+current_date)       
df.to_csv(current_time+".csv")


#df = pd.read_csv("13:47:02.csv")

#df1 = pd.read_csv("17April.csv")

#df2 = pd.read_csv("18April.csv")


# df2 = pd.read_csv("1April.csv")

# df.to_csv("18April.csv")


# daily_deceased = "https://api.covid19india.org/states_daily_csv/deceased.csv"

# df = pd.read_csv(daily_deceased)


daily_confirmed = "http://api.covid19india.org/states_daily_csv/confirmed.csv"

df_conf = pd.read_csv(daily_confirmed)

