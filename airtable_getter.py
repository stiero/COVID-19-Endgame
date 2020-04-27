#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 14:14:49 2020

@author: tauro
"""

import os
from airtable import Airtable

os.environ['AIRTABLE_API_KEY'] = 'keyMvfzeLUynQHhdG'


def airtable():
    table = Airtable('appNoMhZ2h3BqsBvd', 'Districts')
    data = table.get_all()
    
    return data


def red_districts():
    
    data = airtable()
    
    red_districts = []
    for district in data:
           
        try:
            zone_type = district['fields']['Zone Type']
            state = district['fields']['State']
            district_api = district['fields']['DistrictAPI']
            district_original = district['fields']['District']
        except:
            continue
            
        payload = {"district_api": district_api,
                    "district_original": district_original,
                    "state": state,
                    "zone_type": zone_type}
        
        
        if zone_type == 'Red':
            red_districts.append(payload)
            
    
    return red_districts
