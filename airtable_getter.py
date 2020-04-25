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
        
        zone_type = district['fields']['Zone Type']
        
        
        if zone_type == 'Red':
            red_districts.append(district['fields']['DistrictAPI'])
            
    
    return red_districts
