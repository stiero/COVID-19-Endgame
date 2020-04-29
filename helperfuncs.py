#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 20:34:57 2020

@author: tauro
"""


import numpy as np
import pandas as pd

from scipy import stats as sps
from scipy.interpolate import interp1d

from matplotlib import pyplot as plt
from matplotlib.dates import date2num, num2date
from matplotlib import dates as mdates
from matplotlib import ticker
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch



R_T_MAX = 12
r_t_range = np.linspace(0, R_T_MAX, R_T_MAX*100+1)

# Gamma is 1/serial interval
GAMMA = 1/7

cutoff = 1






def prepare_cases_state(state):
    state_data = df[state]
    
    state_data = state_data.rename({state: "delta"}, axis = "columns")
    
    smoothed = state_data.rolling(window = 10, 
                                  win_type = 'gaussian',
                                  min_periods = 1,
                                  center = True).mean(std=3).round()
    
    idx_start = np.searchsorted(smoothed.iloc[:], cutoff)
    
    if idx_start == len(state_data):
        print("Ignored: ", state)
        return "Cannot use this data as it contains zero as its latest value"
    
    smoothed = smoothed.iloc[idx_start:]
    
    state_data = state_data.loc[smoothed.index]
    
    print("Added :", state)
    
    return smoothed





def get_posteriors(sr, sigma=0.15):

    # (1) Calculate Lambda
    lam = sr[:-1].values * np.exp(GAMMA * (r_t_range[:, None] - 1))

    
    # (2) Calculate each day's likelihood
    likelihoods = pd.DataFrame(
        data = sps.poisson.pmf(sr[1:].values, lam),
        index = r_t_range,
        columns = sr.index[1:])
    
    # (3) Create the Gaussian Matrix
    process_matrix = sps.norm(loc=r_t_range,
                              scale=sigma
                             ).pdf(r_t_range[:, None]) 

    # (3a) Normalize all rows to sum to 1
    process_matrix /= process_matrix.sum(axis=0)
    
    # (4) Calculate the initial prior
    prior0 = sps.gamma(a=4).pdf(r_t_range)
    prior0 /= prior0.sum()

    # Create a DataFrame that will hold our posteriors for each day
    # Insert our prior as the first posterior.
    posteriors = pd.DataFrame(
        index=r_t_range,
        columns=sr.index,
        data={sr.index[0]: prior0}
    )
    
    # We said we'd keep track of the sum of the log of the probability
    # of the data for maximum likelihood calculation.
    log_likelihood = 0.0

    # (5) Iteratively apply Bayes' rule
    for previous_day, current_day in zip(sr.index[:-1], sr.index[1:]):

        #(5a) Calculate the new prior
        current_prior = process_matrix @ posteriors[previous_day]
        
        #(5b) Calculate the numerator of Bayes' Rule: P(k|R_t)P(R_t)
        numerator = likelihoods[current_day] * current_prior
        
        #(5c) Calcluate the denominator of Bayes' Rule P(k)
        denominator = np.sum(numerator)
        
        # Execute full Bayes' Rule
        posteriors[current_day] = numerator/denominator
        
        # Add to the running sum of log likelihoods
        log_likelihood += np.log(denominator)
    
    return posteriors, log_likelihood







def highest_density_interval(pmf, p=.9):
    # If we pass a DataFrame, just call this recursively on the columns
    if(isinstance(pmf, pd.DataFrame)):
        return pd.DataFrame([highest_density_interval(pmf[col], p=p) for col in pmf],
                            index=pmf.columns)
    
    cumsum = np.cumsum(pmf.values)
    
    # N x N matrix of total probability mass for each low, high
    total_p = cumsum - cumsum[:, None]
    
    # Return all indices with total_p > p
    lows, highs = (total_p > p).nonzero()
    
    # Find the smallest range (highest density)
    best = (highs - lows).argmin()
    
    low = pmf.index[lows[best]]
    high = pmf.index[highs[best]]
    
    return pd.Series([low, high],
                     index=[f'Low_{p*100:.0f}',
                            f'High_{p*100:.0f}'])





def prepare_cases_district(district, data):

    dates = list(data.keys())
    deltas = []
    
    for date, sub_df in data.items():
        
        district_id = district['district_id']
        state = district['state']
        district_api = district['district_api']
        
        delta = int(sub_df[(sub_df['district']==district_api)
                           & (sub_df['state']==state)]['delta'])
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
        #print("Ignored: ", district)
        payload = {"district": district_api,
         "state": state,
         "smoothed_data": smoothed,
         "status": "ignore"}
        
        return payload
    
    smoothed = smoothed.iloc[idx_start:]
    
    district_data = district_data.loc[smoothed.index]
    
    print("Added :", district)
    
    payload = {"district_api": district_api,
               "state": state,
               "smoothed_data": smoothed,
               "status": "use",
               "district_id": district_id}
    
    return payload








