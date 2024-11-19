# Functions for Space-Time Ordinary Kriging:
# Marie-Christin Juhl, 2024
# @DGFI-TUM

import numpy as np
import xarray as xr
import gstools as gs
import datetime


def sum_metric_model(xy, sill1, range_1, nugget_1, sill2, range_2,nugget_2,range_3, sill3,nugget_3):
    alpha=0.1
    x, y = xy 
    arr=(sill1 * (1 - np.exp(-x / range_1)) + nugget_1) +(sill2 * (1 - np.exp(-y / range_2))+nugget_2)+ (sill3 * (1 - np.exp(-(np.sqrt(x**2+(alpha*y)**2)) / range_3))) -nugget_3 
    return arr 

def empirical_gamma_s_t_(h_t,h_s, Z, T, S):
    gamma=np.zeros((len(h_s)-1,len(h_t)-1))

    for i in range(0,len(h_s)-1): 
        for j in range(0,len(h_t)-1): 

            [id1,id2]  = np.where((S >= h_s[i]) & (S < h_s[i+1]) & (T >= h_t[j]) & (T < h_t[j+1]))
            diff       = np.subtract.outer(np.unique(Z[id1,id2]),np.unique(Z[id1,id2]))
            gamma[i,j] = 0.5*1/(len((diff)))*(np.sum(diff**2))#0.5*np.nanmean(np.abs(diff**2))
    return gamma
    
def select_data_xarray(l3,lat_min, lat_max, lon_min,lon_max, year1, year2):
    return l3.where((l3.longitude>=lon_min) & (l3.longitude<=lon_max) &(l3.latitude>=lat_min) &(l3.latitude<=lat_max) & (l3.time.dt.year>year1-1) & (l3.time.dt.year<year2+1), drop=True)

