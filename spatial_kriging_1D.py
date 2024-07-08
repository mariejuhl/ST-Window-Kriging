###transfer Ordinary  Kriging and functions here ..#

import xarray as xr 
import numpy as np 
import pandas as pd
import numpy.ma as ma
import gstools as gs
import math
import matplotlib.pyplot as plt
import tqdm
import scipy
import sklearn
import warnings
import haversine as haversine 

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
#from tqdm.contrib.concurrent import process_map


def haversine_vectorized(lon1, lat1, lon2, lat2):
    R = 6371  # Earth radius in kilometers
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c



def DISTANCE_MATRIX(l3):
    longitudes = l3.longitude.values
    latitudes = l3.latitude.values
    lon1, lon2 = np.meshgrid(longitudes, longitudes)
    lat1, lat2 = np.meshgrid(latitudes, latitudes)
    
    D = haversine_vectorized(lon1, lat1, lon2, lat2)
    return D



def distance(reference_point, coordinates):

    distances = haversine2(reference_point[0],reference_point[1],coordinates[0,:],coordinates[1,:])  
    return distances


    
def variogram_spatial_exp(coordinates,variable,model,bin_no, max_dist,plot):

    models={"Matern":      gs.Matern, 
            "Gaussian":    gs.Gaussian, 
            "Linear":      gs.Linear,
            "Exponential": gs.Exponential,
            "Spherical":   gs.Spherical, 
            "JBessel":     gs.JBessel}

    bins     = gs.standard_bins(coordinates, max_dist=max_dist, latlon=True, bin_no=bin_no,geo_scale=gs.KM_SCALE)
    bin_center1, gamma1 = gs.vario_estimate(coordinates, variable, bins, latlon=True, estimator="cressie",geo_scale=gs.KM_SCALE)

    if plot==True:
        fig1 = plt.scatter(bin_center1, gamma1, color="k", label="data")
        ax   = plt.gca()
    fit1 = models[model](dim=1, latlon=True,geo_scale=gs.KM_SCALE)
    para, pcov, r21 = fit1.fit_variogram(bin_center1, gamma1, return_r2=True)
    
    if plot==True:
        fit1.plot( ax=ax,x_max=bins[-1]);
        plt.xlabel('distance in km');plt.ylabel('variance')
        plt.title('Primary variogram')
   
    print("Fitting score of variogram: %s" %np.round(r21,2))
    if plot==True:
        plt.show()
    return bin_center1, gamma1, fit1, r21  
    


    
def OrdinaryKriging(l3, grid, fit1):

    Z_pred       = []
    Z_pred_error = []

    print('Make distance matrix ... ')
    distances = DISTANCE_MATRIX(l3)       #Distance between all L3-data
    
    C_zz = fit1.variogram(distances)      #LHS
    n    = len(l3)
    K    = np.zeros((n+1,n+1))             
    b    = np.zeros(n+1)                  #RHS Kriging 
    
    K[:n,:n]  = C_zz 
    K[n,n]    = 0
    K[:n,n]   = 1
    K[n,:n]   = 1 
    np.fill_diagonal(K,0.0) #--> force exact values  

    for i in tqdm.tqdm(range(0,len(grid)), desc='Predict grid points..'):
        if grid.land.values[i] !=1:
            Z_ = np.nan
            Z_pred.append(Z_)
            Z_pred_error.append(np.nan)
            
        else:    
        
            u0_    = np.array((grid.longitude.values[i],grid.latitude.values[i])) 
            points = np.array((l3.longitude.values, l3.latitude.values))
            
            dist = []
            dist=np.zeros((len(points[0])))
            for k in range(0,len(points[0])):
                dist[k] = haversine_(points[0][k], points[1][k],u0_[0],u0_[1])        #haversine distance 

            c_zz = fit1.variogram(dist)
                
            b[:n]     = c_zz  
            b[n]      = 1
            
            #valid_indices = ~np.isnan(b)
            #K_valid = K[valid_indices][:, valid_indices]
            #b_valid = b[valid_indices]
            
            try:
                #w = np.linalg.solve(K_valid,b_valid) #--> not for singular matrices 
                w = np.linalg.solve(K,b)

            except np.linalg.LinAlgError:

                #w = np.dot(scipy.linalg.pinv(K_valid),b_valid) # solve by stimating (Moore-Penrose) pseudo-inverse of a matrix
                w = np.dot(scipy.linalg.pinv(K),b)
            
            Z_ = (np.nansum(w[:n]*l3.resid.values))#*grid['land'].values[i]
            Z_pred.append(Z_)
            Z_pred_error.append((np.dot(w[:n],(c_zz-fit1.nugget))))#*grid['land'].values[i]) ### maybe include Lagrange multiplier ?? 

    return Z_pred, Z_pred_error
    
    
"""def scipy_box_cox(x):
    from scipy import stats
    #x = x[~np.isnan(x)]
    x[np.isnan(x)]=0
    off = np.absolute(x.min())+0.1
    x = x + off
    y, tau = stats.boxcox(x)
    #y = y-off
    return y, tau, off

def inverse_box_cox(y, tau, off):
    
    x = scipy.special.inv_boxcox(y, tau, out=None)
    return x-off

def standard_data(data_in):
     mean = np.nanmean(data_in)
     var = np.nanvar(data_in)

     return mean, var, (data_in-mean)/var

def standard_data_back(mean, var, data):
    return (data*var)+mean """

