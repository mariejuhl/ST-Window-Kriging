#
import numpy as np 
import datetime
import scipy
import xarray as xr
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
import haversine.haversine as haversine
from tqdm.auto import tqdm

def select_data_xarray(l3,lat_min, lat_max, lon_min,lon_max, year1, year2):
    return l3.where((l3.longitude>=lon_min) & (l3.longitude<=lon_max) &(l3.latitude>=lat_min) &(l3.latitude<=lat_max) & (l3.time.dt.year>year1-1) & (l3.time.dt.year<year2+1), drop=True)


def empirical_gamma_s_t_(h_t,h_s, Z, T, S, estimator):
    gamma=np.zeros((len(h_s)-1,len(h_t)-1))

    for i in range(0,len(h_s)-1): 
        for j in range(0,len(h_t)-1): 

            [id1,id2]  = np.where((S >= h_s[i]) & (S < h_s[i+1]) & (T >= h_t[j]) & (T < h_t[j+1]))
            diff       = np.subtract.outer(np.unique(Z[id1,id2]),np.unique((Z[id1,id2])))

            if estimator=='matheron':
                #gamma[i,j] = 0.5*np.unique(Z[id1,id2]).var() #
                gamma[i,j] =  0.5*1/(len(diff)**2)*(np.sum(diff**2))
                #gamma[i,j]  = 0.5*np.var(np.unique(Z[id1,id2]))
            if estimator == 'cressie':
                gamma[i,j] = 0.5*((1/(len(diff))*np.sum(np.abs(diff)**0.5))**4)/(0.457+(0.494/len(diff))+(0.045/(len(diff)**2)))

    return gamma


def sum_metric_model(xy, sill1, range_1, nugget_1, sill2, range_2,nugget_2,range_3, sill3,nugget_3, alpha):
    #alpha = 10
    x, y = xy 
    arr=(sill1 * (1 - np.exp(-x / range_1)) + nugget_1) +(sill2 * (1 - np.exp(-y / range_2))+nugget_2)+ (sill3 * (1 - np.exp(-(np.sqrt(x**2+(alpha*y)**2)) / range_3))) -nugget_3 
    return arr 


def ST_Window_Kriging(xtime,l3,grid=None, n_days=5, hs_range=(0, 220, 11), ht_range=(0, 5, 11), estimator='matheron', func=None, plot=False, Kriging =False,exact= True, save_as_netcdf=None, save_as_csv=None):
    """
    Fits an empirical variogram with temporal and spatial lags using curve fitting.

    Parameters:
        xtime (datetime): Current timestamp for analysis.
        grid (DataFrame): DataFrame containing spatial-temporal grid data with 'time'.
        l3 (DataFrame): DataFrame containing residual data with a time index.
        n_days (int): Number of days for temporal window around each time point.
        hs_range (tuple): Range for spatial lags (min, max, num_bins).
        ht_range (tuple): Range for temporal lags (min, max, num_bins).
        estimator (str): Estimator for variogram calculation.
        func (callable): Function to fit the variogram model.
        plot (bool): Whether to plot the experimental and modeled variograms.
        alpha (float): Alpha value for the fit.

    Returns:
        popt (array): Optimized parameters from curve fitting.
        residual_metrics (dict): Residual analysis metrics including MAE, RMSE, and R^2.
        alpha (float): Input alpha value.
    """
    all_popt = []  

    print(f"Processing time: {xtime}")
    l3_ = l3[((l3.index.values) >= (xtime - np.timedelta64(n_days, 'D'))) & ((l3.index.values) <= (xtime + np.timedelta64(n_days, 'D')))]

    scaler1 = StandardScaler(with_mean=True, with_std=True)
    l3_['resid_'] = scaler1.fit_transform(l3_.resid.values.reshape(-1, 1))

    print("Making temporal matrix...")
    T = l3_.index.values
    grid_T = np.abs(np.subtract.outer(T, T)) / np.timedelta64(1, 'D')  # Temporal lag in days
    
    print("Making distance matrix...")
    l3_['longitude'] = l3_.longitude - 360 
    coords_ = l3_[['latitude', 'longitude']]
    from scipy.spatial import distance 
    import haversine.haversine as haversine
    grid_S = distance.cdist(coords_, coords_, metric=haversine)  # Spatial lag in km

    print("Making value matrix...")
    Z = np.outer(l3_['resid_'], np.ones(len(l3_['resid_'])))

    # Build empirical variogram
    hs = np.linspace(*hs_range)
    ht = np.linspace(*ht_range)
    gamma = empirical_gamma_s_t_(ht, hs, Z, grid_T, grid_S, estimator=estimator)
    
    X, Y = np.meshgrid(hs[:-1], ht[:-1])
    
    # Fit model
    try:
        popt, _ = curve_fit(func, (X.flatten(), Y.flatten()), gamma.T.flatten(), bounds=(0, np.inf))
        all_popt.append(popt)
        
        # Residual analysis
        gamma_model = func((X.flatten(), Y.flatten()), *popt)
        residuals = gamma.T.flatten() - gamma_model
        mae = mean_absolute_error(gamma.T.flatten(), gamma_model)
        rmse = np.sqrt(mean_squared_error(gamma.T.flatten(), gamma_model))
        s_res = np.sum(residuals**2)  # Residual sum of squares
        s_tot = np.sum((gamma.T.flatten() - np.mean(gamma.T.flatten()))**2)  # Total sum of squares
        r_squared = 1 - (s_res / s_tot)
        
        residual_metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'R^2': r_squared
        }

        print(f"Fitting Results for time {xtime}:\n  MAE: {mae:.4f}, RMSE: {rmse:.4f}, R^2: {r_squared:.4f}")
    
    except Exception as e:
        print(f"Curve fitting failed for time {xtime}: {e}")
        residual_metrics = None

    # Optional plot
    if plot and popt is not None:
        fig, axes = plt.subplot_mosaic("""AB""", figsize=(8, 5), layout='constrained', gridspec_kw={'hspace': 0.1, 'wspace': 0.1}, subplot_kw={"projection": "3d"})
        color_map = sns.cubehelix_palette(start=.5, rot=-.75, as_cmap=True)

        # Experimental variogram
        axes['A'].plot_surface(X, Y, gamma.T, cmap=color_map, linewidth=0, antialiased=True)
        axes['A'].view_init(10,220)
        axes['A'].set_title('Experimental Variogram', fontsize=12)

        # Modeled variogram
        X_m, Y_m = np.meshgrid(hs, ht)
        C_sla = func((X_m, Y_m), *popt)

        #axes['B'].view_init(10,220)
        axes['B'].plot_surface(X_m, Y_m, C_sla, cmap=color_map, linewidth=1)
        axes['B'].scatter(X_m, Y_m, C_sla, c='k', marker='o', s=1)
        axes['B'].set_title('Modeled Variogram', fontsize=12)
        #axes['B'].view_init(10,220)#
        axes['B'].view_init(10, 220)    
        #plt.show()

   
    if Kriging == False:
        return all_popt
    else: 
        grid_=grid[grid.time==xtime]
        print(xtime)

        ###KRIGING
        print('Preparing...')
        n = len(l3_)
        K = np.zeros((n+1,n+1))         
        b = np.zeros((n+1)) 
        b[n] = 1

        K[:n,:n]  = func((grid_S, grid_T), *popt)
        K[n,:n]   = 1
        K[n,n]    = 0

        if exact == True: 
            np.fill_diagonal(K,0.0) 

        if len(K[K<0])>0:
            print("WARNING! Kriging matrix has negative values.")

        xlat      = grid_.latitude.values
        xlon      = grid_.longitude.values
        Z_        = np.zeros(len(grid_))
        Z_error   = np.zeros(len(grid_))

        for i in tqdm(range(0,len(grid_))):
            
            if np.isnan(grid_.sla.values[i])==True:
                Z_[i] = np.nan;Z_error[i]= np.nan
            else:    
                point        = pd.DataFrame([{'latitude':xlat[i], 'longitude':xlon[i]}])
                sdist        = distance.cdist(coords_.values,point.values, metric=haversine)
                tdist        = np.abs(( coords_.index.values-xtime.to_datetime64()) / np.timedelta64(1, 'D'))
                b[:n]        = func((sdist.squeeze(), tdist), *popt)
                w            = np.dot(scipy.linalg.pinv(K),b)
                z            = (np.nansum(w[:n]*l3_.resid_.values)) #predict_gridpoint(coords_,grid_,K, popt)
                Z_[i]        = scaler1.inverse_transform(z.reshape(-1,1))
                error        = np.dot((w[:n]),(b[:n]))-w[n]
                Z_error[i]   = scaler1.inverse_transform(error.reshape(-1,1))


        grid_['OK_ST']=Z_
        grid_['OK_ST_error']= Z_error

        if save_as_csv != None: 
            print('Saving grid...')
            grid_.to_csv(save_as_netcdf+'OK_ST_sum_metric_'+xtime.date().strftime("%d_%m_%Y")+'.csv')
            print('Grid saved as csv '+save_as_netcdf+'OK_ST_sum_metric_'+xtime.date().strftime("%d_%m_%Y")+'.nc')

        if save_as_netcdf != None: 
            print('Saving grid...')
            ds = grid_.set_index(['time','latitude','longitude']).to_xarray()
            ds = xr.Dataset({'OK_ST': (('latitude', 'longitude'),Z_),'OK_ST_error': (('latitude', 'longitude'),Z_error)},coords={'time': xtime.to_datetime64(), 'latitude': xlat, 'longitude': xlon})
            ds.to_netcdf(save_as_netcdf+'OK_ST_sum_metric_'+xtime.date().strftime("%d_%m_%Y")+'.nc')
            print('Grid saved as NetCDF '+save_as_netcdf+'OK_ST_sum_metric_'+xtime.date().strftime("%d_%m_%Y")+'.nc')
            

        
    return grid_
