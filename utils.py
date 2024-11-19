from tqdm import tqdm
import numpy as np
import xarray as xr
from scipy import signal
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
lowess = sm.nonparametric.lowess
from datetime import datetime
from time import time
import xskillscore as xs
# for creation of requirements.txt
import matplotlib.pyplot as plt
from time import time
#import geoplot
import pickle 
import geopandas as gpd
#from mpl_toolkits.basemap import Basemap
import matplotlib.cm as cm
import matplotlib as mpl

#world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
plt.rcParams['figure.figsize'] = (10, 7.5)
plt.style.use('ggplot')


#### data loading/saving
def get_datetime_str():
    """get date string of current date and time

    Returns:
        str: Date string.
    """
    return str(datetime.fromtimestamp(time()))

def pickleize(object, path):
    with open(path, "wb") as f:
        pickle.dump(object, f)

def load_attrs():
    with open("Data/meta/attrs.pkl", "rb") as f:
        attrs_load = pickle.load(f)
    return attrs_load

def load_paths():
    with open("Data/meta/paths.pkl", "rb") as f:
        paths = pickle.load(f)
    return paths

def find_nearest(array, value):
    """Get index of element in an array which is closest to the input value.

    Args:
        array (array_type): Search array
        value (float, int): Search value

    Returns:
        int: Index of closest value in array
    """
    array = np.asarray(array)
    idx = (np.abs(array-value)).argmin()
    return idx

def tile_array(array: np.array, repeats: int):
    """expands array in the last dimension by repeating the array and stacking it a defined number of times.

    Args:
        array (np.array): input array
        repeats (int): how often to stack the array along the last dimension

    Returns:
        np.array: expanded array
    """
    array = array.reshape(array.shape+(1,))
    array = np.tile(array,repeats)
    return array

def daily_mean(ds,var_name):
    """Build daily mean of a time series

    Args:
        ds (xarray.DataSet): Data set
        var_name (str): Name of data variable

    Returns:
        ds_out (xarray.DataSet): Output data set with daily averaged data variable
    """
    variable = ds[var_name].where(ds[var_name]!= -99.9999) 
    df = variable.to_pandas().transpose()
    
    # Downsample to daily mean
    df_d = df.resample("D").mean()

    datasets = []
    for s in df_d.columns:
        dataset = xr.Dataset(
                data_vars = {var_name: (["station","time"], np.array([df_d[s]]))},
                coords = dict(station=np.array([s]), time=np.asarray(df_d.index))
            )
        datasets.append(dataset)
        
    ds_out = xr.concat(datasets,dim="station")
    ds_out = ds_out.assign(dict(
        longitude=(["station"], ds.longitude.data),
        latitude=(["station"], ds.latitude.data),
        site_name=(["station"], ds.site_name.data),
        country=(["station"], ds.country.data))
    )
    return ds_out

def count_nan(array):
    """counts consecutive occurences of missing values in a time series and adds counts to a list

    Args:
        array (array_type): time series

    Returns:
        nan_counts(array_type): List with counts
    """
    nan_counts = []
    nans = np.where(np.isnan(array))[0]
    n = 0
    for i in range(1,len(nans)):
        n += 1
        if (nans[i]-nans[i-1])!=1:
            nan_counts.append(n)
            n = 0 
    if n!=0:
        n += 1
        nan_counts.append(n)
    if len(nans)==1:
        nan_counts.append(1)
    return np.array(nan_counts)

def gap_handling(ds,var_name,max_gap,interp="linear"):
    """Handle gaps in time series data. Data gaps that are small enpugh are interpolated. Data with large gaps is deleted.

    Args:
        ds (xarray.DatSet): Data set
        var_name (str): Data variable
        max_gap (int): Maximum number of consecutive data gaps that are tolerated
        interp (str, optional): Interpolation method. Defaults to "linear".

    Returns:
        ds_out (xarray.DataSet): Filtered data set
    """
    datasets = [] 
    for station in tqdm(ds.station,"Handling data gaps..."):
        x = ds[var_name].sel(station=station)
        x = x.where(x!= -99.9999) # replace not valid values (-99.9999) with NaN
        gaps = count_nan(x) # count data gaps 
        if all(gaps<=max_gap): # choose time series with gaps below threshold value
            # Interpolate between gaps
            if interp != None:
                x = x.interpolate_na(dim="time",method=interp) 
            
            dataset = xr.Dataset(
                data_vars = {
                    var_name: (["station","time"], np.array([x])),                    
                    "longitude": (["station"], np.array([ds.longitude.sel(station=station).data])),
                    "latitude": (["station"], np.array([ds.latitude.sel(station=station).data])),
                    "site_name": (["station"], np.array([ds.site_name.sel(station=station).data]).astype(str)),
                    "country": (["station"], np.array([ds.country.sel(station=station).data]).astype(str))
                },
                coords = dict(station=np.array([station]), time=ds.time.data)
            )
            datasets.append(dataset)
    ds_out = xr.concat(datasets, dim="station")
    return ds_out

def lowess_filter(ds,var_name, hours):
    """Applies lowes filter to time series data.

    Args:
        ds (xarray.DatSet): Data set
        var_name (str): Data variable

    Returns:
        xarray.DataSet: Filtered data set
    """
    data = np.zeros_like(ds[var_name]) 
    i = 0
    for station in tqdm(ds.station,"Applying lowess filter to data..."):
        x = ds[var_name].sel(station=station)
        
        #frac = 0.004567
        frac = (hours/(len(ds.time)*24))
        #frac = 1/(len(ds.time)/40)
        data[i,:] = lowess(x, ds.time, is_sorted='false', missing='drop', return_sorted=False, frac=frac)
        i += 1
    ds = ds.assign({var_name:(["station","time"],data)})
    return ds
        
def spatial_match(ds_grid,var_grid,ds_tg,var_tg,land_mask=True):
    """Math gridded data and tide gage data in the space domain.

    Args:
        ds_grid (xarray.DataSet): Gridded data set
        var_grid (str): Variable name
        ds_tg (xarray.DataSet): Tide gauge data set
        var_tg (str): Variable name
        land_mask (bool, optional): Gridded data set contains a land mask. Defaults to True.

    Returns:
        xarray.DataSet: Matched data set
    """
    datasets = [] 
    for station in tqdm(ds_tg.station.data,"Matching geolocation..."):
        lon_tg = ds_tg.lon_tg.sel(station=station).data
        lat_tg = ds_tg.lat_tg.sel(station=station).data
        # find coordinates of data grid, which are nearest to those of the tide gauge station
        if land_mask==True:
            lat,lon,dist = find_nearest_ocean_point(ds_grid,lat_tg,lon_tg)
        else:
            # find coordinates of satellite data grid, which are nearest to those of the tide gauge station
            i_lon = find_nearest(ds_grid.lon,lon_tg)
            i_lat = find_nearest(ds_grid.lat,lat_tg)
            lon = ds_grid.lon[i_lon]
            lat = ds_grid.lat[i_lat]
            dist = haversine_distance(lat,lon,lat_tg,lon_tg)

        # Select closest point
        x = ds_grid[var_grid].sel(lon=lon, lat=lat)
            
        dataset = xr.Dataset(
            data_vars = {
                var_grid: (["station","time"], np.array([x])),                    
                "lon": (["station"], np.array([lon])),
                "lat": (["station"], np.array([lat])),
                "dist2tg": (["station"],np.array([dist])),
             
            },
            coords = dict(station=np.array([station]), time=ds_grid.time.data)
        )
        datasets.append(dataset)
    #ds_out = xr.merge(ds_tg,dataset)
    ds_out = xr.concat((datasets), dim="station")
    ds_out = ds_out.assign_attrs({
        var_grid: "Sea level anomaly from gridded data closest to tide gauge stations.",
        "dist2tg": "Harversine distance of ocean points from gridded data closest to tide gauge stations in km.",
        })
    return ds_out

def find_nearest(array, value):
    """Find index of value within a list closest to a search value

    Args:
        array (array_type): Search array
        value (int,float): Search value

    Returns:
        int: Index of nearest value
    """
    array = np.asarray(array)
    idx = (np.abs(array-value)).argmin()
    return idx

def find_nearest_ocean_point(ds_grid,lat_tg,lon_tg):
    def take_first(elem):
        return elem[0]
    """Computes the coordinates of the gridpoint which is closest to given coordinates, and is on the ocean. The gridded data set requires a land mask for this function.	
    Args:
        ds_grid (xarray.DataSet): Gridded data set
        lat_tg (float): Latitude of search point
        lon_tg (float): Longitude of search point

    Returns:
        lon (float): Longitude of gridded data point closest to search point
        lat (float): Latitude of gridded data point closest to search point
        dist (float): Harversine distance of gridded data point closest to search point
    """
    lon_list = []
    lat_list = []
    dist_list = []
    for lat in ds_grid.lat.data:
        for lon in ds_grid.lon.data:
            #lat_list.append(lat)
            #lon_list.append(lon)
            if np.isnan(ds_grid.land_mask.sel(lon=lon, lat=lat))==False:
                dist_list.append(haversine_distance(lat,lon,lat_tg.data,lon_tg.data))
                lat_list.append(lat)
                lon_list.append(lon)
                
    
    idx = dist_list.index(min(dist_list))
    ilon = lon_list[idx]
    ilat = lat_list[idx]
    idist= dist_list[idx]
 
    return ilat,ilon,idist
            #break
    #dist_list = dist_list[0][:]
    #idx = dist_list.index(min(dist_list))
    #idx = dist_list.argmin()
    #ilon = lon_list[idx]
    #ilat = lat_list[idx]
    #idist= dist_list[idx]
    #dist_list,lon_list,lat_list = zip(*sorted(zip(dist_list,lon_list,lat_list)))
    
    #for idist,ilon,ilat in zip(dist_list,lon_list,lat_list):
    #if np.isnan(ds_grid.land_mask.sel(lon=ilon, lat=ilat))==False:
    #    return ilat,ilon,idist
            #break

def haversine_distance(lat1, lon1, lat2, lon2):
    """Computes the harversine distance between two points.

    Args:
        lat1 (float): Latitude of point 1
        lon1 (float): Longitude of point 1
        lat2 (float): Latitude of point 2
        lon2 (float): Longitude of point 2

    Returns:
        float: harversine distance
    """
    r = 6371
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 -lon1)
    a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2)**2
    res = r * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))
    return np.round(res, 4)

def select_station_time(ds_rev,var_rev,ds,var):
    """Select data from selected stations at selected time spans depending on data availible on a reference data set (ds_rev).

    Args:
        ds_rev (xarray.Dataset): Reference data set
        var_rev (str): Variable name of reference data
        ds (str): Target data set
        var (xarray.Dataset): Variable name of target data set

    Returns:
        xarray.Dataset: Selected data from target data set
    """
    datasets = [] 
    for station in tqdm(ds_rev.station,"selecting data..."):
        x_rev = ds_rev[var_rev].sel(station=station).dropna(dim="time")
        x = ds[var].sel(station=station)
        # t_min = np.where(tg_tm.time==min(x_rev.time))[0][0]
        # t_max = np.where(tg_tm.time==max(x_rev.time))[0][0]
        # time_frame = ds.time.data[t_min:t_max]
        t_min = min(x_rev.time.data)
        t_max = max(x_rev.time.data)
        idx_min = np.where(x.time.data == t_min)[0][0]
        idx_max = np.where(x.time.data == t_max)[0][0]
        time_frame = x.time.data[idx_min:idx_max]
        x = x.data[idx_min:idx_max]
        dataset = xr.Dataset(
            data_vars = {
                var: (["station","time"], np.array([x])),                    
                "longitude": (["station"], np.array([ds.longitude.sel(station=station).data])),
                "latitude": (["station"], np.array([ds.latitude.sel(station=station).data])),
                "site_name": (["station"], np.array([ds.site_name.sel(station=station).data])),
                "country": (["station"], np.array([ds.country.sel(station=station).data])),
            },
            coords = dict(station=np.array([station]), time=time_frame)
        )
        datasets.append(dataset)
    ds_out = xr.concat(datasets, dim="station")
    return ds_out

def correlate(ds_grid,var_grid,ds_tg,var_tg):
    
    """Correlation of two datasets, where the first is taken from a gridded dataset and the second is from tide gauge data.

    Args:
        ds_grid (xarray.DataSet): Dataset taken from gridded product
        var_grid (str): Variable to be correlateed
        ds_tg (xarray.DataSet): Tide gauge dataset
        var_tg (str): Variable to be correlateed

    Returns:
        xarray.DataSet: Output dataset with correlation
    """
    ds_out = xr.Dataset(
        data_vars={
            var_grid: (["station","time"],ds_grid[var_grid].data),
            var_tg: (["station","time"],ds_tg[var_tg].data),
        },
        coords=dict(
            station=(["station"],ds_tg.station.data),
            time=(["time"],ds_tg.time.data),
            )
        )
    #ds_out = ds_out.dropna(dim='time')
    corr = xr.corr(ds_out[var_grid],ds_out[var_tg],dim="time")
    rmse = xs.rmse(ds_out[var_grid],ds_out[var_tg],dim='time', skipna=True)
    corr = corr.assign_attrs({"correlation":"Correlation between "+var_grid+" and "+var_tg})
    rmse = rmse.assign_attrs({"rmse":"RootMean Squared Error between "+var_grid+" and "+var_tg})
    return corr, rmse 


def coherence (ds_grid,var_grid,ds_tg,var_tg,window_len=430):
    """Coherence of two datasets, where the first is taken from a gridded dataset and the second is from tide gauge data.

    Args:
        ds_grid (xarray.DataSet): Dataset taken from gridded product
        var_grid (str): Variable to be correlateed
        ds_tg (xarray.DataSet): Tide gauge dataset
        var_tg (str): Variable to be correlateed

    Returns:
        xarray.DataSet: Output dataset with coherence
    """
    cohs = []
    for s in ds_tg.station:
        
        fq, coh = signal.coherence(
            ds_grid[var_grid].sel(station=s).dropna(dim="time"),
            ds_tg[var_tg].sel(station=s).dropna(dim="time"), 
            fs=1.0, window='hann', nperseg=window_len)
        if fq.size != 0:
            coherence = xr.DataArray(
                data = np.array([coh]),
                dims = ["station","frequency"],
                coords = dict(station=np.array([s]), frequency=fq)                                    
                )
            cohs.append(coherence)
        if np.mean(coh)==1:
            print("station " + str(s.data) +": time series to short, window length of " + str(window_len)+ " is too long. Result of coherence = 1")
    
        coherence = xr.concat(cohs, dim="station")
        coherence = coherence.assign_attrs({"coherence": "Coherence between "+var_grid+" and "+var_tg+". window_len = "+str(window_len)+" days."})
    return coherence

def nrmse(ds_grid,var_grid,ds_tg,var_tg,):
    """Computes the  root mean square error of two datasets, where the first is taken from a gridded dataset and the second is from tide gauge data. The RMSE is normalized by the range of the data.

    Args:
        ds_grid (xarray.DataSet): Dataset taken from gridded product
        var_grid (str): Variable name
        ds_tg (xarray.DataSet): Tide gauge dataset
        var_tg (str): Variable name

    Returns:
        xarray.DataSet: Output dataset with NRMSE
    """
    nrmse_out = []
    for s in ds_tg.station:
        y = ds_tg[var_tg].sel(station=s).dropna(dim="time").data
        y_hat = ds_grid[var_grid].sel(station=s).dropna(dim="time").data
        y_range = np.max([np.max(y),np.max(y_hat)])-np.min([np.min(y),np.min(y_hat)])
        nrmse = mean_squared_error(y,y_hat)/y_range

        temp = xr.DataArray(
            data = np.array([nrmse]),
            dims = ["station"],
            coords = dict(station=np.array([s]))                                    
            )
        nrmse_out.append(temp)
        
    nrmse_out = xr.concat(nrmse_out, dim="station")
    nrmse_out = nrmse_out.assign_attrs({"nrmse": "Normalized root mean squared error between "+var_grid+" and "+var_tg+". Normalized with value range"})
    return nrmse_out

def nan_nrmse(ds_grid,var_grid,ds_tg,var_tg,):
    """Computes the  root mean square error of two datasets, where the first is taken from a gridded dataset and the second is from tide gauge data. The RMSE is normalized by the range of the data.

    Args:
        ds_grid (xarray.DataSet): Dataset taken from gridded product
        var_grid (str): Variable name
        ds_tg (xarray.DataSet): Tide gauge dataset
        var_tg (str): Variable name

    Returns:
        xarray.DataSet: Output dataset with NRMSE
    """
    nrmse_out = []
    for s in ds_tg.station:
        y = ds_tg[var_tg].sel(station=s).data
        y_hat = ds_grid[var_grid].sel(station=s).data
        try:
            y_range = np.nanmax([np.nanmax(y),np.nanmax(y_hat)])-np.nanmin([np.nanmin(y),np.nanmin(y_hat)])
            nrmse = np.sqrt(np.nanmean((y-y_hat)**2))/y_range

            temp = xr.DataArray(
                data = np.array([nrmse]),
                dims = ["station"],
                coords = dict(station=np.array([s]))                                    
                )
            nrmse_out.append(temp)
        except ValueError:
            temp = xr.DataArray(
                data = np.array([np.nan]),
                dims = ["station"],
                coords = dict(station=np.array([s]))                                    
                )
            nrmse_out.append(temp)
    nrmse_out = xr.concat(nrmse_out, dim="station")
    nrmse_out = nrmse_out.assign_attrs({"nrmse": "Normalized root mean squared error between "+var_grid+" and "+var_tg+". Normalized with value range. NaN values ignored."})
    return nrmse_out

def count_non_nan(ds_tg,var_tg,):
    arr_out = []
    for s in ds_tg.station:
        y = ds_tg[var_tg].sel(station=s).dropna(dim="time").data
   
        count = np.count_nonzero(~np.isnan(y))

        temp = xr.DataArray(
            data = np.array([count]),
            dims = ["station"],
            coords = dict(station=np.array([s]))                                    
            )
        arr_out.append(temp)

    arr_out = xr.concat(arr_out, dim="station")
    arr_out = arr_out.assign_attrs({"counts": "Count of valid (Non-NaN) values in timeseries."})
    return arr_out
## Old
class Preprocessing:
    
    def __init__(self, sat_ds, tg_ds, sat_var, tg_var, DAC, min_len, max_gap, interp, lowess, loc, attrs):
        """Initialization of class variables. loading of satellite and tide gauge data and DAC data if needed.

        Args:
            sat_var (str): Name of satellite data variable.
            tg_var (str): Name of tide gauge data variable.
            DAC (bool,None): Perform dynamic atmospheric correction. 
                True: sat data = original; tg data = original - DAC
                False: sat data = original + DAC; tg data = original
                None: sat data = original; tg data = original
            min_len (int, None): Minimmum length of time series.
            max_gap (int, None): Maximum number of consecutive data gaps.
            interp (bool): Choose if linear interpolation of missing values shall be performed.
            lowess (bool): Coose if lowess 40 hour filter for filtering of tides shall be performed.
            loc (str): {"pat", "cal"} Location of tide gauges
                "pat": Patagonia
                "cal": California
        """
        # unpack values of argument dictionary:
        self.__dict__.update(locals())
        
        # initialize data variables
        self.sat_data = sat_ds
        self.tg_data = tg_ds
        if DAC != None:
            self.DAC_data = DAC

        self.attrs_raw = attrs
 
    def process(self):
        """
        Prepares gridded data analysis. Satellite and tide gauge data are matched in spatial and time
            domain so that time series analysis like correlations can be performed.

        Args:
            sat_data (xr.DataArray): Gridded data from satellite.
            tg_data (xr.DataArray): Data from tide gagues.
            interp (bool): Choose if tide gauge data should be interpolated.
            smoothing (bool): Choose if tide gauge data should be smoothed to eliminate tidal contributions.
            min_len (int, optional): Minimum length of time series in days.
            max_gap (int, optional): Maximum number of consecutively occuring missing values.
            t0 (datetime64, optional): Start date of time slice for selection of time series for correlation. 
                                    Defaults to None. In default case the start date is chosen from where the
                                    dates of both time series overlap.
            tend (datetime64, optional): Start date of time slice for selection of time series for correlation. 
                                    Defaults to None. In default case the end date is chosen from where the
                                    dates of both time series overlap.
            

        Returns:
            dataset (xr.Dataset): Dataset containing tide gauge and satellite data.
            start (xr.DataArray, None): Data array containing start date of time series. Returns None if t0 = None
            end (xr.DataArray, None): Data array containing end date of time series. Returns None if tend = None

        """
        
        datasets = []
        
        for station in self.tg_data.station:
            #print(station.data)
            # find coordinates of satellite data grid, which are nearest to those of the tide gauge station
            i_lon = self._find_nearest(self.sat_data.longitude,self.tg_data.longitude.sel(station=station).data)
            i_lat = self._find_nearest(self.sat_data.latitude,self.tg_data.latitude.sel(station=station).data)
            x_sat = self.sat_data[self.sat_var].isel(longitude=i_lon, latitude=i_lat)

            # check if next gridpoint is a land point
            # if true search for next nearest sea gridpoint
            try:
                if self.sat_data.land_mask.isel(longitude=i_lon, latitude=i_lat)==0:
                    lon, lat = self._find_nearest_2(self.sat_data, self.sat_var, self.tg_data, station)
                    x_sat = self.sat_data[self.sat_var].sel(longitude=lon, latitude=lat)
            except AttributeError:
                pass

            # set final satellite coordinates
            sat_lon = x_sat.longitude.data
            sat_lat = x_sat.latitude.data

            x_tg = self.tg_data[self.tg_var].sel(station=station)

            # Select only samples, where time of x_sat and x_tg match 
            # x_tg[2015:2020]
            x_tg = x_tg.where(x_tg.time.isin(x_sat.time),drop=True)

            # continue if x_sat or x_tg are nan vectors
            if np.isnan(np.asarray(x_tg.mean())) or np.isnan(np.asarray(x_sat.mean())):
                continue

            # Select only samples, where time of x_sat and x_tg match - continuation
            # nan values at beginning and eng of x_tg are dropped
            start = x_tg.dropna(dim="time").time.data[0]
            end = x_tg.dropna(dim="time").time.data[-1]
            x_tg = x_tg.sel(time=slice(start,end))
            x_sat = x_sat.sel(time=slice(start,end))

            # Replace -99.9999 with nan
            x_tg = x_tg.where(x_tg!= -99.9999)    

            if self.min_len != None:
                if (len(x_tg) <= self.min_len):
                    continue
            
            # count consecutive occurrences of missing values
            if self.max_gap != None: 
                nan_counts = self._count_nan(x_tg)
                if (all(x <= self.max_gap for x in nan_counts)!=True):
                    continue

            if self.DAC != None:
                # get DAC timeseries corresponding to the nearest x_sat gridpoint
                i_lon = self._find_nearest(self.DAC_data.longitude.data, sat_lon)
                i_lat = self._find_nearest(self.DAC_data.latitude.data, sat_lat)
                x_DAC = self.DAC_data.isel(longitude=i_lon, latitude=i_lat)
                x_DAC["longitude"] = sat_lon
                x_DAC["latitude"] = sat_lat
                # match time of x_sat and x_DAC
                x_DAC = x_DAC.where(x_DAC.time.isin(x_sat.time),drop=True)
                # perform DAC
                if self.DAC == True:
                    # Add correction to sea_level
                    x_tg = x_tg - x_DAC
                elif self.DAC == False:
                    # Remove correction from sla
                    x_sat = x_sat + x_DAC


            # Substract mean so get sea level anomalies
            x_tg = x_tg-x_tg.mean()
            x_sat = x_sat-x_sat.mean()

            # interpolate missing values in x_tg
            if self.interp == True:
                x_tg = x_tg.interpolate_na(method = "linear", dim="time")

            # apply lowess 40 hour filter to x_tg
            if self.lowess == True:
                # filter tide guage data
                nans = sum(self._count_nan(x_tg))
                frac = 1/((len(x_tg.time)-nans)/40)
                x_tg = lowess(x_tg, x_tg.time, is_sorted='false', missing='drop', 
                            return_sorted=False, frac=frac)

            dataset = xr.Dataset(
                data_vars = {
                    self.sat_var: (["station","time"], np.array([x_sat])),                    
                    self.tg_var: (["station","time"], np.array([x_tg])),
                    "lon_tg": (["station"], np.array([self.tg_data.longitude.sel(station=station).data])),
                    "lat_tg": (["station"], np.array([self.tg_data.latitude.sel(station=station).data])),
                    "lon_sat": (["station"], np.array([x_sat.longitude.data])),
                    "lat_sat": (["station"], np.array([x_sat.latitude.data])),
                    "site_name": (["station"], np.array([self.tg_data.site_name.sel(station=station).data]).astype(str)),
                    "country": (["station"], np.array([self.tg_data.country.sel(station=station).data]).astype(str))
                },
                coords = dict(station=np.array([station]), time=x_sat.time.data)
            )

            datasets.append(dataset)
        self.dataset = xr.concat(datasets, dim="station")
        self.dataset = self._write_attrs(self.dataset)
        return self.dataset

    def corr_coh (self, window_len=430):
        # Correlation
        corr = xr.corr(self.dataset[self.sat_var], self.dataset[self.tg_var],dim="time")
        self.dataset = self.dataset.assign({"corr_"+self.sat_var: corr})
        self.dataset = self.dataset.assign_attrs({"corr_"+self.sat_var: "Correlation between "+self.sat_var+" and "+self.tg_var})

        # Coherence
        cohs = []
        for s in self.dataset.station:
            try:
                fq, coh = signal.coherence(self.dataset[self.sat_var].sel(station=s).dropna(dim="time"), 
                                    self.dataset[self.tg_var].sel(station=s).dropna(dim="time"), 
                                    fs=1.0, window='hann', nperseg=window_len)
                if fq.size != 0:
                    coherence = xr.Dataset(
                        data_vars = {"coh_"+self.sat_var: (["station","frequency"], np.array([coh]))},
                        coords = dict(station=np.array([s]), frequency=fq)                                    
                        )
                    cohs.append(coherence)
                if np.mean(coh)==1:
                    print("station " + str(s.data) +": time series to short, window length of " + str(window_len)+ " is too long. Result of coherence = 1")
            except ValueError:
                continue
            coherence = xr.concat(cohs, dim="station")
            self.dataset = xr.merge([self.dataset, coherence], compat="no_conflicts")
            # dataset = dataset.assign_coords(dict(frequency=fq))
            # dataset = dataset.assign({"coh_"+sat_data+app: xr.DataArray(data=data,dims=["station","frequency"])})
            self.dataset = self.dataset.assign_attrs({"coh_"+self.sat_var: "Coherence between "+self.sat_var+" and "+self.tg_var+". window_len = "+str(window_len)+" days."})
        return self.dataset

    def _write_attrs(self, ds):
        """Function for assigning attributes to dataset

        Args:
            ds (xarray dataset): input dataset

        Returns:
            xarray dataset: output dataset
        """
        
        attrs_raw = load_attrs()
        keys = [
            'creation_date',
            'location', 
            'description', 
            'time_match', 
            'min_len', 
            'max_gap', 
            'interp', 
            'lowess', 
            self.sat_var, 
            self.tg_var,
            'DAC', 
            'no_DAC', 
            'lon_tg', 
            'lat_tg', 
            'lon_sat', 
            'lat_sat', 
            'country', 
            'site_name'
            ]

        attrs = {}
        for key,val in self.attrs_raw.items(): 
            if key in keys:
                attrs.update({key: val}) 

        # location
        if self.loc == "pat":
            attrs["location"] = "Patagonia"
        elif self.loc == "cal":
            attrs["location"] = "California"

        # min length
        if self.min_len == None:
            del attrs["min_len"]
        else:
            attrs["min_len"] = str(self.min_len)+" - "+attrs["min_len"]

        # max gap
        if self.max_gap == None:
            del attrs["max_gap"]
        else:
            attrs["max_gap"] = str(self.max_gap)+" - "+attrs["max_gap"]

        # interpolate
        if self.interp == False:
            del attrs["interp"]

        # lowess 40 hour filter
        if self.lowess == False:
            del attrs["lowess"]
        
        # Dynamic Atmospheric Correction
        if self.DAC == None:
            del attrs["DAC"]
            del attrs["no_DAC"]
        elif self.DAC == True:
            del attrs["no_DAC"]
        elif self.DAC == False:
            del attrs["DAC"]

        # creation date
        attrs["creation_date"] = str(datetime.fromtimestamp(time()))

        return ds.assign_attrs(attrs)
            
    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        r = 6371
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 -lon1)
        a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2)**2
        res = r * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))
        return np.round(res, 2)

    def _merge_datasets(self, ds1: xr.Dataset, ds2: xr.Dataset, variables: list):
        datasets = []
        for var in variables:
            ds_temp = xr.concat([ds1[var], ds2[var]],dim="time").to_dataset()
            datasets.append(ds_temp)
        return xr.merge(datasets,"equals")
        
    def _count_nan(self, array):
        """counts consecutive occurences of missing values in a time series and adds counts to a list

        Args:
            array (array_type): time series

        Returns:
            nan_counts(list): List with counts
        """
        nan_counts = []
        nans = np.where(np.isnan(array))[0]
        n = 0
        for i in range(1,len(nans)):
            n += 1
            if (nans[i]-nans[i-1])!=1:
                nan_counts.append(n)
                n = 0 
        if n!=0:
            n += 1
            nan_counts.append(n)
        if len(nans)==1:
            nan_counts.append(1)
        return nan_counts

    def _check_data_overlap(self, x_sat: xr.Dataset,x_tg: xr.Dataset):
        """Check if coordinates of stations in tide gague data set are contained in
        the satelittes coordinate grid.

        Args:
            x_sat (xarray.Dataset): satellite data set
            x_tg (xarray.Dataset): tide gague data set

        Returns:
            boolean: True/False
        """
        # print(x_tg.longitude.min().data,x_tg.longitude.max().data)
        # print(x_sat.longitude.min().data,x_sat.longitude.max().data)

        # print(x_tg.latitude.min().data,x_tg.latitude.max().data)
        # print(x_sat.latitude.min().data,x_sat.latitude.max().data)

        if (x_tg.longitude.min() >= x_sat.longitude.min()) and \
            (x_tg.longitude.max() <= x_sat.longitude.max()) and \
                (x_tg.latitude.min() >= x_sat.latitude.min()) and \
                    (x_tg.latitude.max() <= x_sat.latitude.max()):
                    return True
        else:
            return  False

    def _find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array-value)).argmin()
        return idx

    def _find_nearest_2(self, sat_data, sat_var, tg_data, station):
        lon_sat = sat_data.longitude.data
        lat_sat = sat_data.latitude.data
        lon_tg = tg_data.longitude.sel(station=station).data
        lat_tg = tg_data.latitude.sel(station=station).data

        # find neighboring coordinates to station
        i_lon = np.argsort((np.abs(lon_sat-lon_tg)))[:2]
        i_lat = np.argsort((np.abs(lat_sat-lat_tg)))[:2]
        lon = lon_sat[i_lon]
        lat = lat_sat[i_lat]

        # build all combinations of coordinates from meshgrid
        mesh = np.array(np.meshgrid(lon,lat))
        combs = mesh.reshape((mesh.shape[0],mesh.shape[1]*mesh.shape[2]))
        lon_sat = combs[0,:]
        lat_sat = combs[1,:]
        # calculate distance to neighboring gridpoints
        dist = self._haversine_distance(lat_sat,lon_sat,lat_tg,lon_tg)

        # check if ocean point using land mask
        for i in range(len(dist)):
            idx = np.argsort(dist)[i]
            lon = lon_sat[idx] 
            lat = lat_sat[idx]

            a = sat_data[sat_var].sel(longitude=lon, latitude=lat)
            b = sat_data.land_mask.sel(longitude=lon, latitude=lat)
            if sat_data.land_mask.sel(longitude=lon, latitude=lat)==1:
                return lon, lat

        # if no ocean point found within neares neighbors, next grid point is chosen,
        # regardless wether it is a land point
        if sat_data.land_mask.sel(longitude=lon, latitude=lat)==0:
            idx = np.argmin(dist)
            lon = lon_sat[idx] 
            lat = lat_sat[idx]
            return lon, lat
            
def add_land_mask(ds_grid,var):# create land mask and assign to datasets
    if "land_mask" not in list(ds_grid):
        
        land_mask = ds_grid[var].transpose('time','lat','lon').data[1,:,:].copy()
        land_mask[np.isnan(land_mask)==False]=1 # valid numbers are set so 1
        mask = xr.DataArray(
            data = land_mask,
            dims = ["lat","lon"],
            coords = dict(
                lat = ds_grid.lat.data,
                lon = ds_grid.lon.data
            )
        )
        ds_grid = ds_grid.assign({"land_mask":mask})
    return ds_grid
    
    
def change_latlon(ds_grid):
    if ('latitude' and 'longitude' in list(ds_grid.coords)) and ('lat' and 'lon' not in list(ds_grid.coords)):
        ds_grid = ds_grid.rename_dims({'latitude':'lat','longitude':'lon'})
        ds_grid  = ds_grid.rename_vars({'latitude':'lat','longitude':'lon'})   

    if ('latitude' and 'longitude' in list(ds_grid.data_vars)) and ('lat' and 'lon' not in list(ds_grid.data_vars)):
        ds_grid  = ds_grid.rename_vars({'latitude':'lat','longitude':'lon'})   
    return ds_grid             
            
            
            
"""
Functions for detrending xarray data.
copied from: https://xrft.readthedocs.io/en/latest/_modules/xrft/detrend.html
"""

import scipy.signal as sps
import scipy.linalg as spl

def detrend(da, dim, detrend_type="constant"):
    """
    Detrend a DataArray

    Parameters
    ----------
    da : xarray.DataArray
        The data to detrend
    dim : str or list
        Dimensions along which to apply detrend.
        Can be either one dimension or a list with two dimensions.
        Higher-dimensional detrending is not supported.
        If dask data are passed, the data must be chunked along dim.
    detrend_type : {'constant', 'linear'}
        If ``constant``, a constant offset will be removed from each dim.
        If ``linear``, a linear least-squares fit will be estimated and removed
        from the data.

    Returns
    -------
    da : xarray.DataArray
        The detrended data.

    Notes
    -----
    This function will act lazily in the presence of dask arrays on the
    input.
    """

    if dim is None:
        dim = list(da.dims)
    else:
        if isinstance(dim, str):
            dim = [dim]

    if detrend_type not in ["constant", "linear", None]:
        raise NotImplementedError(
            "%s is not a valid detrending option. Valid "
            "options are: 'constant','linear', or None." % detrend_type
        )

    if detrend_type is None:
        return da
    elif detrend_type == "constant":
        return da - da.mean(dim=dim)
    elif detrend_type == "linear":
        data = da.data
        axis_num = [da.get_axis_num(d) for d in dim]
        chunks = getattr(data, "chunks", None)
        if chunks:
            axis_chunks = [data.chunks[a] for a in axis_num]
            if not all([len(ac) == 1 for ac in axis_chunks]):
                raise ValueError("Contiguous chunks required for detrending.")
        if len(dim) == 1:
            dt = xr.apply_ufunc(
                sps.detrend,
                da,
                axis_num[0],
                output_dtypes=[da.dtype],
                dask="parallelized",
            )
        elif len(dim) == 2:
            dt = xr.apply_ufunc(
                _detrend_2d_ufunc,
                da,
                input_core_dims=[dim],
                output_core_dims=[dim],
                output_dtypes=[da.dtype],
                vectorize=True,
                dask="parallelized",
            )
        else:  # pragma: no cover
            raise NotImplementedError(
                "Only 1D and 2D detrending are implemented so far."
            )

    return dt



def _detrend_2d_ufunc(arr):
    assert arr.ndim == 2
    N = arr.shape

    col0 = np.ones(N[0] * N[1])
    col1 = np.repeat(np.arange(N[0]), N[1]) + 1
    col2 = np.tile(np.arange(N[1]), N[0]) + 1
    G = np.stack([col0, col1, col2]).transpose()

    d_obs = np.reshape(arr, (N[0] * N[1], 1))
    m_est = np.dot(np.dot(spl.inv(np.dot(G.T, G)), G.T), d_obs)
    d_est = np.dot(G, m_est)
    linear_fit = np.reshape(d_est, N)
    return arr - linear_fit
