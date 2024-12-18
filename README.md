# **ST-Window Kriging** 
### **Spatio Temporal Window Kriging** for mapping daily Sea Level Anomalies from satellite altimetry along-track data (L3 to L4)


### Description 
The project **ST-Window-Kriging**, available in Jupyter Notebook format, can be used to produce daily Sea Level Anomalie Grids from along-track satellite altimetry data for California in 2018. **The final gridded data are published at SEANOE under https://doi.org/10.17882/103947.**

The approach $^1$ is based on Ordinary Kriging in a moving $\pm$ 5-day time window around the target day. Kriging is done by building an empirical variogram from the observations (along-track data) and fitting a sum-metric semi-variogram model, which is the base for the Kriging matrix. For details on Kriging, we recommend literature $^2$ listed below. The approach uses along-track data from CMEMS as input data (https://doi.org/10.48670/moi-00146) and produces gridded data comparable to their maps (https://doi.org/10.48670/moi-00149).


### Installation 
Provided that Git, python, and conda are installed: Open Terminal and follow the instructions: 

<u> 1) Clone the repository and go into the directory created: </u>

git clone https://github.com/mariejuhl/ST-Window-Kriging 

cd ST-Window-Kriging

<u> 2) Make a new Environment and activate: </u>
   
conda env create --file OK_ST.yml

conda activate OK-ST

<u> 3) Open *ST-Window-Kriging.ipynb* in Jupyter Notebook, VSCode, or other: </u>
   
jupyter-notebook ST-Window-Kriging.ipynb

<u> 4) Happy Kriging! </u>
   
   
### Usage 
#### Dependencies
Dependencies need to be installed according to the Installation step above. The main Notebook  **ST-Window-Kriging.ipynb** dependencies are loaded under **1. Import Dependencies**, where the main function is loaded from *tools.py*. 

#### Input data:
Input data are Sea Level Anomalies from along-track satellite altimetry (L3) in 1Hz, downloaded from Copernicus Marine Service (https://doi.org/10.48670/moi-00146). We collected them in a data frame containing, e.g., for California, all data within the year 2018 for the study region $\pm$ 2° latitude and longitude to avoid boundary effects. Input data in this study are stored in the directory *sample data*. Data are loaded in the Notebook and prepared under **2. Input data, Area Selection, Mask and Subsampling**. 

#### Variogram 
In **ST-Window-Kriging.ipynb**, variogram modeling is integrated into the *ST_Window_Kriging* function. Select Kriging=False to only fit variogram. Otherwise, the variogram will be built (matheron-like semi-variogram) and fitted (sum-metric model) using the time range (*ht_range*) and space range (*hs_range*). If *plot* is set to True, the experimental variogram and model are plotted for each day. Since this process can take a while, a processing bar will show the progress of the Kriging.

![Alt text](sample_variogram.png)

#### Daily Grids
The function *ST_Window_Kriging* will be executed for all days in *run_time*, which can be defined under **3. Producing daily Maps**. Grids will be plotted alongside the CMEMS L4 grid and can be stored as NetCDF or CSV (by giving a path to *save_as_csv* or *save_as_netcdf* in the function).

![Alt text](sample_output.png)


### Literature 

$^1$ Paper connected to the approach: 

Juhl, M.-C., Passaro, M., Dettmering, D. (tba): Regional daily sea level maps from Multi-mission Altimetry using Space-time Window Kriging (submitted to Advances in Space Research in December 2024)

$^2$ Basics of Kriging:

Heuvelink, G.B.M., Pebesma, E., Gräler, B. (2017). Space-Time Geostatistics. In: Shekhar, S., Xiong, H., Zhou, X. (eds) Encyclopedia of GIS. Springer, Cham. https://doi.org/10.1007/978-3-319-17885-1_1647

Example of Window Block Kriging:
Tadić, J. M., Qiu, X., Yadav, V., and Michalak, A. M.: Mapping of satellite Earth observations using moving window block kriging, Geosci. Model Dev., 8, 3311–3319, https://doi.org/10.5194/gmd-8-3311-2015, 2015.

### Contact

Feel free to contact me via email:

mariechristin.juhl@tum.de

