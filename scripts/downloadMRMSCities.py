import cv2
from PIL import Image
import matplotlib as mpl
from siphon.catalog import TDSCatalog
from datetime import timedelta
from datetime import datetime
import time
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from xarray.backends import NetCDF4DataStore
from netCDF4 import Dataset
import xarray as xr
from dask import array as da
import requests
import matplotlib.colors as mpcolors
from metpy import calc as mpcalc
from metpy.units import units
import pandas as pd
import psycopg2
import shutil
import scipy
import scipy.stats as stats
import requests
from requests import HTTPError
from pathlib import Path
import os
import numpy.matlib
import cartopy.io.shapereader as shpreader

def download_file(url, date):
    local_dir = f"/rstudio/pmykolajtchuk/Observation/"
    local_filename = Path(local_dir + f"{url.split('/')[-1]}")
    if not Path(local_dir).is_dir():
        os.mkdir(local_dir)
    if Path(local_filename).is_file():
        print(f'Already have {local_filename}')
        return local_filename
    else:
        # NOTE the stream=True parameter below
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    # If you have chunk encoded response uncomment if
                    # and set chunk_size parameter to None.
                    #if chunk:
                    f.write(chunk)
        return local_filename


now = datetime.now()
start = datetime(now.year, now.month, now.day, now.hour)
start2 = start + timedelta(hours=4)
start3 = start-timedelta(hours=1)

url_date = start2.strftime('%Y%m%d%H')
url_year = url_date[0:4]
url_month = url_date[4:6]
url_day = url_date[6:8]
url_hour = url_date[8:10]
#print(url_date)
#print(url_year)
#print(url_month)

times = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
times_2 =[0, 5, 0, 5, 0, 5, 0, 5, 0, 5, 0, 5]

for i in enumerate(times):
	try:
		download_file(f'https://mesonet.agron.iastate.edu/archive/data/{url_year}/{url_month}/{url_day}/GIS/uscomp/n0q_{url_year}{url_month}{url_day}{url_hour}{times[i[0]]}{times_2[i[0]]}.png', url_date)
	except HTTPError as e:
        	print(e)

files =  list(Path(f"/rstudio/pmykolajtchuk/Observation/").glob(f'n0q_{url_year}{url_month}{url_day}{url_hour}*.png'))
files.sort(key=os.path.getmtime)
#print(files)

interp_index = pd.date_range(start3, start3+timedelta(minutes=55), freq='5min')
#print(interp_index)

latitude = np.arange(50,23.000,-0.005)
#print(latitude.shape)
latitude_numpy = np.matlib.repmat(latitude,12200,1)
latitude_transpose = np.transpose(latitude_numpy)
print(latitude_transpose.shape)

longitude = np.arange(-126,-65.000,0.005)
longitude_transpose = np.transpose(longitude)
longitude_numpy = np.matlib.repmat(longitude_transpose,5400,1)

print(longitude_numpy.shape)

n = 10
latitude_correct = latitude_transpose[::n,::n]
longitude_correct = longitude_numpy[::n,::n]
counter = 10
col_list=["Latitude","Longitude"]
#print(col_list)
stations = ['FST','MAF','SJT','ABI','SPS','DFW','TYR','ACT','AUS','SAT','IAH','GLS','CRP','BRO','LRD']
Farms = pd.read_csv("wind_farms_latslons_columns.csv", usecols = col_list)
station_coords_lat = [30.8769, 31.9417, 31.3573, 32.4119, 33.9643, 32.8998, 32.3511, 31.6092, 30.1975, 29.5312, 29.9902, 29.2682, 27.7724, 25.9063, 27.5430]
station_coords_lon = [-102.8919,-102.2047, -100.5028,-99.68,-98.4918, -97.0403, -95.4096,-97.2232,-97.6664,-98.4683,-95.3368,-94.8552,-97.5022,-97.4270,-99.4621]
for j in enumerate(files):
	reader = shpreader.Reader('countyl010g.shp')
	counties = list(reader.geometries())
	COUNTIES = cfeature.ShapelyFeature(counties, ccrs.PlateCarree())
	plt.figure(figsize=(18, 12))
	ax = plt.axes(projection=ccrs.PlateCarree())
	ax.add_feature(cfeature.LAND.with_scale('50m'))
	ax.add_feature(cfeature.OCEAN.with_scale('50m'))
	ax.add_feature(cfeature.LAKES.with_scale('50m'))
	ax.add_feature(COUNTIES, facecolor='none', edgecolor='gray')
	degree_sign = u'\N{DEGREE SIGN}'
	ax.coastlines('50m')
	ax.set_extent([-106.5, -93.5, 25.9, 36.5])
	print(files[j[0]])
	radarfile = np.asarray(Image.open(f'{files[j[0]]}'))
#	radarfile = cv2.imread(f'{files[j[0]]}')
	print(radarfile.shape)
	print(radarfile.dtype)
	dbzradarfile = radarfile[::n,::n]
	dbzradarfile_correct1 = [[x*0.5 for x in row] for row in dbzradarfile]
	dbzradarfile_correct2 = [[x-32.5 for x in row] for row in dbzradarfile_correct1]
	## dark blue, dark aquamarine, teal, light blue, dark green, green, lime green, yellow, dark yellow, orange, red, maroon, pink, purple, peach
	cs = plt.contourf(longitude_correct,latitude_correct,dbzradarfile_correct2,levels=[5, 15, 20, 25, 30, 40, 50, 60, 70, 75, 100],colors = ['#add8e6','#00008b','#00FF00','#004100','#FFFF00','#FFA500','#FF0000','#FFC0CB','#800080','#FFE5B4'])
	ax.set_title(f'Observed Reflectivity, valid (CDT): {interp_index[j[0]]}',fontsize=26)
	plt.scatter(station_coords_lon,station_coords_lat,color='black')
	for i in enumerate(stations):
		ax.annotate(stations[i[0]], (station_coords_lon[i[0]],station_coords_lat[i[0]]),fontsize=22)
	##gridlines = ax.gridlines(draw_labels=True)
	#cax = ax.imshow(dbzradarfile, cmap=['#00008b','#40826d','#008080','#','#013220','#32CD32','#FFFF00','#9b870c','#FFA500','#FF0000','#800000','#FFC0CB','#800080','#FFE5B4'])
	#ax.set_title('dBZ',fontsize=22)
	#cbar = fig.colorbar(cax, ticks=[-1, -0.86666666666, -0.73333333332, -0.59999999998, -0.46666666664, -0.3333333333, -0.19999999996, -0.06666666662, 0.06666666672,0.20000000006, 0.3333333334, 0.46666666674, 0.60000000008, 0.73333333342, 0.86666666676, 1], orientation='horizontal')
	#cbar.ax.set_xticklabels(['5', '10', '15', '20', '25', '30', '35', '40', '45', '50', '55', '60', '65', '70', '75', '80'])  # horizontal colorbar
	proxy = [plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0]) for pc in cs.collections]
	plt.legend(proxy, ["5-15 dBZ", "15-20 dBZ", "20-25 dBZ", "25-30 dBZ", "30-40 dBZ", "40-50 dBZ", "50-60 dBZ", "60-70 dBZ", "70-75 dBZ", "75+ dBZ"],fontsize=18,loc='upper left',framealpha=1)
	ax.text(-0.05, 0.55, f'Latitude ({degree_sign})', va='bottom', ha='center',rotation='vertical', rotation_mode='anchor',transform=ax.transAxes,fontsize=22)
	ax.text(0.5, -0.08, f'Longitude ({degree_sign})', va='bottom', ha='center',rotation='horizontal', rotation_mode='anchor',transform=ax.transAxes,fontsize=22)
	plt.show()
	plt.savefig(f'ObservedMRMSReflectivityCities_{counter}.png')
	plt.clf()
	plt.close()
	counter = counter+1

directory = "/rstudio/pmykolajtchuk/Observation/"
directory2 = "/data/rdata/rdatashare/weather/Observation/ERCOT/Cities/Radar/"

osdir = os.listdir(directory)


for item in osdir:
	if item.startswith("n0q"):
		os.remove(os.path.join(directory, item))

for item in osdir:
	if item.startswith("ObservedMRMSReflectivityCities_"):
		shutil.copyfile(f'{item}', f'/data/rdata/rdatashare/weather/Obse
