from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from siphon.catalog import TDSCatalog
import metpy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from metpy.plots import colortables
from metpy.plots import add_timestamp
import os
import shutil
import utm
import cartopy.io.shapereader as shpreader

# Create variables for URL generation
# YOUR CODE GOES HERE

# Construct the data_url string
# YOUR CODE GOES HERE

# Print out your URL and verify it works!
# YOUR CODE GOES HERE

# Create variables for URL generation
image_date = datetime.utcnow().date()
region = 'CONUS'
channel = 10

# We want to match something like:
# https://thredds-test.unidata.ucar.edu/thredds/catalog/satellite/goes16/GOES16/Mesoscale-1/Channel08/20181113/catalog.html

# Construct the data_url string
data_url = ('https://thredds.ucar.edu/thredds/catalog/satellite/goes/east/products/'
            f'CloudAndMoistureImagery/{region}/Channel{channel:02d}/'
            f'{image_date:%Y%m%d}/catalog.xml')

# Print out your URL and verify it works!
#print(data_url)

test_list = [1, 2, 3, 4, 5]
cat = TDSCatalog(data_url)
stations = ['FST','MAF','SJT','ABI','SPS','DFW','TYR','ACT','AUS','SAT','IAH','GLS','CRP','BRO','LRD']
station_coords_lat = [30.8769, 31.9417, 31.3573, 32.4119, 33.9643, 32.8998, 32.3511, 31.6092, 30.1975, 29.5312, 29.9902, 29.2682, 27.7724, 25.9063, 27.5430]
station_coords_lon = [-102.8919,-102.2047, -100.5028,-99.68,-98.4918, -97.0403, -95.4096,-97.2232,-97.6664,-98.4683,-95.3368,-94.8552,-97.5022,-97.4270,-99.4621]
counter = 14
degree_sign = u'\N{DEGREE SIGN}'
x_annotate = []
y_annotate = []

for i in enumerate(test_list):
        reader = shpreader.Reader('countyl010g.shp')
        counties = list(reader.geometries())
        COUNTIES = cfeature.ShapelyFeature(counties, ccrs.PlateCarree())
        dataset = cat.datasets[i[0]]
        print(dataset)
        ds = dataset.remote_access(use_xarray=True)
        dat = ds.metpy.parse_cf('Sectorized_CMI')
        #print(dat)
        proj = dat.metpy.cartopy_crs
        x = dat['x']
        y = dat['y']
        fig = plt.figure(figsize=(24, 24))
        ax = fig.add_subplot(1, 1, 1, projection=proj)
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=2)
        ax.add_feature(cfeature.STATES.with_scale('50m'), linestyle=':', edgecolor='black')
        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=2, edgecolor='black')
        ax.add_feature(COUNTIES, facecolor='none', edgecolor='black')
        im = ax.imshow(dat, interpolation='nearest',extent=(x.min(), x.max(), y.min(), y.max()))

        ax.set_extent([-106.5, -93.5, 25.9, 36.5])
        ax.set_title(f'Observed Satellite (Yellow/Brown: Few Clouds, Blue: Moderate Clouds, \n White/Green: Many Clouds)',fontsize=38)
        ax.text(-0.05, 0.51, f'Latitude ({degree_sign})', va='bottom', ha='center',rotation='vertical', rotation_mode='anchor',transform=ax.transAxes,fontsize=3$
        ax.text(0.5, -0.08, f'Longitude ({degree_sign})', va='bottom', ha='center',rotation='horizontal', rotation_mode='anchor',transform=ax.transAxes,fontsize$
        wv_norm, wv_cmap = colortables.get_with_range('WVCIMSS_r', 195, 290)
        im.set_cmap(wv_cmap)
        im.set_norm(wv_norm)
        ax.plot(station_coords_lon,station_coords_lat, 'ko', markersize=10, transform=ccrs.Geodetic())
        for j in enumerate(stations):
                ax.text(station_coords_lon[j[0]],station_coords_lat[j[0]], stations[j[0]], fontsize=28, transform=ccrs.Geodetic())
        start_time = datetime.strptime(ds.start_date_time, '%Y%j%H%M%S')
        add_timestamp(ax, time=start_time, pretext=f'GOES-16 Ch. {channel} ',
                high_contrast=True, fontsize=28, y=0.01)
        plt.show()
        plt.savefig(f'ObservedSatelliteCities_{counter}.png')
        plt.clf()
        plt.close()
        counter = counter-1
        
directory = "/rstudio/pmykolajtchuk/Observation/"
directory2 = "/data/rdata/rdatashare/weather/Observation/ERCOT/Cities/Satellite/"

osdir = os.listdir(directory)
osdir2 = os.listdir(directory2)


for item in osdir:
        if item.startswith("OR_ABI"):
                os.remove(os.path.join(directory, item))

for item in osdir:
        if item.startswith("ObservedSatelliteCities_"):
                shutil.copyfile(f'{item}', f'/data/rdata/rdatashare/weather/Observation/ERCOT/Cities/Satellite/{item}')
