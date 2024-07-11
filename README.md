# CreateGlobalMaps
/////////////////////////////////////////////////////////Global CPCP Precipitation Map////////////////////////////////////////////////////
#install required libraries
# pip install numpy xarray matplotlib cartopy seaborn netCDF4
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import seaborn as sns
from netCDF4 import Dataset
import cartopy.feature as cfeature
# Instructions and Data Source Information
# This script fetches global precipitation data from the GPCP dataset.
# The data is acquired from the NOAA Physical Sciences Laboratory.
# Base URL: http://www.esrl.noaa.gov
# Catalog URL: /psd/thredds/dodsC/Datasets/gpcp/precip.mon.mean.nc
# Define the base URL and catalog URL for GPCP Precipitation
baseURL = 'http://www.esrl.noaa.gov'
catalogURL = '/psd/thredds/dodsC/Datasets/gpcp/precip.mon.mean.nc'

# Open the dataset using netCDF4
dataset_url = baseURL + catalogURL
nc = Dataset(dataset_url)

# Convert the netCDF4 dataset to an xarray dataset
precipID = xr.open_dataset(xr.backends.NetCDF4DataStore(nc))

# Access the precipitation (precip) variable
precip = precipID['precip']

# Find the most recent time index
mostRecent = len(precip.time.values) - 1

# Select the most recent precipitation data
recentPrecip = precip.isel(time=mostRecent)

# Define contour levels
precipmin = 0
precipmax = 20
levels = np.linspace(precipmin, precipmax, 21)

# Create a figure with Orthographic projection focused on Asia
#If you want to focus on African regions set central_longitude value from 90 to 20
#For focus on South and North America set central_longitude value from 90 to -75
fig = plt.figure(figsize=[12, 6], facecolor='none')
ax = plt.subplot(1, 1, 1, projection=ccrs.Orthographic(central_longitude=90, central_latitude=0), facecolor='none')

# Use seaborn's icefire colormap
cmap = sns.color_palette("icefire", as_cmap=True)

# Plot the precipitation data
contour = recentPrecip.plot.contourf(levels=levels, cmap=cmap, transform=ccrs.PlateCarree(), ax=ax, add_colorbar=False)

# Add coastlines with higher resolution
ax.coastlines('10m')

 
ax.add_feature(cfeature.BORDERS, edgecolor='white')

# Uncomment the following lines to add a color bar
# cbar = plt.colorbar(contour, ax=ax, orientation='horizontal', pad=0.05)
# cbar.set_label('Precipitation (mm/month)', fontsize=12)  # Add 'color='white'' to set the text color to white
# cbar.ax.xaxis.set_tick_params(color='white')  # Add to set tick color to white
# cbar.outline.set_edgecolor('black')  # Add to set outline color to white
# plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color='black')  # Add to set tick labels color to white

# Uncomment the following line to add a title in black color
# plt.title('Global Precipitation Climatology on Most Recent Date', fontsize=14, weight='bold', color='black')  # Change color to 'black' if needed

# Save the figure with high resolution and transparency as PNG
plt.savefig('precip_plot_asia_orthographic.png', dpi=300, bbox_inches='tight', transparent=True)

# Display the plot
plt.show()


///////////////////////////////////////////////////////////Global Sea Surface Temperature/////////////////////////////////////////////////
#install required libraries
# pip install numpy xarray matplotlib cartopy seaborn netCDF4
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import seaborn as sns
from netCDF4 import Dataset
import cartopy.feature as cfeature

# Instructions and Data Source Information
# This script fetches global sea surface temperature (SST) data from the NOAA ERSST v5 dataset.
# The data is acquired from the NOAA Physical Sciences Laboratory.
# Base URL: http://www.esrl.noaa.gov
# Catalog URL: /psd/thredds/dodsC/Datasets/noaa.ersst.v5/sst.mnmean.nc

# Define the base URL and catalog URL for SST data
baseURL = 'http://www.esrl.noaa.gov'
catalogURL = '/psd/thredds/dodsC/Datasets/noaa.ersst.v5/sst.mnmean.nc'

# Open the dataset using netCDF4
dataset_url = baseURL + catalogURL
nc = Dataset(dataset_url)

# Convert the netCDF4 dataset to an xarray dataset
sstID = xr.open_dataset(xr.backends.NetCDF4DataStore(nc))

# Access the sea surface temperature (sst) variable
sst = sstID['sst']

# Find the most recent time index
mostRecent = len(sst.time.values) - 1

# Select the most recent SST data
recentSST = sst.isel(time=mostRecent)

# Define contour levels
sstmin = 0
sstmax = 30
levels = np.linspace(sstmin, sstmax, 21)

# Create a figure with Orthographic projection centered on Asia
#If you want to focus on African regions set central_longitude value from 90 to 20
#For focus on South and North America set central_longitude value from 90 to -75
fig = plt.figure(figsize=[12, 6], facecolor='none')
ax = plt.subplot(1, 1, 1, projection=ccrs.Orthographic(central_longitude=90, central_latitude=0), facecolor='none')

# Use seaborn's icefire colormap
cmap = sns.color_palette("icefire", as_cmap=True)

# Plot the SST data
contour = recentSST.plot.contourf(levels=levels, cmap=cmap, transform=ccrs.PlateCarree(), ax=ax, add_colorbar=False)

# Add coastlines with higher resolution
ax.coastlines('10m')

# Add country borders with black color
ax.add_feature(cfeature.BORDERS, edgecolor='black')

# Fill the countries with a light silver color
countries = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_0_countries',
    scale='10m',
    facecolor='lightgrey')
ax.add_feature(countries, edgecolor='black')

# Uncomment the following lines to remove the gridlines
# gl = ax.gridlines(draw_labels=False, color='white')

# Uncomment the following lines to add a color bar (legend)
# cbar = plt.colorbar(contour, ax=ax, orientation='horizontal', pad=0.05)
# cbar.set_label('Sea Surface Temperature (°C)', fontsize=12)  # Add 'color='white'' to set the text color to white
# cbar.ax.xaxis.set_tick_params(color='white')  # Add to set tick color to white
# cbar.outline.set_edgecolor('white')  # Add to set outline color to white
# plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color='white')  # Add to set tick labels color to white

# Uncomment the following line to add a title in black color
# plt.title('Global Sea Surface Temperature on Most Recent Date', fontsize=14, weight='bold', color='black')  # Change color to 'black' if needed

# Save the figure with high resolution and transparency
plt.savefig('SST_plot_asia.png', dpi=300, bbox_inches='tight', transparent=True)

# Display the plot
plt.show()
//////////////////////////////////////////////////Global Forest Fires///////////////////////////////////////////////////////////////////

#install required libraries
# pip install numpy xarray matplotlib cartopy seaborn netCDF4
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors

# Instructions:
# 1. Download the updated VIIRS data from here: https://firms.modaps.eosdis.nasa.gov/active_fire/
# 2. Save the downloaded file to your local directory.
# 3. Update the path in the 'pd.read_csv' function to the location of your downloaded file in .txt format.

# Load the data
fires = pd.read_csv("/file_path/VIIRSNDE_global2024312.v1.0.txt")

# Define the grid
coverage = [-180.0, -90.0, 180.0, 90.0]
grid_size = 1.0
num_points_x = int((coverage[2] - coverage[0]) / grid_size)
num_points_y = int((coverage[3] - coverage[1]) / grid_size)

# Create the meshgrid
nx = complex(0, num_points_x)
ny = complex(0, num_points_y)
Xnew, Ynew = np.mgrid[coverage[0]:coverage[2]:nx, coverage[1]:coverage[3]:ny]

# Populate the fire count array
fire_count = np.zeros([num_points_x, num_points_y])
for i, lon in enumerate(fires['Lon']):
    lat = fires['Lat'][i]
    adjlat = (lat + 90) / grid_size
    adjlon = (lon + 180) / grid_size
    latbin = int(adjlat)
    lonbin = int(adjlon)
    fire_count[lonbin, latbin] += 1

# Replace zeros with NaN
fire_count[fire_count == 0] = np.nan

# Define the color map
cmap = plt.cm.get_cmap("hot")
norm = mcolors.Normalize(vmin=0, vmax=40)

# Plot the data
fig, ax = plt.subplots(figsize=[15, 15], subplot_kw={'projection': ccrs.Orthographic(central_longitude=90.0, central_latitude=0.0)})

# Add a beautiful basemap
ax.add_feature(cfeature.LAND, edgecolor='black')
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAKES, alpha=0.5)
ax.add_feature(cfeature.RIVERS)

# Plot the scatter data
sc = ax.scatter(fires['Lon'], fires['Lat'], s=1, transform=ccrs.PlateCarree(), color='blue', alpha=0.5, label='Fire Events')

# Plot the pcolormesh data
pcm = ax.pcolormesh(Xnew, Ynew, fire_count, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), alpha=0.6)

# Add color bar
cbar = plt.colorbar(pcm, ax=ax, orientation='horizontal', pad=0.05)
cbar.set_label('Fire Counts')

# Set global and coastlines
ax.set_global()
ax.coastlines()

# Add title
plt.title('Global Fire Events with Binned Counts (1-degree grid)')

# Save the figure with high resolution and transparency
plt.savefig('global_fire_events.png', dpi=300, bbox_inches='tight', transparent=True)

# Display the plot
plt.show()
