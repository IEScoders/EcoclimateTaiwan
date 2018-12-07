import numpy as np
import pandas as pd
import folium
import branca
from folium import plugins
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import geojsoncontour
import scipy as sp
import scipy.ndimage
 
df = pd.read_csv("../output_csv/solenopsis_invicta/2017-01_solenopsis_invicta.csv",index_col=0)
def color_producer(el):
    if el < 0.2:
        return '#2b83ba'
    elif 0.2 <= el < 0.4:
        return '#abdda4'
    elif 0.4 <= el < 0.6:
        return '#ffffbf'
    elif 0.6 <= el < 0.8:
        return '#fdae61'

    else:
        return '#d7191c'    
# Setup
temp_mean = 12
temp_std  = 2
debug     = False
 
# Setup colormap
colors = ['#d7191c',  '#fdae61',  '#ffffbf',  '#abdda4',  '#2b83ba']
colors.reverse()
#vmin   = temp_mean - 2 * temp_std
#vmax   = temp_mean + 2 * temp_std
vmin = 0
vmax = 1
levels = len(colors)
cm     = branca.colormap.LinearColormap(colors, vmin=vmin, vmax=vmax).to_step(levels)
 
# Create a dataframe with fake data
# df = pd.DataFrame({
#     'longitude':   np.random.normal(11.84,     0.15,     1000),
#     'latitude':    np.random.normal(55.55,     0.15,     1000),
#     'temperature': np.random.normal(temp_mean, temp_std, 1000)})
 
# The original data
x_orig = np.asarray(df.long.tolist())
y_orig = np.asarray(df.lat.tolist())
z_orig = np.asarray(df.EI.tolist())
 
# Make a grid
x_arr          = np.linspace(np.min(x_orig), np.max(x_orig), 500)
y_arr          = np.linspace(np.min(y_orig), np.max(y_orig), 500)
x_mesh, y_mesh = np.meshgrid(x_arr, y_arr)
 
# Grid the values
z_mesh = griddata((x_orig, y_orig), z_orig, (x_mesh, y_mesh), method='linear')
 
# Gaussian filter the grid to make it smoother
sigma = [5, 5]
z_mesh = sp.ndimage.filters.gaussian_filter(z_mesh, sigma, mode='constant')
 
# Create the contour
contourf = plt.contourf(x_mesh, y_mesh, z_mesh, levels, alpha=0.5, colors=colors, linestyles='None', vmin=vmin, vmax=vmax)
 
# Convert matplotlib contourf to geojson
geojson = geojsoncontour.contourf_to_geojson(
    contourf=contourf,
    min_angle_deg=3.0,
    ndigits=5,
    stroke_width=1,
    fill_opacity=0.5)
 
# Set up the folium plot
geomap = folium.Map([df.lat.mean(), df.long.mean()], zoom_start=10, tiles="cartodbpositron")
 
# Plot the contour plot on folium
folium.GeoJson(
    geojson,
    style_function=lambda x: {
        'color':     x['properties']['stroke'],
        'weight':    x['properties']['stroke-width'],
        'fillColor': x['properties']['fill'],
        'opacity':   0.6,
    }).add_to(geomap)
 
# Add the colormap to the folium map
cm.caption = 'EI value'
geomap.add_child(cm)

html = """
Station name:
<br><a href="https://www.google.com/search?q=%%22%s%%22" target="_blank">%s</a><br>
EI value: %.2f 
"""

fgv = folium.FeatureGroup(name='Stations')
for lt, ln, ei, nm in zip(df.lat,df.long, df.EI, df.id):
    iframe = folium.IFrame(html=html % (nm,nm,ei), width = 200, height = 100)
    fgv.add_child(folium.CircleMarker(location = [lt,ln], 
                                popup = folium.Popup(iframe),
                                radius = 4,
                                color = 'grey',
                                fill_color=color_producer(ei),
                                fill_opacity = 0.7))
geomap.add_child(fgv)

# Fullscreen mode
plugins.Fullscreen(position='topright', force_separate_button=True).add_to(geomap)
geomap.add_child(folium.LayerControl()) 
# Plot the data
geomap.save('../html/test.html')
