
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import folium
import branca
import matplotlib.pyplot as plt
#from scipy.interpolate import griddata
import geojsoncontour
import base64
from folium.features import DivIcon
from folium import plugins
import json, ast
#import scipy as sp
#import scipy.ndimage

from pykrige.ok import OrdinaryKriging
from pykrige.kriging_tools import write_asc_grid
import pykrige.kriging_tools as kt

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


# In[2]:


def tw_bound():
    df = pd.read_csv("../data/taiwan2.csv",header = None)
    return [[x,y] for x,y in zip(df[0],df[1])]


# In[45]:


def color_producer1(el):
    if el < 0.01:
        return "#FFFFFF"
    elif 0.01 <= el < 0.1:
        return "#FFFFFF"
    elif 0.11 <= el < 0.2:
        return "#AAAAFF"
    elif 0.21 <= el < 0.6:
        return "#6666FF"
    elif 0.6 <= el:
        return "#0000FF"

def color_producer2(el):
    if el < 0.01:
        return "#FFFFFF"
    elif 0.01 <= el < 0.1:
        return "#FFFFFF"
    elif 0.11 <= el < 0.2:
        return "#FFAAAA"
    elif 0.21 <= el < 0.6:
        return "#FF6666"
    else:
        return "#FF0000"
    
def add_layer(df,text,cc): 
    if cc == 1:
        colors = ["#FFFFFF","#CCCCFF", "#AAAAFF", "#6666FF","#0000FF"]
    elif cc == 2:
        colors = ["#FFFFFF","#FFCCCC", "#FFAAAA", "#FF6666","#FF0000"] 
    elif cc == 3:
        colors = ["#FFFFFF","#CCFFCC", "#AAFFAA", "#66FF66","#00FF00"] 
    polygon = Polygon(tw)
    lons = np.asarray(df.long.tolist())
    lats = np.asarray(df.lat.tolist())
    data = np.asarray(df.EI.tolist())
    grid_space = 0.05
    global grid_lon, grid_lat
    grid_lon = np.arange(lons.min()-0.1, lons.max()+0.1, grid_space) 
    grid_lat = np.arange(lats.min()-0.1, lats.max()+0.1, grid_space)
    OK = OrdinaryKriging(lons, lats, data, variogram_model='gaussian', verbose=False, enable_plotting=False,nlags=20)
    global z_mesh,x_mesh,y_mesh
    z_mesh, ss1 = OK.execute('grid', grid_lon, grid_lat)
    x_mesh, y_mesh = np.meshgrid(grid_lon, grid_lat)
    shp = x_mesh.shape
    mtw = np.zeros(shp,dtype=np.float)
    for i in range(shp[0]):
        for j in range(shp[1]):
            point = Point(x_mesh[i][j], y_mesh[i][j])
            if not polygon.contains(point):
                mtw[i][j] = 1        
    z_mesh = np.ma.masked_where(mtw == 1, z_mesh)    
    contourf = plt.contourf(x_mesh, y_mesh, z_mesh, levels=[-0.2,0.01,0.1,0.2,0.6,1.2], alpha=0.9, colors=colors, linestyles='None', vmin=-0.2, vmax=1.2)
    global geojson
    geojson = geojsoncontour.contourf_to_geojson(
        contourf=contourf,
        min_angle_deg=3.0,
        ndigits=5,
        stroke_width=1,
        fill_opacity=0.5)
#     geoj = folium.GeoJson(
#         geojson,
#         style_function=lambda x: {
#             'color':     x['properties']['stroke'],
#             'weight':    x['properties']['stroke-width'],
#             'fillColor': x['properties']['fill'],
#             'opacity':   0.6,
#         })
    geoj = 'none'
    return geoj,geojson



# In[46]:


def add_station(df):
    features = []
    for time in list(set(df.Time)):
        dft = df[df.Time == time]
        long,lat,EI = list(dft.long),list(dft.lat),list(dft.EI)
        for i in range(len(dft)):
            features.append(
                {
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': [
                        long[i],lat[i]
                    ],
                },
                'properties': {
                    'time': time,
                    'popup': "EI value: {:.2f}".format(float(EI[i])),
                    'icon': 'circle',
                    'iconstyle': {
                        'fillColor': color_producer2(EI[i]),
                        'stroke': 'false',
                        'fillOpacity': 0.5,
                        'radius': 4,
                        'color' : 'black'
                    },'style': {'weight': 0.5}
                }
            }
        )    
    return features    


# In[47]:


def add_dictime(geojson,time):
    t = []
    for i,ply in enumerate(ast.literal_eval(geojson)['features']):
        dic = {}
        dic['geometry'] = ply['geometry']
        dic['properties'] = {"times" : ["{}".format(time)], "style": 
                             {"color":ply['properties']['fill'],
                            "opacity" :0,"weight" : 0.5, "fillColor": ply['properties']['fill']
                             }}
        dic['type'] = 'Feature'
        t.append(dic)
    return t

def add_dictime_ref(geojson,time):
    t = []
    for i,ply in enumerate(ast.literal_eval(geojson)['features']):
        dic = {}
        dic['geometry'] = ply['geometry']
        dic['properties'] = {"times" : ["{}".format(time)], "style": 
                             {"color":ply['properties']['fill'],
                            "opacity" :0.5,"weight" : 2.0, "fillOpacity": 0,'dashArray':4
                             }}
        dic['type'] = 'Feature'
        t.append(dic)
    return t


# In[48]:


def get_list(df,cc,ref=False):
    tl = []
    
    for time in list(set(df.Time)):
        dft = df[df.Time == time]
        _,geojson = add_layer(dft,'',cc)
        if ref == False:
            t = add_dictime(geojson,time)
        else:
            t = add_dictime_ref(geojson,time)
        tl += t
    return tl


# In[ ]:


def makemap(lst):

    geomap = folium.Map([23.75, 121], zoom_start=8, tiles="cartodbpositron")
    folium.TileLayer('stamenterrain').add_to(geomap)
    folium.TileLayer('openstreetmap').add_to(geomap)

    colors = ["#FFFFFF", "#FFAAAA", "#FF6666","#FF6666", "#FF6666","#FF6666",
              "#FF0000","#FF0000","#FF0000","#FF0000"] 
    cm     = branca.colormap.LinearColormap(colors, vmin=0, vmax=1).to_step(len(colors))
    cm.caption = 'EI value'
    geomap.add_child(cm)
    sp,*pred = lst
    dfsp = pd.read_csv("../output_csv/combine/{}.csv".format(sp))
    tl1 = get_list(dfsp,2)
    if len(pred) == 1:
        pred1 = pd.read_csv("../output_csv/combine/{}.csv".format(pred[0]))
        tl2 =  get_list(pred1,1,ref=True)
        tl3 = []
    elif len(pred) == 2:
        pred1 = pd.read_csv("../output_csv/combine/{}.csv".format(pred[0]))
        pred2 = pd.read_csv("../output_csv/combine/{}.csv".format(pred[1]))
        tl2 =  get_list(pred1,1,ref=True)
        tl3 =  get_list(pred2,3,ref=True)
    else:
        tl2 = tl3 = []
    features = add_station(dfsp)
    FC = tl1 + tl2 + tl3 + features
    plugins.TimestampedGeoJson(
        {
            'type': 'Feature',
            'features': FC

        },
        period='P1M',
        duration='P15D',
        auto_play=False,
        loop=False,
        loop_button=True,
        date_options='YYYY/MM',
    ).add_to(geomap)


    plugins.Fullscreen(position='topright', force_separate_button=True).add_to(geomap)
    plugins.MiniMap().add_to(geomap)
    plugins.MeasureControl(primary_length_unit='kilometers', 
                         secondary_length_unit='meters', 
                         primary_area_unit='hectares', 
                         secondary_area_unit='sqmeters').add_to(geomap)
    formatter = "function(num) {return L.Util.formatNum(num, 3) + ' º ';};"

    plugins.MousePosition(
        position='bottomleft',
        separator=' | ',
        empty_string='NaN',
        lng_first=True,
        num_digits=20,
        prefix='Coordinates:',
        lat_formatter=formatter,
        lng_formatter=formatter
    ).add_to(geomap)
    geomap.add_child(folium.LayerControl(position="topleft"))
#    return geomap
    geomap.save('../html/{}.html'.format(sp))


# In[50]:


if __name__ == '__main__':
    tw = tw_bound()
    species_df = pd.read_csv("../species.csv")
    species = list(species_df.scientific_name)
    listsp = []
    for sp in species[2:]:
        prel = species_df.predators[species_df.scientific_name == sp].to_string().replace(';','').split()
        if prel[1] != 'NaN':
            pre = prel[1:]
        else:
            pre = []
        listsp.append([sp]+pre)
    for lst in listsp:
        geomap = makemap(lst)

