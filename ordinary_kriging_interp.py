import numpy as np
import pandas as pd
import glob
import sys
from pykrige.ok import OrdinaryKriging 
from pykrige.kriging_tools import write_asc_grid
import pykrige.kriging_tools as kt
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Path, PathPatch
from matplotlib import rcParams
rcParams['font.size'] = 16
rcParams['font.family'] = 'times new roman'

import pylab as plb
import matplotlib
#Define a function to plot the interpolated map of the data on the map of Taiwan


def plot_interpolated_map(datafile='CWB_Stations_171226.csv',drop_index=[],data="R_FACTOR",label='R Factor',grid_space=0.1,counties_plot=True,units=r'MJ mm $ha^{-1}$ $h^{-1}$ $yr^{-1}$'):
#def plot_interpolated_map(datafile='R_FACTOR_WITH_ST_ALTITUDE_GTR_5.txt',drop_index=[],data="R_FACTOR_DENSITY",label='R Factor Density',grid_space=0.01,counties_plot=True,units=r'MJ $ha^{-1}$ $h^{-1}$ $yr^{-1}$'):
#def plot_interpolated_map(datafile='R_FACTOR_WITH_ST_ALTITUDE_GTR_20.txt',drop_index=[],data="mn_anual_precip",label='Mean Anual Precip',grid_space=0.01,counties_plot=True,units='mm'):
    '''
    Plot the interpolated map of the 'data' column of the 'datafile'.
    drop_index drops the given list of indices from the data set.
    'label' labels the colorbar of the map.
    'grid_space' decides the resolution of the interpolation.
    'counties_plot' True tells the program to plot counties on the map else enter False
    'units' plot the unit of the data.
    '''
    print("Reading Data")
    #Read the data in the different columns separated by space
    df=pd.read_csv(datafile,delimiter=',')

    e = np.random.rand(len(df['id']))
    df=df.assign(EI=pd.Series(e).values)
    # noEI = df[df['EI']<=0]
    # drop_index = np.array(noEI.index())
    df.drop(df.index[drop_index],inplace=True)

    print(df)
    sys.exit()

    # del df['St.Name'],df['S.N0.'], df['ALTITUDE'], df['num_of_yrs'], df['no_typ'],df['Acc_Precip']
    df_new=df[(df['St.Lon']>119.8) & (df['St.Lat']<25.7)] #exclude all the points outside the inland of taiwan
    lons=np.array(df_new['St.Lon']) #define an array containing the Longitude information of the data
    lats=np.array(df_new['St.Lat']) #define an array containing the latitude insformation of the data
    data=np.array(df_new[data]) #Save the data from selected column (such as R_FACTOR or mn_precip) in the data variable
    # data1=np.array(df_new['R_FACTOR'])
    # data2=np.array(df_new['mn_precip'])

    # grid_lon = np.arange(lons.min()-0.05, lons.max()+0.1, grid_space) #make the grid for interpolating the data.
    # #The minimum and maximum of the longitude and latitude is chosen based on the data. 
    # grid_lat = np.arange(lats.min(), lats.max()+0.1, grid_space)
    lonmin=119.9
    lonmax=122.15
    latmin=21.8
    latmax=25.4


    grid_lon = np.arange(lonmin, lonmax, grid_space) #make the grid for interpolating the data.
    #The minimum and maximum of the longitude and latitude is chosen based on the data. 
    grid_lat = np.arange(latmin, latmax, grid_space)

    print("Interpolating Data")
    # Create ordinary kriging object
    # change the variogram model, nlags for obtaining different level of fits
    OK = OrdinaryKriging(lons, lats, data, variogram_model='gaussian', verbose=True, enable_plotting=False,nlags=20)
    #print(grid_lon)
    #print(data)
    #print(z1)
 

    # Obtain the interpolation on the grid points using the kriging object
    z1, ss1 = OK.execute('grid', grid_lon, grid_lat)
    write_asc_grid(grid_lon,grid_lat,z1,filename='ordinary_kriging_output_R_FACTOR_GTR_5.asc')
    #write_asc_grid(grid_lon,grid_lat,z1,filename='ordinary_kriging_output_MN_ANUAL_PRECIP_GTR_20.asc')
    #write_asc_grid(grid_lon,grid_lat,z1,filename='ordinary_kriging_R_FACTOR_DENSITY_GTR_20.asc')

 
    # Make a meshgrid for plotting the data on the map
    xintrp, yintrp = np.meshgrid(grid_lon, grid_lat) 
    print('Plotting Data')
    fig, ax = plt.subplots(figsize=(6,8)) #make the figure frame of given size
    colors = [(0.8, 0, 0), (0, 0.8, 0), (0, 0, 0.8)]  # R -> G -> B #select the level of red, blue and green intensity
    cmap_name = 'my_list'
    ncols=33 #select the number of colors for plotting. The whole spectrum of red,blue, green will be divided into ncols bars.
    ## forR factor ncols = 17; for mn_precip ncols = 13
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=ncols) # red, blue and green will be joined to make a single colormap
    #Make a basemap for plotting the map
    m = Basemap(llcrnrlon=lonmin,llcrnrlat=latmin,urcrnrlon=lonmax,urcrnrlat=latmax,
                projection='merc',
                resolution ='h',area_thresh=1000.,ax=ax)
    m.drawcoastlines() #draw coastlines on the map
    x,y=m(xintrp, yintrp) # convert the coordinates into the map scales
    ln,lt=m(lons,lats)
    

  
   # cs = ax.contourf(x,y,z1,np.linspace(300,900,ncols),extend='both',cmap='jet')# mn precip
    cs = ax.contourf(x,y,z1,np.linspace(2000,10000,ncols),extend='both',cmap='jet')# R_factor
   #cs = ax.contourf(x,y,z1,np.linspace(5,9,ncols),extend='both',cmap='jet')# R_factor DENSITY
    # The minimum and maximum can be changed by changing the vmin and vmax!

    cbar=m.colorbar(cs,location='right',pad="7%") #plot the colorbar on the map
    plt.title(label+'\n'+units,fontsize = 14, family ='times new roman',fontweight='bold')
    plt.xlabel('\n\nLongitude', fontsize=16)
    plt.ylabel('Latitude\n\n\n', fontsize=16)

    #cbar.set_label(units,fontsize=16,family='times new roman') # put labels of the colorbar
    ax.text(0.2, 0.95,'(a)', ha='center', va='center', transform=ax.transAxes,fontsize=16)
    
    ## Plotting counties
    if counties_plot:
        counties=glob.glob("counties_polygon/polygon_*.txt") #read all the colunties coastline data separately
        for files in counties:
            df=pd.read_csv(files,delimiter="\s+",names=["longi","lati"],header=None)
            if df.shape[0]>3000: #plot only those counties with size greater than 3000
                lo,la=m(df['longi'].values,df['lati'].values)
                m.plot(lo,la,'k',lw=0.5) #plot the counties on the basemap with black lines and with linewidth 0.5

    # draw parallels.
    #parallels = np.arange(latmin,latmax,0.5)
    parallels = np.arange(21.5,26.0,0.5)
    m.drawparallels(parallels,labels=[1,0,0,0],fontsize=16, linewidth=0.0) #Draw the latitude labels on the map
    # draw meridians
    meridians = np.arange(119.5,122.5,0.5)
    m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=16, linewidth=0.0)
    ##getting the limits of the map
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    map_edges = np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])

    ##getting all polygons used to draw the coastlines of the map
    polys = [p.boundary for p in m.landpolygons]

    ##combining with map edges
    polys = [map_edges]+polys[:]

    ##creating a PathPatch
    codes = [
        [Path.MOVETO] + [Path.LINETO for p in p[1:]]
        for p in polys
    ]
    polys_lin = [v for p in polys for v in p]
    codes_lin = [c for cs in codes for c in cs]
    path = Path(polys_lin, codes_lin)
    patch = PathPatch(path,facecolor='white', lw=0)

    ##masking the data outside the inland of taiwan
    ax.add_patch(patch)
    fignm_array=label.split()
    fignm="_".join(fignm_array)
    fig_prefix=datafile.split(".")[0]
    print(fig_prefix)
    print("Output figures is saved as Figures/{}-{}.png".format(fig_prefix,fignm))
    plt.savefig('Figures/{}-{}.png'.format(fig_prefix,fignm),dpi=100,bbox_inches='tight')


if __name__=="__main__":
    
    plot_interpolated_map(datafile='CWB_Stations_171226.csv',drop_index=[],data='R_FACTOR',label='R FACTOR',grid_space=0.1,counties_plot=True,units=r'MJ $ha^{-1}$$h^{-1}$$yr^{-1}$')
    #plot_interpolated_map(datafile='R_FACTOR_WITH_ST_ALTITUDE_GTR_20.txt',drop_index=[],data='mn_anual_precip',label='Mean_Precip',grid_space=0.01,counties_plot=True,units='mm')
    #plot_interpolated_map(datafile='R_FACTOR_WITH_ST_ALTITUDE_GTR_5.txt',drop_index=[],data='R_FACTOR_DENSITY',label='R FACTOR Density',grid_space=0.01,counties_plot=True,units=r'MJ $ha^{-1}$$h^{-1}$$yr^{-1}$')
    
    