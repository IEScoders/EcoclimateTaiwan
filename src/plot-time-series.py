
# coding: utf-8

# In[14]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys,os
# -*- coding: utf-8 -*-


# In[15]:


def df_value(filename,station):
    df = pd.read_csv(filename,index_col=0)
    dft = df[df.name == station]
    dft = dft.T.drop('name')
    dft.columns = ['EI']
    dft['id'] = [station] * len(dft)
    dft['Time'] = dft.index
    new = dft.Time.str.split('-', n = 0,expand = True)
    dft['Year'] = new[0]
    dft['Month'] = new[1].apply(int)
    return dft
def df_month(df,station):
    df = df[df.id == station]
    df = df.sort_values('Month')
    dfft = [df[df.Month == m] for m in range(1,13)]
    mean = [DF.EI.mean() for DF in dfft]
    std = [DF.EI.std() for DF in dfft]
    return mean, std


# In[22]:


def plot_ts(species,station):
    print("Now making figure for {} {}".format(species,station))
    dfTemp = df_value("../analysis/temperature.csv",station)
    Tmean, Tstd = df_month(dfTemp,station)
    dfTmin = df_value("../analysis/tmin.csv",station)
    Tminmean, Tminstd = df_month(dfTmin,station)
    dfTmax = df_value("../analysis/tmax.csv",station)
    Tmaxmean, Tmaxstd = df_month(dfTmax,station)

    dfPrep = df_value("../analysis/precp.csv",station)
    Pmean, Pstd = df_month(dfPrep,station)
    EImean, EIstd = df_month(pd.read_csv("../output_csv/combine/{}.csv".format(species)),station)
    mm = [1,2,3,4,5,6,7,8,9,10,11,12]
    plt.figure(figsize=[8,8])
    a1 = plt.subplot(311)
    plt.errorbar(mm,EImean,EIstd,fmt="-go")
    plt.xticks(mm)
    plt.ylim([-0.1,1.1])
    plt.ylabel("EI value")
    a2 = plt.subplot(312,sharex=a1)
    plt.errorbar(mm,Tmean,Tstd,fmt="-ro",label="Average Temp")
    plt.errorbar(mm,Tminmean,Tminstd,fmt="--ro", label = "Min Temp")
    plt.errorbar(mm,Tmaxmean,Tmaxstd,fmt="-.ro", label = "Max Temp")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    a3 = plt.subplot(313,sharex=a1)
    plt.errorbar(mm,Pmean,Pstd,fmt="-bo")
    plt.ylabel("Precipitation (mm)")

    path ="../result/{}".format(species)
    if not os.path.exists(path):
        os.mkdir(path)
    plt.xlabel("Month")
    plt.savefig("../result/{}/{}.jpg".format(species,station),dpi=150,bbox_inches="tight")


# In[28]:


df = pd.read_csv("../species.csv")
dfs = pd.read_csv("../output_csv/combine/achatina_fulica.csv")
spl = list(set(df.scientific_name))
stl = list(set(dfs.id))
for sp in spl[-3:]:
    for st in stl:
        plot_ts(sp,st)

