#!/usr/bin/env python3

import numpy as np
import pandas as pd
import os, sys
import get_ei


def main():

    date = '2017-12'
    path_to_rep = '../'
    species = 'solenopsis_invicta'

    name, ei = get_ei.get_ei(path_to_rep,species,date)

    ei_dict = {}
    for n, e in zip(name, ei):
      ei_dict[n] = e

    station_file = 'CWB_Stations_171226.csv'
    df=pd.read_csv(path_to_rep + station_file,delimiter=',')

    station_id = df['id']
    station_lon = df['long']
    station_lat = df['lat']

    ei, lon, lat = [], [], []
    for i, j ,k in zip(station_id,station_lon,station_lat):
      try:
        ei.append(ei_dict[i])
        lon.append(j)
        lat.append(k)
      except:
        pass

    for i, e in enumerate(ei):
      print (lon[i], lat[i], e)


if __name__ == '__main__':

    main()



