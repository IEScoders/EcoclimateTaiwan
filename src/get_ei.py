#!/usr/bin/env python3

import numpy as np
import pandas as pd
import os, sys

def get_mi(prec,sm):
    shp = prec.shape
    mi = np.zeros(shp,dtype=np.float)

    for i in range(shp[0]):
        m = prec[i]
        if m < sm[0]:
            mi_index = 0
        elif m < sm[1]:
            mi_index = (m - sm[0])/(sm[1] - sm[0])
        elif m < sm[2]:
            mi_index = 1.
        elif m < sm[3]:
            mi_index = (m - sm[3])/(sm[2] - sm[3])
        else:
            mi_index = 0.
        mi[i] = mi_index

    return mi

def get_ti(td,tn,dv):
    shp = td.shape
    ti = np.zeros(shp,dtype=np.float)

    for i in range(shp[0]):
        hi_temp = td[i]
        low_temp = tn[i]
        dt = hi_temp - low_temp
        if dt > 0:
            iq = min( 1, 12 * max(0, hi_temp - dv[0]) ** 2 / dt / ((dv[1] - dv[0]) * 24))
        else:
            iq = np.nan
        ih = min( 1 - ( max(0, hi_temp - dv[2]) / (dv[3] - dv[2])) , 1)
        ih = max(0, ih)
        ti[i] = iq * ih
    return ti

def get_ei(path_to_rep,species,date):

  path_to_data = path_to_rep + 'analysis/'

  data_temp_max = pd.read_csv(path_to_data+'tmax.csv',encoding="UTF-8")
  data_temp_min = pd.read_csv(path_to_data+'tmin.csv',encoding="UTF-8")
  data_precp = pd.read_csv(path_to_data+'precp.csv',encoding="UTF-8")
  data_species = pd.read_csv(path_to_rep+'species.csv',encoding="UTF-8")

#  station_id = data_temp_max['id']
  temp_max = data_temp_max[['name',date]]
  temp_min = data_temp_min[['name',date]]
  precp = data_precp[['name',date]]

  ind = (data_species['scientific_name'] == species)
  sm, dv = np.zeros(4), np.zeros(4)
  for i in range(4):
    key = 'sm_' + '%1.1d' % i
    sm[i] =  data_species[key][ind]
    key = 'dv_' + '%1.1d' % i
    dv[i] =  data_species[key][ind]

  raw_data = temp_max.merge(temp_min,left_on='name',right_on='name')
  raw_data = raw_data.merge(precp,left_on='name',right_on='name')

  name = np.array(raw_data['name'])
  tmax = np.array(raw_data[date+'_x'])
  tmin = np.array(raw_data[date+'_y'])
  precp = np.array(raw_data[date])
  ind = (tmax>=-100) * (tmin>=-100) * (precp>=-100)

  name = name[ind]
  tmax = tmax[ind]
  tmin = tmin[ind]
  precp = precp[ind]

  ei = get_ti(tmax,tmin,dv) * get_mi(precp,sm)

  return name, ei

def main():

    date = '2017-12'
    path_to_rep = sys.argv[1]
    species = 'solenopsis_invicta'

    ei = {}
    for i in range(12):
      date = '2017-%2.2d'%(i+1)
      name, ei[date] = get_ei(path_to_rep,species,date)

      ind = (ei[date] < 0.)
      if (ei[date][ind].shape[0] > 0):
        print (mo, name[ind],ei[date][ind])


if __name__ == '__main__':

    main()



