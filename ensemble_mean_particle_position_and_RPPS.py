""" Copyright 2020 Chris Wilson, National Oceanography Centre, United Kingdom (cwi@noc.ac.uk)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 """

# This code is used to calculate ensemble-mean particle position and Root-mean-squared Particle Pair Separation (RPPS)=sqrt(cloud dispersion)
# from particle positions given in longitude and latitude (spherical coordinates) by transforming to Cartesian coordinates to use a uniform weight for the 
# distance metric within the calculation.
 
import sys, os

import numpy as np
import numpy.ma as ma
import os.path
import netCDF4
from netCDF4 import Dataset
import h5py
from pathlib import Path
import xarray as xr
import numpy as np

#Define some parameters

ntoffset=21 #number of initial time offsets (e.g. t0-10, t0-9, ..., t0, ..., t0+9, t0+10 equals 21 offsets)
numyrs=10 #number of individual years that particles are initialised in (e.g. 2005, 2006, ..., 2014 equals 10 years)
numdoyr=391 #number of days of year (period of particle data) in particle tracking 'run' (e.g. 365 + extra to include offsets, etc.)

## Define Cartesian distance (km) between two points on sphere with a given lon, lat

import math
def distance(origin, destination):
    """
    Calculate the Haversine distance.

    Parameters
    ----------
    origin : tuple of float
        (lat, long)
    destination : tuple of float
        (lat, long)

    Returns
    -------
    distance_in_km : float

    Examples
    --------
    >>> origin = (48.1372, 11.5756)  # Munich
    >>> destination = (52.5186, 13.4083)  # Berlin
    >>> round(distance(origin, destination), 1)
    504.2
    """
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371  # km (radius of Earth)

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) * math.sin(dlon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c

    return d

## Define transformation from lon, lat to Cartesian coordinates 

def lon_lat_to_cartesian(lon, lat, R = 1):
    """
    calculates x, y, z coordinates of a point on a sphere with
    radius R
    """
    lon_r = np.radians(lon)
    lat_r = np.radians(lat)

    x =  R * np.cos(lat_r) * np.cos(lon_r)
    y = R * np.cos(lat_r) * np.sin(lon_r)
    z = R * np.sin(lat_r)
    return x,y,z

## ... and the inverse

def cartesian_to_lon_lat(x, y, z, R = 1):
    """
    calculates lon, lat coordinates of a point on a sphere with
    radius R
    """
    lon =  np.degrees(np.arctan2(y,x))
    lat = np.degrees(np.pi/2-np.arctan2((x**2+y**2)**0.5,z))

    return lon,lat


# **Load or define the data*
# 4D arrays : (initial time offset) X (start year) X (day of year) X (particle position label in 25x25 patch used for initial condition)

xypatch=25*25
#pre-allocate
x=np.zeros((ntoffset,numyrs,numdoyr,xypatch))
y=np.zeros((ntoffset,numyrs,numdoyr,xypatch))
# x and y are the longitude and latitude in degrees [-180,180], [-90,90], respectively

# ** LOAD or DEFINE x, y here **



## Calculate the ensemble-mean position for each year, day and start time offset by averaging in Cartesian coordinates across the ensemble positions
## then transform back to lon and lat - This ensures that the correct spatial metric is used and that the points are given uniform distance weighting.

#Initialise
ensmeanposlon=np.zeros((ntoffset,numyrs,numdoyr),dtype=np.float32)
ensmeanposlat=np.zeros((ntoffset,numyrs,numdoyr),dtype=np.float32) 
xs=np.zeros((xypatch,1),dtype=np.float32)
ys=np.zeros((xypatch,1),dtype=np.float32)
zs=np.zeros((xypatch,1),dtype=np.float32)


for nyear in np.arange(numyrs):

    for toffset in np.arange(ntoffset):
        print(nyear,toffset)
        for nday in np.arange(numdoyr):
            for npos in np.arange(xypatch):
                #convert to Cartesian
                xs[npos], ys[npos], zs[npos] = lon_lat_to_cartesian(x[toffset,nyear,nday,npos], y[toffset,nyear,nday,npos])
                
            #calculate the ensemble mean
            ensmeanposx=np.nanmean(xs[:])
            ensmeanposy=np.nanmean(ys[:])
            ensmeanposz=np.nanmean(zs[:])
            #convert back to lat, lon
            ensmeanposlon[toffset,nyear,nday],ensmeanposlat[toffset,nyear,nday]=cartesian_to_lon_lat(ensmeanposx,ensmeanposy,ensmeanposz)




### Calculate the cloud dispersion and Root-mean-squared Particle Pair Separation (RPPS)


# ## Use cloud dispersion, defined in eqn. (3) of LaCasce (2008)
# 
# 
# $$D_x(t)=\frac{1}{2N(N-1)}\sum_{i\ne j}[x_i(t)-x_j(t)]^2$$
# or, extending to 2D, where ${\bf{x}}=(x,y)$,
# $$D_{\bf{x}}(t)=\frac{1}{2N(N-1)}\sum_{i\ne j}|{\bf{x_i}}(t)-{\bf{x_j}}(t)|^2$$


xypatch=25*25 #number of particles in patch used for initial condition 

dx2=np.zeros((ntoffset,numyrs,numdoyr,xypatch),dtype=np.float32) #Cartesian distance squared
dispersion=np.zeros((ntoffset,numyrs,numdoyr),dtype=np.float32) #cloud dispersion

#outer loops over initial time offset and year

for nyear in np.arange(numyrs):

    for toffset in np.arange(ntoffset):
        print(nyear,toffset)
        #inner loops over particle origin, day of year and particle destination (i.e. the other particle in the pair)
        
        for nday in np.arange(numdoyr):
            ncounter=0
            for norigin in np.arange(xypatch): #loop over position i            
                origin=(y[toffset,nyear,nday,norigin],x[toffset,nyear,nday,norigin]) #chosen point in posn i at 1200 hrs, each day
                for npos in np.arange(norigin+1,xypatch,1):  #loop over position j
                    
                        
                    destination=(y[toffset,nyear,nday,npos],x[toffset,nyear,nday,npos])
                        
                    disto=distance(origin, destination) 
                    ncounter=ncounter+1
                    #sum the squared displacement over all the particles other than the 'origin'
                    dx2[toffset,nyear,nday,norigin]=+disto**2

        #dispersion[toffset,nyear,:]=1/(2*N*(N-1))*np.nansum(dx2[toffset,nyear,:,:],axis=1) #cloud dispersion across xypatch    
        #However, one may loop as a triangle, rather than a rectangle and then don't double count and can lose factor of 2 in sum, or 1/2 in the mean:    
        dispersion[toffset,nyear,:]=1/(xypatch*(xypatch-1))*np.nansum(dx2[toffset,nyear,:,:],axis=1) #cloud dispersion across xypatch

        #RPPS = sqrt(dispersion)





