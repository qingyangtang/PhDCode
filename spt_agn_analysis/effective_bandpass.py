import numpy as numpy
from astropy.io import fits
import matplotlib.pyplot as plt
from sptpol_software.util import idl
import astropy.coordinates as coord
import astropy.units as u
from astropy.coordinates import SkyCoord
from sptpol_software.observation import sky
from itertools import product
from sptpol_software.util import files
from sptpol_software.util import fits as Fits
import sptpol_software.analysis.maps_amy as maps
import pickle
from astropy.io import fits
import sptpol_software.scratch.tnatoli.transient.calc_optfilt_oneband_2d_amy as calc
import os
from astropy.table import Table, Column

import scipy.constants as const
from scipy.interpolate import interp1d
from scipy import integrate

c = const.c
k = const.k
h = const.h

def dbdt(f,T):#      #t in K, f in Hz; returns SI units
    x = h*f/(k*T)
    return (2*h*f**3/c**2) / (np.exp(x)-1.0)**2 * (x/T) * np.exp(x)

def dcmbrdt(f):
    return, dbdt(f, 2.726)

 # calculate effective frequency (as defined in Reichardt et al. 2012)
 # given transmission vs. frequency and source spectrum. assumes
 # beam-filling source. will calculate the frequency that gives you the
 # band-weighted average of the conversion factor from T_CMB to MJy/sr.

 #nu is in GHz NOT Hz!!
def calc_nu_eff(nu, transmission, source_spectrum, nonusq=True):
	nu_interp = np.arange(1e4)/1e4*(np.max(nu)-np.min(nu)+20.) + np.min(nu) - 10.
	dbdt = dcmbrdt(nu*1.e9)
	if nonusq:
	    bw_source = integrate.simps(transmission*source_spectrum,nu)
	    bw_cmb = integrate.simps(transmission*dbdt,nu)
	else: 
	    bw_source = integrate.simps(transmission*source_spectrum/nu**2,nu)
	    bw_cmb = integrate.simps(transmission*dbdt/nu**2,nu)

	f = interp1d(nu, source_spectrum)
	ss_interp = f(nu_interp)
	dbdt_interp = dcmbrdt(nu_interp*1e9)
	whlow = np.where(nu_interp <= 40.)[0]

	if np.any(nu_interp<= 40.) == True:
		dbdt_interp[whlow] = 1e12
	ratio1 = bw_source/bw_cmb
	ratio2 = ss_interp/dbdt_interp
	dss = np.abs(ratio1-ratio2)
	mindss = np.min(dss,whmin)
	nu_eff = nu_interp[whmin]
	return nu_eff