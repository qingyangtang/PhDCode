import numpy as numpy
from astropy.io import fits
import matplotlib.pyplot as plt
import astropy.coordinates as coord
import astropy.units as u
import os
from astropy.table import Table, Column
import scipy.constants as const
from scipy.interpolate import interp1d
from scipy import integrate
from scipy.optimize import curve_fit
from scipy.signal import argrelmin

c_c = const.c
k_c = const.k
h_c = const.h

from datetime import datetime

def mod_BB_curve_with_z(v, A, T, z):
    #from Viero et al 
    beta = 2.
    v0 = v*(1+z)
    Bv = (2*h_c*v0**3/c_c**2)*(1/(np.exp(h_c*v0/(k_c*T))-1))
    Sv = A*v0**beta*Bv
    return Sv

def mad(arr):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variabililty of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation
    """
    arr = np.ma.array(arr).compressed() # should be faster to not use masked arrays.
    med = np.median(arr)
    return np.median(np.abs(arr - med))

def bootstrap(n_real, sample_len, galaxy_arr):
    #returns the MAD error using the bootstrap medians
    bs_med = np.zeros((n_real))
    for j in range(n_real):
        indices = np.random.choice(sample_len, sample_len)
        bs_med[j] = np.median(galaxy_arr[indices])
    return mad(bs_med)

def noise_PSD(zp_map, reso_arcmin, dngrid):
    reso = (reso_arcmin/60)*np.pi/180.
    dk = 1./dngrid/reso
    factor = np.sqrt(dngrid**2*dk**2)
    noise_psd = np.sqrt(np.abs(np.fft.ifft2(zp_map)/factor))

def zero_pad_map(masked_map):
    #converting the nans in the masked map to 0s and zero padding map to have square shape
    zeros_masked_map = np.nan_to_num(masked_map)
    if zeros_masked_map.shape[0] != zeros_masked_map.shape[1]:
        zp_mask_map = np.zeros((max(zeros_masked_map.shape),max(zeros_masked_map.shape)))
        if zp_mask_map.shape[0]>zeros_masked_map.shape[0]:
            edge_padding = (zp_mask_map.shape[0]-zeros_masked_map.shape[0])/2
            zp_mask_map[edge_padding:-edge_padding, :] = zeros_masked_map
        elif zp_mask_map.shape[1]>zeros_masked_map.shape[1]: 
            edge_padding = (zp_mask_map.shape[1]-zeros_masked_map.shape[1])/2
            zp_mask_map[:, edge_padding:-edge_padding] = zeros_masked_map
    else:
        zp_mask_map = zeros_masked_map
    pixelmask = np.zeros((max(zeros_masked_map.shape),max(zeros_masked_map.shape)))
    pixelmask[edge_padding:-edge_padding, :] = mask
    zp_mask = np.zeros((zp_mask_map.shape))
    zp_mask[edge_padding:-edge_padding, :] = mask

    return zp_mask_map, zp_mask

def match_src_to_catalog(src_ra, src_dec, cat_ra, cat_dec):
    c = SkyCoord(ra=src_ra*u.degree, dec=src_dec*u.degree)  
    catalog = SkyCoord(ra=cat_ra*u.degree, dec=cat_dec*u.degree)  
    idx, d2d, d3d = c.match_to_catalog_sky(catalog)
    return idx, d2d

### Below is stuff for match filter algorithm

def dbdt(f,T):#      #t in K, f in Hz; returns SI units
    x = h_c*f/(k_c*T)
    return (2*h_c*f**3/c_c**2) / (np.exp(x)-1.0)**2 * (x/T) * np.exp(x)

def dcmbrdt(f):
    return dbdt(f, 2.726)

def calc_nu_eff1(nu, transmission, source_spectrum, nonusq=True):
    #expects input nu in GHz
    nu_interp = np.arange(1e4)/1e4*(np.max(nu)-np.min(nu)+20.) + np.min(nu) - 10.

    dbdt = dcmbrdt(nu*1.e9)

    if nonusq:
        bw_source = integrate.simps(transmission*source_spectrum,nu)
        bw_cmb = integrate.simps(transmission*dbdt,nu)
    else:
        bw_source = integrate.simps(transmission*source_spectrum/nu**2,nu)
        bw_cmb = integrate.simps(transmission*dbdt/nu**2,nu)
    f = interp1d(nu, source_spectrum,fill_value="extrapolate")
    ss_interp = f(nu_interp)
    dbdt_interp = dcmbrdt(nu_interp*1e9)
    whlow = np.where(nu_interp <= 40.)[0]
    if np.any(nu_interp<= 40.) == True:
        dbdt_interp[whlow] = 1e12
    ratio1 = bw_source/bw_cmb
    ratio2 = ss_interp/dbdt_interp
    dss = np.abs(ratio1-ratio2)
    minima_ind = argrelmin(np.log(dss),order=10)[0]
    if len(minima_ind) > 1:
        band_start = np.min(nu[transmission>0.2])
        band_end = np.max(nu[transmission>0.2])
        band_cent = (band_start+band_end)/2.
        true_min = np.where(np.abs(band_cent-nu_interp[minima_ind]) == np.min(np.abs(band_cent-nu_interp[minima_ind])))[0][0]
        nu_eff = nu_interp[minima_ind[true_min]]
    else:
        nu_eff = nu_interp[minima_ind]
    return nu_eff

