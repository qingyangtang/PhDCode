import numpy as numpy
from astropy.io import fits
import matplotlib.pyplot as plt
import astropy.coordinates as coord
import astropy.units as u
import os
from astropy.table import Table, Column
import scipy.constants as const
from scipy.optimize import curve_fit
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
