import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import astropy.coordinates as coord
import astropy.units as u
import os
from astropy.table import Table, Column
from astropy.coordinates import SkyCoord
import scipy.constants as const
from scipy.interpolate import interp1d
from scipy import integrate
from scipy.optimize import curve_fit
from scipy.signal import argrelmin
from sptpol_software.util.math import makeEllGrids
from sptpol_software.simulation.beams import calc_gauss_Bl

c_c = const.c
k_c = const.k
h_c = const.h
Tcmb = 2.725

from datetime import datetime

def func1(X, a, b, c, d,e,f,g,h,i,j,k,l,m,n,o,p, q, r, s, t, u, v, w, x, y):
    aa,bb,cc,dd,ee,ff,gg,hh,ii,jj,kk,ll,mm,nn,oo, pp, qq,rr,ss,tt,uu,vv,ww,xx= X
    return a*aa+b*bb+c*cc+d*dd+e*ee+f*ff+g*gg+h*hh+i*ii+j*jj+k*kk+l*ll+m*mm+n*nn+o*oo+p*pp+q*qq+r*rr+s*ss+t*tt+u*uu+v*vv+w*ww+x*xx+y

# def func1(X, a, b,c, d):
#     aa, bb, cc, dd = X
#     return a*aa+b*bb+c*cc+d*dd

def wrapper_func(X, *args):
    a = list(args)
    return func(X, a)

def func(X, var_list):
    result = X[0] * var_list[0]
    for i in range(len(var_list)-1):
        result += var_list[i+1] * X[i+1]
    return result

def mod_BB_curve_with_z(v, A, T, z):
    #from Viero et al 
    beta = 2.
    v0 = v*(1+z)
    Bv = (2*h_c*v0**3/c_c**2)*(1/(np.exp(h_c*v0/(k_c*T))-1))
    Sv = np.abs(A)*v0**beta*Bv
    return Sv

def mod_BB_curve_with_tsz(v, A, y, T, z, g):
    #from Viero et al 
    beta = 2.
    v0 = v*(1+z)
    Bv = (2*h_c*v0**3/c_c**2)*(1/(np.exp(h_c*v0/(k_c*T))-1)) 
    Sv = np.abs(A)*v0**beta*Bv+ np.abs(y)*g
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
        bs_med[j] = np.mean(galaxy_arr[indices])
    return mad(bs_med)

def noise_PSD(zp_map, reso_arcmin, dngrid):
    '''
    Calculate the noise PSD given a map, code given by Lindsey
    '''
    reso = (reso_arcmin/60)*np.pi/180.
    dk = 1./dngrid/reso
    factor = np.sqrt(dngrid**2*dk**2)
    noise_psd = np.sqrt(np.abs(np.fft.ifft2(zp_map)/factor))
    return noise_psd

def zero_pad_map(masked_map,mask):
    #converting the nans in the masked map to 0s and zero padding map to have square shape
    zeros_masked_map = np.nan_to_num(masked_map)
    if zeros_masked_map.shape[0] != zeros_masked_map.shape[1]:
        zp_mask_map = np.zeros((max(zeros_masked_map.shape),max(zeros_masked_map.shape)))
        zp_mask = np.zeros((zp_mask_map.shape))
        if zp_mask_map.shape[0]>zeros_masked_map.shape[0]:
            edge_padding = (zp_mask_map.shape[0]-zeros_masked_map.shape[0])/2
            zp_mask_map[edge_padding:-edge_padding, :] = zeros_masked_map
            zp_mask[edge_padding:-edge_padding, :] = mask
        elif zp_mask_map.shape[1]>zeros_masked_map.shape[1]: 
            edge_padding = (zp_mask_map.shape[1]-zeros_masked_map.shape[1])/2
            zp_mask_map[:, edge_padding:-edge_padding] = zeros_masked_map
            zp_mask[:,edge_padding:-edge_padding] = mask
    else:
        zp_mask_map = zeros_masked_map
        zp_mask = mask

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

def calc_nu_eff(nu, transmission, source_spectrum, nonusq=True):
    '''
    Based on Tom's IDL code, modified so it can handle 2 minima due to CMB peak
    expects input nu in GHz
    WARNING: only compatible with SciPy versions >=0.17
    '''
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

def herschel_calculate_eff_band(freq, trans, T, z):
    wavelength = c_c/(freq*1.e9)

    eff_bands_500 = calc_nu_eff(freq, trans[0],mod_BB_curve_with_z(wavelength,1, T, z))
    eff_bands_350 = calc_nu_eff(freq, trans[1],mod_BB_curve_with_z(wavelength,1, T, z))
    eff_bands_250 = calc_nu_eff(freq, trans[2],mod_BB_curve_with_z(wavelength,1, T, z))
    return np.asarray([eff_bands_250, eff_bands_350, eff_bands_500])

def get_eff_band(lookup_table, T, z):
    z_set = lookup_table[:,0,0]
    T_set = lookup_table[0,:,0]
    z_ind = np.where(np.abs(z_set-z)==np.min(np.abs(z_set-z)))[0][0]
    T_ind = np.where(np.abs(T_set-T)==np.min(np.abs(T_set-T)))[0][0]
    return lookup_table[z_ind, T_ind, :]

def beamfilt(ell, fwhm_arcmin):
    sig_smooth = fwhm_arcmin/np.sqrt(8.*np.log(2.))/(60.*180.)* np.pi
    b_filter = np.exp(-ell*(ell+1.)*sig_smooth**2/2.)
    return b_filter

def calc_conversion_factor(mapfreq, reso_arcmin, filt_output):
    '''
    Calculates the conversion factor of K -> J 
    mapfreq is given in units of GHz
    filt_output is the filter output
    '''
    x = float(mapfreq)*1.e9*const.h/(const.k*Tcmb)
    reso = (reso_arcmin/60.)*np.pi/180.
    dngrid = filt_output[0].optfilt.shape[0]
    dk = 1./dngrid/reso
    tauprime = filt_output[0].prof2d*filt_output[0].trans_func_grid #* filt_output[0].area_eff
    solid_angle =1./ np.sum(filt_output[0].optfilt*tauprime)  / dk**2
    conversion = solid_angle*1.e26*(2*const.k/(const.c)**2)*((const.k*Tcmb/const.h)**2)*(x**4)*np.exp(x)/(np.exp(x)-1)**2
    return conversion, solid_angle

def map_smoothing(zp_map, trans_func_beam, trans_func, ellgrid, common_fwhm = 1.7):
    fft_masked_map = np.fft.fft2(zp_map)
    beam_divided_fft = fft_masked_map/trans_func_beam

    bfilt = beamfilt(ellgrid, common_fwhm)
    smoothed_map = np.fft.ifft2(beam_divided_fft*bfilt*trans_func).real
    return smoothed_map, bfilt*trans_func

def calc_theor_tsz_T(nu_eff):
    x = nu_eff/56.846 #in GHz
    delta_T = Tcmb * (x * (np.exp(x) + 1.)/(np.exp(x) - 1.) - 4.)
    return delta_T

def tsz_eff_freq_sptsz():
    return np.asarray([99.6, 155.5, 220.0 ])

def tsz_eff_freq_hers():
    return np.asarray([1076.4630, 804.72065, 560.62342])

def tsz_eff_freq_sptpol():
    return np.asarray([95.9000, 148.500])

def theor_tsz_hers_spt_in_T(use_sptpol = True):
    hers_eff_freq = tsz_eff_freq_hers()
    sptsz_eff_freq = tsz_eff_freq_sptsz()
    sptpol_eff_freq = tsz_eff_freq_sptpol()
    if use_sptpol == True:
        total_freq = np.concatenate([hers_eff_freq, np.asarray([sptsz_eff_freq[-1]]), sptpol_eff_freq[::-1]])
    elif use_sptpol == False:
        total_freq = np.concatenate([hers_eff_freq, sptsz_eff_freq[::-1]])
    return calc_theor_tsz_T(total_freq)

def calc_fac_with_error(error):
    random_error = np.random.normal(1, error, 1)[0]
    return random_error

def jackknife_error_est(Npatchx, Npatchy, ypix, xpix, true_fit, filtered_maps, gal_mask, cmap):
    patchsize = np.max(ypix)-np.min(ypix),np.max(xpix)-np.min(xpix) 
    real_patchsize = (np.round(patchsize[0]/Npatchy)+1)*Npatchy, (np.round(patchsize[1]/Npatchx)+1)*Npatchx
    centre_pixel = np.int(np.mean((np.max(ypix),np.min(ypix)))), np.int(np.mean((np.max(xpix),np.min(xpix))))
    small_patch = real_patchsize[0]/Npatchy, real_patchsize[1]/Npatchx
    jk_fits = np.zeros((Npatchy*Npatchx, len(true_fit)))
    jk_mask = np.zeros((len(jk_fits[:,0])),dtype=bool)
    fit_gal_mask = np.array(gal_mask, dtype=bool)

    for i in range(Npatchy):
        for j in range(Npatchx):
            pixel_place = centre_pixel[0]-small_patch[0]*Npatchy/2 + i*small_patch[0], centre_pixel[1]-small_patch[1]*Npatchx/2 + j*small_patch[1]

            filtered_maps_mod = filtered_maps.copy()
            filtered_maps_mod[:,pixel_place[0]:pixel_place[0]+small_patch[0], pixel_place[1]:pixel_place[1]+small_patch[1]] = 0

            cmap_mod = cmap.copy()
            cmap_mod[pixel_place[0]:pixel_place[0]+small_patch[0], pixel_place[1]:pixel_place[1]+small_patch[1]] = 0

            size_maps = len(filtered_maps_mod)
            guess = np.ones(size_maps)
            # popt, pcov =  curve_fit(lambda y, *params_0: wrapper_func(y, *params_0), ([filtered_maps_mod[x][fit_gal_mask].flatten() for x in range(size_maps)]),
            #                         cmap_mod[fit_gal_mask].flatten(), p0=guess)
            popt, pcov =  curve_fit(func1, ([filtered_maps_mod[x][fit_gal_mask].flatten() for x in range(len(filtered_maps_mod))]),\
                cmap_mod[fit_gal_mask].flatten())
            jk_fits[i*Npatchy+j,:] = popt

            mask_patch = np.sum(gal_mask[pixel_place[0]:pixel_place[0]+small_patch[0], pixel_place[1]:pixel_place[1]+small_patch[1]]) > 0.5*small_patch[0]*small_patch[1]
            jk_mask[i*Npatchy+j] = mask_patch

    jk_fits_mean = np.mean(jk_fits, axis=0)
    jk_var = (len(jk_fits[:,0])-1.)/((len(jk_fits[:,0])))*np.sum((jk_fits[jk_mask,:]-true_fit)**2, axis=0)
    return np.sqrt(jk_var)

def create_galaxy_mask(ypix, xpix, cmap, reso_arcmin, beam_arcmin = 5.0, threshold = 0.005):
    gal_hitmap = np.zeros((cmap.shape))
    gal_hitmap[ypix,xpix] = 1.
    reso = (reso_arcmin/60)*np.pi/180.
    ellgrid,ellx,elly = makeEllGrids([cmap.shape[0],cmap.shape[1]],reso)
    bfilt = calc_gauss_Bl(ellgrid,beam_arcmin)
    gal_mask = np.fft.ifft2(np.fft.fft2(gal_hitmap)*bfilt).real
    gal_mask[gal_mask>threshold] = 1.
    gal_mask[gal_mask<threshold] = 0.
    return gal_mask


def filtered_hitmaps(ypix, xpix, trans_func_grid, optfilt):
    hitmaps = np.zeros((optfilt.shape))
    for pix in range(len(ypix)):
        hitmaps[ypix[pix],xpix[pix]] += 1
    fft_maps = np.fft.fft2(hitmaps)
    fft_maps_TF = fft_maps*trans_func_grid
    real_maps = np.fft.ifft2(fft_maps_TF).real
    filtered_map = np.fft.ifft2(fft_maps_TF*optfilt).real
    filt_flux = filtered_hitmaps_norm_flux(trans_func_grid, optfilt)
    filtered_map = filtered_map/filt_flux
    return filtered_map

def filtered_hitmaps_norm_flux(trans_func_grid, optfilt):
    hitmaps = np.zeros((optfilt.shape))
    hitmaps[optfilt.shape[0]/2, optfilt.shape[1]/2] = 1
    fft_maps = np.fft.fft2(hitmaps)
    fft_maps_TF = fft_maps*trans_func_grid
    real_maps = np.fft.ifft2(fft_maps_TF).real
    filtered_map = np.fft.ifft2(fft_maps_TF*optfilt).real  
    return filtered_map[optfilt.shape[0]/2, optfilt.shape[1]/2] 

import multiprocessing
from joblib import Parallel, delayed
num_cores = multiprocessing.cpu_count()

def jackknife_error_parallel(Npatchx, Npatchy, ypix, xpix, true_fit, filtered_maps, gal_mask, cmap):
    patchsize = np.max(ypix)-np.min(ypix),np.max(xpix)-np.min(xpix) 
    real_patchsize = (np.round(patchsize[0]/Npatchy)+1)*Npatchy, (np.round(patchsize[1]/Npatchx)+1)*Npatchx
    centre_pixel = np.int(np.mean((np.max(ypix),np.min(ypix)))), np.int(np.mean((np.max(xpix),np.min(xpix))))
    small_patch = real_patchsize[0]/Npatchy, real_patchsize[1]/Npatchx
    jk_fits = np.zeros((Npatchy*Npatchx, len(true_fit)))
    jk_mask = np.zeros((len(jk_fits[:,0])),dtype=bool)
    fit_gal_mask = np.array(gal_mask, dtype=bool)

    for i in range(Npatchy):
        for j in range(Npatchx):
            pixel_place = centre_pixel[0]-small_patch[0]*Npatchy/2 + i*small_patch[0], centre_pixel[1]-small_patch[1]*Npatchx/2 + j*small_patch[1]

            filtered_maps_mod = filtered_maps.copy()
            filtered_maps_mod[:,pixel_place[0]:pixel_place[0]+small_patch[0], pixel_place[1]:pixel_place[1]+small_patch[1]] = 0

            cmap_mod = cmap.copy()
            cmap_mod[pixel_place[0]:pixel_place[0]+small_patch[0], pixel_place[1]:pixel_place[1]+small_patch[1]] = 0

            size_maps = len(filtered_maps_mod)
            guess = np.ones(size_maps)
            # popt, pcov =  curve_fit(lambda y, *params_0: wrapper_func(y, *params_0), ([filtered_maps_mod[x][fit_gal_mask].flatten() for x in range(size_maps)]),
            #                         cmap_mod[fit_gal_mask].flatten(), p0=guess)
            popt, pcov =  curve_fit(func1, ([filtered_maps_mod[x][fit_gal_mask].flatten() for x in range(len(filtered_maps_mod))]),\
                cmap_mod[fit_gal_mask].flatten())
            jk_fits[i*Npatchy+j,:] = popt

            mask_patch = np.sum(gal_mask[pixel_place[0]:pixel_place[0]+small_patch[0], pixel_place[1]:pixel_place[1]+small_patch[1]]) > 0.5*small_patch[0]*small_patch[1]
            jk_mask[i*Npatchy+j] = mask_patch

    jk_fits_mean = np.mean(jk_fits, axis=0)
    jk_var = (len(jk_fits[:,0])-1.)/((len(jk_fits[:,0])))*np.sum((jk_fits[jk_mask,:]-true_fit)**2, axis=0)
    return np.sqrt(jk_var)


