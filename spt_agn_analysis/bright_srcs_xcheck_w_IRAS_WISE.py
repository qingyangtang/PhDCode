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
from scipy.optimize import curve_fit
c_c = const.c
k_c = const.k
h_c = const.h

from datetime import datetime

today = datetime.today().strftime('%Y%m%d')
srcs_7sig = np.load('/home/tangq/src_7sig_herschel_hist.npy')

idldata = idl.readIDLSav("/data/tangq/IRAS/iras_ptsrc_plus_assoc.sav")['iras']
iras_ra = idldata['ra']
iras_dec = idldata['dec']

c = SkyCoord(ra=srcs_7sig[:,0]*u.degree, dec=srcs_7sig[:,1]*u.degree)  
catalog = SkyCoord(ra=iras_ra*u.degree, dec=iras_dec*u.degree)  
idx, d2d, d3d = c.match_to_catalog_sky(catalog)

src_flag = d2d.arcmin < 2

#############---------- 

idldata = idl.readIDLSav("/data/tangq/IRAS/IRAS_FSC.sav")['iras_fsc']
iras_ra = idldata['ra']
iras_dec = idldata['dec']

c = SkyCoord(ra=srcs_7sig[:,0]*u.degree, dec=srcs_7sig[:,1]*u.degree)  
catalog = SkyCoord(ra=iras_ra*u.degree, dec=iras_dec*u.degree)  
idx, d2d, d3d = c.match_to_catalog_sky(catalog)

src_flag = d2d.arcsec < 30

############------------WISE------------

idldata = idl.readIDLSav("/data/tangq/IRAS/24um_master.sav")['cat']

wise_ra = [idldata[x]['ra'] for x in np.arange(len(idldata))]
wise_dec = [idldata[x]['dec'] for x in np.arange(len(idldata))]
w1mpro = [idldata[x]['w1mpro'] for x in np.arange(len(idldata))]
w1snr = [idldata[x]['w1snr'] for x in np.arange(len(idldata))]
w2mpro = [idldata[x]['w2mpro'] for x in np.arange(len(idldata))]
w2snr = [idldata[x]['w2snr'] for x in np.arange(len(idldata))]
w3mpro = [idldata[x]['w3mpro'] for x in np.arange(len(idldata))]
w3snr = [idldata[x]['w3snr'] for x in np.arange(len(idldata))]
w4mpro = [idldata[x]['w4mpro'] for x in np.arange(len(idldata))]
w4snr = [idldata[x]['w4snr'] for x in np.arange(len(idldata))]

idx, d2d = t.match_src_to_catalog(srcs_7sig[:,0], srcs_7sig[:,1], wise_ra, wise_dec)

src_flag = d2d.arcmin < 10


################## random points from Herschel

idldata = idl.readIDLSav("/data/tangq/IRAS/24um_master.sav")['cat']

wise_ra = [idldata[x]['ra'] for x in np.arange(len(idldata))]
wise_dec = [idldata[x]['dec'] for x in np.arange(len(idldata))]
w1mpro = [idldata[x]['w1mpro'] for x in np.arange(len(idldata))]
w1snr = [idldata[x]['w1snr'] for x in np.arange(len(idldata))]
w2mpro = [idldata[x]['w2mpro'] for x in np.arange(len(idldata))]
w2snr = [idldata[x]['w2snr'] for x in np.arange(len(idldata))]
w3mpro = [idldata[x]['w3mpro'] for x in np.arange(len(idldata))]
w3snr = [idldata[x]['w3snr'] for x in np.arange(len(idldata))]
w4mpro = [idldata[x]['w4mpro'] for x in np.arange(len(idldata))]
w4snr = [idldata[x]['w4snr'] for x in np.arange(len(idldata))]

mask = np.zeros((zp_mask.shape))
mask[zp_mask>0.9999999] = 1
mask = mask.astype(bool)
goodpix = np.random.choice(cmap[mask], 800)
ypix_random = np.zeros((len(goodpix)))
xpix_random = np.zeros((len(goodpix)))

for i in range(len(goodpix)):
	ypix_random[i] = np.where(cmap ==goodpix[i])[0]
	xpix_random[i] = np.where(cmap == goodpix[i])[1]

ypix_random = ypix_random.astype(int)
xpix_random = xpix_random.astype(int)
import sptpol_software.observation.sky_amy1 as sky1
ra_random, dec_random = sky.pix2Ang(np.asarray([ypix_random,xpix_random]), radec0, reso_arcmin, npixels, proj=5)

idx, d2d = t.match_src_to_catalog(ra_random, dec_random, wise_ra, wise_dec)




