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

#Checking match filter code with a random source #

source_dir = '/data/tangq/SPT/test_source.fits'
transfer_dir = '/data/tangq/SPT/test_xfer_with_beam.fits'
psd_dir = '/data/tangq/SPT/test_psd.fits'

mapfreq = "90"

calfacs =[0.722787, 0.830593, 0.731330]
if mapfreq == "150":
	mapindex = 1
elif mapfreq == "90":
	mapindex = 0
elif mapfreq == "220":
	mapindex = 2
calfac = calfacs[mapindex]
data = fits.open(source_dir)[0].data[mapindex]*calfac
mask = maps.makeBorderApodization(data, reso_arcmin =0.25, radius_arcmin = 2)
masked_map = np.multiply(mask,data)

noise_psd = fits.open(psd_dir)[0].data[mapindex]
trans_func = fits.open(transfer_dir)[0].data[mapindex]

#filt_output,filt_map =  calc.call_matched_filt_point(masked_map, cmb = False, skysigvar = np.zeros((25000)), noisepsd=noise_psd, ellmax = 25000, ell_highpass = 10, ell_lowpass = 20000,use_fft_as_noise=False, plot_map=False, return_filtered_map=True, mask = zp_mask, reso_arcmin = reso_arcmin, ngridx = zp_mask_map.shape[0],ngridy = zp_mask_map.shape[0])

reso_arcmin = 0.25
skysigvar = np.zeros((25000))
noisepsd = noise_psd
ngridy = masked_map.shape[1]
ngridx = masked_map.shape[0]

import numpy as np
import pylab as pl
import sptpol_software.constants as constants
import sptpol_software.analysis.maps as maps 
import sptpol_software.analysis.plotting as plotting
from sptpol_software.constants import DTOR 
from sptpol_software.util.tools import struct
from sptpol_software.util.files import readCAMBCls
from sptpol_software.util.math import makeEllGrids
from sptpol_software.simulation.beams import calc_gauss_Bl
from sptpol_software.observation.sky import filterMap
import sptpol_software.util.idl as idl 
import scipy as sp
import cPickle as pkl
import sptpol_software.util.files as files
from glob import glob
import os as os
import sptpol_software.util.time as spttime
import multiprocessing
import itertools as itools

reso_rad = reso_arcmin * constants.DTOR / 60.

map_size = ngridx*ngridy
ellg,ellx,elly = makeEllGrids([ngridx,ngridy],reso_rad)

ellmax = len(skysigvar)-1
ell = np.arange(0,ellmax)

bfilt = None

m_filt = calc.calc_optfilt_oneband_2d([0,0],# profiles is not used with pointsource=True
                                skysigvar, noisepsd, reso_arcmin,
                                bfilt = bfilt,
                                # keeping below for all calls from the function
                                pointsource = True,
                                sigspect = 1,
                                keepdc = False, 
                                trans_func_grid= trans_func,
                                arcmin_sigma = None,
                                norm_unfilt = None,
                                external_profile_norm = None
                                )
Map = masked_map
filtered_map = Map.copy()
uv_filter = m_filt[0].optfilt

# Create a default "mask" of all ones if we didn't get an input mask,
# and calculate the mask normalization.
mask = np.ones((Map.shape))
window_norm_factor = np.sqrt(np.sum(mask**2) / np.product(mask.shape))

# Go through the T maps and filter them.
ft_map = calc.getFTMap(Map, mask)

    # Apply the k-space filter.
ft_map *= uv_filter

  # Transform back to real space, and remove the apodization mask.
filtered_map = np.fft.ifft2(ft_map).real

filt_map = filtered_map


plt.subplot(1,2,1)
plt.imshow(masked_map, vmin=-0.10, vmax = 0.25)
plt.colorbar()
plt.title('Original Map')
plt.subplot(1,2,2)
plt.imshow(filt_map, vmin=-0.10, vmax = 0.25)
plt.colorbar()
plt.title('Filtered map')

print "max of original map: " + str(np.max(masked_map))
print "max of filtered map: " + str(np.max(filt_map))
print "ratio of filtered/original: " + str(np.max(filt_map)/np.max(masked_map))
