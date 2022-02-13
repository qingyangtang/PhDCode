import numpy as np
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
import sptpol_software.constants as constants
from sptpol_software.util.math import makeEllGrids
import sptpol_software.analysis.plotting as plotting
import scipy.constants as const
from scipy.optimize import curve_fit
import tool_func as t
import ConfigParser
from datetime import datetime
from astropy.coordinates import Angle
from sptpol_software.util.files import readCAMBCls
today = datetime.today().strftime('%Y%m%d')


###--------------- CREATING TRANSFER FUNCTIONS -------------------###

# run below to convert the .bin files into .hdf5
# ln -s /data42/bleeml/beam_sims/bundle40_90ghz_beamsim.bin /data/tangq/SPT/SPTpol/bundle40_90ghz_beamsim.hdf5
# ln -s /data42/bleeml/beam_sims/bundle0_90ghz_beamsim.bin /data/tangq/SPT/SPTpol/bundle0_90ghz_beamsim.hdf5
# ln -s /data42/bleeml/beam_sims/bundle40_150ghz_beamsim.bin /data/tangq/SPT/SPTpol/bundle40_150ghz_beamsim.hdf5
# ln -s /data42/bleeml/beam_sims/bundle0_150ghz_beamsim.bin /data/tangq/SPT/SPTpol/bundle0_150ghz_beamsim.hdf5

plot_trans_func = False 

bund0_90 = files.read('/data/tangq/SPT/SPTpol/bundle0_90ghz_beamsim.hdf5')
bund40_90 = files.read('/data/tangq/SPT/SPTpol/bundle40_90ghz_beamsim.hdf5')
bund0_150 = files.read('/data/tangq/SPT/SPTpol/bundle0_150ghz_beamsim.hdf5')
bund40_150 = files.read('/data/tangq/SPT/SPTpol/bundle40_150ghz_beamsim.hdf5')

apod_mask = fits.open('/data/tangq/SPT/SPTpol/ra0hdec-57.5_mask_150_0p8.fits')[1].data['APOD_MASK'][0]

orig_90 = np.rot90(fits.open('/data/tangq/SPT/gaussian_1_65_beam_npixels_11280_6000_0104.fits')[0].data)
orig_150 = np.rot90(fits.open('/data/tangq/SPT/gaussian_1_2_beam_npixels_11280_6000_0104.fits')[0].data)

#orig_small = np.rot90(fits.open('/data/tangq/SPT/gaussian_0_75_beam_npixels_2048_2048_.fits')[0].data)
orig_big = np.rot90(fits.open('/data/tangq/SPT/gaussian_0_75_beam_npixels_11280_6000_.fits')[0].data)

orig_90_masked = orig_big*apod_mask
orig_90_fft = np.fft.fft2(orig_90_masked)
orig_150_masked = orig_big*apod_mask
orig_150_fft = np.fft.fft2(orig_150_masked)
orig_90_fft[np.abs(orig_90_fft) < 0.001*np.max(np.abs(orig_90_fft))] = 0.0005*np.max(np.abs(orig_90_fft))
orig_150_fft[np.abs(orig_150_fft) < 0.001*np.max(np.abs(orig_150_fft))] = 0.0005*np.max(np.abs(orig_150_fft))

xfer90_b0_uw_masked = bund0_90.Map['sim_map_T_0']/bund0_90.Map['weight_map']
xfer90_b0_uw_masked[np.isnan(xfer90_b0_uw_masked)] = 0
xfer90_b0_uw_masked[np.isinf(np.abs((xfer90_b0_uw_masked)))]= 0 
xfer90_b0_uw_masked=xfer90_b0_uw_masked*apod_mask
xfer90_b0_uw_fft = np.fft.fft2(xfer90_b0_uw_masked)

xfer90_b40_uw_masked = bund40_90.Map['sim_map_T_0']/bund40_90.Map['weight_map']
xfer90_b40_uw_masked[np.isnan(xfer90_b40_uw_masked)] = 0
xfer90_b40_uw_masked[np.isinf(np.abs((xfer90_b40_uw_masked)))]= 0 
xfer90_b40_uw_masked=xfer90_b40_uw_masked*apod_mask
xfer90_b40_uw_fft = np.fft.fft2(xfer90_b40_uw_masked)

xfer150_b0_uw_masked = bund0_150.Map['sim_map_T_0']/bund0_150.Map['weight_map']
xfer150_b0_uw_masked[np.isnan(xfer150_b0_uw_masked)] = 0
xfer150_b0_uw_masked[np.isinf(np.abs((xfer150_b0_uw_masked)))]= 0 
xfer150_b0_uw_masked=xfer150_b0_uw_masked*apod_mask
xfer150_b0_uw_fft = np.fft.fft2(xfer150_b0_uw_masked)

xfer150_b40_uw_masked = bund40_150.Map['sim_map_T_0']/bund40_150.Map['weight_map']
xfer150_b40_uw_masked[np.isnan(xfer150_b40_uw_masked)] = 0
xfer150_b40_uw_masked[np.isinf(np.abs((xfer150_b40_uw_masked)))]= 0 
xfer150_b40_uw_masked=xfer150_b40_uw_masked*apod_mask
xfer150_b40_uw_fft = np.fft.fft2(xfer150_b40_uw_masked)

trans_func_90_b0 = np.abs(xfer90_b0_uw_fft/orig_90_fft)
trans_func_90_b0[np.isinf(trans_func_90_b0)] = 0
reso = (0.25/60)*np.pi/180.
ellgrid,ellx5,elly5 = makeEllGrids([trans_func_90_b0.shape[0],trans_func_90_b0.shape[1]],reso)
trans_func_90_b0[np.abs(ellx5)<400] = 0
trans_func_90_b0[np.abs(ellgrid)>20000] = 0

trans_func_90_b40 = np.abs(xfer90_b40_uw_fft/orig_90_fft)
trans_func_90_b40[np.isinf(trans_func_90_b40)] = 0
trans_func_90_b40[np.abs(ellx5)<400] = 0
trans_func_90_b40[np.abs(ellgrid)>20000] = 0

trans_func_150_b0 = np.abs(xfer150_b0_uw_fft/orig_150_fft)
trans_func_150_b0[np.isinf(trans_func_150_b0)] = 0
trans_func_150_b0[np.abs(ellx5)<400] = 0
trans_func_150_b0[np.abs(ellgrid)>20000] = 0

trans_func_150_b40 = np.abs(xfer150_b40_uw_fft/orig_150_fft)
trans_func_150_b40[np.isinf(trans_func_150_b40)] = 0
trans_func_150_b40[np.abs(ellx5)<400] = 0
trans_func_150_b40[np.abs(ellgrid)>20000] = 0

if plot_trans_func == True:
	plt.figure(figsize=(10,10))
	plt.subplot(221)
	plt.imshow(trans_func_90_b0)
	plt.colorbar()
	plt.title('Transfer function 90 bundle 0')
	plt.subplot(222)
	plt.imshow(trans_func_90_b40)
	plt.colorbar()
	plt.title('Transfer function 90 bundle 40')
	plt.subplot(223)
	plt.imshow(trans_func_150_b0)
	plt.colorbar()
	plt.title('Transfer function 150 bundle 0')
	plt.subplot(224)
	plt.imshow(trans_func_150_b40)
	plt.colorbar()
	plt.title('Transfer function 150 bundle 40')
	plt.savefig('/home/tangq/'+today+'_trans_func.png')
	plt.close()

x1, y1 = plotting.plotRadialProfile(np.fft.fftshift(trans_func_90_b0), center_bin=[orig_90_masked.shape[0]/2,orig_90_masked.shape[1]/2], resolution=(ellx5[0,1]-ellx5[0,0]), histogram=True, range=[0,20000],plot=True, bin_resolution_factor=0.1, figure=5,new=False, \
	label="90GHz b0")
x2, y2 = plotting.plotRadialProfile(np.fft.fftshift(trans_func_90_b40), center_bin=[orig_90_masked.shape[0]/2,orig_90_masked.shape[1]/2], resolution=(ellx5[0,1]-ellx5[0,0]), histogram=True, range=[0,20000],plot=True, bin_resolution_factor=0.1, figure=5,new=False, \
	label="90GHz b40")
x3, y3 = plotting.plotRadialProfile(np.fft.fftshift(trans_func_150_b0), center_bin=[orig_90_masked.shape[0]/2,orig_90_masked.shape[1]/2], resolution=(ellx5[0,1]-ellx5[0,0]), histogram=True, range=[0,20000],plot=True, bin_resolution_factor=0.1, figure=5,new=False, \
	label="150GHz b0")
x4, y4 = plotting.plotRadialProfile(np.fft.fftshift(trans_func_150_b40), center_bin=[orig_90_masked.shape[0]/2,orig_90_masked.shape[1]/2], resolution=(ellx5[0,1]-ellx5[0,0]), histogram=True, range=[0,20000],plot=True, bin_resolution_factor=0.1, figure=5,new=False, \
	label="150GHz b40")
plt.legend()
plt.xlabel('ell')
plt.title('Transfer function')

plt.figure()
plt.plot(x1, y1/y2, label='90GHz')
plt.plot(x3, y3/y4, label='150GHz')
plt.title('b0/b40 radial averaged transfer function')
plt.legend()
plt.xlabel('ell')
plt.ylabel('ratio')

trans_func_90_b0[trans_func_90_b0>1] = 1
trans_func_90_b40[trans_func_90_b40>1] = 1
trans_func_150_b0[trans_func_150_b0>1] = 1
trans_func_150_b40[trans_func_150_b40>1] = 1

trans_func_90 = (trans_func_90_b0 + trans_func_90_b40)/2
trans_func_150 = (trans_func_150_b0 + trans_func_150_b40)/2

plt.figure()
plt.subplot(121)
plt.imshow(trans_func_90)
plt.colorbar()
plt.title('90GHz mock xfer func')
plt.subplot(122)
plt.imshow(trans_func_150)
plt.colorbar()
plt.title('150GHz mock xfer func')

x1, y1 = plotting.plotRadialProfile(np.fft.fftshift(trans_func_90), center_bin=[orig_90_masked.shape[0]/2,orig_90_masked.shape[1]/2], resolution=(ellx5[0,1]-ellx5[0,0]), histogram=True, range=[0,20000],plot=True, bin_resolution_factor=0.1, figure=5,new=False, \
	label="90GHz")
x2, y2 = plotting.plotRadialProfile(np.fft.fftshift(trans_func_150), center_bin=[orig_90_masked.shape[0]/2,orig_90_masked.shape[1]/2], resolution=(ellx5[0,1]-ellx5[0,0]), histogram=True, range=[0,20000],plot=True, bin_resolution_factor=0.1, figure=5,new=False, \
	label="150GHz")
plt.legend()
plt.xlabel('ell')
plt.title('Transfer function')

from astropy.io import fits
hdu = fits.PrimaryHDU(trans_func_90)
hdu.writeto('/data/tangq/SPT/SPTpol/trans_func_90.fits')
hdu = fits.PrimaryHDU(trans_func_150)
hdu.writeto('/data/tangq/SPT/SPTpol/trans_func_150.fits')

