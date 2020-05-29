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

import scipy.constants as const
from scipy.optimize import curve_fit
import tool_func as t


def get_SPIRE_ptsrcs(catalog_dir, sn_cut=40):
	#getting all the SPIRE point sources for SN>sn_cut and their RA, dec, flux
	idldata = idl.readIDLSav(catalog_dir)
	cat_ra = np.asarray([d['ra'] for d in idldata.cat])
	cat_dec = np.asarray([d['dec'] for d in idldata.cat])
	cat_f_250 = (np.asarray([d['f_250'] for d in idldata.cat]))
	cat_df_250 = np.asarray([d['df_250'] for d in idldata.cat])
	good_flux = np.where(cat_f_250>0)
	cat_ra = cat_ra[good_flux]
	cat_dec = cat_dec[good_flux]
	cat_f_250 = cat_f_250[good_flux]
	cat_df_250 = cat_df_250[good_flux]
	sn = cat_f_250/cat_df_250
	sn_cut = np.where(sn > sn_cut)
	cat_ra = cat_ra[sn_cut]
	cat_dec = cat_dec[sn_cut]
	cat_f_250 = cat_f_250[sn_cut]
	cat_df_250 = cat_df_250[sn_cut]

	return cat_ra, cat_dec, cat_f_250

def get_Spitzer_DES_cat_with_mass(sp_des_dir,stellar_m_dir,iband_cut=23):
	sp_des_cat = fits.open(sp_des_dir)[1].data
	stellar_cat = np.loadtxt(stellar_m_dir)
	col_z = Column(name = 'stellar_mass_z', data = stellar_cat[:,1])
	col_lmass = Column(name = 'lmass', data = stellar_cat[:,6])
	col_chi2 = Column(name = 'sm_chi2', data=stellar_cat[:,-1])
	new_cols = fits.ColDefs([fits.Column(name='stellar_mass_z', format='D',array=stellar_cat[:,1]),
		fits.Column(name='lmass', format='D',array=stellar_cat[:,6]), 
		fits.Column(name='sm_chi2', format='D',array=stellar_cat[:,-1])])
	hdu = fits.BinTableHDU.from_columns(sp_des_cat.columns + new_cols)
	sp_des_sm_cat = hdu.data

	mask = sp_des_sm_cat['i'] < iband_cut
	clean_cat = sp_des_sm_cat[mask]

	return clean_cat


def make_Herschel_filtered_map(save_map = False, save_map_dir=None, save_map_fname=None):

	beamsizes = np.asarray([18.2, 24.9, 36.3])/60. #beams for 250, 350, 500um in arcmins
	#loading the data
	for mapfreq in ["250", "350","500"]:
		if mapfreq == "250":
			fits0 = fits.open("/data/tangq/SPIRE_maps/spt_zea_itermap_10_iterations_15_arcsec_pixels_PSW.fits")
			beamsize = 0
			source_mask = '/data/tangq/SPIRE_maps/herschel_srcs_40_PSW.txt'

		elif mapfreq == "350":
			fits0 = fits.open("/data/tangq/SPIRE_maps/spt_zea_itermap_10_iterations_15_arcsec_pixels_PMW.fits")
			beamsize = 1
			source_mask = '/data/tangq/SPIRE_maps/herschel_srcs_40_PMW.txt'
		elif mapfreq == "500":
			fits0 = fits.open("/data/tangq/SPIRE_maps/spt_zea_itermap_10_iterations_15_arcsec_pixels_PLW.fits")
			beamsize = 2
			source_mask = '/data/tangq/SPIRE_maps/herschel_srcs_40_PLW.txt'
		else:
			print "Enter valid map freq!!"

		radec0 = np.asarray([fits0[1].header['CRVAL1'], fits0[1].header['CRVAL2']])
		reso_arcmin = abs(fits0[1].header['CD1_1'])*60 
		map0 = np.rot90(np.rot90(fits0[1].data))
		npixels = map0.shape

		mask = maps.makeApodizedPointSourceMask(map0, source_mask, ftype = 'herschel', reso_arcmin=reso_arcmin, center=radec0, radius_arcmin=3)
		masked_map = np.multiply(mask,map0)
		zp_mask_map, zp_mask = t.zero_pad_map(masked_map, mask)

		#calculating noise PSD
		noise_psd = t.noise_PSD(zp_mask_map, reso_arcmin, dngrid=zp_mask_map.shape[0])

		filt_output,filt_map =  calc.call_matched_filt_point(zp_mask_map, cmb = False, skysigvar = np.zeros((25000)), noisepsd=noise_psd, \
								fwhm_beam_arcmin=beamsizes[beamsize], ellmax = 25000, ell_highpass = 10, ell_lowpass = 20000,use_fft_as_noise=False, \
								plot_map=False, return_filtered_map=True, mask = zp_mask, reso_arcmin = reso_arcmin, ngridx = zp_mask_map.shape[0],\
								ngridy = zp_mask_map.shape[0])
		cmap = filt_map
		npixels = np.asarray(cmap.shape)

		if mapfreq == "250":
			map250 = cmap
			reso_arcmin250 = reso_arcmin
			radec0250 = radec0
			npixels250 = npixels
	 	elif mapfreq == "350":
			map350 = cmap
			reso_arcmin350 = reso_arcmin
			radec0350 = radec0
			npixels350 = npixels
		elif mapfreq == "500":
			map500 = cmap
			reso_arcmin500 = reso_arcmin
			radec0500 = radec0
			npixels500 = npixels
	map_arr = [map250, map350, map500]
	reso_arcmin_arr = [reso_arcmin250, reso_arcmin350, reso_arcmin500]
	radec0_arr = [radec0250,radec0350,radec0500]
	npixels_arr = [npixels250,npixels350, npixels500]
	mapfreq_arr = ["250", "350", "500"]

	if save_map:
		np.save(save_map_dir+'/'+save_map_fname+'_250um_map.npy', map_arr[0])
		np.save(save_map_dir+'/'+save_map_fname+'_350um_map.npy', map_arr[1])
		np.save(save_map_dir+'/'+save_map_fname+'_500um_map.npy', map_arr[2])
		np.save(save_map_dir+'/'+save_map_fname+'_zp_mask.npy', zp_mask)
		np.savetxt(save_map_dir+'/'+save_map_fname+'_reso_arcmin.txt', np.asarray(reso_arcmin_arr)) 
		np.savetxt(save_map_dir+'/'+save_map_fname+'_radec.txt', np.asarray(radec0_arr)) 

	return map_arr, reso_arcmin_arr, radec0_arr, npixels_arr, mapfreq_arr, zp_mask

def herschel_trans_curve():
	'''
	Opens the Herschel transmission curves, which are in GHz, returns the freq and transmission curve in GHz
	zero-padded to be compatible with calc_eff_nu
	'''
	hdu = fits.open('/data/tangq/SPIRE_maps/SPIRE_phot_RSRF_14_3.fits')
	freq = hdu[1].data['frequency']
	psw = hdu[1].data['psw']
	pmw = hdu[1].data['pmw']
	plw = hdu[1].data['plw']

	freq = np.concatenate((np.linspace(np.min(freq)-200,np.min(freq)-1, 99), freq))
	trans = np.asarray([plw,pmw,psw])
	trans = np.hstack((np.zeros((3,99)),trans))

	return freq, trans
