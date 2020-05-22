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

	src_ra = cat_ra
	src_dec = cat_dec
	src_flux = cat_f_250

	return cat_ra, cat_dec, cat_flux

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

def Herschel_filtered_maps_

