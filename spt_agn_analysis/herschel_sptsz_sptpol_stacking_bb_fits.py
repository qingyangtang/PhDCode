import numpy as np
import matplotlib.pyplot as plt
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
import tool_func as t
import open_datasets as od
import scipy.constants as const
from scipy.optimize import curve_fit
c_c = const.c
k_c = const.k
h_c = const.h
from datetime import datetime

today = datetime.today().strftime('%Y%m%d')
bands = np.asarray([250.e-6, 350.e-6, 500.e-6])
freqs = c_c/bands
Tcmb = 2.725


sp_des_dir = '/data/tangq/custom_catalogs/spitzer_des_cat.fits'
stellar_m_dir = '/data/tangq/custom_catalogs/combined_ssdf_des_cat_061021.sav'
#stellar_m_dir = '/data/tangq/custom_catalogs/ssdf_matched_to_des_i1_lt_20_stellar_mass.sav'

# stellar_m_dir = '/data/bleeml/stellar_mass_cats/ssdf_des_try1.fout'
#clean_cat = od.get_Spitzer_DES_cat_with_mass(sp_des_dir,stellar_m_dir,iband_cut=23)
# cat = idl.readIDLSav(stellar_m_dir).cat
# cat_dec = np.asarray([cat[x].dec for x in range(len(cat))])
# cat_ra = np.asarray([cat[x].ra for x in range(len(cat))])
# cat_zphot = np.asarray([cat[x].z_phot for x in range(len(cat))])
# cat_sm = np.asarray([cat[x].stellar_mass for x in range(len(cat))])
# cat_smchi2 = np.asarray([cat[x].chi2_sm for x in range(len(cat))])
# cat_lsfr = np.asarray([cat[x].lsfr for x in range(len(cat))])
# cat_lssfr = np.asarray([cat[x].lssfr for x in range(len(cat))])
# mask = (cat_smchi2 < 10) & (cat_zphot >0) & np.invert(np.isnan(cat_sm)) & (cat_smchi2 > 0)
# cat_dec = cat_dec[mask]
# cat_ra = cat_ra[mask]
# cat_zphot = cat_zphot[mask]
# cat_sm = cat_sm[mask]
# cat_smchi2 = cat_smchi2[mask]
# cat_lsfr = cat_lsfr[mask]
# cat_lssfr = cat_lssfr[mask]
# del cat
# clean_cat= Table([cat_ra, cat_dec, cat_zphot, cat_sm, cat_smchi2,cat_lsfr, cat_lssfr], names=('ra', 'dec', 'stellar_mass_z', 'lmass', 'sm_chi2','lsfr','lssfr'))
# clean_cat.write('/data/tangq/custom_catalogs/20210614_cat.fits',format='fits')

clean_cat = fits.open('/data/tangq/custom_catalogs/20210614_cat.fits')[1].data

########## getting rid of galaxies within 4 arcmin of point sources
SPIRE_ptsrc_dir = "/data/tangq/SPIRE_maps/spt_starfndr_catalog.sav"

hers_src_ra, hers_src_dec, hers_src_flux = od.get_SPIRE_ptsrcs(SPIRE_ptsrc_dir, sn_cut=40)
sptsz_src_ra, sptsz_src_dec, sptsz_src_rad = od.get_SPTSZ_ptsrcs('/data/tangq/SPT/SZ/ptsrc_config_ra23h30dec-55_surveyclusters.txt')

src_ra = np.concatenate([hers_src_ra,sptsz_src_ra])
src_dec = np.concatenate([hers_src_dec,sptsz_src_dec])
idx, d2d = t.match_src_to_catalog(clean_cat['ra'], clean_cat['dec'], src_ra, src_dec)
src_mask = d2d.arcmin > 4
clean_cat = clean_cat[src_mask]

## cleaning out area in weird patch
#mask = clean_cat['ra']<357.
#clean_cat = clean_cat[mask]
mask = (clean_cat['lmass']>8.5) & (clean_cat['lmass']<12) & (clean_cat['sm_chi2']>0)
clean_cat = clean_cat[mask]
#sm_bins = np.asarray([8.5, 10.2, 10.6, 10.9, 11.2,12.])
#z_bins = np.asarray([0., 0.66, 1.33, 2.])

sm_bins = np.asarray([9, 9.5, 10., 10.5, 11., 11.5, 12])
z_bins = np.asarray([0., 0.5, 1., 1.5, 2.])
n_zbins = len(z_bins)-1

############SPT-SZ + SPTpol

fname = '/data/tangq/SPT/SZ/products/SPTSZ_filtered_no_smooth_FTfixed_20210401'
fnamepol = '/data/tangq/SPT/SPTpol/products/SPTpol_filtered_no_smooth_20210510'
map90 = np.load(fnamepol + '_90_map.npy')
map150 = np.load(fnamepol + '_150_map.npy')
map220 = np.load(fname + '_220_map.npy')
sptsz_pixelmask = np.load(fname + '_zp_mask.npy')
sptsz_reso_arcmin = np.loadtxt(fname + '_reso_arcmin.txt')
sptsz_radec0 = np.loadtxt(fname + '_radec.txt')
sptsz_cmap = [map90, map150, map220]
sptsz_npixels = [np.asarray(map90.shape),np.asarray(map150.shape), np.asarray(map220.shape)]
sptsz_mapfreq = ["95", "150", "220"]
filt90 = np.load(fnamepol+'_90_filter.npy').item()
filt150 = np.load(fnamepol+'_150_filter.npy').item()
filt220 = np.load(fname+'_220_filter.npy').item()
sptsz_filt = [filt90, filt150, filt220]
spt_K2Jy_conversion = np.zeros((len(sptsz_cmap)))
for x in range(len(sptsz_cmap)):
	spt_K2Jy_conversion[x], dummy = t.calc_conversion_factor(sptsz_mapfreq[x], sptsz_reso_arcmin[x], sptsz_filt[x])
	sptsz_cmap[x] = sptsz_cmap[x]*spt_K2Jy_conversion[x]

########## Hershel

normalize_solid_angles = False

fname = '/data/tangq/SPIRE_maps/products/Herschel_filtered_no_smooth_FTfixed_20210519'
map250 = np.load(fname+'_250_map.npy')
map350 = np.load(fname+'_350_map.npy')
map500 = np.load(fname+'_500_map.npy')
hers_pixelmask = np.load(fname+'_zp_mask.npy')
hers_reso_arcmin = np.loadtxt(fname+'_reso_arcmin.txt')
hers_radec0 = np.loadtxt(fname+'_radec.txt')
hers_cmap = [map250, map350, map500]
hers_npixels = [np.asarray(map250.shape),np.asarray(map350.shape), np.asarray(map500.shape)]
hers_mapfreq = ["250", "350", "500"]
filt250 = np.load(fname+'_250_filter.npy').item()
filt350 = np.load(fname+'_350_filter.npy').item()
filt500 = np.load(fname+'_500_filter.npy').item()
hers_filt = [filt250, filt350, filt500]

### making galaxy mask
gal_mask_dir = '/data/tangq/custom_catalogs/SSDF_galaxy_mask_with_DES.sav'
spt_gal_mask = idl.readIDLSav(gal_mask_dir).mask
hers_gal_mask = np.zeros((spt_gal_mask.shape))
hers_gal_mask[1:, 2:] = spt_gal_mask[:-1,:-2]

spt_gal_mask = spt_gal_mask*sptsz_pixelmask
hers_gal_mask = hers_gal_mask*hers_pixelmask
spt_gal_mask[spt_gal_mask < 0.9999] = 0
hers_gal_mask[hers_gal_mask < 0.9999] = 0
spt_gal_mask_bool = np.asarray(spt_gal_mask, dtype = bool)
hers_gal_mask_bool = np.asarray(hers_gal_mask, dtype = bool)

### getting t-sz spectrum 
hers_tsz_K2Jy_conversion = np.zeros((len(hers_cmap)))
hers_tsz_eff_freq = t.tsz_eff_freq_hers()
spt_tsz_eff_freq = np.concatenate([t.tsz_eff_freq_sptpol(), np.asarray([t.tsz_eff_freq_sptsz()[-1]])])
spt_tsz_K2Jy_conversion = np.zeros((len(sptsz_cmap)))
spt_solid_angle = np.zeros((len(sptsz_cmap)))
hers_solid_angle = np.zeros((len(hers_cmap)))
for x in range(len(sptsz_cmap)):
	hers_tsz_K2Jy_conversion[x], hers_solid_angle[x] = t.calc_conversion_factor(hers_tsz_eff_freq[x], hers_reso_arcmin[x], hers_filt[x])
	spt_tsz_K2Jy_conversion[x], spt_solid_angle[x] = t.calc_conversion_factor(spt_tsz_eff_freq[x], sptsz_reso_arcmin[x], sptsz_filt[x])

theor_tsz_in_T = t.theor_tsz_hers_spt_in_T(use_sptpol = True)
theor_tsz_Jy = np.concatenate([theor_tsz_in_T[:3]*hers_tsz_K2Jy_conversion, theor_tsz_in_T[3:]*spt_K2Jy_conversion[::-1]])

#### cleaning out any galaxies in masked out places
sptsz_ypix = []
sptsz_xpix = []
hers_ypix = []
hers_xpix = []

for i in range(3):
	sptsz_ypix_i, sptsz_xpix_i = sky.ang2Pix(np.asarray([clean_cat['ra'], clean_cat['dec']]), sptsz_radec0[i], sptsz_reso_arcmin[i], sptsz_npixels[i],proj=0)[0]
	sptsz_ypixmask, sptsz_xpixmask = sky.ang2Pix(np.asarray([clean_cat['ra'], clean_cat['dec']]), sptsz_radec0[i], sptsz_reso_arcmin[i], sptsz_npixels[i],proj=0)[1] 
	hers_ypix_i, hers_xpix_i = sky.ang2Pix(np.asarray([clean_cat['ra'], clean_cat['dec']]), hers_radec0[i], hers_reso_arcmin[i], hers_npixels[i],proj=0)[0]
	hers_ypixmask, hers_xpixmask = sky.ang2Pix(np.asarray([clean_cat['ra'], clean_cat['dec']]), hers_radec0[i], hers_reso_arcmin[i], hers_npixels[i],proj=0)[1] 
	ang2pix_mask = (sptsz_ypixmask == True) & (sptsz_xpixmask == True) & (hers_ypixmask == True) & (hers_xpixmask == True)
	clean_cat = clean_cat[ang2pix_mask]
	sptsz_ypix_i = sptsz_ypix_i[ang2pix_mask]
	sptsz_xpix_i = sptsz_xpix_i[ang2pix_mask]
	hers_ypix_i = hers_ypix_i[ang2pix_mask]
	hers_xpix_i = hers_xpix_i[ang2pix_mask]
	pix_mask = (sptsz_pixelmask[sptsz_ypix_i,sptsz_xpix_i]>0.99999999) & (hers_pixelmask[hers_ypix_i,hers_xpix_i]>0.99999999)
	clean_cat = clean_cat[pix_mask]
	sptsz_ypix_i = sptsz_ypix_i[pix_mask]
	sptsz_xpix_i = sptsz_xpix_i[pix_mask]
	hers_ypix_i = hers_ypix_i[pix_mask]
	hers_xpix_i = hers_xpix_i[pix_mask]	
	sptsz_ypix.append(sptsz_ypix_i)
	sptsz_xpix.append(sptsz_xpix_i)
	hers_ypix.append(hers_ypix_i)
	hers_xpix.append(hers_xpix_i)

######

#sm_bins = np.arange(9,11.2,0.2)
n_realizations = 100

herschel_lookup_table = np.load('/data/tangq/SPIRE_maps/Herschel_effective_bands.npy')
sptsz_lookup_table = np.load('/data/tangq/wendymaps/SPTSZ_effective_bands.npy')
sptpol_lookup_table = np.load('/data/tangq/SPT/SPTpol/SPTpol_effective_bands.npy')

hers_stack_flux = np.zeros((len(sm_bins[:-1]), n_zbins, len(hers_mapfreq)))
hers_stack_flux_err = np.zeros((len(sm_bins[:-1]), n_zbins, len(hers_mapfreq)))
spt_stack_flux = np.zeros((len(sm_bins[:-1]), n_zbins, len(sptsz_mapfreq)))
spt_stack_flux_err = np.zeros((len(sm_bins[:-1]), n_zbins, len(sptsz_mapfreq)))
spt_filtered_maps = np.zeros((len(sm_bins[:-1]), n_zbins, len(sptsz_mapfreq), sptsz_cmap[0].shape[0], sptsz_cmap[0].shape[1]))
hers_filtered_maps = np.zeros((len(sm_bins[:-1]), n_zbins, len(hers_mapfreq), hers_cmap[0].shape[0], hers_cmap[0].shape[1]))

T_set = np.arange(3, 45., 0.1)
dof = np.float(len(bands))+np.float(len(sptsz_mapfreq))-1.
num_obj_zbins = np.zeros((len(sm_bins[:-1]),n_zbins))
eff_freq_zarr = np.zeros((n_zbins, len(sptsz_mapfreq)+len(hers_mapfreq)))
stacked_flux_zarr= np.zeros((n_zbins, len(sptsz_mapfreq)+len(hers_mapfreq)))
stacked_flux_err_zarr = np.zeros((n_zbins, len(sptsz_mapfreq)+len(hers_mapfreq)))
median_z = np.zeros((len(sm_bins[:-1]), n_zbins))
mean_z = np.zeros((len(sm_bins[:-1]), n_zbins))


for i in range(len(sm_bins)-1):
	# sm_mask_dummy = ((clean_cat['lmass'] > sm_bins[i]) & (clean_cat['lmass'] < sm_bins[i+1]))
	#z_bins = np.linspace(np.min(clean_cat['stellar_mass_z'][sm_mask_dummy]),np.min([np.max(clean_cat['stellar_mass_z'][sm_mask_dummy]),1.99])+0.01, n_zbins+1)
	for x in range(3):
		cmap = hers_cmap[x]
		ypix = hers_ypix[x]
		xpix = hers_xpix[x]
		mapfreq = hers_mapfreq[x]
		flux_zbins = np.zeros((len(z_bins[:-1])))
		error = np.zeros((len(z_bins[:-1])))
		### creating a galaxy mask
		filtered_maps = np.zeros((len(z_bins[:-1]),cmap.shape[0],cmap.shape[1]))
		for z_i in range(len(z_bins[:-1])):
			sm_mask = (clean_cat['stellar_mass_z'] > z_bins[z_i]) & (clean_cat['stellar_mass_z'] < z_bins[z_i+1]) \
				& (clean_cat['lmass'] > sm_bins[i]) & (clean_cat['lmass'] < sm_bins[i+1])
			ypix_i = ypix[sm_mask]
			xpix_i = xpix[sm_mask]
			num_obj_zbins[i,z_i] = len(ypix_i)
			if num_obj_zbins[i,z_i] != 0:
				hitmap = np.zeros((cmap.shape),dtype=bool)
				hitmap[ypix_i,xpix_i] = True
				flux_zbins[z_i] = np.sum(cmap[hitmap])/num_obj_zbins[i,z_i]
				median_z[i,z_i] = np.median(clean_cat['stellar_mass_z'][sm_mask])
				mean_z[i,z_i] = np.mean(clean_cat['stellar_mass_z'][sm_mask])
				galaxy_pixels = cmap[ypix_i,xpix_i]
				error[z_i] = t.bootstrap(n_realizations,len(ypix_i),galaxy_pixels)
				hers_filtered_maps[i, z_i, x, :, :] = t.filtered_hitmaps(ypix_i, xpix_i, hers_filt[x][0].trans_func_grid, hers_filt[x][0].optfilt)

		hers_stack_flux[i,:, x] = flux_zbins
		hers_stack_flux_err[i,:, x] = error

	for y in range(3):
		cmap = sptsz_cmap[y]
		ypix = sptsz_ypix[y]
		xpix = sptsz_xpix[y]
		flux_zbins = np.zeros((len(z_bins[:-1])))
		error = np.zeros((len(z_bins[:-1])))
		filtered_maps = np.zeros((len(z_bins[:-1]),cmap.shape[0],cmap.shape[1]))
		for z_i in range(len(z_bins[:-1])):
			sm_mask = (clean_cat['stellar_mass_z'] > z_bins[z_i]) & (clean_cat['stellar_mass_z'] < z_bins[z_i+1]) \
				& (clean_cat['lmass'] > sm_bins[i]) & (clean_cat['lmass'] < sm_bins[i+1])
			ypix_i = ypix[sm_mask]
			xpix_i = xpix[sm_mask]
			if num_obj_zbins[i,z_i] != 0:
				hitmap = np.zeros((cmap.shape),dtype=bool)
				hitmap[ypix_i,xpix_i] = True
				flux_zbins[z_i] = np.sum(cmap[hitmap])/num_obj_zbins[i,z_i]
				galaxy_pixels = cmap[ypix_i,xpix_i]
				error[z_i] = t.bootstrap(n_realizations,len(ypix_i),galaxy_pixels)
				spt_filtered_maps[i, z_i, y, :,:] = t.filtered_hitmaps(ypix_i, xpix_i, sptsz_filt[y][0].trans_func_grid, sptsz_filt[y][0].optfilt)
		spt_stack_flux[i,:, y] = flux_zbins
		spt_stack_flux_err[i,:, y] = error

	print "finished stacking SM bin:" +str(i)

# popt, pcov =  curve_fit(t.func, (filtered_maps[0][spt_gal_mask_bool].flatten(),filtered_maps[1][spt_gal_mask_bool].flatten(),\
# 	filtered_maps[2][spt_gal_mask_bool].flatten(),filtered_maps[3][spt_gal_mask_bool].flatten()), cmap[spt_gal_mask_bool].flatten())
# spt_gal_clus_fit[i,y,:] = popt
# spt_gal_clus_fit_err[i,y,:] = t.jackknife_error_est(Npatchx=4, Npatchy=4, ypix=ypix, xpix=xpix, true_fit=popt, filtered_maps=filtered_maps, \
# 	gal_mask=spt_gal_mask, cmap=cmap)
hers_gal_clus_fit = np.zeros((len(sm_bins[:-1]),n_zbins, len(hers_mapfreq)))
hers_gal_clus_fit_err = np.zeros((len(sm_bins[:-1]),n_zbins, len(hers_mapfreq)))
spt_gal_clus_fit = np.zeros((len(sm_bins[:-1]),n_zbins, len(sptsz_mapfreq)))
spt_gal_clus_fit_err = np.zeros((len(sm_bins[:-1]),n_zbins, len(sptsz_mapfreq)))
size_maps = (len(sm_bins)-1)*(len(z_bins)-1)
guess = np.ones(size_maps)
noise_term = np.zeros((len(hers_mapfreq)+len(sptsz_mapfreq)))
for spt_x in range(3):
	cmap = sptsz_cmap[spt_x]
	ypix = sptsz_ypix[spt_x]
	xpix = sptsz_xpix[spt_x]
	spt_filtered_maps_new = np.zeros((size_maps, sptsz_cmap[spt_x].shape[0], sptsz_cmap[spt_x].shape[1]))
	for i in range(len(sm_bins)-1):
		for j in range(len(z_bins)-1):
			spt_filtered_maps_new[i*(len(z_bins)-1)+j, :, :]= spt_filtered_maps[i,j,spt_x,:,:]
	# popt,pcov = curve_fit(lambda y, *params_0: wrapper_func(y, *params_0), ([spt_filtered_maps_new[x][spt_gal_mask_bool].flatten() for x in range(size_maps)]),
	#                                     sptsz_cmap[spt_x][spt_gal_mask_bool].flatten(), p0=guess)

	popt,pcov = curve_fit(t.func1, ([spt_filtered_maps_new[x][spt_gal_mask_bool].flatten() for x in range(size_maps)]),
	                                    sptsz_cmap[spt_x][spt_gal_mask_bool].flatten())
	noise_term[spt_x] = popt[-1]
	spt_gal_clus_fit[:,:,spt_x] = popt[:-1].reshape(len(sm_bins[:-1]),n_zbins)
	spt_gal_clus_fit_err[:,:,spt_x] = (t.jackknife_error_est(Npatchx=4, Npatchy=4, ypix=ypix, xpix=xpix, true_fit=popt, filtered_maps=spt_filtered_maps_new, \
		gal_mask=spt_gal_mask, cmap=cmap))[:-1].reshape(len(sm_bins[:-1]),n_zbins)
	print "finished hitmap fitting for SPT data for " +str(spt_x) +" freq"
np.save('/data/tangq/spt_gal_clus_fit.npy', spt_gal_clus_fit)
np.save('/data/tangq/spt_gal_clus_fit_err.npy',spt_gal_clus_fit_err)
for hers_x in range(3):
	cmap = hers_cmap[hers_x]
	ypix = hers_ypix[hers_x]
	xpix = hers_xpix[hers_x]
	hers_filtered_maps_new = np.zeros((size_maps, hers_cmap[hers_x].shape[0], hers_cmap[hers_x].shape[1]))
	for i in range(len(sm_bins)-1):
		for j in range(len(z_bins)-1):
			hers_filtered_maps_new[i*(len(z_bins)-1)+j, :, :]= hers_filtered_maps[i, j,hers_x,:,:]
	popt,pcov = curve_fit(t.func1, ([hers_filtered_maps_new[x][hers_gal_mask_bool].flatten() for x in range(size_maps)]),
	                                    hers_cmap[hers_x][hers_gal_mask_bool].flatten())
	hers_gal_clus_fit[:,:,hers_x] = popt[:-1].reshape(len(sm_bins[:-1]),n_zbins)
	noise_term[hers_x+3] = popt[-1]
	hers_gal_clus_fit_err[:,:,hers_x] = (t.jackknife_error_est(Npatchx=4, Npatchy=4, ypix=ypix, xpix=xpix, true_fit=popt, filtered_maps=hers_filtered_maps_new, \
		gal_mask=hers_gal_mask, cmap=cmap))[:-1].reshape(len(sm_bins[:-1]),n_zbins)
	print "finished hitmap fitting for Herschel data for "+str(spt_x) +" freq"

np.save('/data/tangq/hers_gal_clus_fit.npy', hers_gal_clus_fit)
np.save('/data/tangq/hers_gal_clus_fit_err.npy',hers_gal_clus_fit_err)
np.save('/data/tangq/noise_term.npy',noise_term)

spt_gal_clus_fit = np.load('/data/tangq/spt_gal_clus_fit.npy')
spt_gal_clus_fit_err = np.load('/data/tangq/spt_gal_clus_fit_err.npy')
hers_gal_clus_fit = np.load('/data/tangq/hers_gal_clus_fit.npy')
hers_gal_clus_fit_err = np.load('/data/tangq/hers_gal_clus_fit_err.npy')
noise_term = np.load('/data/tangq/noise_term.npy')

eff_freq_gc_zarr = np.zeros((n_zbins, len(sptsz_mapfreq)+len(hers_mapfreq)))
gc_flux_zarr = np.zeros((n_zbins, len(sptsz_mapfreq)+len(hers_mapfreq)))
gc_flux_err_zarr = np.zeros((n_zbins, len(sptsz_mapfreq)+len(hers_mapfreq)))

for i in range(len(sm_bins)-1):
	chi2_arr_all = []
	chi2_arr_gc_all = []
	fig, axarr1 = plt.subplots(1,len(z_bins[:-1]), squeeze=True,sharex=True, sharey=False, figsize=(5*len(z_bins[:-1]),5))
	for z_i in range(len(z_bins)-1):
		stacked_flux = np.concatenate([hers_stack_flux[i,z_i,:], spt_stack_flux[i,z_i,:][::-1] ])
		stacked_flux_err = np.concatenate([ hers_stack_flux_err[i,z_i,:], spt_stack_flux_err[i,z_i,:][::-1] ])
		gal_clus_flux = np.concatenate([hers_gal_clus_fit[i,z_i,:],spt_gal_clus_fit[i,z_i,:][::-1]])
		gal_clus_flux_err = np.concatenate([hers_gal_clus_fit_err[i,z_i,:],spt_gal_clus_fit_err[i,z_i,:][::-1]])
		z = mean_z[i, z_i]
		eff_freq_arr = np.zeros((len(hers_mapfreq)+len(sptsz_mapfreq), len(T_set)))
		chi2_arr = np.zeros((len(T_set)))
		A_arr = np.zeros((len(T_set)))
		y_arr = np.zeros((len(T_set)))
		A_gc_arr = np.zeros((len(T_set)))
		y_gc_arr = np.zeros((len(T_set)))
		chi2_arr = np.zeros((len(T_set)))
		chi2_arr_gc = np.zeros((len(T_set)))
		for n, T in enumerate(T_set):
			hers_eff_freq = t.get_eff_band(herschel_lookup_table, T, z)*1.e9
			sptsz_eff_freq = t.get_eff_band(sptsz_lookup_table, T, z)*1.e9
			sptpol_eff_freq = t.get_eff_band(sptpol_lookup_table, T, z)*1.e9
			eff_freq = np.concatenate([hers_eff_freq, np.asarray([sptsz_eff_freq[-1]]), sptpol_eff_freq[::-1]])
			eff_freq_arr[:,n] = eff_freq
			popt, pcov = curve_fit(lambda p1, p2, p3: t.mod_BB_curve_with_tsz(p1, p2, p3, T, z, theor_tsz_Jy), eff_freq, stacked_flux, \
			               sigma = stacked_flux_err, p0 = [1.e-10, 1.e-8], \
			               maxfev=1000000)
			A_arr[n], y_arr[n] = popt
			chi2_arr[n] = np.sum((t.mod_BB_curve_with_tsz(eff_freq, popt[0],popt[1],T,z, theor_tsz_Jy)-stacked_flux)**2/stacked_flux_err**2/dof)
			popt, pcov = curve_fit(lambda p1, p2, p3: t.mod_BB_curve_with_tsz(p1, p2, p3, T, z, theor_tsz_Jy), eff_freq, gal_clus_flux, \
			               sigma = gal_clus_flux_err, p0 = [1.e-10, 1.e-8], \
			               maxfev=1000000)
			A_gc_arr[n], y_gc_arr[n] = popt
			chi2_arr_gc[n] = np.sum((t.mod_BB_curve_with_tsz(eff_freq, popt[0],popt[1],T,z, theor_tsz_Jy)-gal_clus_flux)**2/gal_clus_flux_err**2/dof)
				
		T_ind = np.where(chi2_arr == np.min(chi2_arr))
		T = T_set[T_ind[0]][0]
		A = A_arr[T_ind[0]][0]
		y = y_arr[T_ind[0]][0]
		eff_freq = eff_freq_arr[:,T_ind[0]][:,0]
		eff_freq_zarr[z_i,:] = eff_freq
		stacked_flux_zarr[z_i,:] = stacked_flux
		stacked_flux_err_zarr[z_i,:] = stacked_flux_err
		chi2_arr_all.append(chi2_arr)

		T_ind = np.where(chi2_arr_gc == np.min(chi2_arr_gc))
		T_gc = T_set[T_ind[0]][0]
		A_gc = A_gc_arr[T_ind[0]][0]
		y_gc = y_gc_arr[T_ind[0]][0]
		eff_freq_gc = eff_freq_arr[:,T_ind[0]][:,0]
		eff_freq_gc_zarr[z_i,:] = eff_freq_gc
		gc_flux_zarr[z_i,:] = gal_clus_flux
		gc_flux_err_zarr[z_i,:] = gal_clus_flux_err
		chi2_arr_gc_all.append(chi2_arr_gc)

		### BB SED plots
		axarr1[z_i % 4].errorbar(eff_freq, stacked_flux, yerr=stacked_flux_err, color='blue',marker = 'o',ls='none', \
			label='mean stacked flux')
		axarr1[ z_i % 4].errorbar(eff_freq_gc, gal_clus_flux, yerr=gal_clus_flux_err, color='green',marker = 'o',ls='none', \
			label='flux from hitmap fit')
		axarr1[ z_i % 4].plot(eff_freq, t.mod_BB_curve_with_tsz(eff_freq, A, y, T, z,  theor_tsz_Jy),\
			color='blue', label='A=' + '{0:1.2e}'.format(A) +',z='+"{:.2f}".format(z)+', T='+"{:.0f}".format(T)+'K,y='+'{0:1.2e}'.format(y))
		axarr1[z_i % 4].plot(eff_freq_gc, t.mod_BB_curve_with_tsz(eff_freq, A_gc, y_gc, T_gc, z, theor_tsz_Jy),\
			color='green', label='A=' + '{0:1.2e}'.format(A_gc)+',z=' +"{:.2f}".format(z)+', T='+"{:.0f}".format(T_gc)+'K,y='+'{0:1.2e}'.format(y_gc))
		axarr1[z_i % 4].legend(loc='best', prop={'size': 10}, framealpha=0.5)
#		axarr1[int(z_i/4), z_i % 4].set_xlim(100., 3500.)
		axarr1[z_i % 4].set_title('z='+str(z)+',Nobj='+str(num_obj_zbins[i,z_i]))
	axarr1[0].set_ylabel('Flux')
#	plt.xscale('log')
	axarr1[(z_i % 4)].set_xlabel('Freq (Hz)')
	fig.suptitle('BB curvefit to galaxy stacked flux at SM=' +str((sm_bins[i]+sm_bins[i+1])/2.))
	plt.savefig('/data/tangq/blackbody_plots/'+today+'_gal_stacking_SM'+str(i)+'_no_smooth.png')
	plt.close()
		#--------- for 1D plots of chi2 over T for different z ---------##
	fig, axarr1 = plt.subplots(1,len(z_bins[:-1]), squeeze=True,sharex=True, sharey=False, figsize=(5*len(z_bins[:-1]),5))
	for z_i in range(len(z_bins)-1):
		axarr1[z_i % 4].plot(T_set, chi2_arr_all[z_i], label='z'+"{:.2f}".format(mean_z[i,z_i]))
		axarr1[z_i % 4].set_title(str(z_bins[z_i])+'< z <'+str(z_bins[z_i+1])+',Nobj=' + str(num_obj_zbins[i, z_i]))
		axarr1[z_i % 4].set_xlabel('T')
		axarr1[z_i % 4].legend(loc='best', ncol=2, fontsize='small', framealpha=0.5)
	axarr1[0].set_ylabel('chi2')
	fig.suptitle('reduced chi2 to BB curvefit to galaxy AVG stacked flux at SM=' +str((sm_bins[i]+sm_bins[i+1])/2.)+',binned in redshifts')
	plt.savefig('/data/tangq/blackbody_plots/'+today+'gal_stacking_SM'+str(i)+'chi2_BBfits_1D.png')
	plt.close()

	fig, axarr1 = plt.subplots(1,len(z_bins[:-1]), squeeze=True,sharex=True, sharey=False, figsize=(5*len(z_bins[:-1]),5))
	for z_i in range(len(z_bins)-1):
		axarr1[z_i % 4].plot(T_set, chi2_arr_gc_all[z_i], label='z'+"{:.2f}".format(mean_z[i,z_i]))
		axarr1[ z_i % 4].set_title(str(z_bins[z_i])+'< z <'+str(z_bins[z_i+1])+',Nobj=' + str(num_obj_zbins[i, z_i]))
		axarr1[z_i % 4].set_xlabel('T')
		axarr1[z_i % 4].legend(loc='best', ncol=2, fontsize='x-small', framealpha=0.5)
	axarr1[0].set_ylabel('chi2')
	fig.suptitle('reduced chi2 to BB curvefit to galaxy hitmap fitted flux at SM=' +str((sm_bins[i]+sm_bins[i+1])/2.)+',binned in redshifts')
	plt.savefig('/data/tangq/blackbody_plots/'+today+'gal_hitmap_fit_SM'+str(i)+'chi2_BBfits_1D.png')
	plt.close()
	#np.save('/home/tangq/chi2_array_SM'+str(i)+'.npy', chi2_arr)
	print "SM Bin " + str(i)

plt.figure()
plt.plot(np.ones((15))*eff_freq[-1], np.ravel(spt_stack_flux[:,:,0]/spt_gal_clus_fit[:,:,0]),'.')
plt.plot(np.ones((15))*eff_freq[-2],np.ravel(spt_stack_flux[:,:,1]/spt_gal_clus_fit[:,:,1]),'.')
plt.plot(np.ones((15))*eff_freq[-3],np.ravel(spt_stack_flux[:,:,2]/spt_gal_clus_fit[:,:,2]),'.')
plt.plot(np.ones((15))*eff_freq[0],np.ravel(hers_stack_flux[:,:,0]/hers_gal_clus_fit[:,:,0]),'.')
plt.plot(np.ones((15))*eff_freq[1],np.ravel(hers_stack_flux[:,:,1]/hers_gal_clus_fit[:,:,1]),'.')
plt.plot(np.ones((15))*eff_freq[2],np.ravel(hers_stack_flux[:,:,2]/hers_gal_clus_fit[:,:,2]),'.')
plt.legend(loc='best')
plt.xlabel('Freq')
plt.ylabel('stacked flux:hitmap fitting flux')

sum_filtered_map3 = np.zeros((hers_cmap[0].shape))
for i in range(len(sm_bins[:-1])):
    for j in range(len(z_bins[:-1])):
        sum_filtered_map3 += (hers_gal_clus_fit[i,j,0]*hers_filtered_maps[i, j,0,:,:]-4.14656331e-05)
res_map = hers_cmap[0] -  sum_filtered_map3

plt.subplot(121)
plt.imshow(res_map)
plt.colorbar()
plt.title('Res map')
plt.subplot(122)
plt.imshow(res_map[2000:2200,2000:2200])
plt.colorbar()
plt.title('Res map (zoom into centre)')
