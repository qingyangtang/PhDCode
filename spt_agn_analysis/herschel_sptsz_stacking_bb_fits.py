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



sp_des_dir = '/data/tangq/custom_catalogs/spitzer_des_cat.fits'
stellar_m_dir = '/data/bleeml/stellar_mass_cats/ssdf_des_try1.fout'

clean_cat = od.get_Spitzer_DES_cat_with_mass(sp_des_dir,stellar_m_dir,iband_cut=23)


########## getting rid of galaxies within 4 arcmin of point sources
SPIRE_ptsrc_dir = "/data/tangq/SPIRE_maps/spt_starfndr_catalog.sav"

hers_src_ra, hers_src_dec, hers_src_flux = od.get_SPIRE_ptsrcs(SPIRE_ptsrc_dir, sn_cut=40)

sptsz_src_ra, sptsz_src_dec, sptsz_src_rad = od.get_SPTSZ_ptsrcs('/data/tangq/SPT/SZ/ptsrc_config_ra23h30dec-55_surveyclusters.txt')

src_ra = np.concatenate([hers_src_ra,sptsz_src_ra])
src_dec = np.concatenate([hers_src_dec,sptsz_src_dec])

idx, d2d = t.match_src_to_catalog(clean_cat['ra'], clean_cat['dec'], src_ra, src_dec)
src_mask = d2d.arcmin > 4
clean_cat = clean_cat[src_mask]

############SPT-SZ
make_SPTSZ_filtered_map = False


if make_SPTSZ_filtered_map:
	sptsz_cmap, sptsz_reso_arcmin,sptsz_radec,sptsz_npixels, sptsz_mapfreq, sptsz_pixelmask = od.make_SPTSZ_filtered_map(save_map = True, \
		save_map_dir='/data/tangq/SPT/SZ/products/', save_map_fname='SPTSZ_filtered_1_75arcmin_beam', common_beam = 1.75)	
else:
	fname = '/data/tangq/SPT/SZ/products/SPTSZ_filtered_no_smooth_FTfixed_20210401'
	map90 = np.load(fname + '_90_map.npy')
	map150 = np.load(fname + '_150_map.npy')
	map220 = np.load(fname + '_220_map.npy')
	sptsz_pixelmask = np.load(fname + '_zp_mask.npy')
	sptsz_reso_arcmin = np.loadtxt(fname + '_reso_arcmin.txt')
	sptsz_radec = np.loadtxt(fname + '_radec.txt')
	sptsz_cmap = [map90, map150, map220]
	sptsz_npixels = [np.asarray(map90.shape),np.asarray(map150.shape), np.asarray(map220.shape)]
	sptsz_mapfreq = ["90", "150", "220"]

spt_herschel_factor = np.asarray([ 86.63667295,  89.79231335,  95.52984136])*1.e6/1.e3/1.e3
for x in range(len(sptsz_cmap)):
	sptsz_cmap[x] = sptsz_cmap[x]*spt_herschel_factor[x]

#### cleaning out any galaxies in masked out places

sptsz_ypix = []
sptsz_xpix = []
for y in range(3):
	pixelmask = sptsz_pixelmask
	radec0 = sptsz_radec[y]
	reso_arcmin = sptsz_reso_arcmin[y]
	npixels = sptsz_npixels[y]

	ypix, xpix = sky.ang2Pix(np.asarray([clean_cat['ra'], clean_cat['dec']]), radec0, reso_arcmin, npixels,proj=0)[0]
	ypixmask, xpixmask = sky.ang2Pix(np.asarray([clean_cat['ra'], clean_cat['dec']]), radec0, reso_arcmin, npixels,proj=0)[1] 
	ang2pix_mask = (ypixmask == True) & (xpixmask == True) & (pixelmask[ypix,xpix]>0.99999999)
	clean_cat = clean_cat[ang2pix_mask]
	ypix = ypix[ang2pix_mask]
	xpix = xpix[ang2pix_mask]
	sptsz_ypix.append(ypix)
	sptsz_xpix.append(xpix)

########## Hershel

make_Herschel_filtered_map = False
normalize_solid_angles = True

if make_Herschel_filtered_map:
	map_arr, reso_arcmin_arr, radec0_arr, npixels_arr, mapfreq_arr, zp_mask = od.make_Herschel_filtered_map(save_map = True, \
		save_map_dir='/data/tangq/SPIRE_maps/products/', save_map_fname='Herschel_filtered_1_75arcmin_beam', common_beam = 1.75)
else:
	fname = '/data/tangq/SPIRE_maps/products/Herschel_filtered_no_smooth_20201130'
	map250 = np.load(fname+'_250_map.npy')
	map350 = np.load(fname+'_350_map.npy')
	map500 = np.load(fname+'_500_map.npy')
	zp_mask = np.load(fname+'_zp_mask.npy')
	reso_arcmin_arr = np.loadtxt(fname+'_reso_arcmin.txt')
	radec0_arr = np.loadtxt(fname+'_radec.txt')
	map_arr = [map250, map350, map500]
	npixels_arr = [np.asarray(map250.shape),np.asarray(map350.shape), np.asarray(map500.shape)]
	mapfreq_arr = ["250", "350", "500"]
if normalize_solid_angles == True:
	filt250 = np.load(fname+'_250_filter.npy').item()
	filt350 = np.load(fname+'_350_filter.npy').item()
	filt500 = np.load(fname+'_500_filter.npy').item()
	filt = np.asarray([filt250, filt350, filt500])
	fname_unsmoothed = '/data/tangq/SPIRE_maps/products/Herschel_filtered_no_smooth_20201130'
	filt250_0 = np.load(fname_unsmoothed+'_250_filter.npy').item()
	filt350_0 = np.load(fname_unsmoothed+'_350_filter.npy').item()
	filt500_0 = np.load(fname_unsmoothed+'_500_filter.npy').item()
	filt_unsmoothed = np.asarray([filt250_0, filt350_0, filt500_0])
	for x in range(3):
		reso = (reso_arcmin_arr[x]/60)*np.pi/180.
		dngrid = map_arr[x].shape[0]
		dk = 1./dngrid/reso
		tauprime = filt[x][0].prof2d*filt[x][0].trans_func_grid #* filt_output[0].area_eff
		solid_angle =1./ np.sum(filt[x][0].optfilt*tauprime)  / dk**2

		tauprime0 = filt_unsmoothed[x][0].prof2d*filt_unsmoothed[x][0].trans_func_grid
		solid_angle0 =1./ np.sum(filt_unsmoothed[x][0].optfilt*tauprime0)  / dk**2
		map_arr[x] = map_arr[x]*(solid_angle/solid_angle0)

#### cleaning out any galaxies in masked out places
hers_ypix = []
hers_xpix = []
for x in range(3):
	cmap = map_arr[x]
	reso_arcmin = reso_arcmin_arr[x]
	radec0 = radec0_arr[x]
	npixels = npixels_arr[x]
	ypix, xpix = sky.ang2Pix(np.asarray([clean_cat['ra'], clean_cat['dec']]), radec0, reso_arcmin, npixels,proj=0)[0]
	ypixmask, xpixmask = sky.ang2Pix(np.asarray([clean_cat['ra'], clean_cat['dec']]), radec0, reso_arcmin, npixels,proj=0)[1] 
	ang2pix_mask = (ypixmask == True) & (xpixmask == True) & (zp_mask[ypix,xpix]>0.99999999)
	clean_cat = clean_cat[ang2pix_mask]
	ypix = ypix[ang2pix_mask]
	xpix = xpix[ang2pix_mask]
	hers_ypix.append(ypix)
	hers_xpix.append(xpix)


######

sm_bins = np.arange(9,11.2,0.2)

herschel_lookup_table = np.load('/data/tangq/SPIRE_maps/Herschel_effective_bands.npy')
sptsz_lookup_table = np.load('/data/tangq/wendymaps/SPTSZ_effective_bands.npy')
flux_250 = np.zeros((len(sm_bins[:-1])))
error_250 = np.zeros((len(sm_bins[:-1])))
num_obj_250 = np.zeros((len(sm_bins[:-1])))
flux_350 = np.zeros((len(sm_bins[:-1])))
error_350 = np.zeros((len(sm_bins[:-1])))
num_obj_350 = np.zeros((len(sm_bins[:-1])))
flux_500 = np.zeros((len(sm_bins[:-1])))
error_500 = np.zeros((len(sm_bins[:-1])))
num_obj_500 = np.zeros((len(sm_bins[:-1])))
T_set = np.arange(3, 45., 0.1)
dof = np.float(len(bands))+np.float(len(sptsz_mapfreq))-1.

clean_cat_master = clean_cat

for i in range(len(sm_bins)-1):
	for x in range(3):
		cmap = map_arr[x]
		ypix = hers_ypix[x]
		xpix = hers_xpix[x]
		mapfreq = mapfreq_arr[x]
		sm_mask_dummy = ((clean_cat['lmass'] > sm_bins[i]) & (clean_cat['lmass'] < sm_bins[i+1]))
		z_bins = np.arange(np.min(clean_cat['stellar_mass_z'][sm_mask_dummy]),np.min([np.max(clean_cat['stellar_mass_z'][sm_mask_dummy]),1.99])+0.2, 0.2)
		flux_zbins = np.zeros((len(z_bins[:-1])))
		num_obj_zbins = np.zeros((len(z_bins[:-1])))
		n_realizations = 1000
		error = np.zeros((len(z_bins[:-1])))
		median_z = np.zeros((len(z_bins[:-1])))
		for z_i in range(len(z_bins)-1):
			sm_mask = (clean_cat['stellar_mass_z'] > z_bins[z_i]) & (clean_cat['stellar_mass_z'] < z_bins[z_i+1]) \
				& (clean_cat['lmass'] > sm_bins[i]) & (clean_cat['lmass'] < sm_bins[i+1])
			ypix_i = ypix[sm_mask]
			xpix_i = xpix[sm_mask]
			num_obj_zbins[z_i] = len(ypix_i)
			flux_zbins[z_i] = np.sum(cmap[ypix_i,xpix_i])/num_obj_zbins[z_i]
			median_z[z_i] = np.median(clean_cat['stellar_mass_z'][sm_mask])
			galaxy_pixels = cmap[ypix_i,xpix_i]
			error[z_i] = t.bootstrap(n_realizations,len(ypix_i),galaxy_pixels)
		if mapfreq == "250":
			flux_250 = flux_zbins
			error_250 = error
			num_obj_250 = num_obj_zbins
	 	elif mapfreq == "350":
			flux_350 = flux_zbins
			error_350 = error
			num_obj_350 = num_obj_zbins
		elif mapfreq == "500":
			flux_500 = flux_zbins
			error_500 = error
			num_obj_500 = num_obj_zbins


	for y in range(3):
		cmap = sptsz_cmap[y]
		ypix = sptsz_ypix[y]
		xpix = sptsz_xpix[y]
		sm_mask_dummy = ((clean_cat['lmass'] > sm_bins[i]) & (clean_cat['lmass'] < sm_bins[i+1]))
		z_bins = np.arange(np.min(clean_cat['stellar_mass_z'][sm_mask_dummy]),np.min([np.max(clean_cat['stellar_mass_z'][sm_mask_dummy]),1.99])+0.2, 0.2)
		flux_zbins = np.zeros((len(z_bins[:-1])))
		num_obj_zbins = np.zeros((len(z_bins[:-1])))
		n_realizations = 1000
		error = np.zeros((len(z_bins[:-1])))
		median_z = np.zeros((len(z_bins[:-1])))
		for z_i in range(len(z_bins)-1):
			sm_mask = (clean_cat['stellar_mass_z'] > z_bins[z_i]) & (clean_cat['stellar_mass_z'] < z_bins[z_i+1]) \
				& (clean_cat['lmass'] > sm_bins[i]) & (clean_cat['lmass'] < sm_bins[i+1])
			ypix_i = ypix[sm_mask]
			xpix_i = xpix[sm_mask]
			num_obj_zbins[z_i] = len(ypix_i)
			flux_zbins[z_i] = np.mean(cmap[ypix_i,xpix_i])
			median_z[z_i] = np.median(clean_cat['stellar_mass_z'][sm_mask])
			galaxy_pixels = cmap[ypix_i,xpix_i]
			error[z_i] = t.bootstrap(n_realizations,len(ypix_i),galaxy_pixels)
		if y==0:
			flux_90 = flux_zbins
			error_90 = error
			num_obj_90 = num_obj_zbins
	 	elif y==1:
			flux_150 = flux_zbins
			error_150 = error
			num_obj_150 = num_obj_zbins
		elif y==2:
			flux_220 = flux_zbins
			error_220 = error
			num_obj_220 = num_obj_zbins

	### STRAIGHTFORWARD FITS USING THE Z GIVEN FROM REDSHFIT BINS
	figpsd1, axarr1 = plt.subplots(int(ceil(len(z_bins)/4.)),4, squeeze=True,sharex=True, sharey=False, figsize=(20,5*int(ceil(len(z_bins)/4.))))
	all_bands = np.concatenate([bands,c_c/np.asarray([220.e9,150.e9,90.e9])])
	freqs = c_c/all_bands
	for z_i in range(len(z_bins)-1):
		z = median_z[z_i]
		stacked_flux_err = np.asarray([ error_250[z_i], error_350[z_i], error_500[z_i], error_220[z_i], error_150[z_i], error_90[z_i] ])
		stacked_flux = np.asarray([flux_250[z_i], flux_350[z_i], flux_500[z_i] ,flux_220[z_i], flux_150[z_i], flux_90[z_i]])
		#plt.errorbar(bands, [flux_250[z_i], flux_350[z_i], flux_500[z_i]], \
		#	yerr=stacked_flux_err, marker = 'o',lw=1, label='Data')
		#z = (sm_bins[i]+sm_bins[i+1])/2.
		axarr1[int(z_i/4), z_i % 4].errorbar(all_bands*1.e6, stacked_flux, yerr=stacked_flux_err, marker = 'o',lw=1, \
			label='z='+str(z)+',Nobj='+str(num_obj_zbins[z_i]))
		popt, pcov = curve_fit(lambda p1, p2, p3: t.mod_BB_curve_with_z(p1, p2, p3, z), freqs, stacked_flux, \
		                   sigma = stacked_flux_err, p0 = [1.e-11, 30.], \
		                   maxfev=1000000)#bounds = ([0, 0], [1., 100]))
		axarr1[int(z_i/4), z_i % 4].plot(all_bands*1.e6, t.mod_BB_curve_with_z(freqs, popt[0], popt[1], z), \
			label='A=' + '{0:1.2e}'.format(popt[0]) +', T='+"{:.0f}".format(popt[1])+'K')
		axarr1[int(z_i/4), z_i % 4].legend(loc='best', prop={'size': 10})
		axarr1[int(z_i/4), z_i % 4].set_xlim(100., 3500.)
	axarr1[0,0].set_ylabel('Flux')
	plt.xscale('log')
	axarr1[int(z_i/4), (z_i % 4)].set_xlabel('Wavelength (um)')
	axarr1[0,1].set_title('BB curvefit to galaxy stacked flux at SM=' +str((sm_bins[i]+sm_bins[i+1])/2.))

	plt.savefig('/data/tangq/blackbody_plots/'+today+'_gal_stacking_SM'+str(i)+'_no_smooth.png')
	plt.close()


	fig, axarr1 = plt.subplots(int(ceil((len(z_bins)-1)/4.)),4, squeeze=True,sharex=True, sharey=False, figsize=(20,5*int(ceil((len(z_bins)-1)/4.))))
	chi2_arr = np.zeros((len(z_bins)-1,21,len(T_set)))
	for z_i in range(len(z_bins)-1):
		stacked_flux_err = np.asarray([ error_250[z_i], error_350[z_i], error_500[z_i], error_220[z_i], error_150[z_i], error_90[z_i] ])
		stacked_flux = np.asarray([flux_250[z_i], flux_350[z_i], flux_500[z_i], flux_220[z_i], flux_150[z_i], flux_90[z_i]])
		z_set = np.arange(z_bins[z_i],z_bins[z_i+1],0.01)
		#z_set = np.arange(0.1,2,0.05)

		colors = plt.cm.jet(np.linspace(0,1,len(z_set)))
		chi2_arr = np.zeros((len(z_set),len(T_set)))
		for m, z in enumerate(z_set):
		    for n, T in enumerate(T_set):
		    	hers_eff_freq = t.get_eff_band(herschel_lookup_table, T, z)*1.e9
		    	sptsz_eff_freq = t.get_eff_band(sptsz_lookup_table, T, z)*1.e9
		    	eff_freq = np.concatenate([hers_eff_freq, sptsz_eff_freq[::-1]])
		        popt, pcov = curve_fit(lambda p1, p2: t.mod_BB_curve_with_z(p1, p2, T, z), eff_freq, stacked_flux, \
		                       sigma = stacked_flux_err, p0 = [1.e-10], \
		                       maxfev=10000)
		        chi2_arr[m,n] = np.sum((t.mod_BB_curve_with_z(eff_freq, popt[0],T,z)-stacked_flux)**2/stacked_flux_err**2/dof)

	# 	##--------- for 2D images of chi2 ---------##
	# 	im = axarr1[int(z_i/4), z_i % 4].imshow(chi2_arr, interpolation=None,vmin=0,vmax=min([1.e6, np.max(chi2_arr)]),extent=[z_set[0], z_set[-1], T_set[-1], T_set[0]])
	# 	axarr1[int(z_i/4), z_i % 4].set_aspect(chi2_arr.shape[0]/float(chi2_arr.shape[1]))
	# 	plt.colorbar(im,ax=axarr1[int(z_i/4), z_i % 4])
	# 	axarr1[int(z_i/4), z_i % 4].vlines(z_bins[z_i], min(T_set), max(T_set))
	# 	axarr1[int(z_i/4), z_i % 4].vlines(z_bins[z_i+1], min(T_set), max(T_set))
	# axarr1[0,0].set_ylabel('Temperature (K)')
	# axarr1[int(z_i/4), z_i % 4].set_xlabel('redshift')
	# fig.suptitle('reduced chi2 to BB curvefit to galaxy stacked flux at SM=' +str((sm_bins[i]+sm_bins[i+1])/2.)+',binned in redshifts,Nobj=' + str(num_obj_zbins[z_i]))
	# plt.savefig('/data/tangq/blackbody_plots/'+today+'Herschel_gal_stacking_SM'+str(i)+'chi2_BBfits.png')
	# plt.close()

		#--------- for 1D plots of chi2 over T for different z ---------##
		colors = plt.cm.jet(np.linspace(0,1,len(z_set)))
		for m in range(len(z_set)):
			axarr1[int(z_i/4), z_i % 4].plot(T_set, chi2_arr[m,:], color=colors[m], label='z'+"{:.2f}".format(z_set[m]))
		axarr1[int(z_i/4), z_i % 4].set_title(str(z_bins[z_i])+'< z <'+str(z_bins[z_i+1])+',Nobj=' + str(num_obj_zbins[z_i]))
		axarr1[int(z_i/4), z_i % 4].set_xlabel('T')
		axarr1[int(z_i/4), z_i % 4].legend(loc='best', ncol=2, fontsize='small')
	axarr1[0,0].set_ylabel('chi2')
	fig.suptitle('reduced chi2 to BB curvefit to galaxy stacked flux at SM=' +str((sm_bins[i]+sm_bins[i+1])/2.)+',binned in redshifts')
	plt.savefig('/data/tangq/blackbody_plots/'+today+'gal_stacking_SM'+str(i)+'chi2_BBfits_1D.png')
	plt.close()

	#np.save('/home/tangq/chi2_array_SM'+str(i)+'.npy', chi2_arr)
	print "SM Bin " + str(i)

print "min redshift: " +str(min_z)+  ", max redshift: " + str(max_z)

plt.figure()
plt.plot(sm_bins, num_obj_zbins)
plt.xlabel('Stellar mass')
plt.ylabel('Num objs')
plt.title('z < 0.5, map: ' + mapfreq)
plt.figure()
plt.errorbar(sm_bins[:-1], flux_zbins[:-1], yerr=error[:, 0:-1], fmt='o', label='median flux')
plt.xlabel('stellar mass')
plt.ylabel('Flux mJy')
plt.title('z < 0.5, map: ' + mapfreq)
plt.grid(True)

