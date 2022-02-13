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
import sptpol_software.constants as constants
from sptpol_software.util.math import makeEllGrids
import sptpol_software.analysis.plotting as plotting
from sptpol_software.simulation.beams import calc_gauss_Bl

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



### loading the data ###

fname = '/data/tangq/SPT/SZ/products/SPTSZ_filtered_no_smooth_FTfixed_20210401'
fnamepol = '/data/tangq/SPT/SPTpol/products/SPTpol_filtered_no_smooth_20210510'
map90 = np.load(fnamepol + '_90_map.npy')
map150 = np.load(fnamepol + '_150_map.npy')
map220 = np.load(fname + '_220_map.npy')
sptsz_pixelmask = np.load(fname + '_zp_mask.npy')
sptsz_reso_arcmin = np.loadtxt(fname + '_reso_arcmin.txt')
sptsz_radec = np.loadtxt(fname + '_radec.txt')
sptsz_cmap = [map90, map150, map220]
sptsz_npixels = [np.asarray(map90.shape),np.asarray(map150.shape), np.asarray(map220.shape)]
sptsz_mapfreq = ["90", "150", "220"]
filt90 = np.load(fnamepol+'_90_filter.npy').item()
filt150 = np.load(fnamepol+'_150_filter.npy').item()
filt220 = np.load(fname+'_220_filter.npy').item()
sptsz_filt = [filt90, filt150, filt220]
#spt_herschel_factor = np.asarray([ 86.63667295,  89.79231335,  95.52984136])*1.e6/1.e3/1.e3
spt_K2Jy_conversion = np.zeros((len(sptsz_cmap)))
for x in range(len(sptsz_cmap)):
	spt_K2Jy_conversion[x], dummy = t.calc_conversion_factor(sptsz_mapfreq[x], sptsz_reso_arcmin[x], sptsz_filt[x])
	sptsz_cmap[x] = sptsz_cmap[x]*spt_K2Jy_conversion[x]

fname = '/data/tangq/SPIRE_maps/products/Herschel_filtered_no_smooth_FTfixed_20210502'
map250 = np.load(fname+'_250_map.npy')
map350 = np.load(fname+'_350_map.npy')
map500 = np.load(fname+'_500_map.npy')
zp_mask = np.load(fname+'_zp_mask.npy')
reso_arcmin_arr = np.loadtxt(fname+'_reso_arcmin.txt')
radec0_arr = np.loadtxt(fname+'_radec.txt')
map_arr = [map250, map350, map500]
npixels_arr = [np.asarray(map250.shape),np.asarray(map350.shape), np.asarray(map500.shape)]
mapfreq_arr = ["250", "350", "500"]
filt250 = np.load(fname+'_250_filter.npy').item()
filt350 = np.load(fname+'_350_filter.npy').item()
filt500 = np.load(fname+'_500_filter.npy').item()
hers_filt = [filt250, filt350, filt500]
hers_tsz_K2Jy_conversion = np.zeros((len(sptsz_cmap)))
hers_tsz_eff_freq = t.tsz_eff_freq_hers()
spt_tsz_eff_freq = np.concatenate([t.tsz_eff_freq_sptpol(), np.asarray([t.tsz_eff_freq_sptsz()[-1]])])
spt_tsz_K2Jy_conversion = np.zeros((len(sptsz_cmap)))
spt_solid_angle = np.zeros((len(sptsz_cmap)))
hers_solid_angle = np.zeros((len(sptsz_cmap)))
for x in range(len(sptsz_cmap)):
	hers_tsz_K2Jy_conversion[x], hers_solid_angle[x] = t.calc_conversion_factor(hers_tsz_eff_freq[x], reso_arcmin_arr[x], hers_filt[x])
	spt_tsz_K2Jy_conversion[x], spt_solid_angle[x] = t.calc_conversion_factor(spt_tsz_eff_freq[x], sptsz_reso_arcmin[x], sptsz_filt[x])

theor_tsz_in_T = t.theor_tsz_hers_spt_in_T(use_sptpol = True)
theor_tsz_Jy = np.concatenate([theor_tsz_in_T[:3]*hers_tsz_K2Jy_conversion, theor_tsz_in_T[3:]*spt_K2Jy_conversion[::-1]])

### catalog
cluster_cat = fits.open('/data/tangq/SPT/SPTpol/sptpol100d_catalog_huang19.fits')[1].data
mask = cluster_cat['redshift'] > 0
cluster_ra = cluster_cat['RA'][mask]
cluster_dec = cluster_cat['DEC'][mask]
cluster_z = cluster_cat['redshift'][mask]

spt_ypix, spt_xpix = sky.ang2Pix(np.asarray([cluster_ra, cluster_dec]), sptsz_radec[0], sptsz_reso_arcmin[0], sptsz_npixels[0],proj=0)[0]
spt_ypixmask, spt_xpixmask = sky.ang2Pix(np.asarray([cluster_ra, cluster_dec]), sptsz_radec[0], sptsz_reso_arcmin[0], sptsz_npixels[0],proj=0)[1] 
hers_ypix, hers_xpix = sky.ang2Pix(np.asarray([cluster_ra, cluster_dec]), radec0_arr[0], reso_arcmin_arr[0], npixels_arr[0],proj=0)[0]
hers_ypixmask, hers_xpixmask = sky.ang2Pix(np.asarray([cluster_ra, cluster_dec]), radec0_arr[0], reso_arcmin_arr[0], npixels_arr[0],proj=0)[1] 
ang2pix_mask = (spt_ypixmask == True) & (spt_xpixmask == True) & (hers_ypixmask == True) & (hers_xpixmask == True) & \
				(sptsz_pixelmask[spt_ypix,spt_xpix]>0.99999999) & (zp_mask[hers_ypix,hers_xpix]>0.99999999)
spt_ypix = spt_ypix[ang2pix_mask]
spt_xpix = spt_xpix[ang2pix_mask]
hers_ypix = hers_ypix[ang2pix_mask]
hers_xpix = hers_xpix[ang2pix_mask]
cluster_z = cluster_z[ang2pix_mask]


#plt.figure()
n_realizations = 50
herschel_lookup_table = np.load('/data/tangq/SPIRE_maps/Herschel_effective_bands.npy')
sptsz_lookup_table = np.load('/data/tangq/wendymaps/SPTSZ_effective_bands.npy')
sptpol_lookup_table = np.load('/data/tangq/SPT/SPTpol/SPTpol_effective_bands.npy')
z_bins = np.arange(np.min(cluster_z),np.max(cluster_z)+0.4, 0.4)
T_set = np.arange(3, 45., 1.)
dof = np.float(len(bands))+np.float(len(sptsz_mapfreq))-1.
nobj = np.zeros((len(z_bins[:-1])))
eff_freq_zarr = np.zeros((len(z_bins)-1,len(mapfreq_arr)+len(sptsz_mapfreq)))
stacked_flux_zarr = np.zeros((len(z_bins)-1,len(mapfreq_arr)+len(sptsz_mapfreq)))
stacked_flux_err_zarr = np.zeros((len(z_bins)-1,len(mapfreq_arr)+len(sptsz_mapfreq)))

for z_i in range(len(z_bins[:-1])):
	z_mask = (cluster_z > z_bins[z_i]) & (cluster_z < z_bins[z_i+1])
	nobj[z_i] = len(cluster_z[z_mask])
	spt_cluster_flux = np.zeros((len(sptsz_mapfreq)))
	spt_cluster_ferr = np.zeros((len(sptsz_mapfreq)))
	for i in range(len(sptsz_mapfreq)):
		spt_cluster_flux[i] = np.sum(sptsz_cmap[i][spt_ypix[z_mask],spt_xpix[z_mask]])/nobj[z_i]
		spt_cluster_ferr[i] = t.bootstrap(n_realizations,len(spt_ypix[z_mask]),sptsz_cmap[i][spt_ypix[z_mask],spt_xpix[z_mask]])
	hers_cluster_flux = np.zeros((len(mapfreq_arr)))
	hers_cluster_ferr = np.zeros((len(mapfreq_arr)))
	for i in range(len(mapfreq_arr)):
		hers_cluster_flux[i] = np.sum(map_arr[i][hers_ypix[z_mask],hers_xpix[z_mask]])/nobj[z_i]
		hers_cluster_ferr[i] = t.bootstrap(n_realizations,len(hers_ypix[z_mask]),map_arr[i][hers_ypix[z_mask],hers_xpix[z_mask]])

	z_set = np.arange(z_bins[z_i],z_bins[z_i+1],0.1)
	eff_freq_arr = np.zeros((len(mapfreq_arr)+len(sptsz_mapfreq), len(z_set), len(T_set)))
	stacked_flux = np.concatenate([hers_cluster_flux, spt_cluster_flux[::-1]])
	stacked_flux_err = np.concatenate([hers_cluster_ferr, spt_cluster_ferr[::-1]])
	A_arr = np.zeros((len(z_set),len(T_set)))
	y_arr = np.zeros((len(z_set),len(T_set)))
	chi2_arr = np.zeros((len(z_set),len(T_set)))
	for m, z in enumerate(z_set):
		for n, T in enumerate(T_set):
			hers_eff_freq = t.get_eff_band(herschel_lookup_table, T, z)*1.e9
			sptsz_eff_freq = t.get_eff_band(sptsz_lookup_table, T, z)*1.e9
			sptpol_eff_freq = t.get_eff_band(sptpol_lookup_table, T, z)*1.e9
			eff_freq = np.concatenate([hers_eff_freq, np.asarray([sptsz_eff_freq[-1]]), sptpol_eff_freq[::-1]])
			eff_freq_arr[:,m,n] = eff_freq
			popt, pcov = curve_fit(lambda p1, p2, p3: t.mod_BB_curve_with_tsz(p1, p2, p3, T, z, theor_tsz_Jy), eff_freq, stacked_flux, \
			               sigma = stacked_flux_err, p0 = [1.e-10, 1.], \
			               maxfev=1000000)
			A_arr[m,n], y_arr[m,n] = popt
			chi2_arr[m,n] = np.sum((t.mod_BB_curve_with_tsz(eff_freq, popt[0],popt[1],T,z, theor_tsz_Jy)-stacked_flux)**2/stacked_flux_err**2/dof)
	### find what m and n are
	z_ind,T_ind = np.where(chi2_arr == np.min(chi2_arr))
	z = z_set[z_ind[0]]
	T = T_set[T_ind[0]]
	A = A_arr[z_ind[0],T_ind[0]]
	y = y_arr[z_ind[0],T_ind[0]]
	eff_freq = eff_freq_arr[:,z_ind[0],T_ind[0]]

	eff_freq_zarr[z_i,:] = eff_freq
	stacked_flux_zarr[z_i,:] = stacked_flux
	stacked_flux_err_zarr[z_i,:] = stacked_flux_err

# 	plt.subplot(2,2,z_i+1)
# 	plt.errorbar(eff_freq, stacked_flux,yerr=stacked_flux_err, label='data', ls='none')
# 	plt.plot(eff_freq, t.mod_BB_curve_with_tsz(eff_freq, A, y, T, z, theor_tsz_Jy), label='full fit')
# 	plt.plot(eff_freq, popt[1]*theor_tsz_Jy, ls='--',label='t-sz fit')
# 	plt.plot(eff_freq, t.mod_BB_curve_with_z(eff_freq, A,T, z),ls='--',label='BB fit')
# #	plt.errorbar(sptsz_eff_freq_zarr[z_i], sptsz_stacked_flux_zarr[z_i], yerr=sptsz_stacked_flux_err_zarr[z_i], label='SPT-SZ data', ls='none')
# 	plt.legend(loc='best')
# 	plt.xlabel('Freq')
# 	plt.ylabel('flux (Jy)')
# 	plt.title('t-sz + BB fit,Nobj='+str(nobj[z_i])+',A='+ '{0:1.2e}'.format(A)+ ',T='+str(T)+',z='+'{0:.2f}'.format(z)+',y='+'{0:1.2e}'.format(popt[0]))
# 	plt.xlim(10.e9,250.e9)
sptpol_eff_freq_zarr = eff_freq_zarr
sptpol_stacked_flux_zarr = stacked_flux_zarr
sptpol_stacked_flux_err_zarr = stacked_flux_err_zarr

solid_angles = np.concatenate([hers_solid_angle, spt_solid_angle[::-1]])
theor_tsz_Jy_sr = theor_tsz_Jy/solid_angles
#plt.figure()
n_realizations = 100
herschel_lookup_table = np.load('/data/tangq/SPIRE_maps/Herschel_effective_bands.npy')
sptsz_lookup_table = np.load('/data/tangq/wendymaps/SPTSZ_effective_bands.npy')
z_bins = np.arange(np.min(cluster_z),np.max(cluster_z)+0.4, 0.4)
T_set = np.arange(3, 45., 1.)
dof = np.float(len(bands))+np.float(len(sptsz_mapfreq))-1.
nobj = np.zeros((len(z_bins[:-1])))
#for z_i in range(len(z_bins[:-1])):
z_i = 0
z_mask = (cluster_z > z_bins[z_i]) & (cluster_z < z_bins[z_i+1])
nobj[z_i] = len(cluster_z[z_mask])
spt_cluster_flux = np.zeros((len(sptsz_mapfreq)))
spt_cluster_ferr = np.zeros((len(sptsz_mapfreq)))
for i in range(len(sptsz_mapfreq)):
	spt_cluster_flux[i] = np.mean(sptsz_cmap[i][spt_ypix[z_mask],spt_xpix[z_mask]])
	spt_cluster_ferr[i] = t.bootstrap(n_realizations,len(spt_ypix[z_mask]),sptsz_cmap[i][spt_ypix[z_mask],spt_xpix[z_mask]])
hers_cluster_flux = np.zeros((len(mapfreq_arr)))
hers_cluster_ferr = np.zeros((len(mapfreq_arr)))
for i in range(len(mapfreq_arr)):
	hers_cluster_flux[i] = np.mean(map_arr[i][hers_ypix[z_mask],hers_xpix[z_mask]])
	hers_cluster_ferr[i] = t.bootstrap(n_realizations,len(hers_ypix[z_mask]),map_arr[i][hers_ypix[z_mask],hers_xpix[z_mask]])

z_set = np.arange(z_bins[z_i],z_bins[z_i+1],0.1)
eff_freq_arr = np.zeros((len(mapfreq_arr)+len(sptsz_mapfreq), len(z_set), len(T_set)))
stacked_flux = np.concatenate([hers_cluster_flux, spt_cluster_flux[::-1]])
mod_stacked_flux = stacked_flux/solid_angles
mod_stacked_flux_err = np.concatenate([hers_cluster_ferr, spt_cluster_ferr[::-1]])/solid_angles
A_arr = np.zeros((len(z_set),len(T_set)))
y_arr = np.zeros((len(z_set),len(T_set)))
chi2_arr = np.zeros((len(z_set),len(T_set)))
for m, z in enumerate(z_set):
	for n, T in enumerate(T_set):
		hers_eff_freq = t.get_eff_band(herschel_lookup_table, T, z)*1.e9
		sptsz_eff_freq = t.get_eff_band(sptsz_lookup_table, T, z)*1.e9
		eff_freq = np.concatenate([hers_eff_freq, sptsz_eff_freq[::-1]])
		eff_freq_arr[:,m,n] = eff_freq
		popt, pcov = curve_fit(lambda p1, p2, p3: t.mod_BB_curve_with_tsz(p1, p2, p3, T, z, theor_tsz_Jy_sr), eff_freq, mod_stacked_flux, \
		               sigma = mod_stacked_flux_err, p0 = [1.e-10*1.e8, 1.], \
		               maxfev=1000000)
		A_arr[m,n], y_arr[m,n] = popt
		chi2_arr[m,n] = np.sum((t.mod_BB_curve_with_tsz(eff_freq, popt[0],popt[1],T,z, theor_tsz_Jy_sr)-mod_stacked_flux)**2/mod_stacked_flux_err**2/dof)
### find what m and n are
z_ind,T_ind = np.where(chi2_arr == np.min(chi2_arr))
z = z_set[z_ind[0]]
T = T_set[T_ind[0]]
A = A_arr[z_ind[0],T_ind[0]]
y = y_arr[z_ind[0],T_ind[0]]
eff_freq = eff_freq_arr[:,z_ind[0],T_ind[0]]

#plt.subplot(2,2,z_i+1)
xval = np.arange(50,1200,1.)*1.e9
x = (xval/1.e9)*1.e9*const.h/(const.k*Tcmb)
theor_tsz_Jy_full = t.calc_theor_tsz_T(xval/1.e9)*1.e26*(2*const.k/(const.c)**2)*((const.k*Tcmb/const.h)**2)*(x**4)*np.exp(x)/(np.exp(x)-1)**2
plt.figure()
plt.plot(xval/1.e9, t.mod_BB_curve_with_tsz(xval, A, y, T, z, theor_tsz_Jy_full)/1.e3, lw = 2, color='purple', label='Combined fit')
plt.plot((xval+4)/1.e9, y*theor_tsz_Jy_full/1.e3, ls='--', color = 'blue',lw = 2, label='tSZ fit')
plt.plot(xval/1.e9, t.mod_BB_curve_with_z(xval, A,T, z)/1.e3,ls='--', color='red',lw = 2,label='BB fit')
plt.scatter(eff_freq/1.e9, mod_stacked_flux/1.e3,color='black',label='Stacked fluxes')
plt.errorbar(eff_freq/1.e9, mod_stacked_flux/1.e3,yerr=mod_stacked_flux_err/1.e3, color='black',lw = 2, ls='none')
plt.legend(loc='best')
plt.xlabel('Frequency (GHz)')
plt.ylabel('Intensity/beam (kJy/sr)')
plt.xlim(xval[0]/1.e9, xval[-1]/1.e9)
plt.text(100, 50, r'$\mathregular{N_{clus}}$='+str(nobj[z_i])+'\n'+'{0:.2f}'.format(z_bins[z_i])+'<z<'+'{0:.2f}'.format(z_bins[z_i+1])+'\nT='+str(T)+'K')
plt.title('Stacked SED of SPTpol clusters')
plt.savefig('/data/tangq/SED_SPTpol_clusters.png')
#plt.title('t-sz + BB fit,Nobj='+str(nobj[z_i])+',A='+ '{0:1.2e}'.format(A)+ ',T='+str(T)+',z='+'{0:.2f}'.format(z)+',y='+'{0:1.2e}'.format(popt[0]))


# herschel_lookup_table = np.load('/data/tangq/SPIRE_maps/Herschel_effective_bands.npy')
# sptsz_lookup_table = np.load('/data/tangq/wendymaps/SPTSZ_effective_bands.npy')
# z_set = np.asarray([np.mean(cluster_z)])
# T_set = np.arange(3, 45., 1.)
# chi2_arr = np.zeros((len(z_set),len(T_set)))
# eff_freq_arr = np.zeros((len(mapfreq_arr)+len(sptsz_mapfreq), len(z_set), len(T_set)))
# A_arr = np.zeros((len(z_set),len(T_set)))
# y_arr = np.zeros((len(z_set),len(T_set)))
# dof = np.float(len(bands))+np.float(len(sptsz_mapfreq))-1.
# mod_stacked_flux = stacked_flux/solid_angles
# mod_stacked_flux_err = stacked_flux_err/solid_angles
# theor_tsz_Jy_sr = theor_tsz_Jy/solid_angles
# for m, z in enumerate(z_set):
#     for n, T in enumerate(T_set):
# 		hers_eff_freq = t.get_eff_band(herschel_lookup_table, T, z)*1.e9
# 		sptsz_eff_freq = t.get_eff_band(sptsz_lookup_table, T, z)*1.e9
# 		eff_freq = np.concatenate([hers_eff_freq, sptsz_eff_freq[::-1]])
# 		eff_freq_arr[:,m,n] = eff_freq
# 		popt, pcov = curve_fit(lambda p1, p2, p3: mod_BB_curve_with_tsz(p1, p2, p3, T, z, theor_tsz_Jy_sr), eff_freq, mod_stacked_flux, \
# 		               sigma = mod_stacked_flux_err, p0 = [1.e-10*1.e8, 1.], \
# 		               maxfev=1000000)
# 		A_arr[m,n], y_arr[m,n] = popt
# 		chi2_arr[m,n] = np.sum((mod_BB_curve_with_tsz(eff_freq, popt[0],popt[1],T,z, theor_tsz_Jy_sr)-mod_stacked_flux)**2/mod_stacked_flux_err**2/dof)
# ### find what m and n are
# z_ind,T_ind = np.where(chi2_arr == np.min(chi2_arr))
# z = z_set[z_ind[0]]
# T = T_set[T_ind[0]]
# A = A_arr[z_ind[0],T_ind[0]]
# y = y_arr[z_ind[0],T_ind[0]]
# eff_freq = eff_freq_arr[:,z_ind[0],T_ind[0]]

# xval = np.arange(50,1200,1.)*1.e9
# x = (xval/1.e9)*1.e9*const.h/(const.k*Tcmb)
# theor_tsz_Jy_full = t.calc_theor_tsz_T(xval/1.e9)*1.e26*(2*const.k/(const.c)**2)*((const.k*Tcmb/const.h)**2)*(x**4)*np.exp(x)/(np.exp(x)-1)**2
# plt.errorbar(eff_freq, mod_stacked_flux,yerr=mod_stacked_flux_err, label='data', ls='none')
# plt.plot(xval, mod_BB_curve_with_tsz(xval, A, y, T, z, theor_tsz_Jy_full), label='full fit')
# plt.plot(xval+4, popt[1]*theor_tsz_Jy_full, ls='--',label='t-sz fit')
# plt.plot(xval, t.mod_BB_curve_with_z(xval, A,T, z),ls='--',label='BB fit')
# plt.legend(loc='best')
# plt.xlabel('Freq')
# plt.ylabel('Jy/solid angle')
# plt.title('t-sz + BB fit,A='+ '{0:1.2e}'.format(A)+ ',T='+str(T)+',z='+'{0:.2f}'.format(z)+',y='+'{0:1.2e}'.format(popt[0]))




