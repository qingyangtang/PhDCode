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

#### cleaning out any galaxies in the area off the main observing area
mask = clean_cat['ra']<357.
clean_cat = clean_cat[mask]

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

#### Hershel

make_Herschel_filtered_map = False
normalize_solid_angles = False

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

#making galaxy mask

spt_gal_hitmap = np.zeros((sptsz_cmap[1].shape))
spt_gal_hitmap[sptsz_ypix[1],sptsz_xpix[1]] = 1.
reso = (0.25/60)*np.pi/180.
ellgrid,ellx5,elly5 = makeEllGrids([sptsz_cmap[1].shape[0],sptsz_cmap[1].shape[1]],reso)
bfilt = calc_gauss_Bl(ellgrid,5.0)
spt_gal_mask = np.fft.ifft2(np.fft.fft2(spt_gal_hitmap)*bfilt).real
spt_gal_mask[spt_gal_mask>=0.005] = 1.
spt_gal_mask[spt_gal_mask<0.005] = 0.

for i in range(3):
	sptsz_cmap[i] = sptsz_cmap[i]*spt_gal_mask

hers_gal_hitmap = np.zeros((map_arr[0].shape))
hers_gal_hitmap[hers_ypix[0],hers_xpix[0]] = 1.
reso = (reso_arcmin_arr[0]/60)*np.pi/180.
ellgrid,ellx5,elly5 = makeEllGrids([map_arr[0].shape[0],map_arr[0].shape[1]],reso)
bfilt = calc_gauss_Bl(ellgrid,5.0)
hers_gal_mask = np.fft.ifft2(np.fft.fft2(hers_gal_hitmap)*bfilt).real
hers_gal_mask[hers_gal_mask>0.005] = 1.
hers_gal_mask[hers_gal_mask<0.005] = 0.

for i in range(3):
	map_arr[i] = map_arr[i]*hers_gal_mask

sm_mask =  (clean_cat['lmass'] > 10) & (clean_cat['lmass'] < 11) #(clean_cat['stellar_mass_z'] > 0.5) & (clean_cat['stellar_mass_z'] <0.6)
sm_list = clean_cat['lmass'][sm_mask]
z_list = clean_cat['stellar_mass_z'][sm_mask]
ypix_i = sptsz_ypix[1][sm_mask]
xpix_i = sptsz_xpix[1][sm_mask]
ylow = np.min(ypix_i)
yhigh = np.max(ypix_i)
xlow = np.min(xpix_i)
xhigh = np.max(xpix_i)
num_obj_zbins = len(ypix_i)

trans_func_beam = idl.readIDLSav('/data/tangq/SPT/SZ/ra23h30dec-55_2year_xfer_function_with_beam_padded_map.sav').transfer_with_beam[1]
optfilt = np.load('/data/tangq/SPT/SZ/products/SPTSZ_filtered_no_smooth_FTfixed_20210401_150_filter.npy').item()[0].optfilt

z_groups = np.asarray([0, 0.66, 1., 1.33, 2.])
z_objs = np.zeros((len(z_groups)-1))
hitmaps = np.zeros((len(z_groups)-1,cmap.shape[0],cmap.shape[1]))
fft_maps = np.zeros((hitmaps.shape), dtype = complex)
fft_maps_TF = np.zeros((hitmaps.shape),dtype=complex)
real_maps = np.zeros((hitmaps.shape))
filtered_maps = np.zeros((hitmaps.shape))
flux_zbins = np.zeros((len(z_groups)-1))
flux_zbins_med = np.zeros((len(z_groups)-1))
n_realizations = 100
error = np.zeros((len(z_groups)-1))
spt_filt_flux = np.zeros((len(z_groups)-1))

for z_i in range(len(z_groups)-1):
	z_mask = (z_list >z_groups[z_i]) & (z_list <z_groups[z_i+1])
	ypix_z = ypix_i[z_mask]
	xpix_z = xpix_i[z_mask]
	z_objs[z_i] = len(ypix_z)
	for i in range(len(ypix_z)):
		hitmaps[z_i][ypix_z[i],xpix_z[i]] += 1
	fft_maps[z_i] = np.fft.fft2(hitmaps[z_i])
	fft_maps_TF[z_i] = fft_maps[z_i]*trans_func_beam
	real_maps[z_i] = np.fft.ifft2(fft_maps_TF[z_i]).real
	filtered_maps[z_i] = np.fft.ifft2(fft_maps_TF[z_i]*optfilt).real
	spt_filt_flux[z_i] = np.sum(filtered_maps[z_i][ypix_z,xpix_z])/z_objs[z_i]
	filtered_maps[z_i] = filtered_maps[z_i]*spt_gal_mask/spt_filt_flux[z_i]

	flux_zbins[z_i] = np.sum(sptsz_cmap[1][ypix_z,xpix_z])/z_objs[z_i]
	flux_zbins_med[z_i] = np.median(sptsz_cmap[1][ypix_z,xpix_z])
	galaxy_pixels = sptsz_cmap[1][ypix_z,xpix_z]
	error[z_i] = t.bootstrap(n_realizations,len(ypix_z),galaxy_pixels)

def func(X, a, b, c, d):
	x,y,z,xx = X
	return a*x+b*y+c*z+d*xx


popt, pcov =  curve_fit(func, (filtered_maps[0].flatten(),filtered_maps[1].flatten(),filtered_maps[2].flatten(),filtered_maps[3].flatten()), sptsz_cmap[1].flatten())
a,b,c,d= popt

#### estimating the error
Npatchx = 4
Npatchy = 4
patchsize = np.max(sptsz_ypix)-np.min(sptsz_ypix),np.max(sptsz_xpix)-np.min(sptsz_xpix) 
real_patchsize = (np.round(patchsize[0]/Npatchy)+1)*Npatchy, (np.round(patchsize[1]/Npatchx)+1)*Npatchx
centre_pixel = np.int(np.mean((np.max(sptsz_ypix),np.min(sptsz_ypix)))), np.int(np.mean((np.max(sptsz_xpix),np.min(sptsz_xpix))))
small_patch = real_patchsize[0]/Npatchy, real_patchsize[1]/Npatchx
jk_fits = np.zeros((Npatchy*Npatchx, len(popt)))
jk_mask = np.zeros((len(jk_fits[:,0])),dtype=bool)

sptsz_cmap_mod = np.zeros((3, sptsz_cmap[0].shape[0], sptsz_cmap[0].shape[1]))
filtered_maps_mod = np.zeros((len(z_groups)-1,filtered_maps.shape[1],filtered_maps.shape[2]))
for i in range(Npatchy):
	for j in range(Npatchx):
		pixel_place = centre_pixel[0]-small_patch[0]*Npatchy/2 + i*small_patch[0], centre_pixel[1]-small_patch[1]*Npatchx/2 + j*small_patch[1]

		filtered_maps_mod[0] = filtered_maps[0]
		filtered_maps_mod[1] = filtered_maps[1]
		filtered_maps_mod[2] = filtered_maps[2]
		filtered_maps_mod[3] = filtered_maps[3]
		filtered_maps_mod[:, pixel_place[0]:pixel_place[0]+small_patch[0], pixel_place[1]:pixel_place[1]+small_patch[1]] = 0

		sptsz_cmap_mod[0,:,:] = sptsz_cmap[0]
		sptsz_cmap_mod[1,:,:] = sptsz_cmap[1]
		sptsz_cmap_mod[2,:,:] = sptsz_cmap[2]
		sptsz_cmap_mod[0][pixel_place[0]:pixel_place[0]+small_patch[0], pixel_place[1]:pixel_place[1]+small_patch[1]] = 0
		sptsz_cmap_mod[1][pixel_place[0]:pixel_place[0]+small_patch[0], pixel_place[1]:pixel_place[1]+small_patch[1]] = 0
		sptsz_cmap_mod[2][pixel_place[0]:pixel_place[0]+small_patch[0], pixel_place[1]:pixel_place[1]+small_patch[1]] = 0

		popt, pcov =  curve_fit(func, (filtered_maps_mod[0].flatten(),filtered_maps_mod[1].flatten(),filtered_maps_mod[2].flatten(),filtered_maps_mod[3].flatten()), sptsz_cmap_mod[1].flatten())
		jk_fits[i*Npatchy+j,:] = popt

		mask_patch = np.sum(spt_gal_mask[pixel_place[0]:pixel_place[0]+small_patch[0], pixel_place[1]:pixel_place[1]+small_patch[1]]) > 0.5*small_patch[0]*small_patch[1]
		jk_mask[i*Npatchy+j] = mask_patch

jk_fits_mean = np.mean(jk_fits, axis=0)
jk_var = (len(jk_fits[:,0])-1.)/((len(jk_fits[:,0])))*np.sum((jk_fits[jk_mask,:]-np.asarray([a,b,c,d]))**2, axis=0)


x = (z_groups[1:]+z_groups[:-1])/2
plt.errorbar(x, flux_zbins,yerr = error,label='Stacked average flux')
plt.errorbar(x, [a,b,c,d], yerr=np.sqrt(jk_var), label='flux from fitter')
#plt.semilogy()
plt.xlabel('z')
plt.ylabel('Flux')
plt.legend(loc='best')
plt.title('SPT 150GHz map')

res = sptsz_cmap[1] - flux_zbins[0]*filtered_maps[0] - flux_zbins[1]*filtered_maps[1] - flux_zbins[2]*filtered_maps[2] - flux_zbins[3]*filtered_maps[3]

plt.figure()
plt.subplot(121)
plt.imshow(res)
plt.colorbar()
plt.title('Res SPT map (entire map), sum|res| = ' + str(np.sum(np.abs(res))))
plt.subplot(122)
reso = (0.25/60)*np.pi/180.
ellgrid,ellx5,elly5 = makeEllGrids([sptsz_cmap[1].shape[0],sptsz_cmap[1].shape[1]],reso)
x, y = plotting.plotRadialProfile(np.abs(np.fft.fftshift(np.fft.fft2(res).real)), center_bin=[res.shape[1]/2,res.shape[0]/2], resolution=(ellx5[0,1]-ellx5[0,0]), histogram=True, range=[0,20000],plot=True, bin_resolution_factor=0.1,new=False, label="whole map")
plt.legend(loc='best')
plt.title('Radially averaged FFT of SPT residual map')
plt.xlabel('ell')
###herschel 

beamfile = idl.readIDLSav('/data/tangq/SPIRE_maps/spire_beams_for_matched_filter.sav')
beam = beamfile['spire_beams']
beams = []
beam_x = beam[0]
reso_rad = reso_arcmin_arr[0] * constants.DTOR / 60.
ellg,ellx,elly = makeEllGrids([npixels[1],npixels[0]],reso_rad)
bl2d = beam_x[(np.round(ellg)).astype(int)]
beams.append(bl2d)

sm_mask =  (clean_cat['lmass'] > 10) & (clean_cat['lmass'] < 11) #(clean_cat['stellar_mass_z'] > 0.5) & (clean_cat['stellar_mass_z'] <0.6)
sm_list = clean_cat['lmass'][sm_mask]
z_list = clean_cat['stellar_mass_z'][sm_mask]
ypix_i = hers_ypix[0][sm_mask]
xpix_i = hers_xpix[0][sm_mask]
num_obj_zbins = len(ypix_i)
ylow = np.min(ypix_i)
yhigh = np.max(ypix_i)
xlow = np.min(xpix_i)
xhigh = np.max(xpix_i)

trans_func_file = idl.readIDLSav('/data/tangq/SPT/SZ/ra23h30dec-55_2year_xfer_function_no_beam_padded_map.sav')
trans_func = trans_func_file.tfgrids_nobeam[2]
fname_unsmoothed = '/data/tangq/SPIRE_maps/products/Herschel_filtered_no_smooth_20201130'
filt250_0 = np.load(fname_unsmoothed+'_250_filter.npy').item()
optfilt = filt250_0[0].optfilt


z_groups = np.asarray([0, 0.66, 1., 1.33, 2.])
z_objs = np.zeros((len(z_groups)-1))
hitmaps = np.zeros((len(z_groups)-1,cmap.shape[0],cmap.shape[1]))
fft_maps = np.zeros((hitmaps.shape), dtype = complex)
fft_maps_TF = np.zeros((hitmaps.shape),dtype=complex)
real_maps = np.zeros((hitmaps.shape))
filtered_maps = np.zeros((hitmaps.shape))
flux_zbins = np.zeros((len(z_groups)-1))
flux_zbins_med = np.zeros((len(z_groups)-1))
n_realizations = 100
error = np.zeros((len(z_groups)-1))
hers_filt_flux = np.zeros((len(z_groups)-1))

for z_i in range(len(z_groups)-1):
	z_mask = (z_list >z_groups[z_i]) & (z_list <z_groups[z_i+1])
	ypix_z = ypix_i[z_mask]
	xpix_z = xpix_i[z_mask]
	z_objs[z_i] = len(ypix_z)
	for i in range(len(ypix_z)):
		hitmaps[z_i][ypix_z[i],xpix_z[i]] += 1
	fft_maps[z_i] = np.fft.fft2(hitmaps[z_i])
	fft_maps_TF[z_i] = fft_maps[z_i]*bl2d*trans_func
	real_maps[z_i] = np.fft.ifft2(fft_maps_TF[z_i]).real
	filtered_maps[z_i] = np.fft.ifft2(fft_maps_TF[z_i]*optfilt).real
	hers_filt_flux[z_i] = np.sum(filtered_maps[z_i][ypix_z,xpix_z])/z_objs[z_i]

	filtered_maps[z_i] = filtered_maps[z_i]*hers_gal_mask/hers_filt_flux[z_i]

	flux_zbins[z_i] = np.sum(map_arr[0][ypix_z,xpix_z])/z_objs[z_i]
	flux_zbins_med[z_i] = np.median(map_arr[0][ypix_z,xpix_z])
	galaxy_pixels = map_arr[0][ypix_z,xpix_z]
	error[z_i] = t.bootstrap(n_realizations,len(ypix_z),galaxy_pixels)

def func(X, a, b, c, d):
	x,y,z,xx = X
	return np.abs(a)*x+np.abs(b)*y+np.abs(c)*z+np.abs(d)*xx

popt, pcov =  curve_fit(func, (filtered_maps[0].flatten(),filtered_maps[1].flatten(),filtered_maps[2].flatten(),filtered_maps[3].flatten()), map_arr[0].flatten())
a,b,c,d= popt

Npatchx = 4
Npatchy = 4
patchsize = np.max(hers_ypix)-np.min(hers_ypix),np.max(hers_xpix)-np.min(hers_xpix) 
real_patchsize = (np.round(patchsize[0]/Npatchy)+1)*Npatchy, (np.round(patchsize[1]/Npatchx)+1)*Npatchx
centre_pixel = np.int(np.mean((np.max(hers_ypix),np.min(hers_ypix)))), np.int(np.mean((np.max(hers_xpix),np.min(hers_xpix))))
small_patch = real_patchsize[0]/Npatchy, real_patchsize[1]/Npatchx
jk_fits = np.zeros((Npatchy*Npatchx, len(popt)))
jk_mask = np.zeros((len(jk_fits[:,0])),dtype=bool)

hers_cmap_mod = np.zeros((3, map_arr[0].shape[0], map_arr[0].shape[1]))
filtered_maps_mod = np.zeros((len(z_groups)-1,filtered_maps.shape[1],filtered_maps.shape[2]))
for i in range(Npatchy):
	for j in range(Npatchx):
		pixel_place = centre_pixel[0]-small_patch[0]*Npatchy/2 + i*small_patch[0], centre_pixel[1]-small_patch[1]*Npatchx/2 + j*small_patch[1]

		filtered_maps_mod[0] = filtered_maps[0]
		filtered_maps_mod[1] = filtered_maps[1]
		filtered_maps_mod[2] = filtered_maps[2]
		filtered_maps_mod[3] = filtered_maps[3]
		filtered_maps_mod[:, pixel_place[0]:pixel_place[0]+small_patch[0], pixel_place[1]:pixel_place[1]+small_patch[1]] = 0

		hers_cmap_mod[0,:,:] = map_arr[0]
		hers_cmap_mod[1,:,:] = map_arr[1]
		hers_cmap_mod[2,:,:] = map_arr[2]
		hers_cmap_mod[0][pixel_place[0]:pixel_place[0]+small_patch[0], pixel_place[1]:pixel_place[1]+small_patch[1]] = 0
		hers_cmap_mod[1][pixel_place[0]:pixel_place[0]+small_patch[0], pixel_place[1]:pixel_place[1]+small_patch[1]] = 0
		hers_cmap_mod[2][pixel_place[0]:pixel_place[0]+small_patch[0], pixel_place[1]:pixel_place[1]+small_patch[1]] = 0

		popt, pcov =  curve_fit(func, (filtered_maps_mod[0].flatten(),filtered_maps_mod[1].flatten(),filtered_maps_mod[2].flatten(),filtered_maps_mod[3].flatten()), hers_cmap_mod[0].flatten())
		jk_fits[i*Npatchy+j,:] = popt
		mask_patch = np.sum(spt_gal_mask[pixel_place[0]:pixel_place[0]+small_patch[0], pixel_place[1]:pixel_place[1]+small_patch[1]]) > 0.5*small_patch[0]*small_patch[1]
		jk_mask[i*Npatchy+j] = mask_patch

jk_fits_mean = np.mean(jk_fits, axis=0)
jk_var = (len(jk_fits[:,0])-1.)/((len(jk_fits[:,0])))*np.sum((jk_fits[jk_mask,:]-np.asarray([a,b,c,d]))**2, axis=0)

x = (z_groups[1:]+z_groups[:-1])/2
plt.errorbar(x, flux_zbins,yerr = error,label='Stacked average flux')
plt.errorbar(x, [a,b,c,d], yerr=np.sqrt(jk_var), label='flux from fitter')
plt.xlabel('z')
plt.ylabel('Flux')
plt.legend(loc='best')
plt.title('Herschel 250um map')

res = map_arr[0] - flux_zbins[0]*filtered_maps[0] - flux_zbins[1]*filtered_maps[1] - flux_zbins[2]*filtered_maps[2] - flux_zbins[3]*filtered_maps[3]
plt.figure()
plt.subplot(121)
plt.imshow(res)
plt.colorbar()
plt.title('Res Herschel map (entire map), sum|res| = ' + str(np.sum(np.abs(res))))

plt.subplot(122)
reso = (0.25/60)*np.pi/180.
ellgrid,ellx5,elly5 = makeEllGrids([map_arr[0].shape[0],map_arr[0].shape[1]],reso)
x, y = plotting.plotRadialProfile(np.abs(np.fft.fftshift(np.fft.fft2(res).real)), center_bin=[res.shape[1]/2,res.shape[0]/2], resolution=(ellx5[0,1]-ellx5[0,0]), histogram=True, range=[0,20000],plot=True, bin_resolution_factor=0.1, new=False, label="whole map")
plt.legend(loc='best')
plt.title('Radially averaged FFT of Herschel residual map')
plt.xlabel('ell')

