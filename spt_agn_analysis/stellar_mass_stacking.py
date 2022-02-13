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


sp_des_dir = '/data/tangq/custom_catalogs/spitzer_des_cat.fits'
stellar_m_dir = '/data/bleeml/stellar_mass_cats/ssdf_des_try1.fout'

#### SPT stacking:

#for mapfreq in ["90", "150", "220"]:
mapfreq = "90"
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
del sp_des_cat, stellar_cat, col_z, col_lmass, col_chi2

mask = sp_des_sm_cat['i'] < 23
clean_cat = sp_des_sm_cat[mask]

field = "ra23h30dec-55_allyears"
idldata = idl.readIDLSav("/data/tangq/wendymaps/"+field+"/maps_planck15cal_allyears_"+mapfreq+".sav")
pixelmask = idldata.pixel_mask
radec0 = idldata.radec0
reso_arcmin = idldata.reso_arcmin
npixels = np.asarray(idldata.cmap_mjy.shape)

cmap = idldata.cmap_mjy
if npixels[0] > npixels[1]:
	y_coords = np.arange(npixels[0])
	x_coords = np.zeros((len(y_coords)))
	x_coords[:npixels[1]] = np.arange(npixels[1])
	x_coords[npixels[1]:] = np.arange(npixels[1])[-1]
elif npixels[0] < npixels[1]:
	x_coords = np.arange(npixels[1])
	y_coords = np.zeros((len(x_coords)))
	y_coords[:npixels[0]] = np.arange(npixels[0])
	y_coords[npixels[0]:] = np.arange(npixels[0])[-1]
else:	
	y_coords = np.arange(npixels[0]); x_coords=np.arange(npixels[1])
y_coords = y_coords.astype(int)
x_coords = x_coords.astype(int)
ra_coords, dec_coords = sky.pix2Ang(np.asarray([y_coords,x_coords]), radec0, reso_arcmin, npixels,proj=5)
ra_edge = [ra_coords[0], ra_coords[-1]]
dec_edge = [dec_coords[0],dec_coords[-1]]

source_mask = '/data/tangq/wendymaps/'+field+'/source_list_3sig_allyears_'+mapfreq+'.dat'
source_list = np.loadtxt(source_mask)
src_ra = source_list[:,3]
src_dec = source_list[:,4]
src_flux = source_list[:,6]

ypix, xpix = sky.ang2Pix(np.asarray([clean_cat['ra'], clean_cat['dec']]), radec0, reso_arcmin, npixels,proj=5)[0]
ypixmask, xpixmask = sky.ang2Pix(np.asarray([clean_cat['ra'], clean_cat['dec']]), radec0, reso_arcmin, npixels,proj=5)[1] 
ang2pix_mask = (ypixmask == True) & (xpixmask == True) & (pixelmask[ypix,xpix]==1)
clean_cat = clean_cat[ang2pix_mask]

c = SkyCoord(ra=clean_cat['ra']*u.degree, dec=clean_cat['dec']*u.degree)  
catalog = SkyCoord(ra=src_ra*u.degree, dec=src_dec*u.degree)  
idx, d2d, d3d = c.match_to_catalog_sky(catalog)

src_mask = d2d.arcmin > 4
ypix = ypix[src_mask]
xpix = xpix[src_mask]
d2d = d2d[src_mask]
idx = idx[src_mask]
clean_cat = clean_cat[src_mask]

nearest_src_flux = src_flux[idx]
gtr100srcmask = np.invert((nearest_src_flux > 100) & (d2d.arcminute < 8))
ypix = ypix[gtr100srcmask]
xpix = xpix[gtr100srcmask]
d2d = d2d[gtr100srcmask]
idx = idx[gtr100srcmask]
clean_cat= clean_cat[gtr100srcmask]

# randpixdata = np.load('/data/tangq/wendymaps/rand_pix_flux_freq_' + mapfreq + '.npy')
# nreal = 100
# LZLM_mask = (clean_cat['stellar_mass_z'] < 0.5) & (clean_cat['lmass'] > 0 ) & (clean_cat['lmass'] < 10)
# ypix_lzlm = ypix[LZLM_mask]
# xpix_lzlm = xpix[LZLM_mask]
# flux_lzlm = np.mean(cmap[ypix_lzlm, xpix_lzlm])
# map_lzlm = np.zeros((40,40))
# for i in range(len(ypix_lzlm)):
# 	map_lzlm += cmap[ypix_lzlm[i]-20:ypix_lzlm[i]+20, xpix_lzlm[i]-20:xpix_lzlm[i]+20]
# print "for LOW redshift LOW stellar mass (z<0.5, SM < 10)"
# print "Number of objects is: " + str(len(ypix_lzlm))
# print "Mean flux is: " + str(flux_lzlm)
# rand_mean = np.zeros((nreal))
# rand_med = np.zeros((nreal))
# for j in range(nreal):
# 	randpix = randpixdata[(np.round(np.random.rand(len(ypix_lzlm))*(len(randpixdata)-1))).astype(int)]
# 	rand_mean[j] = np.mean(randpix)
# 	rand_med[j] = np.median(randpix)
# print "The signal to noise is (mean): " + str(np.mean(cmap[ypix_lzlm,xpix_lzlm])/np.std(rand_mean))
# print "The signal to noise is (median): " + str(np.median(cmap[ypix_lzlm,xpix_lzlm])/np.std(rand_med))

# LZHM_mask = (clean_cat['stellar_mass_z'] < 0.5) & (clean_cat['lmass'] > 10 ) & (clean_cat['lmass'] < 15)
# ypix_lzhm = ypix[LZHM_mask]
# xpix_lzhm = xpix[LZHM_mask]
# flux_lzhm = np.mean(cmap[ypix_lzhm, xpix_lzhm])
# print "for LOW redshift HIGH stellar mass (z<0.5, SM > 10)"
# print "Number of objects is: " + str(len(ypix_lzhm))
# print "Mean flux is: " + str(flux_lzhm)
# map_lzhm = np.zeros((40,40))
# for i in range(len(ypix_lzhm)):
# 	map_lzhm += cmap[ypix_lzhm[i]-20:ypix_lzhm[i]+20, xpix_lzhm[i]-20:xpix_lzhm[i]+20]
# map_lzhm = map_lzhm/len(ypix_lzhm)
# rand_mean = np.zeros((nreal))
# rand_med = np.zeros((nreal))
# for j in range(nreal):
# 	randpix = randpixdata[(np.round(np.random.rand(len(ypix_lzhm))*(len(randpixdata)-1))).astype(int)]
# 	rand_mean[j] = np.mean(randpix)
# 	rand_med[j] = np.median(randpix)
# print "The signal to noise is (mean): " + str(np.mean(cmap[ypix_lzhm,xpix_lzhm])/np.std(rand_mean))
# print "The signal to noise is (median): " + str(np.median(cmap[ypix_lzhm,xpix_lzhm])/np.std(rand_med))

# HZLM_mask = (clean_cat['stellar_mass_z'] > 0.5) & (clean_cat['stellar_mass_z'] < 1) & (clean_cat['lmass'] > 0 ) & (clean_cat['lmass'] < 10)
# ypix_hzlm = ypix[HZLM_mask]
# xpix_hzlm = xpix[HZLM_mask]
# flux_hzlm = np.mean(cmap[ypix_hzlm, xpix_hzlm])
# print "for HIGH redshift LOW stellar mass (0.5<z<1, SM < 10)"
# print "Number of objects is: " + str(len(ypix_hzlm))
# print "Mean flux is: " + str(flux_hzlm)
# map_hzlm = np.zeros((40,40))
# for i in range(len(ypix_hzlm)):
# 	map_hzlm += cmap[ypix_hzlm[i]-20:ypix_hzlm[i]+20, xpix_hzlm[i]-20:xpix_hzlm[i]+20]
# rand_mean = np.zeros((nreal))
# rand_med = np.zeros((nreal))
# for j in range(nreal):
# 	randpix = randpixdata[(np.round(np.random.rand(len(ypix_hzlm))*(len(randpixdata)-1))).astype(int)]
# 	rand_mean[j] = np.mean(randpix)
# 	rand_med[j] = np.median(randpix)
# print "The signal to noise is (mean): " + str(np.mean(cmap[ypix_hzlm,xpix_hzlm])/np.std(rand_mean))
# print "The signal to noise is (median): " + str(np.median(cmap[ypix_hzlm,xpix_hzlm])/np.std(rand_med))

# HZHM_mask = (clean_cat['stellar_mass_z'] > 0.5) & (clean_cat['stellar_mass_z'] < 1) & (clean_cat['lmass'] > 10 ) & (clean_cat['lmass'] < 15)
# ypix_hzhm = ypix[HZHM_mask]
# xpix_hzhm = xpix[HZHM_mask]
# flux_hzhm = np.mean(cmap[ypix_hzhm, xpix_hzhm])
# print "for HIGH redshift HIGH stellar mass (0.5<z<1, SM > 10)"
# print "Number of objects is: " + str(len(ypix_hzhm))
# print "Mean flux is: " + str(flux_hzhm)
# map_hzhm = np.zeros((40,40))
# for i in range(len(ypix_hzhm)):
# 	map_hzhm += cmap[ypix_hzhm[i]-20:ypix_hzhm[i]+20, xpix_hzhm[i]-20:xpix_hzhm[i]+20]
# rand_mean = np.zeros((nreal))
# rand_med = np.zeros((nreal))
# for j in range(nreal):
# 	randpix = randpixdata[(np.round(np.random.rand(len(ypix_hzhm))*(len(randpixdata)-1))).astype(int)]
# 	rand_mean[j] = np.mean(randpix)
# 	rand_med[j] = np.median(randpix)
# print "The signal to noise is (mean): " + str(np.mean(cmap[ypix_hzhm,xpix_hzhm])/np.std(rand_mean))
# print "The signal to noise is (median): " + str(np.median(cmap[ypix_hzhm,xpix_hzhm])/np.std(rand_med))

#	del clean_cat

lz_mask = (clean_cat['stellar_mass_z'] < 0.5)
ypix_lz = ypix[lz_mask]
xpix_lz = xpix[lz_mask]
clean_cat_lz = clean_cat[lz_mask]

sm_bins = np.arange(8.5,11.5,0.1)
sm_bins = sm_bins[:-1]
flux_zbins = np.zeros(len(sm_bins))
num_obj_zbins = np.zeros(len(sm_bins))
min_z = np.zeros((len(sm_bins)))
max_z = np.zeros((len(sm_bins)))
n_realizations = 1000
bs_med = np.zeros((len(sm_bins), n_realizations))

for i in range(len(sm_bins)-1):
	sm_mask = (clean_cat_lz['lmass'] > sm_bins[i]) & (clean_cat_lz['lmass'] < sm_bins[i+1])
	ypix_i = ypix_lz[sm_mask]
	xpix_i = xpix_lz[sm_mask]
	num_obj_zbins[i] = len(ypix_i)
	flux_zbins[i] = np.mean(cmap[ypix_i,xpix_i])
	min_z[i] = min(clean_cat_lz['stellar_mass_z'][sm_mask])
	max_z[i] = max(clean_cat_lz['stellar_mass_z'][sm_mask])
	galaxy_pixels = cmap[ypix_i,xpix_i]

	#bootstrapping
	for j in range(n_realizations):
		indices = np.random.choice(len(ypix_i), len(ypix_i))
		bs_med[i,j] = np.median(galaxy_pixels[indices])
ylow = np.zeros((len(sm_bins)))
yhigh = np.zeros((len(sm_bins)))
for i in range(len(ylow)):
	ylow[i] = -1*(np.percentile(bs_med[i,:], 15.9)-np.median(bs_med[i,:]))
	yhigh[i] = -1*(np.median(bs_med[i,:])-np.percentile(bs_med[i,:], 84.1))
error = np.asarray([ylow, yhigh])

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


