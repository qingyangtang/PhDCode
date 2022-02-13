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

########## Hershel:
mapfreq = "500"
field = "Herschel_15arcsecs"

#getting the position of sources from Marco's catalog
idldata = idl.readIDLSav("/data/tangq/SPIRE_maps/spt_starfndr_catalog.sav")
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
sn_cut = np.where(sn > 40)
cat_ra = cat_ra[sn_cut]
cat_dec = cat_dec[sn_cut]
cat_f_250 = cat_f_250[sn_cut]
cat_df_250 = cat_df_250[sn_cut]

src_ra = cat_ra
src_dec = cat_dec
src_flux = cat_f_250

beamsizes = np.asarray([18.2, 24.9, 36.3])/60. #beams for 250, 350, 500um in arcmins
#loading the data
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
#converting the nans in the masked map to 0s and zero padding map to have square shape
zeros_masked_map = np.nan_to_num(masked_map)
zp_mask_map = np.zeros((max(zeros_masked_map.shape),max(zeros_masked_map.shape)))
edge_padding = (zp_mask_map.shape[0]-zeros_masked_map.shape[0])/2
zp_mask_map[edge_padding:-edge_padding, :] = zeros_masked_map
pixelmask = np.zeros((max(zeros_masked_map.shape),max(zeros_masked_map.shape)))
pixelmask[edge_padding:-edge_padding, :] = mask
zp_mask = np.zeros((zp_mask_map.shape))
zp_mask[edge_padding:-edge_padding, :] = mask

#calculating noise PSD
reso = (reso_arcmin/60)*np.pi/180.
dngrid = zp_mask_map.shape[0]
dk = 1./dngrid/reso
factor = np.sqrt(dngrid**2*dk**2)
noise_psd = np.sqrt(np.abs(np.fft.ifft2(zp_mask_map)/factor))

filt_output,filt_map =  calc.call_matched_filt_point(zp_mask_map, cmb = False, skysigvar = np.zeros((25000)), noisepsd=noise_psd, fwhm_beam_arcmin=beamsizes[beamsize], ellmax = 25000, ell_highpass = 10, ell_lowpass = 20000,use_fft_as_noise=False, plot_map=False, return_filtered_map=True, mask = zp_mask, reso_arcmin = reso_arcmin, ngridx = zp_mask_map.shape[0],ngridy = zp_mask_map.shape[0])
cmap = filt_map
npixels = np.asarray(cmap.shape)

sp_des_dir = '/data/tangq/custom_catalogs/spitzer_des_cat.fits'
stellar_m_dir = '/data/bleeml/stellar_mass_cats/ssdf_des_try1.fout'

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

ypix, xpix = sky.ang2Pix(np.asarray([clean_cat['ra'], clean_cat['dec']]), radec0, reso_arcmin, npixels,proj=5)[0]
ypixmask, xpixmask = sky.ang2Pix(np.asarray([clean_cat['ra'], clean_cat['dec']]), radec0, reso_arcmin, npixels,proj=5)[1] 
ang2pix_mask = (ypixmask == True) & (xpixmask == True) & (pixelmask[ypix,xpix]>0.99999999)
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


# randpixdata = np.load('/data/tangq/SPIRE_maps/rand_pix_flux_freq_' + mapfreq + '.npy')
# nreal = 100
# rand_mean = np.zeros((nreal))
# rand_med = np.zeros((nreal))
# print "for observing wavelength: " + mapfreq
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
# rand_mean = np.zeros((nreal))
# rand_med = np.zeros((nreal))
# for j in range(nreal):
# 	randpix = randpixdata[(np.round(np.random.rand(len(ypix_hzhm))*(len(randpixdata)-1))).astype(int)]
# 	rand_mean[j] = np.mean(randpix)
# 	rand_med[j] = np.median(randpix)
# print "The signal to noise is (mean): " + str(np.mean(cmap[ypix_hzhm,xpix_hzhm])/np.std(rand_mean))
# print "The signal to noise is (median): " + str(np.median(cmap[ypix_hzhm,xpix_hzhm])/np.std(rand_med))



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

