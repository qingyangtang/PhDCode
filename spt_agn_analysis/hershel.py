import numpy as numpy
from astropy.io import fits
import matplotlib.pyplot as plt
from sptpol_software.util import idl
import astropy.coordinates as coord
import astropy.units as u
from sptpol_software.observation import sky
from itertools import product
from sptpol_software.util import files
from sptpol_software.util import fits
import sptpol_software.analysis.maps_amy as maps
import pickle
from astropy.io import fits
import sptpol_software.scratch.tnatoli.transient.calc_optfilt_oneband_2d_amy as calc
import sptpol_software.analysis.maps_amy as maps
from matplotlib.colors import LogNorm

show_hdulist_info = False
plot_ptsrc_cutout = False
make_ptsrc_catalogue = False
check_point_sources_masked = False


def add_hotcold_pixels_to_ptsrc_cat(datafile, src_file, pix_flux = 1.5e-1, pix_dflux = 3.5e-3):
	files = ["PSW","PMW","PLW"]
	for freq in range(len(files)):
		fits0 = fits.open("/data/tangq/SPIRE_maps/spt_zea_itermap_10_iterations_15_arcsec_pixels_" + files[freq] + ".fits")
		radec0 = np.asarray([fits0[1].header['CRVAL1'], fits0[1].header['CRVAL2']])
		reso_arcmin = abs(fits0[1].header['CD1_1'])*60 
		map0 = np.rot90(np.rot90(fits0[1].data))
		npixels = map0.shape
		if files[freq] == "PSW":
			hotpixels = np.where(map0 > 1)
			coldpixels = np.where(map0 < -1)
		elif files[freq] == "PMW":
			hotpixels = []
			coldpixles = []
			ptsrc = np.loadtxt('/data/tangq/SPIRE_maps/herschel_srcs_40.txt')
			np.savetxt('/data/tangq/SPIRE_maps/herschel_srcs_40_'+files[freq]+'.txt',ptsrc)
			continue
		elif files[freq] == "PLW":
				hotpixels = np.where(map0 > 0.5)
				coldpixels = np.where(map0 < -0.5)
		ypixels = np.concatenate([hotpixels[0],coldpixels[0]])
		xpixels = np.concatenate([hotpixels[1],coldpixels[1]])
		pix_ra, pix_dec = sky.pix2Ang(np.asarray([ypixels,xpixels]), radec0, reso_arcmin, npixels,proj=5)
		pixelstack = np.vstack([pix_ra,pix_dec,np.ones(len(pix_ra))*pix_flux,np.ones(len(pix_ra))*pix_dflux])
		ptsrc = np.loadtxt('/data/tangq/SPIRE_maps/herschel_srcs_40.txt')
		ptsrc_new = np.vstack([ptsrc, pixelstack.T])
		np.savetxt('/data/tangq/SPIRE_maps/herschel_srcs_40_'+files[freq]+'.txt',ptsrc_new)

mapfreq = "350"
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

if make_ptsrc_catalogue == True:
	ptsrc_array = np.vstack((cat_ra, cat_dec, cat_f_250, cat_df_250))
	np.savetxt('/data/tangq/SPIRE_maps/herschel_srcs_40.txt',ptsrc_array.T)


beamsizes = np.asarray([18.2, 24.9, 36.3])/60. #beams for 250, 350, 500um in arcmins

#loading the data
if mapfreq == "250":
	fits0 = fits.open("/data/tangq/SPIRE_maps/spt_zea_itermap_10_iterations_15_arcsec_pixels_PSW.fits")
	# hotpixels = np.where(fits0[1].data > 1)
	# fits0[1].data[hotpixels] = 0
	# coldpixels = np.where(fits0[1].data < -1)
	# fits0[1].data[coldpixels] = 0
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
	# hotpixels = np.where(fits0[1].data > 0.5)
	# fits0[1].data[hotpixels] = 0
	# coldpixels = np.where(fits0[1].data < -0.5)
	# fits0[1].data[coldpixels] = 0
else:
	print "Enter valid map freq!!"



#printing what's in the fits file
# if show_hdulist_info == True:
# 	print fits0.info() 
# 	print fits0[1].header

radec0 = np.asarray([fits0[1].header['CRVAL1'], fits0[1].header['CRVAL2']])
reso_arcmin = abs(fits0[1].header['CD1_1'])*60 

map0 = np.rot90(np.rot90(fits0[1].data))

npixels = map0.shape


# if plot_ptsrc_cutout == True:
# 	cat_ypix, cat_xpix = sky.ang2Pix(np.asarray([cat_ra, cat_dec]), radec0, reso_arcmin, npixels,proj=5)[0]
# 	cat_ypixmask, cat_xpixmask = sky.ang2Pix(np.asarray([cat_ra, cat_dec]), radec0, reso_arcmin, npixels,proj=5)[1] 
# 	for i in range(10):
# 		plt.figure()
# 		plt.imshow(map0[cat_ypix[i]-50:cat_ypix[i]+50, cat_xpix[i]-50:cat_xpix[i]+50])
# 		plt.colorbar()

mask = maps.makeApodizedPointSourceMask(map0, source_mask, ftype = 'herschel_nan', reso_arcmin=reso_arcmin, center=radec0, radius_arcmin=3)
masked_map = np.multiply(mask,map0)

if check_point_sources_masked ==True:
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
	cat_ypix, cat_xpix = sky.ang2Pix(np.asarray([cat_ra, cat_dec]), radec0, reso_arcmin, npixels,proj=0)[0]
	cat_ypixmask, cat_xpixmask = sky.ang2Pix(np.asarray([cat_ra, cat_dec]), radec0, reso_arcmin, npixels,proj=0)[1] 
	for i in range(10):
		plt.figure()
		plt.subplot(1,2,1)
		plt.imshow(map0[cat_ypix[i]-40:cat_ypix[i]+40, cat_xpix[i]-40:cat_xpix[i]+40])
		plt.subplot(1,2,2)
		plt.imshow(masked_map[cat_ypix[i]-40:cat_ypix[i]+40, cat_xpix[i]-40:cat_xpix[i]+40])

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
# Checking the point sources
# idldata = idl.readIDLSav("/data/tangq/SPIRE_maps/spt_starfndr_catalog.sav")
# cat_ra = np.asarray([d['ra'] for d in idldata.cat])
# cat_dec = np.asarray([d['dec'] for d in idldata.cat])
# cat_f_250 = (np.asarray([d['f_250'] for d in idldata.cat]))
# cat_df_250 = np.asarray([d['df_250'] for d in idldata.cat])

# cat_ypix, cat_xpix = sky.ang2Pix(np.asarray([cat_ra, cat_dec]), radec0, reso_arcmin, zp_mask_map.shape,proj=5)[0]
# cat_ypixmask, cat_xpixmask = sky.ang2Pix(np.asarray([cat_ra, cat_dec]), radec0, reso_arcmin, zp_mask_map.shape,proj=5)[1] 
# for i in range(10):
# 	plt.figure()
# 	plt.subplot(1,2,1)
# 	plt.imshow(zp_mask_map[cat_ypix[i]-40:cat_ypix[i]+40, cat_xpix[i]-40:cat_xpix[i]+40])
# 	plt.colorbar()
# 	plt.title('Orig Map')
# 	plt.subplot(1,2,2)
# 	plt.imshow(filt_map[cat_ypix[i]-40:cat_ypix[i]+40, cat_xpix[i]-40:cat_xpix[i]+40])
# 	plt.colorbar()
# 	plt.title('Filtered Map')

############# Getting the DES galaxies catalogue
show_plot = False

hdulist = fits.open('/data/bleeml/for_amy/sva1_gold_r1.0_catalog.fits')
data = hdulist[1]

ra = data.data['RA']
dec = data.data['DEC']
coadd_ID = data.data['COADD_OBJECTS_ID']
I_band = data.data['MAG_AUTO_I']
I_flag = data.data['FLAGS_I']

hdulist = fits.open('/data/bleeml/for_amy/sva1_gold_r1.0_bpz_point.fits')
data = hdulist[1]
coadd_ID_z = data.data['COADD_OBJECTS_ID']
z_mean = data.data['Z_MEAN']
y_coords = np.arange(npixels[0]); x_coords=np.arange(npixels[1])
ra_coords, dec_coords = sky.pix2Ang(np.asarray([y_coords,x_coords]), radec0, reso_arcmin, npixels,proj=5)
ra_edge = [ra_coords[0], ra_coords[-1]]
dec_edge = [dec_coords[0],dec_coords[-1]]

ra_cut = np.concatenate((np.where(ra < ra_edge[1])[0], np.where(ra > ra_edge[0])[0]))
ra = ra[ra_cut]
dec = dec[ra_cut]
I_band = I_band[ra_cut]
I_flag = I_flag[ra_cut]
z_mean = z_mean[ra_cut]
coadd_ID = coadd_ID[ra_cut]
dec_cut = np.where((dec < dec_edge[0]) & (dec > dec_edge[1]))[0]
ra = ra[dec_cut]
dec = dec[dec_cut]
I_band = I_band[dec_cut]
I_flag = I_flag[dec_cut]
z_mean = z_mean[dec_cut]
coadd_ID = coadd_ID[dec_cut]

zgood = np.invert(np.isnan(z_mean))
ra = ra[zgood]
dec = dec[zgood]
I_band = I_band[zgood]
I_flag = I_flag[zgood]
z_mean = z_mean[zgood]
coadd_ID = coadd_ID[zgood]

Igood = np.where((I_flag == 0) & (I_band != 99))[0]
ra = ra[Igood]
dec = dec[Igood]
I_band = I_band[Igood]
I_flag = I_flag[Igood]
z_mean = z_mean[Igood]
coadd_ID = coadd_ID[Igood]

zbin_width = 0.1
zbins = np.arange(zbin_width,2,zbin_width)
colors = plt.cm.jet(np.linspace(0,1,len(zbins)))
Ibins = np.arange(floor(min(I_band)), ceil(max(I_band)), 0.1)
Ibins = np.arange(15,30,0.1)
for z in range(len(zbins)):
	z_indices = np.where((z_mean < zbins[z]) & (z_mean > (zbins[z]-zbin_width)))[0]
	plt.hist(I_band[z_indices], bins = Ibins, color = colors[z], alpha=2./len(zbins),label='z='+str(zbins[z]))
plt.legend(loc='best')
plt.xlabel('I mag')
plt.savefig('/home/tangq/data/DES_galaxies_Imag_redshift.pdf')
if show_plot == False:
	plt.close()
Icut = np.where(I_band < 21)[0]
ra = ra[Icut]
dec = dec[Icut]
I_band = I_band[Icut]
I_flag = I_flag[Icut]
z_mean = z_mean[Icut]
coadd_ID = coadd_ID[Icut]
zcut1 = np.where (z_mean < 0.25)[0]
zcut2 = np.where((z_mean > 0.25) & (z_mean <0.75))[0]
zcut3 = np.where(z_mean > 0.75)[0]
zbands = [zcut1, zcut2, zcut3]

ra_good = ra
dec_good = dec
I_band_good = I_band
I_flag_good = I_flag
z_mean_good = z_mean
coadd_ID_good = coadd_ID

ypix, xpix = sky.ang2Pix(np.asarray([ra, dec]), radec0, reso_arcmin, npixels,proj=5)[0]
ypixmask, xpixmask = sky.ang2Pix(np.asarray([ra, dec]), radec0, reso_arcmin, npixels,proj=5)[1] 
ang2pixind = np.where((ypixmask ==True) & (xpixmask == True))[0]
ypix = ypix[ang2pixind]
xpix = xpix[ang2pixind]
ra = ra[ang2pixind]
dec = dec[ang2pixind]
I_band = I_band[ang2pixind]
I_flag = I_flag[ang2pixind]
z_mean = z_mean[ang2pixind]
coadd_ID = coadd_ID[ang2pixind]
del ang2pixind
pixelmaskind = np.where(pixelmask[ypix,xpix] > 0.99999999)[0]
ypix = ypix[pixelmaskind]
xpix = xpix[pixelmaskind]
ra = ra[pixelmaskind]
dec = dec[pixelmaskind]
I_band = I_band[pixelmaskind]
I_flag = I_flag[pixelmaskind]
z_mean = z_mean[pixelmaskind]
coadd_ID = coadd_ID[pixelmaskind]

from astropy.coordinates import SkyCoord
from astropy import units as u
c = SkyCoord(ra=ra*u.degree, dec=dec*u.degree)  
catalog = SkyCoord(ra=src_ra*u.degree, dec=src_dec*u.degree)  
idx, d2d, d3d = c.match_to_catalog_sky(catalog)

#cutting out the gals where it's within 4 arcmins of a source
dist_cut = np.where(d2d.arcminute > 4)[0]
ypix = ypix[dist_cut]
xpix = xpix[dist_cut]
idx = idx[dist_cut]
d2d = d2d[dist_cut]
ra = ra[dist_cut]
dec = dec[dist_cut]
I_band = I_band[dist_cut]
I_flag = I_flag[dist_cut]
z_mean = z_mean[dist_cut]
coadd_ID = coadd_ID[dist_cut]

# #cutting out the gal that are within 8 arcmins of > 100mJy source
# nearest_src_flux = src_flux[idx]
# gtr100fluxind = np.where((nearest_src_flux > 100) & (d2d.arcminute < 8))[0]

# ypix = ypix[np.in1d(np.arange(len(ypix)), gtr100fluxind, invert=True)]
# xpix = xpix[np.in1d(np.arange(len(xpix)), gtr100fluxind, invert=True)]
# idx = idx[np.in1d(np.arange(len(idx)), gtr100fluxind, invert=True)]
# d2d = d2d[np.in1d(np.arange(len(d2d)), gtr100fluxind, invert=True)]
# ra = ra[np.in1d(np.arange(len(ra)), gtr100fluxind, invert=True)]
# dec = dec[np.in1d(np.arange(len(dec)), gtr100fluxind, invert=True)]
# I_band = I_band[np.in1d(np.arange(len(I_band)), gtr100fluxind, invert=True)]
# I_flag = I_flag[np.in1d(np.arange(len(I_flag)), gtr100fluxind, invert=True)]
# z_mean = z_mean[np.in1d(np.arange(len(z_mean)), gtr100fluxind, invert=True)]
# coadd_ID = coadd_ID[np.in1d(np.arange(len(coadd_ID)), gtr100fluxind, invert=True)]


print "The freq " + mapfreq
print "The signal is (mean): " + str(np.mean(cmap[ypix,xpix]))
print "The signal is (median): " + str(np.median(cmap[ypix,xpix]))

plt.figure()
plt.imshow(cmap)
plt.plot(xpix,ypix,'.')
plt.ylabel('Y pix')
plt.xlabel('X pix')
plt.colorbar()
plt.title('DES gal location overplotted on top of map at ' + mapfreq + 'GHz')
plt.savefig('/home/tangq/data/'+field+'_'+mapfreq+'GHz_DES_gal_locations.pdf')
if show_plot == False:
	plt.close()
plt.figure()
plt.hist(cmap[ypix, xpix],bins=np.linspace(np.min(cmap[ypix,xpix]), np.max(cmap[ypix,xpix]), 200))
plt.xlabel('DES gal flux')
plt.title('Field: ' + field + '_' + mapfreq+'GHz')
plt.savefig('/home/tangq/data/'+field+'_'+mapfreq+'GHz_DES_gal_flux_hist.pdf')
if show_plot == False:
	plt.close()


# randommean = np.zeros((n_realizations))
# randommedian = np.zeros((n_realizations))
# for j in range(n_realizations):
# 	data = np.load("/data/tangq/wendymaps/"+field+"/Random_"+mapfreq+"GHz_Sim"+str(j)+".npy")
# 	rand_flux = data[0:len(ypix)]
# 	randommean[j] = np.mean(rand_flux)
# 	randommedian[j] = np.median(rand_flux)
	
n_realizations = 200
randommean = np.zeros((n_realizations))
randommedian = np.zeros((n_realizations))

for j in range(n_realizations):
	rand_ypix = (np.round(np.random.rand(len(ypix)*3)*(npixels[0]-1))).astype(int)
	rand_xpix = (np.round(np.random.rand(len(ypix)*3)*(npixels[1]-1))).astype(int)
	pixelmaskind = np.where(pixelmask[rand_ypix,rand_xpix]> 0.99999999)[0]
	rand_ypix = rand_ypix[pixelmaskind]
	rand_xpix = rand_xpix[pixelmaskind]

	ra_coords, dec_coords = sky.pix2Ang(np.asarray([rand_ypix,rand_xpix]), radec0, reso_arcmin, npixels,proj=5)
	c = SkyCoord(ra=ra_coords*u.degree, dec=dec_coords*u.degree)  
	catalog = SkyCoord(ra=src_ra*u.degree, dec=src_dec*u.degree)  
	idx, d2d, d3d = c.match_to_catalog_sky(catalog)
	dist_cut = np.where(d2d.arcminute > 4)[0]
	ra_coords = ra_coords[dist_cut]
	dec_coords = dec_coords[dist_cut]
	rand_ypix = rand_ypix[dist_cut]
	rand_xpix = rand_xpix[dist_cut]
	d2d = d2d[dist_cut]
	idx = idx[dist_cut]
	nearest_src_flux = src_flux[idx]
	gtr100fluxind = np.where((nearest_src_flux > 100) & (d2d.arcminute < 8))[0]
	rand_ypix = rand_ypix[np.in1d(np.arange(len(rand_ypix)), gtr100fluxind, invert=True)]
	rand_xpix = rand_xpix[np.in1d(np.arange(len(rand_xpix)), gtr100fluxind, invert=True)]
	idx = idx[np.in1d(np.arange(len(idx)), gtr100fluxind, invert=True)]
	d2d = d2d[np.in1d(np.arange(len(d2d)), gtr100fluxind, invert=True)]
	ra_coords = ra_coords[np.in1d(np.arange(len(ra_coords)), gtr100fluxind, invert=True)]
	dec_coords = dec_coords[np.in1d(np.arange(len(dec_coords)), gtr100fluxind, invert=True)]
	rand_flux = cmap[rand_ypix,rand_xpix]
	#np.save("/data/tangq/wendymaps/"+field+"/Random_"+mapfreq+"GHz_Sim"+str(j)+".npy", rand_flux)

	#print "Done Iteration: " + str(j)
	if len(ypix) < len(rand_flux):
		rand_flux = rand_flux[0:len(ypix)]
	randommean[j] = np.mean(rand_flux)
	randommedian[j] = np.median(rand_flux)
print "The field is " + str(field) + " and freq " + mapfreq
print "redshifts between " + str(min(z_mean)) + " and " + str(max(z_mean))
print "The signal is (mean): " + str(np.mean(cmap[ypix,xpix]))
print "The noise is: " + str(np.std(randommean))
print "The signal to noise is (mean): " + str(np.mean(cmap[ypix,xpix])/np.std(randommean))
print "The signal is (median): " + str(np.median(cmap[ypix,xpix]))
print "The noise is: " + str(np.std(randommedian))
print "The signal to noise is: " + str(np.median(cmap[ypix,xpix])/np.std(randommedian))


