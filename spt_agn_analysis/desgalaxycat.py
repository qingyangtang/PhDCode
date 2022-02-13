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

from astropy.io import fits
import sptpol_software.scratch.tnatoli.transient.calc_optfilt_oneband_2d_amy as calc

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

radec0 = np.asarray([352.50226, -55.000390])
reso_arcmin= 0.25
idldata = idl.readIDLSav("/home/tangq/data/coadds_latest.sav")
map90 = idldata.maps[0,:,:]
map150 = idldata.maps[1,:,:]
map220 = idldata.maps[2,:,:]
npixels = np.asarray(map150.shape)
y_coords = np.arange(npixels[0]); x_coords=np.arange(npixels[1])
ra_coords, dec_coords = sky.pix2Ang(np.asarray([y_coords,x_coords]), radec0, reso_arcmin, npixels)
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

Icut = np.where(I_band < 21)[0]
ra = ra[Icut]
dec = dec[Icut]
I_band = I_band[Icut]
I_flag = I_flag[Icut]
z_mean = z_mean[Icut]
coadd_ID = coadd_ID[Icut]

#########

idldata = idl.readIDLSav("/home/tangq/data/coadds_latest.sav")
idlpsd90 = idl.readIDLSav("/home/tangq/data/psd_ra23h30dec-55_2year_90_wtd.sav")
idlpsd150 = idl.readIDLSav("/home/tangq/data/psd_ra23h30dec-55_2year_150_wtd.sav")
fitsdata = files.readSptData('/home/tangq/data/masks_allbands_0p8.fits')

map150 = idldata.maps[1,:,:]
weights = idldata.weights[1,:,:]
radec0 = np.asarray([352.50226, -55.000390])
reso_arcmin= 0.25
npixels = np.asarray(map150.shape)
psd90 = idlpsd90.psd
psd150 = idlpsd150.psd
apodmask = fitsdata.masks['apod_mask']
pixelmask = fitsdata.masks['pixel_mask']

import sptpol_software.analysis.maps_amy as maps
source_mask = '/home/tnatoli/gits_sptpol/sptpol_software/config_files/ptsrc_config_ra23h30dec-55_surveyclusters.txt'
mask = maps.makeApodizedPointSourceMask(map150, source_mask, reso_arcmin=0.25, center=np.asarray([352.50226, -55.000390]), radius_arcmin=5)

masked_map = np.multiply(mask,map150)

#############MAKE SURE THAT calc_optfilt_oneband_2d_amy.py CHANGES SO THAT I PASS ON THE ARCMIN RESO AND INFO LIKE THAT

filt_output,filt_map =  calc.call_matched_filt_point(masked_map,noisepsd=psd150, fwhm_beam_arcmin=1.1, use_fft_as_noise=False, plot_map=False, return_filtered_map=True, mask=apodmask)

#### Getting the XXL survey AGN objects list and locating them on the map


ypix, xpix = sky.ang2Pix(np.asarray([ra, dec]), radec0, reso_arcmin, npixels,proj=0)[0]
ypixmask, xpixmask = sky.ang2Pix(np.asarray([ra, dec]), radec0, reso_arcmin, npixels,proj=0)[1] 

goodpixy = []
goodpixx = []
DESgal_flux = []

for i in range(len(ypix)):
	if ypixmask[i] == True & xpixmask[i] == True:
		if pixelmask[ypix[i],xpix[i]] == 1:
			if mask[ypix[i],xpix[i]] > 0.5:
				goodpixy.append(ypix[i])
				goodpixx.append(xpix[i])
				DESgal_flux.append(filt_map[ypix[i],xpix[i]])


# import glob
# flist = glob.glob('/home/tangq/data/Random_90GHz_Sim*')
# randommean = np.zeros((len(flist)))
# for j in range(len(flist)):
# 	data = np.load(flist[j])
# 	randommean[j] = np.mean(data[:len(DESgal_flux)])
# 	print str(j)
n_realizations = 1000
randommean = np.zeros((n_realizations))
for j in range(n_realizations):
	rand_ypix = np.round(np.random.rand(len(DESgal_flux)*2)*(npixels[0]-1))
	rand_xpix = np.round(np.random.rand(len(DESgal_flux)*2)*(npixels[0]-1))
	rand_flux = []
	rand_goodpixy = []
	rand_goodpixx = []
	for i in range(len(rand_xpix)):
		if pixelmask[rand_ypix[i],rand_xpix[i]] == 1:
			if mask[rand_ypix[i],rand_xpix[i]] >0.5:
				rand_goodpixy.append(rand_ypix)
				rand_goodpixx.append(rand_xpix)
				rand_flux.append(filt_map[rand_ypix[i],rand_xpix[i]])
	np.save("/data/tangq/Random_150GHz_Sim"+str(j)+"_Length_"+ str(len(rand_flux))+".npy", rand_flux)
	print "Done Iteration: " + str(j)
	if len(DESgal_flux) < len(rand_flux):
		rand_flux = rand_flux[0:len(DESgal_flux)]
	randommean[j] = np.mean(rand_flux)


