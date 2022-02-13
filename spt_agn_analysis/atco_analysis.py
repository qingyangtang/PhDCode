from sptpol_software.util import idl
import astropy.coordinates as coord
import astropy.units as u
from sptpol_software.observation import sky
from itertools import product
from sptpol_software.util import files
from sptpol_software.util import fits

from astropy.io import fits
import sptpol_software.scratch.tnatoli.transient.calc_optfilt_oneband_2d_amy as calc


savedir = '/home/tangq/data/atco_ptsrc/'

#opening the atco data and getting the info
idldata = idl.readIDLSav("/home/tangq/data/at20g.sav")

atco_ra = np.asarray([d['ra'] for d in idldata.at20g])
atco_dec = np.asarray([d['dec'] for d in idldata.at20g])
atco_flux = np.asarray([d['flux'] for d in idldata.at20g])
atco_dflux = np.asarray([d['dflux'] for d in idldata.at20g])
at_ra = coord.Angle(atco_ra*u.degree)
at_ra = at_ra.wrap_at(180*u.degree)
at_dec = coord.Angle(atco_dec*u.degree)


save_file = True
cm = plt.cm.get_cmap('OrRd')
cm2 = plt.cm.get_cmap('Blues')
fig = plt.figure(figsize=(18,16))
ax = fig.add_subplot(111, projection="mollweide")
ax.grid(True)
sc = ax.scatter(at_ra.radian, at_dec.radian, c=atco_flux, s=10, cmap=cm, alpha=1, edgecolors='none')

#opening sptsz data, finding coordinates of the edges of the map
radec0 = np.asarray([352.50226, -55.000390])
reso_arcmin= 0.25
idldata = idl.readIDLSav("/home/tangq/data/coadds_latest.sav")
map90 = idldata.maps[0,:,:]
map150 = idldata.maps[1,:,:]
map220 = idldata.maps[2,:,:]
npixels = np.asarray(map90.shape)
y_coords = np.arange(npixels[0]); x_coords=np.arange(npixels[1])
ra_coords, dec_coords = sky.pix2Ang(np.asarray([y_coords,x_coords]), radec0, reso_arcmin, npixels)
ra = [ra_coords[0], ra_coords[-1]]
dec = [dec_coords[0],dec_coords[-1]]

#finding where the spt patch maps to in the atco data
patch_ra= np.where((atco_ra >=ra[0]) | (atco_ra <=ra[1]))[0]
patch_dec = np.where((atco_dec <= dec[0]) & (atco_dec >= dec[1]))[0]
patch = np.asarray(list(set(patch_ra).intersection(patch_dec)))
sc = ax.scatter(at_ra[patch].radian, at_dec[patch].radian, c=atco_flux[patch], s=10, cmap=cm2, alpha=1, edgecolors='none')

#Getting the top 30 brightest sources in the patch
bright30 = sort(atco_flux[patch])[-30]
b30index = np.where(atco_flux[patch] >= bright30)[0]

sc = ax.scatter(at_ra[patch][b30index].radian, at_dec[patch][b30index].radian, color='yellow', marker='*', edgecolors='none')
b30_ra =  atco_ra[patch][b30index]
b30_dec =atco_dec[patch][b30index]
b30_flux =  atco_flux[patch][b30index]
#plt.colorbar(sc,shrink=0.5)
plt.title('atco20, flux')
if save_file==True:
	plt.savefig("atco20_fluxmap.png")

plt.close()
#opening sptsz data, finding coordinates of the map
idldata = idl.readIDLSav("/home/tangq/data/coadds_latest.sav")
map90 = idldata.maps[0,:,:]
map150 = idldata.maps[1,:,:]
map220 = idldata.maps[2,:,:]
ypix30, xpix30 = sky.ang2Pix(np.asarray([b30_ra, b30_dec]), radec0, reso_arcmin, npixels,proj=0)[0]
ypixmask, xpixmask = sky.ang2Pix(np.asarray([b30_ra, b30_dec]), radec0, reso_arcmin, npixels,proj=0)[1] #tells me if the pixel is in the map or not (theoretically should be but apparently not..?)

#getting the pixel mask which tells if the pixel is good or not 
fitsdata = files.readSptData('/home/tangq/data/masks_allbands_0p8.fits')
pixelmask = fitsdata.masks['pixel_mask']
goodpixy = []
goodpixx = []

#checking that the pixel is good
for i in range(len(ypix30)):
	if ypixmask[i] == True & xpixmask[i] == True:
		if pixelmask[ypix30[i],xpix30[i]] == 1:
			goodpixy.append(ypix30[i])
			goodpixx.append(xpix30[i])
print goodpixy, goodpixx

for source in range(len(np.array(goodpixy))):

#plotting area around selected point sources
	f, (ax1, ax2, ax3) = plt.subplots(1, 3)
	ax1.imshow(map90[goodpixy[source]-40:goodpixy[source]+40,goodpixx[source]-40:goodpixx[source]+40])
	ax2.imshow(map150[goodpixy[source]-40:goodpixy[source]+40,goodpixx[source]-40:goodpixx[source]+40])
	ax3.imshow(map220[goodpixy[source]-40:goodpixy[source]+40,goodpixx[source]-40:goodpixx[source]+40])
	ax1.set_title('90GHz map')
	ax2.set_title('150GHz map')
	ax3.set_title('220GHz map, src'+str(source))
	ax1.set_ylabel('X pixel')
	ax2.set_xlabel('Y pixel')
	if save_file == True:
		plt.savefig(savedir + "atco_src" + str(source)+"_in_SZmap.png")
	plt.close()

####Matched filtering on data#####

idldata = idl.readIDLSav("/home/tangq/data/coadds_latest.sav")
idlpsd = idl.readIDLSav("/home/tangq/data/psd_ra23h30dec-55_2year_90_wtd.sav")
fitsdata = files.readSptData('/home/tangq/data/masks_allbands_0p8.fits')

map90 = idldata.maps[0,:,:]
weights = idldata.weights[0,:,:]
radec0 = np.asarray([352.50226, -55.000390])
npixels = np.asarray(map90.shape)
psd90 = idlpsd.psd
apodmask = fitsdata.masks['apod_mask']

#############MAKE SURE THAT calc_optfilt_oneband_2d_amy.py CHANGES SO THAT I PASS ON THE ARCMIN RESO AND INFO LIKE THAT

filt_output,filt_map =  calc.call_matched_filt_point(map90,noisepsd=psd90, fwhm_beam_arcmin=1.1, use_fft_as_noise=False, plot_map=False, return_filtered_map=True, mask=apodmask)


for source in range(len(np.array(goodpixy))):
#plotting area around selected point sources
	f, (ax1, ax2) = plt.subplots(1, 2)
	ax1.imshow(map90[goodpixy[source]-40:goodpixy[source]+40,goodpixx[source]-40:goodpixx[source]+40])
	ax2.imshow(filt_map[goodpixy[source]-40:goodpixy[source]+40,goodpixx[source]-40:goodpixx[source]+40])
	ax1.set_title('SZ 90GHz map')
	ax2.set_title('Match-filtered map, src' + str(source)+' Ypix:'+str(goodpixy[source])+' Xpix:'+str(goodpixx[source]))
	ax1.set_ylabel('X pixel')
	ax1.set_xlabel('Y pixel')
	if save_file == True:
		plt.savefig(savedir + "atco_src" + str(source)+"_in_SZmap_and_filtered.png")
	plt.close()
atca_pixy = goodpixy
atca_pixx = goodpixx
del goodpixx
del goodpixy

###checking ptsrc mask actually masks everything
idldata = idl.readIDLSav("/home/tangq/data/coadds_latest.sav")
idlpsd = idl.readIDLSav("/home/tangq/data/psd_ra23h30dec-55_2year_90_wtd.sav")
fitsdata = files.readSptData('/home/tangq/data/masks_allbands_0p8.fits')

map90 = idldata.maps[0,:,:]
weights = idldata.weights[0,:,:]
radec0 = np.asarray([352.50226, -55.000390])
reso_arcmin= 0.25
npixels = np.asarray(map90.shape)
psd90 = idlpsd.psd
apodmask = fitsdata.masks['apod_mask']
pixelmask = fitsdata.masks['pixel_mask']

import sptpol_software.analysis.maps_amy as maps
source_mask = '/home/tnatoli/gits_sptpol/sptpol_software/config_files/ptsrc_config_ra23h30dec-55_surveyclusters.txt'
mask = maps.makeApodizedPointSourceMask(map90, source_mask, reso_arcmin=0.25, center=np.asarray([352.50226, -55.000390]),radius_arcmin=5)

masked_map = np.multiply(mask,map90)

#############MAKE SURE THAT calc_optfilt_oneband_2d_amy.py CHANGES SO THAT I PASS ON THE ARCMIN RESO AND INFO LIKE THAT

filt_output,filt_map =  calc.call_matched_filt_point(masked_map,noisepsd=psd90, fwhm_beam_arcmin=1.1, use_fft_as_noise=False, plot_map=False, return_filtered_map=True, mask=apodmask)

for source in range(len(atca_pixy)):
	f, ax = plt.subplots(2, 2)
	ax[0,0].imshow(map90[atca_pixy[source]-50:atca_pixy[source]+50,atca_pixx[source]-50:atca_pixx[source]+50])
	ax[0,1].imshow(mask[atca_pixy[source]-50:atca_pixy[source]+50,atca_pixx[source]-50:atca_pixx[source]+50])
	ax[1,0].imshow(masked_map[atca_pixy[source]-50:atca_pixy[source]+50,atca_pixx[source]-50:atca_pixx[source]+50])
	ax[1,1].imshow(filt_map[atca_pixy[source]-50:atca_pixy[source]+50,atca_pixx[source]-50:atca_pixx[source]+50])
	ax[0,0].set_title('SZ 90GHz map')
	ax[0,1].set_title('Point source mask')
	ax[1,0].set_title('Masked Map')
	ax[1,1].set_title('Match-filtered map, src' + str(source))
	ax[0,0].set_ylabel('X pixel')
	ax[1,1].set_xlabel('Y pixel')
	if save_file == True:
		plt.savefig(savedir + "atca_src" + str(source)+"_in_SZmap_and_filtered_check_mask.png")
#	plt.close()


#reading the SPIRE map
fits250 = fits.open("/home/tangq/data/spt_zea_itermap_10_iterations_15_arcsec_pixels_PSW.fits")
fits350 = fits.open("/home/tangq/data/spt_zea_itermap_10_iterations_15_arcsec_pixels_PMW.fits")
fits500 = fits.open("/home/tangq/data/spt_zea_itermap_10_iterations_15_arcsec_pixels_PLW.fits")
