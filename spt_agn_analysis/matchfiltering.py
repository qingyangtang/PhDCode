from sptpol_software.util import idl
import astropy.coordinates as coord
import astropy.units as u
from sptpol_software.observation import sky
from itertools import product
from sptpol_software.util import files
from sptpol_software.util import hdf
from sptpol_software.util import fits
#from astropy.io import fits
import sptpol_software.scratch.tnatoli.transient.calc_optfilt_oneband_2d_amy as calc
from astropy.io import fits
import matplotlib.pyplot as plt
import sptpol_software.analysis.maps_amy as maps


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

source_mask = '/home/tnatoli/gits_sptpol/sptpol_software/config_files/ptsrc_config_ra23h30dec-55_surveyclusters.txt'
mask = maps.makeApodizedPointSourceMask(map90, source_mask,ftype = 'tom', reso_arcmin=0.25, center=np.asarray([352.50226, -55.000390]), radius_arcmin=5)

masked_map = np.multiply(mask,map90)

#############MAKE SURE THAT calc_optfilt_oneband_2d_amy.py CHANGES SO THAT I PASS ON THE ARCMIN RESO AND INFO LIKE THAT

filt_output,filt_map =  calc.call_matched_filt_point(masked_map,noisepsd=psd90, fwhm_beam_arcmin=1.1, use_fft_as_noise=False, plot_map=False, return_filtered_map=True, mask=apodmask, reso_arcmin =reso_arcmin,ngridx = npixels[0],ngridy=npixels[1])



#### Getting the XXL survey AGN objects list and locating them on the map

hdulist = fits.open('/home/tangq/data/aatxxlpasa2015nov05.fits')
data = hdulist[1].data
XXL_ra = data.field(1)
XXL_dec = data.field(2)

ypix, xpix = sky.ang2Pix(np.asarray([XXL_ra, XXL_dec]), radec0, reso_arcmin, npixels,proj=0)[0]
ypixmask, xpixmask = sky.ang2Pix(np.asarray([XXL_ra, XXL_dec]), radec0, reso_arcmin, npixels,proj=0)[1] 

goodpixy = []
goodpixx = []
XXL_flux = []

for i in range(len(ypix)):
	if ypixmask[i] == True & xpixmask[i] == True:
		if pixelmask[ypix[i],xpix[i]] == 1:
			if mask[ypix[i],xpix[i]] > 0.5:
				goodpixy.append(ypix[i])
				goodpixx.append(xpix[i])
				XXL_flux.append(filt_map[ypix[i],xpix[i]])

## Check with radio sources in XXL (ATCA)###
hdulist = fits.open('/home/tangq/data/XXL_ATCA_15.fits')
data = hdulist[1].data
XXLATCA_ra = data.field(5)
XXLATCA_dec = data.field(6)
XXLATCA_tableflux = data.field(12)

xxlatca_ypix, xxlatca_xpix = sky.ang2Pix(np.asarray([XXLATCA_ra, XXLATCA_dec]), radec0, reso_arcmin, npixels,proj=0)[0]
xxlatca_ypixmask, xxlatca_xpixmask = sky.ang2Pix(np.asarray([XXLATCA_ra, XXLATCA_dec]), radec0, reso_arcmin, npixels,proj=0)[1] 

XXLATCA_goodpixy = []
XXLATCA_goodpixx = []
XXLATCA_flux = []
XXLATCA_radioflux = []

for i in range(len(xxlatca_ypix)):
	if xxlatca_ypixmask[i] == True & xxlatca_xpixmask[i] == True:
		if pixelmask[xxlatca_ypix[i],xxlatca_xpix[i]] == 1:
			if mask[xxlatca_ypix[i],xxlatca_xpix[i]] > 0.5:
				XXLATCA_goodpixx.append(xxlatca_xpix[i])
				XXLATCA_goodpixy.append(xxlatca_ypix[i])
				XXLATCA_flux.append(filt_map[xxlatca_ypix[i],xxlatca_xpix[i]])
				XXLATCA_radioflux.append(XXLATCA_tableflux[i])
XXLATCA_radioflux = np.array(XXLATCA_radioflux)
XXLATCA_flux = np.array(XXLATCA_flux)
#random pixel generator
rand_ypix = np.round(np.random.rand(6000)*(npixels[0]-1))
rand_xpix = np.round(np.random.rand(6000)*(npixels[0]-1))

rand_goodpixy = []
rand_goodpixx = []
rand_flux = []
goodpixy_array = np.array(goodpixy)
goodpixx_array = np.array(goodpixx)
for i in range(len(rand_xpix)):
	if pixelmask[rand_ypix[i],rand_xpix[i]] == 1:
		if mask[rand_ypix[i],rand_xpix[i]] >0.5:
			if rand_ypix[i] in goodpixy:
				idx = np.where(rand_ypix[i] == goodpixy)[0]
				if not np.in1d(rand_xpix[i], goodpixx_array[idx]):
					rand_goodpixy.append(rand_ypix)
					rand_goodpixx.append(rand_xpix)
					rand_flux.append(filt_map[rand_ypix[i],rand_xpix[i]])
			else:
				rand_goodpixy.append(rand_ypix)
				rand_goodpixx.append(rand_xpix)
				rand_flux.append(filt_map[rand_ypix[i],rand_xpix[i]])


bins = np.linspace(-3.e-4,3.e-4,50)
below30 = np.where(XXLATCA_radioflux<30)[0]
over30to60 = np.where((XXLATCA_radioflux<60) & (XXLATCA_radioflux>30))[0]
over60to90 = np.where((XXLATCA_radioflux<90) & (XXLATCA_radioflux>60))[0]
over90 = np.where(XXLATCA_radioflux>90)[0]

plt.figure()
plt.hist(XXLATCA_flux[below30],bins,alpha=0.5,label='XXL ATCA obj < 30mJ',normed=True)
plt.hist(XXLATCA_flux[over30to60],bins,alpha=0.5,label='30 < XXL ATCA obj < 60mJ ',normed=True)
plt.hist(XXLATCA_flux[over60to90],bins,alpha=0.5,label='60 < XXL ATCA obj < 90mJ ',normed=True)
plt.hist(XXLATCA_flux[over90],bins,alpha=0.5,label='XXL ATCA obj > 90mJ ',normed=True)

#plt.hist(XXL_flux, bins, alpha=0.5, label='XXL AGN flux',normed=True)
#plt.hist(XXLATCA_flux,bins,alpha=0.5,label='XXL ATCA flux',normed = True)
plt.hist(rand_flux, bins, alpha=0.5, label='random pix flux',normed=True)
plt.legend(loc = 'best')
plt.ylabel('Number')
plt.xlabel('Flux value from filtered map')


## Plotting the XXL ATCA 15's radio flux from catalogue with fluxes from SPT map
below30mean = np.mean(XXLATCA_flux[below30])
below30std = np.std(XXLATCA_flux[below30])
over30to60mean = np.mean(XXLATCA_flux[over30to60])
over30to60std = np.std(XXLATCA_flux[over30to60])
over60to90mean = np.mean(XXLATCA_flux[over60to90])
over60to90std = np.std(XXLATCA_flux[over60to90])
over90mean = np.mean(XXLATCA_flux[over90])
over90std = np.std(XXLATCA_flux[over90])
mean = np.asarray([below30mean, over30to60mean, over60to90mean, over90mean])
std = np.asarray([below30std, over30to60std,over60to90std,over90std])
radioflux = np.asarray([15, 45, 75, 150])
plt.figure()
plt.plot(XXLATCA_radioflux,XXLATCA_flux,'.', label='data')
plt.errorbar(radioflux,mean,yerr=std,fmt='o',label='mean and std in each coloured bin')
plt.axvspan(0,30, color='red', alpha=0.3)
plt.axvspan(30,60, color='green', alpha=0.3)
plt.axvspan(60, 90, color='blue', alpha=0.3)
plt.axvspan(90,250,color='yellow',alpha=0.3)
plt.legend(loc='best')
plt.title('XXL ATCA 15 sources')
plt.xlabel('Radio flux (mJy)')
plt.ylabel('Flux from SPT map')


XXLATCA_goodpixy = np.array(XXLATCA_goodpixy)
XXLATCA_goodpixx = np.array(XXLATCA_goodpixx)
mask_with_XXLATCA = mask.copy()
mask_with_XXLATCA[XXLATCA_goodpixy, XXLATCA_goodpixx] = 0

n_realizations = 1000
rndbelow30mean = np.zeros((n_realizations))
rndover30to60mean = np.zeros((n_realizations))
rndover60to90mean = np.zeros((n_realizations))
rndover90mean = np.zeros((n_realizations))

for j in range(n_realizations):
	rand_ypix = np.round(np.random.rand(4000)*(npixels[0]-1))
	rand_xpix = np.round(np.random.rand(4000)*(npixels[0]-1))
	rand_goodpixy = []
	rand_goodpixx = []
	rand_flux = []
	for i in range(len(rand_xpix)):
		if pixelmask[rand_ypix[i],rand_xpix[i]] == 1:
			if mask_with_XXLATCA[rand_ypix[i],rand_xpix[i]] >0.5:
				rand_goodpixy.append(rand_ypix)
				rand_goodpixx.append(rand_xpix)
				rand_flux.append(filt_map[rand_ypix[i],rand_xpix[i]])
	rndbelow30mean[j] = np.mean(rand_flux[0:len(below30)])
	rndover30to60mean[j] = np.mean(rand_flux[len(below30):len(below30)+len(over30to60)])
	rndover60to90mean[j] = np.mean(rand_flux[len(below30)+len(over30to60):len(below30)+len(over30to60)+len(over60to90)])
	rndover90mean[j] = np.mean(rand_flux[len(below30)+len(over30to60)+len(over60to90):len(below30)+len(over30to60)+len(over60to90)+len(over90)])

randommean = np.asarray([np.mean(rndbelow30mean), np.mean(rndover30to60mean), np.mean(rndover60to90mean), np.mean(rndover90mean)])
randomstd = np.asarray([np.std(rndbelow30mean), np.std(rndover30to60mean), np.std(rndover60to90mean), np.std(rndover90mean)])
SN = mean/randomstd
plt.figure()
plt.plot(XXLATCA_radioflux,XXLATCA_flux,'.', label='data')
plt.errorbar(radioflux,mean,yerr=std,fmt='o',label='mean and std in each coloured bin')
plt.errorbar(radioflux,randommean,yerr=randomstd,fmt='o',label='mean and std over 1000 reals')
plt.axvspan(0,30, color='red', alpha=0.3)
plt.axvspan(30,60, color='green', alpha=0.3)
plt.axvspan(60, 90, color='blue', alpha=0.3)
plt.axvspan(90,250,color='yellow',alpha=0.3)
plt.legend(loc='best')
plt.title('XXL ATCA 15 sources')
plt.xlabel('Radio flux (mJy)')
plt.ylabel('Flux from SPT map')
