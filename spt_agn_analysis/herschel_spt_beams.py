import numpy as np
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
from sptpol_software.simulation.quick_flatsky_routines import make_fft_grid

import scipy.constants as const
from scipy.optimize import curve_fit
c_c = const.c
k_c = const.k
h_c = const.h

from datetime import datetime


mapfreq = "250"

beamsizes = np.asarray([18.2, 24.9, 36.3])/60. #beams for 250, 350, 500um in arcmins
#loading the data
beamfile = "/data/tangq/SPIRE_maps/spire_beams_for_matched_filter.sav"
beam_idldata = idl.readIDLSav(beamfile)
b250 = beam_idldata.spire_beams[0,:]
b350 = beam_idldata.spire_beams[1,:]
b500 = beam_idldata.spire_beams[2,:]


for mapfreq in ["250", "350","500"]:
	if mapfreq == "250":
		fits0 = fits.open("/data/tangq/SPIRE_maps/spt_zea_itermap_10_iterations_15_arcsec_pixels_PSW.fits")
		beamsize = 0
		source_mask = '/data/tangq/SPIRE_maps/herschel_srcs_40_PSW.txt'
		beam = b250

	elif mapfreq == "350":
		fits0 = fits.open("/data/tangq/SPIRE_maps/spt_zea_itermap_10_iterations_15_arcsec_pixels_PMW.fits")
		beamsize = 1
		source_mask = '/data/tangq/SPIRE_maps/herschel_srcs_40_PMW.txt'
		beam = b350
	elif mapfreq == "500":
		fits0 = fits.open("/data/tangq/SPIRE_maps/spt_zea_itermap_10_iterations_15_arcsec_pixels_PLW.fits")
		beamsize = 2
		source_mask = '/data/tangq/SPIRE_maps/herschel_srcs_40_PLW.txt'
		beam = b500
	else:
		print "Enter valid map freq!!"

	radec0 = np.asarray([fits0[1].header['CRVAL1'], fits0[1].header['CRVAL2']])
	reso_arcmin = abs(fits0[1].header['CD1_1'])*60 
	map0 = np.rot90(np.rot90(fits0[1].data))
	npixels = map0.shape

	mask = maps.makeApodizedPointSourceMask(map0, source_mask=source_mask, ftype = 'herschel', reso_arcmin=reso_arcmin, center=radec0, radius_arcmin=3)
	masked_map = np.multiply(mask,map0)
	masked_map = np.nan_to_num(masked_map)

	print "For map " + mapfreq + " the RMS value is: "+ str(np.std(masked_map[mask>0.1]))

	fft_masked_map = np.fft.fft2(masked_map)

	
	ellgrid = make_fft_grid(np.radians(reso_arcmin)/60.,npixels[1],npixels[0])*2.*np.pi
	bl2d = beam[(np.round(ellgrid)).astype(int)]

	beam_divided_fft = fft_masked_map/bl2d

	if mapfreq == "250":
		plt.subplot(3,3,1)
		plt.imshow(np.abs(beam_divided_fft))
		plt.colorbar()
		plt.title('FFT 250um map, beam divided')

		plt.subplot(3,3,4)
		plt.imshow(np.abs(fft_masked_map))
		plt.colorbar()
		plt.title('FFT 250um map')

		plt.subplot(3,3,7)
		plt.imshow(bl2d)
		plt.colorbar()
		plt.title('Beam, 250um')

		bdf_map250 = beam_divided_fft
		ellgrid_250 = ellgrid
	elif mapfreq == "350":
		plt.subplot(3,3,2)
		plt.imshow(np.abs(beam_divided_fft))
		plt.colorbar()
		plt.title('FFT 350um map, beam divided')

		plt.subplot(3,3,5)
		plt.imshow(np.abs(fft_masked_map))
		plt.colorbar()
		plt.title('FFT 350um map')

		plt.subplot(3,3,8)
		plt.imshow(bl2d)
		plt.colorbar()
		plt.title('Beam, 350um')
		bdf_map350 = beam_divided_fft
		ellgrid_350 = ellgrid
	elif mapfreq == "500":
		plt.subplot(3,3,3)
		plt.imshow(np.abs(beam_divided_fft))
		plt.colorbar()
		plt.title('FFT 500um map, beam divided')

		plt.subplot(3,3,6)
		plt.imshow(np.abs(fft_masked_map))
		plt.colorbar()
		plt.title('FFT 500um map')

		plt.subplot(3,3,9)
		plt.imshow(bl2d)
		plt.colorbar()
		plt.title('Beam, 500um')

		bdf_map500 = beam_divided_fft
		ellgrid_500 = ellgrid


common_fwhm = 1.7 #in arcmin
bfilt250 = beamfilt(ellgrid_250, common_fwhm)
bfilt350 = beamfilt(ellgrid_350, common_fwhm)
bfilt500 = beamfilt(ellgrid_500, common_fwhm)

cb_map250 = bdf_map250*bfilt250
cb_map350 = bdf_map350*bfilt350
cb_map500 = bdf_map500*bfilt500

plt.figure()
plt.subplot(2,3,1)
plt.imshow(np.abs(cb_map250))
plt.colorbar()
plt.title('FFT 250um map with '+str(common_fwhm)+' arcmin beam')
plt.subplot(2,3,2)
plt.imshow(np.abs(cb_map350))
plt.colorbar()
plt.title('FFT 350um map with '+str(common_fwhm)+' arcmin beam')
plt.subplot(2,3,3)
plt.imshow(np.abs(cb_map500))
plt.colorbar()
plt.title('FFT 500um map with '+str(common_fwhm)+' arcmin beam')

cb2_map250 = np.fft.ifft2(cb_map250)
cb2_map350 = np.fft.ifft2(cb_map350)
cb2_map500 = np.fft.ifft2(cb_map500)

plt.subplot(2,3,4)
plt.imshow(np.real(cb2_map250),vmax =0.042, vmin = -0.012)
plt.colorbar()
plt.title('250um map with '+str(common_fwhm)+' arcmin beam, real')
plt.subplot(2,3,5)
plt.imshow(np.real(cb2_map350), vmax=0.036,vmin=-0.012)
plt.colorbar()
plt.title('350um map with '+str(common_fwhm)+' arcmin beam, real')
plt.subplot(2,3,6)
plt.imshow(np.real(cb2_map500),vmax=0.06, vmin=-0.02)
plt.colorbar()
plt.title('500um map with '+str(common_fwhm)+' arcmin beam, real')


print "For map 250um the RMS value is: "+ str(np.std(cb2_map250[mask>0.1]))
print "For map 350um the RMS value is: "+ str(np.std(cb2_map350[mask>0.1]))
print "For map 500um the RMS value is: "+ str(np.std(cb2_map500[mask>0.1]))


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



import sptpol_software.scratch.tnatoli.transient.calc_optfilt_oneband_2d_amy as calc
#filt_output,filt_map =  calc.call_matched_filt_point(zp_mask_map, cmb = False, skysigvar = np.zeros((25000)), noisepsd=noise_psd, ellmax = 25000, ell_highpass = 10, ell_lowpass = 20000,use_fft_as_noise=False, plot_map=False, return_filtered_map=True, mask = zp_mask, reso_arcmin = reso_arcmin, ngridx = zp_mask_map.shape[0],ngridy = zp_mask_map.shape[0])
m_filt =calc.calc_optfilt_oneband_2d([0,0], np.zeros((25000)), noise_psd, reso_arcmin,fwhm_beam_arcmin = 1E-6, ell_highpass = 10,ell_lowpass = 20000,bfilt = bfilt,pointsource = True,sigspect = 1,keepdc = False, arcmin_sigma = None,norm_unfilt = None,external_profile_norm = None,f_highpass = None, f_lowpass = None,scanspeed = None)


plt.subplot(1,3,1)
plt.imshow((ptsrc_map250 - cb2_map250 ).real)
plt.colorbar()
plt.title('250um, map w/ ptsrc - map w/o')
plt.subplot(1,3,2)
plt.imshow((ptsrc_map350 - cb2_map350 ).real)
plt.colorbar()
plt.title('350um, map w/ ptsrc - map w/o')
plt.subplot(1,3,3)
plt.imshow((ptsrc_map500 - cb2_map500 ).real)
plt.colorbar()
plt.title('500um, map w/ ptsrc - map w/o')

b90 = np.loadtxt('/data/tangq/SPT/bl_2010_90.txt')
b150 = np.loadtxt('/data/tangq/SPT/bl_2010_150.txt')
b220 = np.loadtxt('/data/tangq/SPT/bl_2010_220.txt')

plt.plot(b90[:,0], b90[:,1], label='90GHz')
plt.plot(b150[:,0], b150[:,1], label='150GHz')
plt.plot(b220[:,0], b220[:,1], label='220GHz')
plt.xlabel('ell')
plt.legend()
plt.ylabel('Beam response')
plt.title('SPT beams')
plt.figure()
plt.plot(b250, label='250um')
plt.plot(b350, label='350um')
plt.plot(b500, label='500um')
plt.legend()
plt.ylabel('beam response')
plt.title('Herschel beams')

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


########## Hershel:
mapfreq = "500"
field = "Herschel_15arcsecs"


idldata = idl.readIDLSav("/data/tangq/SPIRE_maps/spire_beams_for_matched_filter.sav")


#loading the data
if mapfreq == "250":
	fits0 = fits.open("/data/tangq/SPIRE_maps/spt_zea_itermap_10_iterations_15_arcsec_pixels_PSW.fits")
	beam = idldata.spire_beams[0,:]
	source_mask = '/data/tangq/SPIRE_maps/herschel_srcs_40_PSW.txt'

elif mapfreq == "350":
	fits0 = fits.open("/data/tangq/SPIRE_maps/spt_zea_itermap_10_iterations_15_arcsec_pixels_PMW.fits")
	beam = idldata.spire_beams[1,:]
	source_mask = '/data/tangq/SPIRE_maps/herschel_srcs_40_PMW.txt'
elif mapfreq == "500":
	fits0 = fits.open("/data/tangq/SPIRE_maps/spt_zea_itermap_10_iterations_15_arcsec_pixels_PLW.fits")
	beam = idldata.spire_beams[2,:]
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

filt_output,filt_map =  calc.call_matched_filt_point_common_beam(zp_mask_map, cmb = False, skysigvar = np.zeros((25000)), noisepsd=noise_psd, beam = beam, fwhm_beam_arcmin=1.7, ellmax = 25000, ell_highpass = 10, ell_lowpass = 20000,use_fft_as_noise=False, plot_map=False, return_filtered_map=True, mask = zp_mask, reso_arcmin = reso_arcmin, ngridx = zp_mask_map.shape[0],ngridy = zp_mask_map.shape[0])
cmap = filt_map

trans_func_beam[trans_func_beam < 0.01] = 0.01

plt.subplot(2,3,1)
plt.imshow(np.abs(fft_masked_map))
plt.colorbar()
plt.title('FFT of masked map')

plt.subplot(2,3,4)
plt.imshow(trans_func_beam)
plt.colorbar()
plt.title('Trans func+beam')

plt.subplot(2,3,2)
plt.imshow(np.abs(fft_masked_map/trans_func_beam))
plt.colorbar()
plt.title('FFT masked map/trans func')

common_beam = 1.7
bfilt = t.beamfilt(ellg, common_beam)
plt.subplot(2,3,5)
plt.imshow(bfilt)
plt.colorbar()
plt.title(str(common_beam)+' arcmin beam')

plt.subplot(2,3,3)
plt.imshow(np.abs(fft_masked_map/trans_func_beam*bfilt))
plt.colorbar()
plt.title('FFT map * ' +str(common_beam)+' arcmin beam')

plt.subplot(2,3,6)
plt.imshow(np.fft.ifft2(fft_masked_map/trans_func_beam*bfilt).real)
plt.colorbar()
plt.title('IFFT of FFT of map with common beam')

mask > 0.1
orig_map_rms = np.std(masked_map[mask>0.1])
ifft_map = np.fft.ifft2(fft_masked_map/trans_func_beam*bfilt).real
print str(common_beam) + ' arcmin beam map has rms ' + str(np.std(ifft_map[mask>0.1]))


common_beam = 2
bfilt = t.beamfilt(ellg, common_beam)
bfilt += 1.e-10
plt.imshow(trans_func_beam/bfilt, vmin = 0, vmax =10)
plt.colorbar()


#####
zp_mask > 0.1
orig_map_rms = np.std(zp_mask_map[zp_mask>0.1])
"original Herschel masked map rms = 0.011101934391575531"


