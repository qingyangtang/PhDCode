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
import sptpol_software.constants as constants
from sptpol_software.util.math import makeEllGrids
from sptpol_software.analysis import plotting


idldata = idl.readIDLSav("/data/tangq/SPIRE_maps/spire_beams_for_matched_filter.sav")

beamsizes = np.asarray([18.2, 24.9, 36.3])/60. #beams for 250, 350, 500um in arcmins

trans_nobeam_idldata = idl.readIDLSav('/data/tangq/SPT/SZ/ra23h30dec-55_2year_xfer_function_no_beam_padded_map.sav')
trans_func = trans_nobeam_idldata.tfgrids_nobeam[2]
#loading the data
mapfreq = "250"
common_beam = 1.7
if mapfreq == "250":
	idlmap = idl.readIDLSav("/data/tangq/SPIRE_maps/proj0_herschel_maps/spt_zea_itermap_10_iterations_15_arcsec_pixels_PSW.fits_v2.sav")
	beamsize = 0
	beam = idldata.spire_beams[0,:]
	source_mask = '/data/tangq/SPIRE_maps/herschel_srcs_40_PSW_proj0.txt'

elif mapfreq == "350":
	idlmap = idl.readIDLSav("/data/tangq/SPIRE_maps/proj0_herschel_maps/spt_zea_itermap_10_iterations_15_arcsec_pixels_PMW.fits_v2.sav")
	beamsize = 1
	beam = idldata.spire_beams[1,:]
	source_mask = '/data/tangq/SPIRE_maps/herschel_srcs_40_PMW_proj0.txt'
elif mapfreq == "500":
	idlmap = idl.readIDLSav("/data/tangq/SPIRE_maps/proj0_herschel_maps/spt_zea_itermap_10_iterations_15_arcsec_pixels_PLW.fits_v2.sav")
	beamsize = 2
	beam = idldata.spire_beams[2,:]
	source_mask = '/data/tangq/SPIRE_maps/herschel_srcs_40_PLW_proj0.txt'
else:
	print "Enter valid map freq!!"

radec0 = idlmap.values()[1]
reso_arcmin = idlmap.values()[0]
map0 = np.rot90(np.rot90(idlmap.values()[2]))
npixels = map0.shape

mask = maps.makeApodizedPointSourceMask(map0, source_mask, ftype = 'herschel', reso_arcmin=np.float(reso_arcmin), center=radec0, radius_arcmin=3)
masked_map = mask*map0

#initial_map, dummy_mask = t.zero_pad_map(map0, np.ones(map0.shape))
reso_rad = reso_arcmin * constants.DTOR / 60.
ellg,ellx,elly = makeEllGrids([npixels[1],npixels[0]],reso_rad)
bl2d = beam[(np.round(ellg)).astype(int)]
smoothed_map, cb_trans_func = t.map_smoothing(masked_map,bl2d, trans_func,ellg, common_fwhm=1.7)
#map0 = smoothed_map
zp_mask_map, zp_mask = t.zero_pad_map(smoothed_map, mask)

#calculating noise PSD
noise_psd = t.noise_PSD(zp_mask_map, reso_arcmin, dngrid=zp_mask_map.shape[0])

if common_beam == None:
	filt_output,filt_map =  calc.call_matched_filt_point(zp_mask_map, cmb = False, skysigvar = np.zeros((25000)), noisepsd=noise_psd, \
							fwhm_beam_arcmin=beamsizes[beamsize], ellmax = 25000, ell_highpass = 10, ell_lowpass = 20000,use_fft_as_noise=False, \
							plot_map=False, return_filtered_map=True, mask = zp_mask, reso_arcmin = reso_arcmin, ngridx = zp_mask_map.shape[0],\
							ngridy = zp_mask_map.shape[0])
else:
	filt_output,filt_map =  calc.call_matched_filt_point(smoothed_map, cmb = False, skysigvar = np.zeros((25000)), noisepsd=noise_psd, \
							beam=bl2d,trans_func_grid = cb_trans_func, use_fft_as_noise=False, \
							plot_map=False, return_filtered_map=True, mask = zp_mask, reso_arcmin = reso_arcmin, ngridx = zp_mask_map.shape[0],\
							ngridy = zp_mask_map.shape[0])		


mapfreq = ['90', '150', '220']
mapindex = [0,1,2]
idldata = idl.readIDLSav("/data/tangq/SPT/coadds_latest.sav")
SZ_data = idl.readIDLSav("/data/tangq/SPT/SZ/ra23h30dec-55_2year_padded.sav")
apod_mask = idl.readIDLSav("/data/tangq/SPT/SZ/ra23h30dec-55_2year_padded_apod.sav").apmask_field

for f in range(3):
	
	idlpsd = idl.readIDLSav("/data/tangq/SPT/psd_ra23h30dec-55_2year_"+mapfreq[f]+"_wtd.sav")

	tom_map = idldata.maps[mapindex[f]]
	weights = idldata.weights[mapindex[f],:,:]
	fitsdata = files.readSptData('/home/tangq/data/masks_allbands_0p8.fits')
	apodmask = fitsdata.masks['apod_mask']

	fft_tom = np.fft.fft2(apodmask*tom_map)
	reso_arcmin = 0.25
	reso_rad = reso_arcmin * constants.DTOR / 60.
	tom_ellg,t_ellx,t_elly = makeEllGrids([tom_map.shape[1],tom_map.shape[0]],reso_rad)

	fft_tom_high_ell = fft_tom
	fft_tom_high_ell[tom_ellg < 4000] = 0
	fft_tom_high_ell[tom_ellg > 20000] = 0

	# plt.figure()
	# plt.subplot(231)
	# plt.imshow(np.abs(fft_tom_high_ell))
	# plt.colorbar()
	# plt.title('Tom ' + mapfreq[f] + ' GHz FFT map, l<4000 masked')

	# plt.subplot(234)
	# plt.imshow(tom_ellg)
	# plt.colorbar()
	# plt.title('Ell Grid for Tom map')

	SZ_map = SZ_data.maps[mapindex[f]]
	fft_sz = np.fft.fft2(apod_mask*SZ_map)
	sz_ellg,s_ellx,s_elly = makeEllGrids([SZ_map.shape[1],SZ_map.shape[0]],reso_rad)

	fft_sz_high_ell = fft_sz
	fft_sz_high_ell[sz_ellg < 4000] = 0
	fft_sz_high_ell[sz_ellg > 20000] = 0

	# plt.subplot(232)
	# plt.imshow(np.abs(fft_sz_high_ell))
	# plt.colorbar()
	# plt.title('SZ ' + mapfreq[f] + ' GHz FFT map, l<4000 masked')

	# plt.subplot(235)
	# plt.imshow(sz_ellg)
	# plt.colorbar()
	# plt.title('Ell Grid for SZ map')

	from scipy.interpolate import interp1d

	tom_ellg_1d = tom_ellg.flatten() 
	fft_tom_high_ell_1d = fft_tom_high_ell.flatten()

	func = interp1d(tom_ellg_1d, fft_tom_high_ell_1d)

	sz_ellg_1d = sz_ellg.flatten()
	fft_tom_interp_1d = func(sz_ellg_1d)

	fft_tom_rebinned = fft_tom_interp_1d.reshape(fft_sz_high_ell.shape)

	# plt.subplot(233)
	# plt.imshow(np.abs(fft_tom_rebinned))
	# plt.colorbar()
	# plt.title('Tom ' + mapfreq[f] + ' GHz FFT map, rebinned')

	a = maps.bin2DPowerSpectrum(fft_tom_high_ell,tom_ellg)
	b = maps.bin2DPowerSpectrum(fft_tom_rebinned, sz_ellg)
	c = maps.bin2DPowerSpectrum(fft_sz_high_ell, sz_ellg)

	from sptpol_software.simulation import quick_flatsky_routines as qfr

	cl_dict1 = qfr.cl_flatsky(tom_map, reso_arcmin = 0.25, apod_mask=apodmask)
	plt.plot(cl_dict1['ell'], cl_dict1['cl']['TT'], color='blue', ls = 'solid',label='Tom map')
	cl_dict = qfr.cl_flatsky(SZ_map, reso_arcmin = 0.25, apod_mask=apod_mask)
	plt.plot(cl_dict['ell'], cl_dict['cl']['TT']/factor, color='orange', ls = 'solid',label='SZ map')
	trans_nobeam_idldata = idl.readIDLSav('/data/tangq/SPT/SZ/ra23h30dec-55_2year_xfer_function_no_beam_padded_map.sav')
	trans_func = trans_nobeam_idldata.tfgrids_nobeam[f]
	#cl_dict = qfr.cl_flatsky(SZ_map, reso_arcmin = 0.25, apod_mask=apod_mask, trans_func_grid = trans_func)
	#plt.plot(cl_dict['ell'], cl_dict['cl']['TT'], color='orange', ls = 'dotted',label='SZ map, trans func')
	plt.legend(loc='best')
	plt.xlabel('ell')
	plt.xlim(0,25000)
	plt.semilogy()



	plotting.plotRadialProfile(np.abs(fft_tom_high_ell), center_bin=[tom_ellg.shape[0]/2,tom_ellg.shape[1]/2], \
		resolution=(t_ellx[0,1]-t_ellx[0,0]), histogram=True, range=[0,25000],plot=True, bin_resolution_factor=1,new=False)

	sz_ellg_1d = sz_ellg.flatten() 
	fft_sz_high_ell_1d = fft_sz_high_ell.flatten()

	func1 = interp1d(sz_ellg_1d, fft_sz_high_ell_1d,fill_value="extrapolate")

	tom_ellg_1d = tom_ellg.flatten() 
	fft_sz_interp_1d = func1(tom_ellg_1d)

	fft_sz_rebinned = fft_sz_interp_1d.reshape(fft_tom_high_ell.shape)

	plt.subplot(236)
	plt.imshow(np.abs(fft_sz_rebinned))
	plt.colorbar()
	plt.title('SZ ' + mapfreq[f] + ' GHz FFT map, rebinned')



	plt.plot(np.diagonal(tom_ellg), np.diagonal(np.abs(fft_tom_high_ell)), alpha=0.3, label='Tom')
	plt.plot(np.diagonal(sz_ellg), np.diagonal(np.abs(fft_tom_rebinned)), alpha=0.3,label='Tom rebinned')
	plt.plot(np.diagonal(sz_ellg), np.diagonal(np.abs(fft_sz_high_ell)), alpha=0.3,label='SZ')
	plt.legend()

	plt.xlabel('ell')





	# fft_sz_rebinned = np.zeros((fft_tom_high_ell.shape),dtype = complex)

	# # for i in range(fft_tom_high_ell.shape[0]):
	# # 	for j in range(fft_tom_high_ell.shape[1]):
	# # 		idx_x, idx_y = np.where(np.abs(sz_ellg-tom_ellg[i,j])==np.min(np.abs(sz_ellg-tom_ellg[i,j])))
	# # 		fft_sz_rebinned[i,j] = fft_tom_high_ell[idx_x[0], idx_y[0]]

	# for i in range(fft_tom_high_ell.shape[0]/2 + 1):
	# 	for j in range(i, fft_tom_high_ell.shape[1]/2 +1 ):
	# 		idx = (np.abs(sz_ellg[:sz_ellg.shape[0]/2,:sz_ellg.shape[1]/2] - tom_ellg[i,j])).argmin()
	# 		fft_sz_rebinned[i,j] = fft_tom_high_ell[idx / sz_ellg[:sz_ellg.shape[0]/2,:sz_ellg.shape[1]/2].shape[1], idx % sz_ellg[:sz_ellg.shape[0]/2,:sz_ellg.shape[1]/2].shape[1]]
	# 	print i

idldata = idl.readIDLSav("/data/tangq/SPT/coadds_latest.sav")
SZ_data = idl.readIDLSav("/data/tangq/SPT/SZ/ra23h30dec-55_2year_padded.sav")
apod_mask = idl.readIDLSav("/data/tangq/SPT/SZ/ra23h30dec-55_2year_padded_apod.sav").apmask_field
mapinfo_idldata = idl.readIDLSav("/data/tangq/SPT/SZ/ra23h30dec-55_2year_map_info.sav")
fitsdata = files.readSptData('/home/tangq/data/masks_allbands_0p8.fits') 

tom_map = idldata.maps[mapindex[f]]
SZ_map = SZ_data.maps[mapindex[f]]

SZ_radec0 = mapinfo_idldata.radec0
SZ_reso_arcmin = mapinfo_idldata.reso_arcmin
SZ_proj = mapinfo_idldata.projection 
SZ_npixels = SZ_map.shape
SZ_mask = apod_mask

tom_radec0 = np.asarray( [352.50226, -55.000390])
tom_proj = 0
tom_reso_arcmin = 0.25
tom_npixels = tom_map.shape
tom_mask = fitsdata.masks['apod_mask']

plt.figure(figsize=(15,5))
plt.subplot(131)
plt.imshow(tom_map*tom_mask)
plt.colorbar()
plt.title('Tom map')
plt.subplot(132)
plt.imshow(SZ_map*SZ_mask)
plt.colorbar()
plt.title('SZ map')
plt.subplot(133)
plt.imshow(tom_map*tom_mask-(SZ_map*SZ_mask)[420:3780,420:3780])
plt.colorbar()
plt.title('Tom-SZ')
plt.savefig('/data/tangq/debug/20200926_tom_SZ_map_cutout.png')

map_array = ['90', '150', '220']
mapfreq = map_array[f]
field = "ra23h30dec-55_allyears"
source_mask = '/data/tangq/wendymaps/'+field+'/source_list_3sig_allyears_'+mapfreq+'.dat'
source_list = np.loadtxt(source_mask)
src_ra = source_list[:,3]
src_dec = source_list[:,4]
src_sn = source_list[:,5]
src_flux = source_list[:,6]
src_cat =Table([src_ra, src_dec, src_sn, src_flux],names = ('ra', 'dec', 'sn','flux'))
sig5_src = source_list[src_sn>5,:]
sig4_mask = (src_cat['sn'] > 4) & (src_cat['sn'] < 5)
sig4_src = src_cat[sig4_mask]

sig4_ra = np.asarray(sig4_src['ra'][:5])
sig4_dec = np.asarray(sig4_src['dec'][:5])
sig5_ra = sig5_src[:,3][:5]
sig5_dec = sig5_src[:,4][:5]

goodpix = []

tom_ypix, tom_xpix = sky.ang2Pix(np.asarray([sig5_ra, sig5_dec]), tom_radec0, tom_reso_arcmin, tom_npixels,proj=0)[0]
tom_ypixmask, tom_xpixmask = sky.ang2Pix(np.asarray([sig5_ra, sig5_dec]), tom_radec0, tom_reso_arcmin, tom_npixels,proj=0)[1] #tells me if the pixel is in the map or not (theoretically should be but apparently not..?)

SZ_ypix, SZ_xpix = sky.ang2Pix(np.asarray([sig5_ra, sig5_dec]), SZ_radec0, SZ_reso_arcmin, SZ_npixels,proj=0)[0]
SZ_ypixmask, SZ_xpixmask = sky.ang2Pix(np.asarray([sig5_ra, sig5_dec]), SZ_radec0, SZ_reso_arcmin, SZ_npixels,proj=0)[1] #tells me if the pixel is in the map or not (theoretically should be but apparently not..?)

for i in range(len(tom_ypix)):
	if tom_ypixmask[i] == True & tom_xpixmask[i] == True & SZ_ypixmask[i] == True & SZ_xpixmask[i] == True:
		if np.bool(tom_mask[tom_ypix[i],tom_xpix[i]]) == True & np.bool(SZ_mask[SZ_ypix[i],SZ_xpix[i]]) == True:
			goodpix.append(i)

for i in goodpix:
	plt.figure(figsize=(15,5))
	plt.subplot(131)
	plt.imshow(tom_map[tom_ypix[i]-20:tom_ypix[i]+20, tom_xpix[i]-20:tom_xpix[i]+20])
	plt.colorbar()
	plt.title('Tom map, at [' + str(round(sig5_ra[i], 2)) + ',' + str(round(sig5_dec[i], 2))+']')
	plt.subplot(132)
	plt.imshow(SZ_map[SZ_ypix[i]-20:SZ_ypix[i]+20, SZ_xpix[i]-20:SZ_xpix[i]+20])
	plt.colorbar()
	plt.title('SZ map, at [' + str(round(sig5_ra[i], 2)) + ',' + str(round(sig5_dec[i], 2))+']')
	plt.subplot(133)
	plt.imshow(tom_map[tom_ypix[i]-20:tom_ypix[i]+20, tom_xpix[i]-20:tom_xpix[i]+20]-SZ_map[SZ_ypix[i]-20:SZ_ypix[i]+20, SZ_xpix[i]-20:SZ_xpix[i]+20])
	plt.colorbar()
	plt.title('Tom-SZ, at [' + str(round(sig5_ra[i], 2)) + ',' + str(round(sig5_dec[i], 2))+']')
	plt.savefig('/data/tangq/debug/20200926_sig5_src'+str(i)+'_tom_SZ_map_cutout.png')


sig4_ra = np.asarray([349.33, 344.13, 350.27, 358.3, 350.49])
sig4_dec = np.asarray([-53.11, -60.32, -50.45, -53.19, -51.43])
sig5_ra = np.asarray([359.47,352.34,348.94, 354.05, 353.69])
sig5_dec = np.asarray([-53.19, -49.93, -50.31, -52.6, -52.85])

cat = np.loadtxt(source_mask)
cat_ra = cat[:,0]
cat_dec= cat[:,1]
cat_ypix, cat_xpix = sky.ang2Pix(np.asarray([cat_ra, cat_dec]), radec0, reso_arcmin, npixels,proj=0)[0]
cat_ypixmask, cat_xpixmask = sky.ang2Pix(np.asarray([cat_ra, cat_dec]), radec0, reso_arcmin, npixels,proj=0)[1] 
for i in range(10):
	plt.figure()
	plt.subplot(1,2,1)
	plt.imshow(map0[cat_ypix[i]-40:cat_ypix[i]+40, cat_xpix[i]-40:cat_xpix[i]+40])
	# plt.subplot(1,2,2)
	# plt.imshow(masked_map[cat_ypix[i]-40:cat_ypix[i]+40, cat_xpix[i]-40:cat_xpix[i]+40])
