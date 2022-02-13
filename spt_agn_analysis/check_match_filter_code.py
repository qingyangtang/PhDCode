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
from sptpol_software.util.math import makeEllGrids
from sptpol_software.analysis import plotting
import sptpol_software.constants as constants
from sptpol_software.util.files import readCAMBCls


mapfreq = "150"
field = "ra23h30dec-55_allyears"

idldata = idl.readIDLSav("/data/tangq/SPT/SZ/ra23h30dec-55_2year_padded.sav")
apod_mask = idl.readIDLSav("/data/tangq/SPT/SZ/ra23h30dec-55_2year_padded_apod.sav").apmask_field
ap_mask = idl.readIDLSav("/data/tangq/SPT/SZ/ra23h30dec-55_2year_padded_apod.sav").apmask
trans_idldata = idl.readIDLSav("/data/tangq/SPT/SZ/ra23h30dec-55_2year_xfer_function_with_beam_padded_map.sav")
mapinfo_idldata = idl.readIDLSav("/data/tangq/SPT/SZ/ra23h30dec-55_2year_map_info.sav")
source_mask = '/home/tnatoli/gits_sptpol/sptpol_software/config_files/ptsrc_config_ra23h30dec-55_surveyclusters.txt'
trans_nobeam_idldata = idl.readIDLSav('/data/tangq/SPT/SZ/ra23h30dec-55_2year_xfer_function_no_beam_padded_map.sav')
radec0 = mapinfo_idldata.radec0
reso_arcmin = mapinfo_idldata.reso_arcmin
proj = mapinfo_idldata.projection


for mapfreq in ["90", "150", "220"]:

	source_mask = '/data/tangq/wendymaps/'+field+'/source_list_3sig_allyears_'+mapfreq+'.dat'
	source_list = np.loadtxt(source_mask)
	src_ra = source_list[:,3]
	src_dec = source_list[:,4]
	src_sn = source_list[:,5]
	src_flux = source_list[:,6]
	src_cat =Table([src_ra, src_dec, src_sn, src_flux],names = ('ra', 'dec', 'sn','flux'))
	sig5_src = source_list[src_sn>5,:]
	sig5_src1 = np.vstack((np.arange(len(sig5_src[:,0]))+1, sig5_src[:,3], sig5_src[:,4], np.ones(len(sig5_src[:,0]))*0.0833)).T
	np.savetxt('/data/tangq/wendymaps/'+field+'/source_list_3sig_allyears_'+mapfreq+'5sig.txt', sig5_src1)
	sig4_mask = (src_cat['sn'] > 4) & (src_cat['sn'] < 5)
	sig4_src = src_cat[sig4_mask]

#	sig4_src =od.get_SPTSZ_ptsrcs_wendy(mapfreq,field)

	### running match filtering on SPT-SZ data
	noise_psd = idl.readIDLSav("/data/tangq/SPT/SZ/psd_ra23h30dec-55_2year_"+mapfreq+"_wtd.sav").psd
	calfacs =[0.722787, 0.830593, 0.731330]
	if mapfreq =="90":
		cmap = idldata.maps[0]
		trans_func_beam = trans_idldata.transfer_with_beam[0]
		trans_func = trans_nobeam_idldata.tfgrids_nobeam[0]
		calfac = calfacs[0]
	elif mapfreq =="150":
		cmap = idldata.maps[1]
		trans_func_beam = trans_idldata.transfer_with_beam[1]
		trans_func = trans_nobeam_idldata.tfgrids_nobeam[1]
		calfac = calfacs[1]
	elif mapfreq =="220":
		cmap = idldata.maps[2]
		trans_func_beam = trans_idldata.transfer_with_beam[2]
		trans_func = trans_nobeam_idldata.tfgrids_nobeam[2]
		calfac = calfacs[2]
	npixels = np.asarray(cmap.shape)
	cmap = cmap * calfac
	source_mask = '/data/tangq/wendymaps/'+field+'/source_list_3sig_allyears_'+mapfreq+'5sig.txt'
	ptsrc_mask = maps.makePointSourceMask(cmap, source_mask, ftype = 'tom', reso_arcmin=reso_arcmin, center=radec0, radius_arcmin=4)
	mask = ptsrc_mask*apod_mask

	masked_map = mask*cmap	

	reso_rad = reso_arcmin * constants.DTOR / 60.
	ellg,ellx,elly = makeEllGrids([npixels[1],npixels[0]],reso_rad)

	if mapfreq == 220:
		mask_ellx = np.abs(ellx)<400
		noise_psd[mask_ellx] = 0


	# idldata = idl.readIDLSav("/data/tangq/SPT/coadds_latest.sav")
	# idlpsd = idl.readIDLSav("/data/tangq/SPT/psd_ra23h30dec-55_2year_"+mapfreq+"_wtd.sav")
	# calfacs =[0.722787, 0.830593, 0.731330]
	# if mapfreq == "150":
	# 	calfac = calfacs[1]
	# 	mapindex = 1
	# elif mapfreq == "90":
	# 	calfac = calfacs[0]
	# 	mapindex = 0
	# elif mapfreq == "220":
	# 	calfac = calfacs[2]
	# 	mapindex = 2
	# calfac = 1.
	# map150 = idldata.maps[mapindex,:,:]*calfac
	# weights = idldata.weights[mapindex,:,:]
	# radec0 = np.asarray([352.50226, -55.000390])
	# reso_arcmin= 0.25
	# npixels = np.asarray(map150.shape)
	# psd150 = idlpsd.psd
	# source_mask = '/data/tangq/wendymaps/'+field+'/source_list_3sig_allyears_'+mapfreq+'5sig.txt'
	# mask = maps.makeApodizedPointSourceMask(map150, source_mask, ftype = 'tom', reso_arcmin=reso_arcmin, center=radec0, radius_arcmin=4)
	# masked_map = map150*mask

	#checking that the 5sig sources are actually masked:
	ypix5sig, xpix5sig = sky.ang2Pix(np.asarray([sig5_src[:,3], sig5_src[:,4]]), radec0, reso_arcmin, npixels,proj=0)[0]
	# plt.imshow(masked_map)
	# plt.colorbar()
	# plt.plot(xpix5sig, ypix5sig, '.')
	beam = '/data/tangq/SPT/bl_2010_'+mapfreq+'.txt'
	skysig0 = idl.readIDLSav("/data/tangq/SPT/viero_cr20_cov_95_150_220_100_143_217_353_.sav")
	mapfreqs = ['90','150','220']
	indx = mapfreqs.index(mapfreq)
	skysig = skysig0['test'][:,indx,indx]
	ellmax = len(skysig)-1
	cls = readCAMBCls('/data/tnatoli/camb/planck_lensing_wp_highL_bestFit_20130627_massless0p046_massive3_lensedtotCls.dat',
	as_cls = True, extrapolate = True, as_uksq = False, max_ell = ellmax-1)
	skysig[cls['ell']] += cls['cl']['TT']
	#############MAKE SURE THAT calc_optfilt_oneband_2d_amy.py CHANGES SO THAT I PASS ON THE ARCMIN RESO AND INFO LIKE THAT
	#filt_output,filt_map =  calc.call_matched_filt_point(masked_map,noisepsd=psd150, beam = beam, fwhm_beam_arcmin=1.1, usbe_fft_as_noise=False, plot_map=False, return_filtered_map=True, mask=mask, f_highpass = 0.2, scanspeed = 0.25)
	filt_output,filt_map =  calc.call_matched_filt_point(cmap, cmb = False, skysigvar = skysig, noisepsd=noise_psd, \
						beam=trans_func_beam,trans_func_grid = trans_func_beam,use_fft_as_noise=False, \
						plot_map=False, return_filtered_map=True, mask = mask, reso_arcmin = reso_arcmin, ngridx = masked_map.shape[0],\
						ngridy = masked_map.shape[0])


	import scipy.constants as const
	Tcmb = 2.725
	x = float(mapfreq)*1.e9*const.h/(const.k*Tcmb)

	reso = (reso_arcmin/60)*np.pi/180.
	dngrid = filt_map.shape[0]
	dk = 1./dngrid/reso
	tauprime = filt_output[0].prof2d*filt_output[0].trans_func_grid #* filt_output[0].area_eff
	solid_angle =1./ np.sum(filt_output[0].optfilt*tauprime)  / dk**2
	conversion = solid_angle*1.e26*(2*const.k/(const.c)**2)*((const.k*Tcmb/const.h)**2)*(x**4)*np.exp(x)/(np.exp(x)-1)**2	

	# if mapfreq =="90":
	# 	beam_fwhm = 1.65
	# elif mapfreq == "150":
	# 	beam_fwhm = 1.2
	# elif mapfreq == "220":
	# 	beam_fwhm = 1.0
	# theta = beam_fwhm/60*(np.pi/180)
	# solid_angle = theta**2*np.pi/(4*np.log(2))
	# conversion = solid_angle*1.e26*(2*const.k/(const.c)**2)*((const.k*Tcmb/const.h)**2)*(x**4)*np.exp(x)/(np.exp(x)-1)**2

	c = SkyCoord(ra=sig4_src['ra']*u.degree, dec=sig4_src['dec']*u.degree)  
	catalog = SkyCoord(ra=sig5_src[:,3]*u.degree, dec=sig5_src[:,4]*u.degree)  
	idx, d2d, d3d = c.match_to_catalog_sky(catalog)

	src_mask = d2d.arcmin > 4
	sig4_ra = sig4_src['ra'][src_mask]
	sig4_dec = sig4_src['dec'][src_mask]
	sig4_flux = sig4_src['flux'][src_mask]

	#for each ptsrc > 4 and <5, get the flux from Wendy's map and from filtered Tom's map:
	ypix4sig, xpix4sig = sky.ang2Pix(np.asarray([sig4_ra, sig4_dec]), radec0, reso_arcmin, npixels,proj=0)[0]
	ypix4sigmask, xpix4sigmask = sky.ang2Pix(np.asarray([sig4_src['ra'], sig4_src['dec']]), radec0, reso_arcmin, npixels,proj=0)[1]

	flux4sig_Tom = np.zeros((len(ypix4sig)))

	# if mapfreq =="150":
	# 	uK2mJy = 0.0735198
	# elif mapfreq == "90":
	# 	uK2mJy = 0.0843136
	# elif mapfreq == "220":
	# 	uK2mJy = 0.0694527
	for i in range(len(ypix4sig)):
		flux4sig_Tom[i] = filt_map[ypix4sig[i], xpix4sig[i]]

	# wendy_factors = np.asarray([84.3136, 73.5198, 69.4527])
	# conversion = wendy_factors[indx]

	uK2mJy = conversion/1000.
	print "For " + mapfreq + " GHz, calculated conversion factor is: " + str(uK2mJy)
	flux4sig_Tom_mjy = flux4sig_Tom*1.e6*uK2mJy

	wendy4sig = sig4_src['flux'][src_mask]

	if mapfreq == "90":
		ratio90 = wendy4sig/flux4sig_Tom_mjy
	elif mapfreq == "150":
		ratio150 = wendy4sig/flux4sig_Tom_mjy
	elif mapfreq == "220":
		ratio220 = wendy4sig/flux4sig_Tom_mjy

plt.figure()
colors = plt.cm.jet(np.linspace(0,1,3))
plt.hist(ratio90, bins = np.arange(0, 2, 0.05), color = colors[0], alpha=2./10,label='90 GHz')
plt.hist(ratio150, bins = np.arange(0, 2, 0.05), color = colors[1], alpha=2./10,label='150 GHz')
plt.hist(ratio220, bins = np.arange(0, 2, 0.05), color = colors[2], alpha=2./10,label='220 GHz')
plt.legend(loc='best')
#plt.title('Wendy map/SZ map, for 4$\sigma$ sources, with calculated uK->mJy factor')
plt.title('Wendy map/SZ map, for 4$\sigma$ sources, with my uK->mJy factor')

plotting.kspaceImshow(idl.readIDLSav("/data/tangq/SPT/SZ/psd_ra23h30dec-55_2year_90_wtd.sav").psd,reso_arcmin=reso_arcmin,log=False,title='90')
plotting.kspaceImshow(idl.readIDLSav("/data/tangq/SPT/SZ/psd_ra23h30dec-55_2year_150_wtd.sav").psd,reso_arcmin=reso_arcmin,log=False,title='150')
plotting.kspaceImshow(idl.readIDLSav("/data/tangq/SPT/SZ/psd_ra23h30dec-55_2year_220_wtd.sav").psd,reso_arcmin=reso_arcmin,log=False,title='220')

plt.figure()
plotting.kspaceImshow(filt_220[0].optfilt,reso_arcmin=reso_arcmin,log=False, figure=3,subplot=131)
plt.title('k-space filter')
plt.subplot(132)
plotting.plotRadialProfile(np.fft.fftshift(filt_220[0].optfilt), center_bin=[npixels[1]/2,npixels[0]/2], resolution=(ellx[0,1]-ellx[0,0]), histogram=True, range=[0,20000],plot=True, bin_resolution_factor=0.1, figure=5,new=False, label=mapfreq + "GHz")
plt.xlabel('ell')
plt.title('Radial Profile of filter')
plt.subplot(133)
plt.imshow(filt_220[0].optfilt)
plt.colorbar()
plt.title('Optimal filter')

plt.suptitle(mapfreq + "noise=0 where kx<400")
