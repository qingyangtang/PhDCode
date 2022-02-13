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
import sptpol_software.constants as constants
from sptpol_software.util.math import makeEllGrids
import sptpol_software.analysis.plotting as plotting
import scipy.constants as const
from scipy.optimize import curve_fit
import tool_func as t
import ConfigParser
from datetime import datetime
from astropy.coordinates import Angle
from sptpol_software.util.files import readCAMBCls
today = datetime.today().strftime('%Y%m%d')


def get_SPIRE_ptsrcs(catalog_dir, sn_cut=40):
	#getting all the SPIRE point sources for SN>sn_cut and their RA, dec, flux
	idldata = idl.readIDLSav(catalog_dir)
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
	sn_cut = np.where(sn > sn_cut)
	cat_ra = cat_ra[sn_cut]
	cat_dec = cat_dec[sn_cut]
	cat_f_250 = cat_f_250[sn_cut]
	cat_df_250 = cat_df_250[sn_cut]

	return cat_ra, cat_dec, cat_f_250

def get_SPTSZ_ptsrcs_wendy(mapfreq, field="ra23h30dec-55_allyears", flux_cut=6.4):
	source_mask = '/data/tangq/wendymaps/'+field+'/source_list_3sig_allyears_'+mapfreq+'.dat'
	source_list = np.loadtxt(source_mask)
	src_ra = source_list[:,3]
	src_dec = source_list[:,4]
	src_flux = source_list[:,6]

	mask = src_flux > flux_cut

	return src_ra[mask],src_dec[mask],src_flux[mask]

def get_SPTSZ_ptsrcs(catalog_dir):
	source_list = np.genfromtxt(catalog_dir, skip_header=2)
	src_ra = source_list[:,1]
	src_dec = source_list[:,2]
	src_rad = source_list[:,3]

	return src_ra,src_dec,src_rad

def add_hotcold_pixels_to_ptsrc_cat(fname, ftype,pix_flux = 1.5e-1, pix_dflux = 3.5e-3):
	files = ["PSW","PMW","PLW"]
	for freq in range(len(files)):
		if ftype == 'fits':
			fits0 = fits.open("/data/tangq/SPIRE_maps/spt_zea_itermap_10_iterations_15_arcsec_pixels_" + files[freq] + ".fits")
			radec0 = np.asarray([fits0[1].header['CRVAL1'], fits0[1].header['CRVAL2']])
			reso_arcmin = abs(fits0[1].header['CD1_1'])*60 
			map0 = np.rot90(np.rot90(fits0[1].data))
			proj = 5
		elif ftype == 'sav':
			idlmap = idl.readIDLSav("/data/tangq/SPIRE_maps/proj0_herschel_maps/spt_zea_itermap_10_iterations_15_arcsec_pixels_"+files[freq]+".fits_v2.sav")
			radec0 = idlmap.values()[1]
			reso_arcmin = idlmap.values()[0]
			map0 = idlmap.values()[2]
			proj=0
		npixels = map0.shape
		if files[freq] == "PSW":
			hotpixels = np.where(map0 > 1)
			coldpixels = np.where(map0 < -1)
		elif files[freq] == "PMW":
			hotpixels = []
			coldpixles = []
			ptsrc = np.loadtxt('/data/tangq/SPIRE_maps/herschel_srcs_40.txt')
			np.savetxt('/data/tangq/SPIRE_maps/herschel_srcs_40_'+files[freq]+'_'+fname+'.txt',ptsrc)
			continue
		elif files[freq] == "PLW":
			hotpixels = np.where(map0 > 0.5)
			coldpixels = np.where(map0 < -0.5)
		ypixels = np.concatenate([hotpixels[0],coldpixels[0]])
		xpixels = np.concatenate([hotpixels[1],coldpixels[1]])
		pix_ra, pix_dec = sky.pix2Ang(np.asarray([ypixels,xpixels]), radec0, reso_arcmin, npixels,proj=proj)
		pixelstack = np.vstack([pix_ra,pix_dec,np.ones(len(pix_ra))*pix_flux,np.ones(len(pix_ra))*pix_dflux])
		ptsrc = np.loadtxt('/data/tangq/SPIRE_maps/herschel_srcs_40.txt')
		ptsrc_new = np.vstack([ptsrc, pixelstack.T])
		np.savetxt('/data/tangq/SPIRE_maps/herschel_srcs_40_'+files[freq]+'_'+fname+'.txt',ptsrc_new)

def get_Spitzer_DES_cat_with_mass(sp_des_dir,stellar_m_dir,iband_cut=23):
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

	mask = sp_des_sm_cat['i'] < iband_cut
	clean_cat = sp_des_sm_cat[mask]

	return clean_cat

def get_SPTSZ_filtered_map_wendy(units='uK',field="ra23h30dec-55_allyears"):
	for mapfreq in ["90","150","220"]:
		idldata = idl.readIDLSav("/data/tangq/wendymaps/"+field+"/maps_planck15cal_allyears_"+mapfreq+".sav")
		pixelmask = idldata.pixel_mask
		radec0 = idldata.radec0
		reso_arcmin = idldata.reso_arcmin
		npixels = np.asarray(idldata.cmap_k.shape)
		# ra = np.linspace(radec0[0]-0.5, radec0[0]+0.5, 10)
		# dec = np.linspace(radec0[1]-0.5, radec0[1]+0.5, 10)
		# ypix30, xpix30 = sky.ang2Pix(np.asarray([ra, dec]), radec0, reso_arcmin, npixels,proj=5)[0]
		if units=='uK':
			cmap = idldata.cmap_k
		elif units =='mJy':
			cmap = idldata.cmap_mjy
		if mapfreq =="90":
			cmap90 = cmap
			pixelmask90 = pixelmask
			radec090 = radec0
			reso_arcmin90 = reso_arcmin
			npixels90 = npixels
		elif mapfreq =="150":
			cmap150 = cmap
			pixelmask150 = pixelmask
			radec0150 = radec0
			reso_arcmin150 = reso_arcmin
			npixels150 = npixels
		elif mapfreq =="220":
			cmap220 = cmap
			pixelmask220 = pixelmask
			radec0220 = radec0
			reso_arcmin220 = reso_arcmin
			npixels220 = npixels
	cmap = [cmap90, cmap150, cmap220]
	pixelmask = [pixelmask90,pixelmask150,pixelmask220]
	reso_arcmin = [reso_arcmin90, reso_arcmin150, reso_arcmin220]
	radec0 = [radec090,radec0150,radec0220]
	npixels = [npixels90,npixels150, npixels220]
	mapfreq = ["90", "150", "220"]
	return cmap, pixelmask, radec0, reso_arcmin, npixels, mapfreq

def make_filtered_map(map0s, source_masks, radec0s, reso_arcmins, npixels_arr, ftype,
						fwhm_beams = None, mapfreqs = None,
						beams=None, trans_func_beams = None, trans_funcs = None, common_beam=0, noise_psds=None, run_full_masking=True, run_ptsrc_masking=False, map_mask = None,
						save_map = False, skysig = None,
						save_map_dir='/data/tangq/SPIRE_maps/products/', save_map_fname='Herschel_filtered_1_75arcmin_beam'):

	for x in range(len(mapfreqs)):
		map0 = map0s[x]
		source_mask = source_masks[x]
		radec0 = radec0s[x]
		reso_arcmin = reso_arcmins[x]
		npixels = npixels_arr[x]
		trans_func_beam = trans_func_beams[x]
		trans_func = trans_funcs[x]
		beam = beams[x]
		fwhm_beam = fwhm_beams[x]
		mapfreq = mapfreqs[x]

		reso_rad = reso_arcmin * constants.DTOR / 60.
		ellg,ellx,elly = makeEllGrids([npixels[1],npixels[0]],reso_rad)
		if common_beam != 0:
			smoothed_map, cb_trans_func = t.map_smoothing(map0, trans_func_beam, trans_func,ellg, common_fwhm=common_beam)
			# need to zero transfer function where kx<400 if using SPT-SZ 220 GHz trans func
			if ftype == 'herschel':
				mask_ellx = np.abs(ellx)<400
				cb_trans_func[mask_ellx] = 0
			elif mapfreq == "220":
				mask_ellx = np.abs(ellx)<400
				cb_trans_func[mask_ellx] = 0				
		elif common_beam ==0:
			smoothed_map = map0
			cb_trans_func = trans_func_beam
			if ftype == 'tom':
				mask_ellx = np.abs(ellx)<400
				cb_trans_func[mask_ellx] = 0

		mask = map_mask
		if run_full_masking == True:
			mask = maps.makeApodizedPointSourceMask(smoothed_map, source_mask, ftype = ftype, reso_arcmin=np.float(reso_arcmin), center=radec0, radius_arcmin=3)
			masked_map = mask*smoothed_map
		else:
			if run_ptsrc_masking == True:
				ptsrc_mask = maps.makePointSourceMask(map0, source_mask, ftype = ftype, reso_arcmin=reso_arcmin, center=radec0, radius_arcmin=3)
				mask = ptsrc_mask*map_mask
				masked_map = mask*smoothed_map
			else:
				mask = np.ones((map0.shape))
				masked_map = smoothed_map

		zp_mask_map, zp_mask = t.zero_pad_map(masked_map, mask)

		#calculating noise PSD
		if noise_psds == None:
			noise_psd = t.noise_PSD(zp_mask_map, reso_arcmin, dngrid=zp_mask_map.shape[0])
			noise_psd[noise_psd==0] = np.std(noise_psd)*1.e-3
		else:
			noise_psd = noise_psds[x]
			if ftype == 'tom':
				mask_ellx = np.abs(ellx)<400
				noise_psd[mask_ellx] = 1.e9

		#cb_trans_func[cb_trans_func<0.01] = 0.01
		# if common_beam == None:
		# 	filt_output,filt_map =  calc.call_matched_filt_point(zp_mask_map, cmb = False, skysigvar = np.zeros((25000)), noisepsd=noise_psd, \
		# 							fwhm_beam_arcmin=beamsizes[beamsize], ellmax = 25000, ell_highpass = 10, ell_lowpass = 20000,use_fft_as_noise=False, \
		# 							plot_map=False, return_filtered_map=True, mask = zp_mask, reso_arcmin = reso_arcmin, ngridx = zp_mask_map.shape[0],\
		# 							ngridy = zp_mask_map.shape[0])
		# else:


		# if common_beam != 0:
		# 	if ftype == 'herschel':
		# 		filt_output,filt_map =  calc.call_matched_filt_point(zp_mask_map, cmb = False, skysigvar = skysig[x], noisepsd=noise_psd, \
		# 							beam=None,trans_func_grid = cb_trans_func, use_fft_as_noise=False, \
		# 							plot_map=False, return_filtered_map=True, mask = zp_mask, reso_arcmin = reso_arcmin, ngridx = zp_mask_map.shape[1],\
		# 							ngridy = zp_mask_map.shape[0])
		# 	elif ftype == 'tom':
		# 		filt_output,filt_map =  calc.call_matched_filt_point(zp_mask_map, cmb = False, skysigvar = skysig[x], noisepsd=noise_psd, \
		# 						beam=trans_func_beam,trans_func_grid = cb_trans_func,use_fft_as_noise=False, \
		# 						plot_map=False, return_filtered_map=True, mask = zp_mask, reso_arcmin = reso_arcmin, ngridx = zp_mask_map.shape[1],\
		# 						ngridy = zp_mask_map.shape[0])					
		# else:
		# 	if ftype == 'herschel':
		# 		filt_output,filt_map =  calc.call_matched_filt_point(zp_mask_map, cmb = False, skysigvar = skysig[x], noisepsd=noise_psd, \
		# 							fwhm_beam_arcmin=fwhm_beam, beam_grid = trans_func_beam, ellmax = 25000, ell_highpass = 10, ell_lowpass = 20000,use_fft_as_noise=False, \
		# 							plot_map=False, return_filtered_map=True, mask = zp_mask, reso_arcmin = reso_arcmin, ngridx = zp_mask_map.shape[1], \
		# 							ngridy = zp_mask_map.shape[0])
		# 	elif ftype =='tom':
		# 		filt_output,filt_map =  calc.call_matched_filt_point(zp_mask_map, cmb = False, skysigvar = skysig[x], noisepsd=noise_psd, \
		# 							beam=np.ones(zp_mask_map.shape),trans_func_grid = cb_trans_func,use_fft_as_noise=False, \
		# 							plot_map=False, return_filtered_map=True, mask = zp_mask, reso_arcmin = reso_arcmin, ngridx = zp_mask_map.shape[1],\
		# 							ngridy = zp_mask_map.shape[0])				
		## this is for smoothing where we multiply by 150GHz SPTpol filter
		filt_output = np.zeros((zp_mask_map.shape))
		uv_filter = np.load('/data/tangq/SPT/SZ/products/SPTSZ_filtered_1_smooth_FTfixed_20210611'+'_220_filter.npy').item()[0].optfilt
		ft_map = calc.getFTMap(zp_mask_map, zp_mask)
		ft_map *= uv_filter
		filtered_map = np.fft.ifft2(ft_map).real
		filt_map = filtered_map

		##################

		cmap = filt_map
		npixels = np.asarray(cmap.shape)

		if x == 0:
			filt_map0 = cmap
			npixels0 = npixels
			filt_0 = filt_output
	 	elif x == 1:
			filt_map1 = cmap
			npixels1 = npixels
			filt_1 = filt_output
		elif x == 2:
			filt_map2 = cmap
			npixels2 = npixels
			filt_2 = filt_output

	reso_arcmin_arr = reso_arcmins
	radec0_arr = radec0s	
	mapfreq_arr = mapfreqs
	if len(mapfreqs) ==3 :
		map_arr = [filt_map0,filt_map1,filt_map2]
		npixels_arr = [npixels0,npixels1, npixels2]
		if save_map:
			np.save(save_map_dir+'/'+save_map_fname+'_'+mapfreqs[0]+'_map.npy', map_arr[0])
			np.save(save_map_dir+'/'+save_map_fname+'_'+mapfreqs[1]+'_map.npy', map_arr[1])
			np.save(save_map_dir+'/'+save_map_fname+'_'+mapfreqs[2]+'_map.npy', map_arr[2])
			np.save(save_map_dir+'/'+save_map_fname+'_zp_mask.npy', zp_mask)
			np.save(save_map_dir+'/'+save_map_fname+'_'+mapfreqs[0]+'_filter.npy', filt_0) 
			np.save(save_map_dir+'/'+save_map_fname+'_'+mapfreqs[1]+'_filter.npy', filt_1) 
			np.save(save_map_dir+'/'+save_map_fname+'_'+mapfreqs[2]+'_filter.npy', filt_2) 
			np.savetxt(save_map_dir+'/'+save_map_fname+'_reso_arcmin.txt', np.asarray(reso_arcmin_arr)) 
			np.savetxt(save_map_dir+'/'+save_map_fname+'_radec.txt', np.asarray(radec0_arr)) 
		return map_arr, filt_0, filt_1, filt_2
	elif len(mapfreqs) ==2 :
		map_arr = [filt_map0,filt_map1]
		npixels_arr = [npixels0,npixels1]
		if save_map:
			np.save(save_map_dir+'/'+save_map_fname+'_'+mapfreqs[0]+'_map.npy', map_arr[0])
			np.save(save_map_dir+'/'+save_map_fname+'_'+mapfreqs[1]+'_map.npy', map_arr[1])
			np.save(save_map_dir+'/'+save_map_fname+'_zp_mask.npy', zp_mask)
			np.save(save_map_dir+'/'+save_map_fname+'_'+mapfreqs[0]+'_filter.npy', filt_0) 
			np.save(save_map_dir+'/'+save_map_fname+'_'+mapfreqs[1]+'_filter.npy', filt_1) 
			np.savetxt(save_map_dir+'/'+save_map_fname+'_reso_arcmin.txt', np.asarray(reso_arcmin_arr)) 
			np.savetxt(save_map_dir+'/'+save_map_fname+'_radec.txt', np.asarray(radec0_arr)) 
		return map_arr, filt_0, filt_1
	#we can read the filter files later as: 
	# a = np.load('/data/tangq/SPIRE_maps/products/Herschel_filtered_1_75_arcmin_20201130_250_filter.npy').item()
	# a[0].optfilt 

def herschel_trans_curve():
	'''
	Opens the Herschel transmission curves, which are in GHz, returns the freq and transmission curve in GHz
	zero-padded to be compatible with calc_eff_nu
	'''
	hdu = fits.open('/data/tangq/SPIRE_maps/SPIRE_phot_RSRF_14_3.fits')
	freq = hdu[1].data['frequency']
	psw = hdu[1].data['psw']
	pmw = hdu[1].data['pmw']
	plw = hdu[1].data['plw']

	freq = np.concatenate((np.linspace(np.min(freq)-200,np.min(freq)-1, 99), freq))
	trans = np.asarray([plw,pmw,psw])
	trans = np.hstack((np.zeros((3,99)),trans))

	return freq, trans

def make_SPTSZ_filtered_map(spt_config='/home/tangq/scripts/spt_config.txt', run_match_filter = True, save_map_dir='/data/tangq/SPT/SZ/products/', save_map_fname='SPTSZ_filtered_1_smooth_FTfixed_'):
	configParser = ConfigParser.RawConfigParser()   
	configFilePath = spt_config
	configParser.read(configFilePath)

	source_mask = [configParser.get('map_and_files', 'source_mask'),configParser.get('map_and_files', 'source_mask'),configParser.get('map_and_files', 'source_mask')]
	map0s = idl.readIDLSav(configParser.get('map_and_files', 'map')).maps
	calfacs =[0.722787, 0.830593, 0.731330]
	for x in range(len(calfacs)):
		map0s[x] = map0s[x]*calfacs[x]

	ftype = 'tom'
	apod_mask = idl.readIDLSav(configParser.get('map_and_files', 'apod_mask')).apmask_field
	ap_mask = idl.readIDLSav(configParser.get('map_and_files', 'apod_mask')).apmask
	trans_func_beam = idl.readIDLSav(configParser.get('map_and_files', 'trans_func_beam')).transfer_with_beam
	trans_func = idl.readIDLSav(configParser.get('map_and_files', 'trans_func')).tfgrids_nobeam

	beam90 = np.loadtxt(configParser.get('map_and_files', 'beam90'))
	beam150 = np.loadtxt(configParser.get('map_and_files', 'beam150'))
	beam220 = np.loadtxt(configParser.get('map_and_files', 'beam220'))
	beams = np.asarray([beam90, beam150,beam220])

	mapinfo = idl.readIDLSav(configParser.get('map_and_files', 'mapinfo'))
	radec0 = np.asarray([mapinfo.radec0,mapinfo.radec0,mapinfo.radec0])
	reso_arcmin = np.asarray([mapinfo.reso_arcmin,mapinfo.reso_arcmin,mapinfo.reso_arcmin])
	proj = mapinfo.projection
	npixels = np.asarray([map0s[0].shape, map0s[1].shape, map0s[2].shape])

	noise_psd_90 = idl.readIDLSav(configParser.get('map_and_files', 'noise_psd_90')).psd
	noise_psd_150 = idl.readIDLSav(configParser.get('map_and_files', 'noise_psd_150')).psd
	noise_psd_220 = idl.readIDLSav(configParser.get('map_and_files', 'noise_psd_220')).psd
	noise_psd = np.asarray([noise_psd_90, noise_psd_150, noise_psd_220])
	for x in range(len(calfacs)):
		noise_psd[x] = noise_psd[x]*calfacs[x]

	initial_map = ap_mask*map0s

	common_beam = float(configParser.get('filter_map_settings', 'common_beam'))
	if common_beam != 0:
		trans_func_beam[trans_func_beam<0.01] = 0.01
	beamsizes = np.asarray([1.65, 1.2, 1.0])

	mapfreqs = np.asarray(["90","150","220"])

	skysigfile = configParser.get('filter_map_settings', 'skysig')
	if skysigfile != "None":
		skysig0 = idl.readIDLSav(skysigfile)
		skysig = np.asarray([skysig0['test'][:,0,0],skysig0['test'][:,1,1],skysig0['test'][:,2,2]])
	else:
		skysig = np.asarray([np.zeros((25000)),np.zeros((25000)),np.zeros((25000))])

	set_cmb0 = configParser.get('filter_map_settings','set_cmb')
	if set_cmb0 == 'True':
		ellmax = len(skysig[0])-1
		cmbfile = configParser.get('filter_map_settings', 'cmb_file')
		cls = readCAMBCls('/data/tnatoli/camb/planck_lensing_wp_highL_bestFit_20130627_massless0p046_massive3_lensedtotCls.dat',
			as_cls = True, extrapolate = True, as_uksq = False, max_ell = ellmax-1)
		skysig[0][cls['ell']] += cls['cl']['TT']
		skysig[1][cls['ell']] += cls['cl']['TT']
		skysig[2][cls['ell']] += cls['cl']['TT']

	if run_match_filter:
		map_arr, filt_90, filt_150, filt_220 = make_filtered_map(map0s=initial_map, source_masks = source_mask, radec0s = radec0, reso_arcmins = reso_arcmin, npixels_arr=npixels, ftype = ftype,
						fwhm_beams = beamsizes, mapfreqs = mapfreqs,
						beams=beams, trans_func_beams = trans_func_beam, trans_funcs = trans_func, common_beam=common_beam, noise_psds=noise_psd, run_full_masking=False, run_ptsrc_masking=True, map_mask = apod_mask,
						save_map = True, skysig = skysig,
						save_map_dir=save_map_dir, save_map_fname=save_map_fname+today)

def make_SPTpol_filtered_map(sptpol_config='/home/tangq/scripts/sptpol_config.txt', run_match_filter = True, save_map = True, save_map_dir='/data/tangq/SPT/SPTpol/products/', save_map_fname='SPTpol_filtered_1_7_smooth_FTfixed_'):
	configParser = ConfigParser.RawConfigParser()   
	configFilePath = sptpol_config
	configParser.read(configFilePath)

	source_mask = [configParser.get('map_and_files', 'source_mask'),configParser.get('map_and_files', 'source_mask'),configParser.get('map_and_files', 'source_mask')]
	map0s = idl.readIDLSav(configParser.get('map_and_files', 'map')).out[0:2,:,:]
	calfacs = [0.776500,0.909700]
	for x in range(len(calfacs)):
		map0s[x] = map0s[x]*calfacs[x]

	ftype = 'tom'
	apod_mask = idl.readIDLSav(configParser.get('map_and_files', 'apod_mask')).apmask_field
	ap_mask = idl.readIDLSav(configParser.get('map_and_files', 'apod_mask')).apmask
	trans_func_90_no_beam = idl.readIDLSav(configParser.get('map_and_files', 'trans_func_90_no_beam')).tf90
	trans_func_150_no_beam = idl.readIDLSav(configParser.get('map_and_files', 'trans_func_150_no_beam')).tf150
	trans_func = np.asarray([trans_func_90_no_beam,trans_func_150_no_beam])

	beam90 = idl.readIDLSav(configParser.get('map_and_files', 'beam90')).bl
	beam150 = idl.readIDLSav(configParser.get('map_and_files', 'beam150')).bl
	beam = [np.concatenate([beam90,np.zeros((50000))]),np.concatenate([beam150,np.zeros((50000))])]

	mapinfo = idl.readIDLSav(configParser.get('map_and_files', 'map'))
	radec0 = np.asarray([mapinfo.radec0,mapinfo.radec0])
	reso_arcmin = np.asarray([mapinfo.reso_arcmin,mapinfo.reso_arcmin])
	proj = int(configParser.get('map_and_files', 'proj'))
	npixels = np.asarray([map0s[0].shape, map0s[1].shape])
	beams = []
	for x in range(len(beam)):
		beam_x = beam[x]
		reso_rad = reso_arcmin[x] * constants.DTOR / 60.
		ellg,ellx,elly = makeEllGrids([npixels[x][1],npixels[x][0]],reso_rad)
		bl2d = beam_x[(np.round(ellg)).astype(int)]
		beams.append(bl2d)
	trans_func_beam = np.asarray([trans_func[0]*beams[0], trans_func[1]*beams[1]])


	noise_psd_90 = idl.readIDLSav(configParser.get('map_and_files', 'noise_psd_90')).psd
	noise_psd_150 = idl.readIDLSav(configParser.get('map_and_files', 'noise_psd_150')).psd

	noise_psd = np.asarray([noise_psd_90, noise_psd_150])
	for x in range(len(calfacs)):
		noise_psd[x] = noise_psd[x]*calfacs[x]

	initial_map = ap_mask*map0s

	common_beam = float(configParser.get('filter_map_settings', 'common_beam'))
	if common_beam != 0:
		trans_func_beam[trans_func_beam<0.01] = 0.01
	beamsizes = np.asarray([1.65, 1.2])

	mapfreqs = np.asarray(["90","150"])

	skysigfile = configParser.get('filter_map_settings', 'skysig')
	if skysigfile != "None":
		skysig0 = idl.readIDLSav(skysigfile)
		skysig = np.asarray([skysig0['test'][:,0,0],skysig0['test'][:,1,1]])
	else:
		skysig = np.asarray([np.zeros((25000)),np.zeros((25000))])

	set_cmb0 = configParser.get('filter_map_settings','set_cmb')
	if set_cmb0 == 'True':
		ellmax = len(skysig[0])-1
		cmbfile = configParser.get('filter_map_settings', 'cmb_file')
		cls = readCAMBCls('/data/tnatoli/camb/planck_lensing_wp_highL_bestFit_20130627_massless0p046_massive3_lensedtotCls.dat',
			as_cls = True, extrapolate = True, as_uksq = False, max_ell = ellmax-1)
		skysig[0][cls['ell']] += cls['cl']['TT']
		skysig[1][cls['ell']] += cls['cl']['TT']

	if run_match_filter:
		map_arr, filt_90, filt_150 = make_filtered_map(map0s=initial_map, source_masks = source_mask, radec0s = radec0, reso_arcmins = reso_arcmin, npixels_arr=npixels, ftype = ftype,
						fwhm_beams = beamsizes, mapfreqs = mapfreqs,
						beams=beams, trans_func_beams = trans_func_beam, trans_funcs = trans_func, common_beam=common_beam, noise_psds=noise_psd, run_full_masking=False, run_ptsrc_masking=True, map_mask = apod_mask,
						save_map = True, skysig = skysig,
						save_map_dir=save_map_dir, save_map_fname=save_map_fname+today)

def read_Herschel_config_file(herschel_config='/home/tangq/scripts/herschel_config.txt', run_match_filter = True, save_map_dir='/data/tangq/SPIRE_maps/products/', save_map_fname='Herschel_filtered_1_smooth_FTfixed_'):
	configParser = ConfigParser.RawConfigParser()   
	configFilePath = herschel_config
	configParser.read(configFilePath)

	proj = int(configParser.get('map_and_ptsrc_list', 'proj'))

	ptsrc250 = configParser.get('map_and_ptsrc_list', 'ptsrc250')
	ptsrc350 = configParser.get('map_and_ptsrc_list', 'ptsrc350')
	ptsrc500 = configParser.get('map_and_ptsrc_list', 'ptsrc500')
	source_mask = [ptsrc250, ptsrc350, ptsrc500]

	mapfreqs = ["250", "350", "500"]

	if proj==0:
		map250 = idl.readIDLSav(configParser.get('map_and_ptsrc_list', 'map250'))
		map350 = idl.readIDLSav(configParser.get('map_and_ptsrc_list', 'map350'))
		map500 = idl.readIDLSav(configParser.get('map_and_ptsrc_list', 'map500'))

		radec0 = [map250.values()[1], map350.values()[1], map500.values()[1]]
		reso_arcmin = [map250.values()[0], map350.values()[0], map500.values()[0]]
		map0 = [map250.values()[2],	map350.values()[2],map500.values()[2]]
		ftype = 'herschel'
	elif proj==5:
		map250 = fits.open(configParser.get('map_and_ptsrc_list', 'map250'))
		map350 = fits.open(configParser.get('map_and_ptsrc_list', 'map350'))
		map500 = fits.open(configParser.get('map_and_ptsrc_list', 'map500'))

		radec0 = [np.asarray([map250[1].header['CRVAL1'], map250[1].header['CRVAL2']]),np.asarray([map350[1].header['CRVAL1'], map350[1].header['CRVAL2']]), np.asarray([map500[1].header['CRVAL1'], map500[1].header['CRVAL2']])  ]
		reso_arcmin = [abs(map250[1].header['CD1_1'])*60,abs(map350[1].header['CD1_1'])*60,abs(map500[1].header['CD1_1'])*60]
		map0 = [np.rot90(np.rot90(fits250[1].data)), np.rot90(np.rot90(fits350[1].data)), np.rot90(np.rot90(fits500[1].data))]
		ftype = 'herschel_nan'
	else:
		raise ValueError("Invalid proj input")
	npixels = [map0[0].shape, map0[1].shape,map0[2].shape]
	if configParser.get('map_and_ptsrc_list', 'shift_radec0') == "True":
		for x in range(3):
			radec0[x][0]-=(Angle(np.asarray(reso_arcmin), u.arcminute).degree[x])*2 
			radec0[x][1]+=Angle(np.asarray(reso_arcmin), u.arcminute).degree[x] 

	if configParser.get('beam_and_transfer_func', 'beamfile') != '0':
		beamfile = idl.readIDLSav(configParser.get('beam_and_transfer_func', 'beamfile'))
		beam = beamfile['spire_beams']
		beams = []
		for x in range(3):
			beam_x = beam[x]
			reso_rad = reso_arcmin[x] * constants.DTOR / 60.
			ellg,ellx,elly = makeEllGrids([npixels[x][1],npixels[x][0]],reso_rad)
			bl2d = beam_x[(np.round(ellg)).astype(int)]
			beams.append(bl2d)
	else:
		beams = None

	beamsizes = np.asarray([18.2, 24.9, 36.3])/60. #beams for 250, 350, 500um in arcmins

	if configParser.get('beam_and_transfer_func', 'trans_func_no_beam') != '0':
		#using SPT-SZ 220 GHz transfer function for all 3 frequencies
		trans_func_file = idl.readIDLSav(configParser.get('beam_and_transfer_func', 'trans_func_no_beam'))
		trans_func = [trans_func_file.tfgrids_nobeam[2],trans_func_file.tfgrids_nobeam[2],trans_func_file.tfgrids_nobeam[2]]
	else:
		trans_func = None

	#get settings for match filter algorithm
	common_beam = float(configParser.get('filter_map_settings', 'common_beam'))
	noise_psd = float(configParser.get('filter_map_settings', 'noise_psd'))
	if noise_psd == 0:
		noise_psd = None

	skysigfile = configParser.get('filter_map_settings', 'skysig')
	if skysigfile != "None":
		skysig0 = idl.readIDLSav(skysigfile)
		skysig = np.asarray([skysig0['test'][:,0,0],skysig0['test'][:,1,1],skysig0['test'][:,2,2]])
	else:
		skysig = np.asarray([np.zeros((25000)),np.zeros((25000)),np.zeros((25000))])

	set_cmb0 = configParser.get('filter_map_settings','set_cmb')
	if set_cmb0 == 'True':
		ellmax = len(skysig[0])-1
		cmbfile = configParser.get('filter_map_settings', 'cmb_file')
		cls = readCAMBCls('/data/tnatoli/camb/planck_lensing_wp_highL_bestFit_20130627_massless0p046_massive3_lensedtotCls.dat',
			as_cls = True, extrapolate = True, as_uksq = False, max_ell = ellmax-1)
		skysig[0][cls['ell']] += cls['cl']['TT']
		skysig[1][cls['ell']] += cls['cl']['TT']
		skysig[2][cls['ell']] += cls['cl']['TT']

	if run_match_filter:
		map_arr, filt_250, filt_350, filt_500 = make_filtered_map(map0s=map0, source_masks = source_mask, radec0s = radec0, reso_arcmins = reso_arcmin, npixels_arr=npixels, ftype = ftype,
						fwhm_beams = beamsizes, mapfreqs = mapfreqs,
						beams=beam, trans_func_beams = beams, trans_funcs = trans_func, common_beam=common_beam, noise_psds=None, run_full_masking=True, run_ptsrc_masking=False, map_mask = None,
						save_map = True, skysig = skysig, save_map_dir=save_map_dir, save_map_fname=save_map_fname+today)

