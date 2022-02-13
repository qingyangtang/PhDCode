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


save_dir = '/data/tangq/wendymaps/'
#### SPT stacking:

for mapfreq in ["90", "150", "220"]:
	show_plot = False
	make_mask = False

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

	nrands = 5.e5

	rand_ypix = (np.round(np.random.rand(nrands*3)*(npixels[0]-1))).astype(int)
	rand_xpix = (np.round(np.random.rand(nrands*3)*(npixels[1]-1))).astype(int)
	pixelmaskind = np.where(pixelmask[rand_ypix,rand_xpix]==1)[0]
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
	print mapfreq + " " + str(len(np.ravel(rand_flux)))
	np.save(save_dir + 'rand_pix_flux_freq_' + mapfreq, np.ravel(rand_flux))



save_dir1 = '/data/tangq/SPIRE_maps/'
mapfreq = "350"
field = "Herschel_15arcsecs"
for mapfreq in ["250", "350", "500"]:
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

	nrands1 = 5.e5
	rand_ypix = (np.round(np.random.rand(nrands1*3)*(npixels[0]-1))).astype(int)
	rand_xpix = (np.round(np.random.rand(nrands1*3)*(npixels[1]-1))).astype(int)
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
	print mapfreq + " " + str(len(np.ravel(rand_flux)))
	np.save(save_dir1 + 'rand_pix_flux_freq_' + mapfreq, np.ravel(rand_flux))