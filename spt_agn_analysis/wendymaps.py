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

for mapfreq in ["90","150","220"]:
	show_plot = False
	make_mask = False

	field = "ra5h30dec-55_allyears"
	idldata = idl.readIDLSav("/data/tangq/wendymaps/"+field+"/maps_planck15cal_allyears_"+mapfreq+".sav")
	pixelmask = idldata.pixel_mask
	radec0 = idldata.radec0
	reso_arcmin = idldata.reso_arcmin
	npixels = np.asarray(idldata.cmap_k.shape)
	# ra = np.linspace(radec0[0]-0.5, radec0[0]+0.5, 10)
	# dec = np.linspace(radec0[1]-0.5, radec0[1]+0.5, 10)
	# ypix30, xpix30 = sky.ang2Pix(np.asarray([ra, dec]), radec0, reso_arcmin, npixels,proj=5)[0]
	cmap = idldata.cmap_k
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

	# if make_mask == True:
	# 	### CHANGE THE DIR_OUT IN MAKE MASKS_AMY_WENDYMAP.PY
	# 	mask = maps.makeApodizedPointSourceMask(cmap, source_mask, reso_arcmin=reso_arcmin, center=radec0, radius_arcmin=5, ftype = 'wendy')
	# else:
	# 	mask = pickle.load(open('/home/tangq/data/wendymap/'+field+'/'+mapfreq+'GHz_ptsrc_mask.pkl','r'))

	source_list = np.loadtxt(source_mask)
	src_ra = source_list[:,3]
	src_dec = source_list[:,4]
	src_flux = source_list[:,6]

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

	for zcut in zbands:
		ra = ra_good
		dec = dec_good
		I_band = I_band_good
		I_flag = I_flag_good
		z_mean = z_mean_good
		coadd_ID = coadd_ID_good

		ra = ra[zcut]
		dec = dec[zcut]
		I_band = I_band[zcut]
		I_flag = I_flag[zcut]
		z_mean = z_mean[zcut]
		coadd_ID = coadd_ID[zcut]

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
		pixelmaskind = np.where(pixelmask[ypix,xpix]==1)[0]
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

		#cutting out the gal that are within 8 arcmins of > 100mJy source
		nearest_src_flux = src_flux[idx]
		gtr100fluxind = np.where((nearest_src_flux > 100) & (d2d.arcminute < 8))[0]

		ypix = ypix[np.in1d(np.arange(len(ypix)), gtr100fluxind, invert=True)]
		xpix = xpix[np.in1d(np.arange(len(xpix)), gtr100fluxind, invert=True)]
		idx = idx[np.in1d(np.arange(len(idx)), gtr100fluxind, invert=True)]
		d2d = d2d[np.in1d(np.arange(len(d2d)), gtr100fluxind, invert=True)]
		ra = ra[np.in1d(np.arange(len(ra)), gtr100fluxind, invert=True)]
		dec = dec[np.in1d(np.arange(len(dec)), gtr100fluxind, invert=True)]
		I_band = I_band[np.in1d(np.arange(len(I_band)), gtr100fluxind, invert=True)]
		I_flag = I_flag[np.in1d(np.arange(len(I_flag)), gtr100fluxind, invert=True)]
		z_mean = z_mean[np.in1d(np.arange(len(z_mean)), gtr100fluxind, invert=True)]
		coadd_ID = coadd_ID[np.in1d(np.arange(len(coadd_ID)), gtr100fluxind, invert=True)]


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
		plt.hist(cmap[ypix, xpix],bins=np.arange(-1e-4,1e-4,1e-6))
		plt.xlabel('DES gal flux (K)')
		plt.title('Field: ' + field + '_' + mapfreq+'GHz')
		plt.savefig('/home/tangq/data/'+field+'_'+mapfreq+'GHz_DES_gal_flux_hist.pdf')
		if show_plot == False:
			plt.close()

		n_realizations = 1000

		# randommean = np.zeros((n_realizations))
		# randommedian = np.zeros((n_realizations))
		# for j in range(n_realizations):
		# 	data = np.load("/data/tangq/wendymaps/"+field+"/Random_"+mapfreq+"GHz_Sim"+str(j)+".npy")
		# 	rand_flux = data[0:len(ypix)]
		# 	randommean[j] = np.mean(rand_flux)
		# 	randommedian[j] = np.median(rand_flux)
			
		n_realizations = 1000
		randommean = np.zeros((n_realizations))
		randommedian = np.zeros((n_realizations))

		for j in range(n_realizations):
			rand_ypix = (np.round(np.random.rand(len(ypix)*3)*(npixels[0]-1))).astype(int)
			rand_xpix = (np.round(np.random.rand(len(ypix)*3)*(npixels[1]-1))).astype(int)
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

