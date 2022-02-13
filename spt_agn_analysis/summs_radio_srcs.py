import numpy as np
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

for mapfreq in ["90", "150", "220"]:
	show_plot = False
	make_mask = False


	field = "ra23h30dec-55_allyears"
	idldata = idl.readIDLSav("/data/tangq/wendymaps/"+field+"/maps_planck15cal_allyears_"+mapfreq+".sav")
	pixelmask = idldata.pixel_mask
	radec0 = idldata.radec0
	reso_arcmin = idldata.reso_arcmin
	npixels = np.asarray(idldata.cmap_k.shape)
	# ra = np.linspace(radec0[0]-0.5, radec0[0]+0.5, 10)
	# dec = np.linspace(radec0[1]-0.5, radec0[1]+0.5, 10)
	# ypix30, xpix30 = sky.ang2Pix(np.asarray([ra, dec]), radec0, reso_arcmin, npixels,proj=5)[0]
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

	# if make_mask == True:
	# 	### CHANGE THE DIR_OUT IN MAKE MASKS_AMY_WENDYMAP.PY
	# 	mask = maps.makeApodizedPointSourceMask(cmap, source_mask, reso_arcmin=reso_arcmin, center=radec0, radius_arcmin=5, ftype = 'wendy')
	# else:
	# 	mask = pickle.load(open('/home/tangq/data/wendymap/'+field+'/'+mapfreq+'GHz_ptsrc_mask.pkl','r'))

	source_list = np.loadtxt(source_mask)
	src_ra = source_list[:,3]
	src_dec = source_list[:,4]
	src_flux = source_list[:,6]

	idldata = idl.readIDLSav("/data55/bleeml/data0/sptpol/source_catalogs/sumss.sav")
	data = idldata.values()[0]
	sumss_ra = np.asarray([d['ra'] for d in data])
	sumss_dec = np.asarray([d['dec'] for d in data])
	sumss_flux = np.asarray([d['mjy'] for d in data])

	# ss_ra = coord.Angle(sumss_ra*u.degree)
	# ss_ra = ss_ra.wrap_at(180*u.degree)
	# ss_dec = coord.Angle(sumss_dec*u.degree)
	# save_file = True
	# cm = plt.cm.get_cmap('OrRd')
	# cm2 = plt.cm.get_cmap('Blues')
	# fig = plt.figure(figsize=(18,16))
	# ax = fig.add_subplot(111, projection="mollweide")
	# ax.grid(True)
	# sc = ax.scatter(ss_ra.radian, ss_dec.radian, c=sumss_flux, s=2, cmap=cm, alpha=1, edgecolors='none')

	ra_cut = np.concatenate((np.where(sumss_ra < ra_edge[1])[0], np.where(sumss_ra > ra_edge[0])[0]))
	sumss_ra = sumss_ra[ra_cut]
	sumss_dec = sumss_dec[ra_cut]
	sumss_flux = sumss_flux[ra_cut]
	dec_cut = np.where((sumss_dec < dec_edge[0]) & (sumss_dec > dec_edge[1]))[0]
	sumss_ra = sumss_ra[dec_cut]
	sumss_dec = sumss_dec[dec_cut]
	sumss_flux = sumss_flux[dec_cut]

	ra = sumss_ra
	dec = sumss_dec

	ypix, xpix = sky.ang2Pix(np.asarray([ra, dec]), radec0, reso_arcmin, npixels,proj=5)[0]
	ypixmask, xpixmask = sky.ang2Pix(np.asarray([ra, dec]), radec0, reso_arcmin, npixels,proj=5)[1] 
	ang2pixind = np.where((ypixmask ==True) & (xpixmask == True))[0]
	ypix = ypix[ang2pixind]
	xpix = xpix[ang2pixind]
	ra = ra[ang2pixind]
	dec = dec[ang2pixind]
	del ang2pixind
	pixelmaskind = np.where(pixelmask[ypix,xpix]==1)[0]
	ypix = ypix[pixelmaskind]
	xpix = xpix[pixelmaskind]
	ra = ra[pixelmaskind]
	dec = dec[pixelmaskind]

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




	#cutting out the gal that are within 8 arcmins of > 100mJy source
	nearest_src_flux = src_flux[idx]
	gtr100fluxind = np.where((nearest_src_flux > 100) & (d2d.arcminute < 8))[0]

	ypix = ypix[np.in1d(np.arange(len(ypix)), gtr100fluxind, invert=True)]
	xpix = xpix[np.in1d(np.arange(len(xpix)), gtr100fluxind, invert=True)]
	idx = idx[np.in1d(np.arange(len(idx)), gtr100fluxind, invert=True)]
	d2d = d2d[np.in1d(np.arange(len(d2d)), gtr100fluxind, invert=True)]
	ra = ra[np.in1d(np.arange(len(ra)), gtr100fluxind, invert=True)]
	dec = dec[np.in1d(np.arange(len(dec)), gtr100fluxind, invert=True)]

	if mapfreq == "90":
		ypix90 = ypix
		xpix90 = xpix
		ra90 = ra
		dec90 = dec
		cmap90 = cmap
	elif mapfreq == "150":
		ypix150 = ypix
		xpix150 = xpix
		ra150 = ra
		dec150 = dec
		cmap150 = cmap
	elif mapfreq == "220":
		ypix220 = ypix
		xpix220 = xpix
		ra220 = ra
		dec220 = dec
		cmap220 = cmap

# randommean = np.zeros((n_realizations))
# randommedian = np.zeros((n_realizations))
# for j in range(n_realizations):
# 	data = np.load("/data/tangq/wendymaps/"+field+"/Random_"+mapfreq+"GHz_Sim"+str(j)+".npy")
# 	rand_flux = data[0:len(ypix)]
# 	randommean[j] = np.mean(rand_flux)
# 	randommedian[j] = np.median(rand_flux)

def running_sims(n_realizations, ypix, xpix, cmap, radec0, reso_arcmin, npixels):
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
	#	np.save("/data/tangq/wendymaps/"+field+"/Random_"+mapfreq+"GHz_Sim"+str(j)+".npy", rand_flux)

		print "Done Iteration: " + str(j)
		if len(ypix) < len(rand_flux):
			rand_flux = rand_flux[0:len(ypix)]
		randommean[j] = np.mean(rand_flux)
		randommedian[j] = np.median(rand_flux)
	print "The field is " + str(field) + " and freq " + mapfreq
	print "The signal is (mean): " + str(np.mean(cmap[ypix,xpix]))
	print "The noise is: " + str(np.std(randommean))
	print "The signal to noise is (mean): " + str(np.mean(cmap[ypix,xpix])/np.std(randommean))


	print "The signal is (median): " + str(np.median(cmap[ypix,xpix]))
	print "The noise is: " + str(np.std(randommedian))
	print "The signal to noise is: " + str(np.median(cmap[ypix,xpix])/np.std(randommedian))

####### Getting SUMSS sources found in all maps####

loc90 = zip(ra90, dec90)
loc150 = zip(ra150, dec150)
loc220 = zip(ra220, dec220)
common_srcs = np.array(list(set(loc90) & set(loc150)& set(loc220)))

cmn_idx90 = np.zeros((common_srcs.shape[0]), dtype = int)
cmn_idx150 = np.zeros((common_srcs.shape[0]),dtype = int)
cmn_idx220 = np.zeros((common_srcs.shape[0]),dtype = int)

for src in range(len(common_srcs)):
	idx = np.where(loc90 == common_srcs[src,:])[0]
	for dx in idx:
		if len(idx) == 2:
			cmn_idx90[src] = dx
		else:
			if list(idx).count(dx) == 2:
				cmn_idx90[src] = dx

	idx = np.where(loc150 == common_srcs[src,:])[0]
	for dx in idx:
		if len(idx) == 2:
			cmn_idx150[src] = dx
		else:
			if list(idx).count(dx) == 2:
				cmn_idx150[src] = dx
	idx = np.where(loc220 == common_srcs[src,:])[0]
	for dx in idx:
		if len(idx) == 2:
			cmn_idx220[src] = dx
		else:
			if list(idx).count(dx) == 2:
				cmn_idx220[src] = dx	

freqs = np.array([90., 150., 220.])*1.e9
good90 = cmap90[ypix90[cmn_idx90],xpix90[cmn_idx90]]
good150 = cmap150[ypix150[cmn_idx150],xpix150[cmn_idx150]]
good220 = cmap220[ypix220[cmn_idx220],xpix90[cmn_idx220]]

plt.subplot(1,2,1)
for i in range(len(common_srcs)):
	plt.plot(freqs, np.asarray([good90[i],good150[i],good220[i]]),'--')
plt.plot(freqs, np.asarray([np.sum(good90), np.sum(good150), np.sum(good220)])/len(good220), '-k', linewidth=5, label='Averaged flux')
plt.legend()
plt.title('Plot of all the SUMSS srcs in SPT bands + averaged')
plt.xlabel('freq')
plt.ylabel('mJy')
plt.subplot(1,2,2)
plt.plot(freqs, np.asarray([np.median(good90), np.median(good150), np.median(good220)]), '-k', linewidth=5, label='median flux')
plt.legend()
plt.title('Plot of just the median SUMSS srcs fluxes in each bands')
plt.xlabel('freq')
plt.ylabel('mJy')

n_bs = 1000
bs_med90 = np.zeros((n_bs))
bs_med150 = np.zeros((n_bs))
bs_med220 = np.zeros((n_bs))

#bootstrapping
for i in range(n_bs):
	indices90 = np.random.choice(len(good90), len(good90))
	indices150 = np.random.choice(len(good150), len(good150))
	indices220 = np.random.choice(len(good220), len(good220))
	bs_med90[i] = np.median(good90[indices90])
	bs_med150[i] = np.median(good150[indices150])
	bs_med220[i] = np.median(good220[indices220])

yhigh = np.array([np.percentile(bs_med90, 15.9),np.percentile(bs_med150, 15.9),np.percentile(bs_med220, 15.9)]) #gives the 1 sigma lower bound and 
ylow = np.array([np.percentile(bs_med90, 84.1),np.percentile(bs_med150, 84.1),np.percentile(bs_med220, 84.1)])
ymed = np.asarray([np.median(good90), np.median(good150), np.median(good220)])
yerr90=np.array([np.percentile(bs_med90, 15.9)-ymed[0], ymed[0]-np.percentile(bs_med90, 84.1)])
yerr150=np.array([np.percentile(bs_med150, 15.9)-ymed[1], ymed[1]-np.percentile(bs_med150, 84.1)])
yerr220=np.array([np.percentile(bs_med220, 15.9)-ymed[2], ymed[2]-np.percentile(bs_med220, 84.1)])

plt.errorbar(freqs, ymed, yerr=np.array([yerr90, yerr150, yerr220]).T, fmt='o', label='median flux')
plt.legend()
plt.title('Plot of just the median SUMSS srcs fluxes in each bands')
plt.xlabel('freq')
plt.ylabel('mJy')

yerr = np.array([np.std(bs_med90), np.std(bs_med150), np.std(bs_med220)])

def spec_fit(x, A, f0, alpha):
	return A*(x/f0)**alpha

from scipy.optimize import curve_fit, least_squares
popt, pcov = least_squares(spec_fit, freqs, ymed, sigma = yerr, bounds=((0, 0, -1), (np.inf, np.inf, -3)))