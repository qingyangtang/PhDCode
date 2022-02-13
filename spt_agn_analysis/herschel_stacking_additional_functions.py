
i=0

ypix, xpix = sky.ang2Pix(np.asarray([clean_cat['ra'], clean_cat['dec']]), radec0, reso_arcmin, npixels,proj=5)[0]
ypixmask, xpixmask = sky.ang2Pix(np.asarray([clean_cat['ra'], clean_cat['dec']]), radec0, reso_arcmin, npixels,proj=5)[1] 
ang2pix_mask = (ypixmask == True) & (xpixmask == True) & (pixelmask[ypix,xpix]>0.99999999)
clean_cat = clean_cat[ang2pix_mask]

idx, d2d = t.match_src_to_catalog(clean_cat['ra'], clean_cat['dec'], src_ra, src_dec)

src_mask = d2d.arcmin > 4
ypix = ypix[src_mask]
xpix = xpix[src_mask]
d2d = d2d[src_mask]
idx = idx[src_mask]
clean_cat = clean_cat[src_mask]

# lz_mask = (clean_cat['stellar_mass_z'] > 0.5) & (clean_cat['stellar_mass_z'] < 1) 
# ypix_lz = ypix[lz_mask]
# xpix_lz = xpix[lz_mask]
# clean_cat_lz = clean_cat[lz_mask]
sm_mask_dummy = ((clean_cat['lmass'] > sm_bins[i]) & (clean_cat['lmass'] < sm_bins[i+1]))
z_bins = np.arange(np.min(clean_cat['stellar_mass_z'][sm_mask_dummy]),np.min([np.max(clean_cat['stellar_mass_z'][sm_mask_dummy]),1.99])+0.2, 0.2)
flux_zbins = np.zeros((len(z_bins[:-1])))
num_obj_zbins = np.zeros((len(z_bins[:-1])))
n_realizations = 1000
bs_med = np.zeros((len(z_bins[:-1]), n_realizations))
median_z = np.zeros((len(z_bins[:-1])))
for x in range(3):
	cmap = map_arr[x]
	reso_arcmin = reso_arcmin_arr[x]
	radec0 = radec0_arr[x]
	npixels = npixels_arr[x]
	mapfreq = mapfreq_arr[x]
	for z_i in [0]:
		sm_mask = (clean_cat['stellar_mass_z'] > z_bins[z_i]) & (clean_cat['stellar_mass_z'] < z_bins[z_i+1]) \
			& (clean_cat['lmass'] > sm_bins[i]) & (clean_cat['lmass'] < sm_bins[i+1])
		ypix_i = ypix[sm_mask]
		xpix_i = xpix[sm_mask]
		num_obj_zbins[z_i] = len(ypix_i)

		flux_zbins[z_i] = np.mean(cmap[ypix_i,xpix_i])
		median_z[z_i] = np.median(clean_cat['stellar_mass_z'][sm_mask])
		galaxy_pixels = cmap[ypix_i,xpix_i]

	nonzero_pix = cmap[zp_mask!=0]
	ran_samples = np.random.choice(nonzero_pix, 1000)
	print ran_samples[:10]
	error_std = np.std(ran_samples)
	error_mad = mad(samples)

	if mapfreq == "250":
		flux_250 = cmap[ypix_i,xpix_i]
		z_250 = clean_cat['stellar_mass_z'][sm_mask]
		num_obj_250 = num_obj_zbins
		error_250 = error
		error_250_mad = error_mad
 	elif mapfreq == "350":
		flux_350 = cmap[ypix_i,xpix_i]
		z_350 = clean_cat['stellar_mass_z'][sm_mask]
		num_obj_350 = num_obj_zbins
		error_350 = error
		error_350_mad = error_mad
	elif mapfreq == "500":
		flux_500 = cmap[ypix_i,xpix_i]
		z_500 = clean_cat['stellar_mass_z'][sm_mask]
		num_obj_500 = num_obj_zbins
		error_500 = error
		error_500_mad = error_mad

flux_err = np.asarray([error_250, error_350, error_500])
flux_err_mad = np.asarray([error_250_mad, error_350_mad, error_500_mad])

z_i=0
T_fit = np.zeros((len(flux_250)))

num_plots = len(flux_250)/25
for j in range(num_plots):
	fig, axarr1 = plt.subplots(5,5, squeeze=True,sharex=True, sharey=False, figsize=(25,25))
	for gal in np.arange(25):
		stacked_flux = np.asarray([flux_250[gal+j*25], flux_350[gal+j*25], flux_500[gal+j*25]])
		try: 
			popt, pcov = curve_fit(lambda p1, p2,p3: mod_BB_curve_with_z(p1, p2, p3, z_250[gal+j*25]), freqs, stacked_flux, \
				               sigma = flux_err, p0 = [1.e-10,10], \
				               maxfev=10000)
			axarr1[int(gal/5), gal % 5].errorbar(bands*1.e6, stacked_flux, yerr=flux_err,marker = 'o',lw=1, label='z='+str(z_250[gal+j*25]))
			axarr1[int(gal/5), gal % 5].plot(bands*1.e6, mod_BB_curve_with_z(freqs, popt[0], popt[1], z_250[gal+j*25]), \
				label='A=' + '{0:1.2e}'.format(popt[0]) +', T='+"{:.0f}".format(popt[1])+'K')
			axarr1[int(gal/5), gal % 5].legend(loc='best', prop={'size': 10})
			T_fit[j*25+gal] = popt[1]
			chi2 = np.sum((mod_BB_curve_with_z(freqs, popt[0],popt[1],z_250[gal+j*25])-stacked_flux)**2/flux_err**2)
			axarr1[int(gal/5), gal % 5].set_title('chi2=' + "{:.3f}".format(chi2))
			axarr1[int(gal/5), gal % 5].set_xlim(100., 600.)
		except:
			pass
	axarr1[0,0].set_ylabel('Flux') 
	axarr1[int(gal/5), (gal % 5)].set_xlabel('Wavelength (um)')
	fig.suptitle('BB curvefit to individual galaxy flux at SM=' +str((sm_bins[i]+sm_bins[i+1])/2.))
	plt.savefig('/data/tangq/'+today+'_Herschel_gal_ind_'+str(j)+'.png')
	plt.close()
	print j


##----- COMPUTING GLOBAL CHI2 VALS ---------##


for i in range(len(sm_bins)-1):
	for x in range(3):
		cmap = map_arr[x]
		reso_arcmin = reso_arcmin_arr[x]
		radec0 = radec0_arr[x]
		npixels = npixels_arr[x]
		mapfreq = mapfreq_arr[x]

		ypix, xpix = sky.ang2Pix(np.asarray([clean_cat['ra'], clean_cat['dec']]), radec0, reso_arcmin, npixels,proj=5)[0]
		ypixmask, xpixmask = sky.ang2Pix(np.asarray([clean_cat['ra'], clean_cat['dec']]), radec0, reso_arcmin, npixels,proj=5)[1] 
		ang2pix_mask = (ypixmask == True) & (xpixmask == True) & (pixelmask[ypix,xpix]>0.99999999)
		clean_cat = clean_cat[ang2pix_mask]

		idx, d2d = t.match_src_to_catalog(clean_cat['ra'], clean_cat['dec'], src_ra, src_dec)

		src_mask = d2d.arcmin > 4
		ypix = ypix[src_mask]
		xpix = xpix[src_mask]
		d2d = d2d[src_mask]
		idx = idx[src_mask]
		clean_cat = clean_cat[src_mask]

		sm_mask_dummy = ((clean_cat['lmass'] > sm_bins[i]) & (clean_cat['lmass'] < sm_bins[i+1]))
		z_bins = np.arange(np.min(clean_cat['stellar_mass_z'][sm_mask_dummy]),np.min([np.max(clean_cat['stellar_mass_z'][sm_mask_dummy]),1.99])+0.2, 0.2)
		flux_zbins = np.zeros((len(z_bins[:-1])))
		num_obj_zbins = np.zeros((len(z_bins[:-1])))
		n_realizations = 1000
		bs_med = np.zeros((len(z_bins[:-1]), n_realizations))
		median_z = np.zeros((len(z_bins[:-1])))
		for z_i in range(len(z_bins)-1):
			sm_mask = (clean_cat['stellar_mass_z'] > z_bins[z_i]) & (clean_cat['stellar_mass_z'] < z_bins[z_i+1]) \
				& (clean_cat['lmass'] > sm_bins[i]) & (clean_cat['lmass'] < sm_bins[i+1])
			ypix_i = ypix[sm_mask]
			xpix_i = xpix[sm_mask]
			num_obj_zbins[z_i] = len(ypix_i)
			flux_zbins[z_i] = np.mean(cmap[ypix_i,xpix_i])
			median_z[z_i] = np.median(clean_cat['stellar_mass_z'][sm_mask])
			galaxy_pixels = cmap[ypix_i,xpix_i]

			#bootstrapping
			for j in range(n_realizations):
				indices = np.random.choice(len(ypix_i), len(ypix_i))
				bs_med[z_i,j] = np.median(galaxy_pixels[indices])

		ylow = np.zeros((len(z_bins[:-1])))
		yhigh = np.zeros((len(z_bins[:-1])))

		mad_error = np.zeros((len(z_bins[:-1])))
		for y_i in range(len(ylow)):
			mad_error[y_i] = mad(bs_med[y_i,:])
		error = mad_error

		if mapfreq == "250":
			flux_250 = flux_zbins
			error_250 = error
			error_250_mad = mad_error
			num_obj_250 = num_obj_zbins
	 	elif mapfreq == "350":
			flux_350 = flux_zbins
			error_350 = error
			error_350_mad = mad_error
			num_obj_350 = num_obj_zbins
		elif mapfreq == "500":
			flux_500 = flux_zbins
			error_500 = error
			error_500_mad = mad_error
			num_obj_500 = num_obj_zbins
	chi2_T = np.zeros((len(T_set)))
	for n, T in enumerate(T_set):
		chi2_z = np.zeros((len(z_bins)-1))

		for z_i in range(len(z_bins)-1):
			z = median_z[z_i]
			stacked_flux_err = np.asarray([error_250[z_i], error_350[z_i], error_500[z_i]])
			stacked_flux = [flux_250[z_i], flux_350[z_i], flux_500[z_i]]
			# colors = plt.cm.jet(np.linspace(0,1,len(z_set)))
			popt, pcov = curve_fit(lambda p1, p2: mod_BB_curve_with_z(p1, p2, T, z), freqs, stacked_flux, \
	        	sigma = stacked_flux_err, p0 = [1.e-10], maxfev=10000)
			chi2_z[z_i] = np.sum((mod_BB_curve_with_z(freqs, popt[0],T,z)-stacked_flux)**2/stacked_flux_err**2/dof)

		chi2_T[n] = np.sum(chi2_z)
	plt.figure()
	plt.plot(T_set, chi2_T,'.')
	plt.xlabel('Temperature (K)')
	plt.ylabel('reduced chi2')
	plt.suptitle('global reduced chi2 for temperature of BB curvefit to galaxy stacked flux at SM=' +str((sm_bins[i]+sm_bins[i+1])/2.))
	plt.savefig('/data/tangq/blackbody_plots/'+today+'Global_chi2_SM'+str(i)+'_BBfits.png')
	plt.close()

	print "SM Bin " + str(i)


##----------- STACKING LARGE AMOUNT OF DATA --------------##
for i in range(len(sm_bins)-1):
	sn_arr = np.zeros((3))
	noise_arr = np.zeros((3))
	signal_arr = np.zeros((3))
	sn_arr_med = np.zeros((3))
	plt.figure(figsize=(25,5))

	for x in range(3):
		cmap = map_arr[x]
		reso_arcmin = reso_arcmin_arr[x]
		radec0 = radec0_arr[x]
		npixels = npixels_arr[x]
		mapfreq = mapfreq_arr[x]

		ypix, xpix = sky.ang2Pix(np.asarray([clean_cat['ra'], clean_cat['dec']]), radec0, reso_arcmin, npixels,proj=5)[0]
		ypixmask, xpixmask = sky.ang2Pix(np.asarray([clean_cat['ra'], clean_cat['dec']]), radec0, reso_arcmin, npixels,proj=5)[1] 
		ang2pix_mask = (ypixmask == True) & (xpixmask == True) & (pixelmask[ypix,xpix]>0.99999999)
		clean_cat = clean_cat[ang2pix_mask]

		idx, d2d = t.match_src_to_catalog(clean_cat['ra'], clean_cat['dec'], src_ra, src_dec)

		src_mask = d2d.arcmin > 4
		ypix = ypix[src_mask]
		xpix = xpix[src_mask]
		d2d = d2d[src_mask]
		idx = idx[src_mask]
		clean_cat = clean_cat[src_mask]

		n_realizations = 1000
		bs_med = np.zeros((n_realizations))
		sm_mask = (clean_cat['lmass'] > sm_bins[i]) & (clean_cat['lmass'] < sm_bins[i+1])
		ypix_i = ypix[sm_mask]
		xpix_i = xpix[sm_mask]
		num_obj_zbins = len(ypix_i)
		flux_zbins = np.mean(cmap[ypix_i,xpix_i])
		median_z = np.median(clean_cat['stellar_mass_z'][sm_mask])
		galaxy_pixels = cmap[ypix_i,xpix_i]

		# #bootstrapping
		# for j in range(n_realizations):
		# 	indices = np.random.choice(len(ypix_i), len(ypix_i))
		# 	bs_med[j] = np.median(galaxy_pixels[indices])

		noise_mad = mad(galaxy_pixels)
		sn_arr[x] = flux_zbins
		sn_arr_med[x] = np.median(cmap[ypix_i,xpix_i])

		plt.subplot(1,3,x+1)
		gal_bins = np.arange(np.min(galaxy_pixels), np.max(galaxy_pixels), 0.001)
		plt.hist(galaxy_pixels, bins=gal_bins, alpha=0.3, color='g')
		plt.xlabel('Galaxy flux (mJy)')
		plt.title('Hist of gal fluxes at SM=' +str((sm_bins[i]+sm_bins[i+1])/2.)+','+str(bands[x]*1.e6)+'um')
		plt.axvline(sn_arr_med[x], color='r', linestyle='solid', linewidth=2, label='med='+'{0:1.2e}'.format(sn_arr_med[x]))
		plt.axvline(sn_arr[x], color='g', linestyle='solid', linewidth=2,label='mean='+'{0:1.2e}'.format(sn_arr[x]))
		plt.axvline(sn_arr_med[x]+5*noise_mad, color='c', linestyle='dashed', linewidth=1,label='5sig\n1sig='+'{0:1.2e}'.format(noise_mad))
		plt.axvline(sn_arr_med[x]-5*noise_mad, color='c', linestyle='dashed', linewidth=1)
		sm_mask = (clean_cat['stellar_mass_z'] > 0.3) & (clean_cat['stellar_mass_z'] < 0.6) \
					& (clean_cat['lmass'] > sm_bins[i]) & (clean_cat['lmass'] < sm_bins[i+1])
		ypix_i = ypix[sm_mask]
		xpix_i = xpix[sm_mask]
		flux_zbins = np.mean(cmap[ypix_i,xpix_i])
		galaxy_pixels = cmap[ypix_i,xpix_i]
		plt.hist(galaxy_pixels, bins=gal_bins, alpha=0.3, color='m', label='0.3<z<0.6')
		plt.yscale('log', nonposy='clip')
		plt.legend(loc='best')


	plt.savefig('/data/tangq/blackbody_plots/'+today+'Histogram_gal_flux_SM'+str(i)+'.png')
	plt.close()


ra_7sig = []
dec_7sig = []


##----------- STACKING LARGE AMOUNT OF DATA --------------##
sn_arr_mean_all = []
sn_arr_med_all = []
num_obj_zbins_all = []

bw_colors = ['blue', 'green', 'orange', 'red']

### masking out the 7sig outliers### 
srcs_7sig = np.load('/home/tangq/src_7sig_herschel_hist.npy')
idx, d2d = t.match_src_to_catalog(clean_cat['ra'], clean_cat['dec'], srcs_7sig[:,0],srcs_7sig[:,1])
src_flag = d2d.arcsec > 5.
clean_cat = clean_cat[src_flag]
### masking out any pixels within 4 arcmins of the point sources in Herschel ptsrc catalog
idx, d2d = t.match_src_to_catalog(clean_cat['ra'], clean_cat['dec'], src_ra, src_dec)
src_mask = d2d.arcmin > 4
clean_cat = clean_cat[src_mask]

z_bins = np.arange(0, 2, 0.2)
for i in range (len(z_bins)-1):
#for i in range(len(sm_bins)-1):

	fig, axarr1 = plt.subplots(4,3, squeeze=True,sharex=True, sharey=False, figsize=(20,15))
	for b, bin_width in enumerate([0.1, 0.2, 0.5]):
		#sm_mask_dummy = ((clean_cat['lmass'] > sm_bins[i]) & (clean_cat['lmass'] < sm_bins[i+1]))
		#z_bins = np.arange(np.min(clean_cat['stellar_mass_z'][sm_mask_dummy]),np.min([np.max(clean_cat['stellar_mass_z'][sm_mask_dummy]),1.99])+bin_width, bin_width)
		sm_mask_dummy = ((clean_cat['stellar_mass_z'] > z_bins[i]) & (clean_cat['stellar_mass_z'] < z_bins[i+1]))
		sm_bins = np.arange(np.max([np.min(clean_cat['lmass'][sm_mask_dummy]), 7]),np.min([np.max(clean_cat['lmass'][sm_mask_dummy]),11])+bin_width, bin_width)

		flux_zbins = np.zeros((len(sm_bins[:-1])))
		num_obj_zbins = np.zeros((len(sm_bins[:-1])))
		n_realizations = 1000
		sn_arr_mean = np.zeros((len(sm_bins[:-1]), 3))
		sn_arr_med = np.zeros((len(sm_bins[:-1]), 3))

		for x in range(3):
			cmap = map_arr[x]
			reso_arcmin = reso_arcmin_arr[x]
			radec0 = radec0_arr[x]
			npixels = npixels_arr[x]
			mapfreq = mapfreq_arr[x]
			ypix, xpix = sky.ang2Pix(np.asarray([clean_cat['ra'], clean_cat['dec']]), radec0, reso_arcmin, npixels,proj=5)[0]
			ypixmask, xpixmask = sky.ang2Pix(np.asarray([clean_cat['ra'], clean_cat['dec']]), radec0, reso_arcmin, npixels,proj=5)[1] 
			ang2pix_mask = (ypixmask == True) & (xpixmask == True) & (zp_mask[ypix,xpix]>0.99999999)
			clean_cat1 = clean_cat[ang2pix_mask]
			ypix = ypix[ang2pix_mask]
			xpix = xpix[ang2pix_mask]

			for z_i in range(len(sm_bins)-1):
				# sm_mask = (clean_cat1['stellar_mass_z'] > z_bins[z_i]) & (clean_cat1['stellar_mass_z'] < z_bins[z_i+1]) \
				# 	& (clean_cat1['lmass'] > sm_bins[i]) & (clean_cat1['lmass'] < sm_bins[i+1])
				sm_mask = (clean_cat1['stellar_mass_z'] > z_bins[i]) & (clean_cat1['stellar_mass_z'] < z_bins[i+1]) \
					& (clean_cat1['lmass'] > sm_bins[z_i]) & (clean_cat1['lmass'] < sm_bins[z_i+1])

				ypix_i = ypix[sm_mask]
				xpix_i = xpix[sm_mask]
				num_obj_zbins[z_i] = len(ypix_i)
				flux_zbins = np.mean(cmap[ypix_i,xpix_i])
				median_z = np.median(clean_cat1['stellar_mass_z'][sm_mask])
				galaxy_pixels = cmap[ypix_i,xpix_i]

				noise_mad = t.mad(galaxy_pixels)
				sn_arr_med[z_i,x] = np.median(cmap[ypix_i,xpix_i])/noise_mad
				sn_arr_mean[z_i,x] = np.mean(cmap[ypix_i,xpix_i])/noise_mad
		for j in range(3):
			axarr1[0, j].plot(sm_bins[:-1], sn_arr_mean[:,j], color=bw_colors[b], label=r'$\Delta$SM='+str(bin_width))
			axarr1[0, j].plot(sm_bins[:-1], sn_arr_med[:,j], ls = 'dotted', color=bw_colors[b])
			axarr1[1, j].plot(sm_bins[:-1], sn_arr_mean[:,j]/num_obj_zbins, color=bw_colors[b], label=r'$\Delta$SM='+str(bin_width))
			axarr1[1, j].plot(sm_bins[:-1], sn_arr_med[:,j]/num_obj_zbins, ls = 'dotted', color=bw_colors[b])
			axarr1[2, j].plot(sm_bins[:-1], 5./(sn_arr_mean[:,j]/num_obj_zbins), color=bw_colors[b], label=r'$\Delta$SM='+str(bin_width))
			axarr1[2, j].plot(sm_bins[:-1], 5./(sn_arr_med[:,j]/num_obj_zbins), ls = 'dotted', color=bw_colors[b])
			axarr1[3, j].plot(sm_bins[:-1], num_obj_zbins, color=bw_colors[b], label=r'$\Delta$z='+str(bin_width))
	for l in range(3):
		axarr1[0, l].set_ylim(-0.3,1.3)
		axarr1[1, l].set_ylim(5.e-6, 1.e-2)
		axarr1[1, l].set_yscale('log')
		axarr1[2, l].set_ylim(1,3.e5)
	axarr1[0, 0].set_ylabel('S/N')
	axarr1[1, 0].set_ylabel('SN/obj')
	axarr1[2, 0].set_ylabel('# objs needed for S/N=5')
	axarr1[3, 0].set_ylabel('# objs in each z bin')
	axarr1[3, 0].set_xlabel('SM (250um data)')
	axarr1[3, 1].set_xlabel('SM (350um data)')
	axarr1[3, 2].set_xlabel('SM (500um data)')
	axarr1[3, 2].legend(loc='best',fontsize="x-small")
	axarr1[2, 2].legend(loc='best',fontsize="x-small")
	axarr1[1, 2].legend(loc='best',fontsize="x-small")
	axarr1[0, 2].legend(loc='best',fontsize="x-small")
	fig.suptitle('Mean (solid) or median (dotted) S/N per obj for varying redshift bins at z=' +str((z_bins[i]+z_bins[i+1])/2.))
	plt.savefig('/data/tangq/blackbody_plots/'+today+'Herschel_SN_mean_SM_bins_z'+str(i)+'.png')
	plt.close()


ra_7sig_1D = []
dec_7sig_1D = []
for i in range(30):
	for j in range(len(ra_7sig[i])):
		if (ra_7sig[i][j] in ra_7sig_1D) == False:
			ra_7sig_1D.append(ra_7sig[i][j])
			dec_7sig_1D.append(dec_7sig[i][j])


for i in range(len(ra_7sig_1D)):
	print str(ra_7sig_1D[i]) +',' + str(dec_7sig_1D[i])


