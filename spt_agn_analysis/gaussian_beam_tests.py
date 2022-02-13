import numpy as np
import matplotlib.pyplot as pyplot



# fluxstack = np.load("/home/tangq/data/Herschel_brightest_400_in_250umband_500350250.npy")
# flux500 = fluxstack[0]

pixelsize = 15#(in arcsecs)
size =  50*pixelsize
fwhm_beam = 0.75*60 #arcsecs
sigma_x,sigma_y = fwhm_beam/np.sqrt(8*np.log(2)), fwhm_beam/np.sqrt(8*np.log(2))
A = 1
x = np.array([np.arange(size)]) - size/2
y = np.transpose(np.array([np.arange(size)])) - size /2
psf  = A *np.exp(-(x**2/(2*sigma_x**2) + y**2/(2*sigma_y**2)))
small = int(size/pixelsize)
bin_psf = psf.reshape((small,psf.shape[0]//small, small,psf.shape[1]//small)).mean(axis=3).mean(1)
plt.subplot(121)
plt.imshow(psf)
plt.colorbar()
plt.title(str(fwhm_beam/60) + 'beam, X & Y units are arcsecs')
plt.subplot(122)
plt.imshow(bin_psf)
plt.colorbar()
plt.title(str(fwhm_beam/60) + 'beam,  X & Y are pixel units (15")')
real_map = np.zeros(( [2048,2048]))
real_map[(real_map.shape[0]/2-bin_psf.shape[0]/2):(real_map.shape[0]/2+bin_psf.shape[0]/2), (real_map.shape[1]/2-bin_psf.shape[1]/2):(real_map.shape[1]/2+bin_psf.shape[1]/2)] = bin_psf


from astropy.io import fits
hdu = fits.PrimaryHDU(real_map)
hdu.writeto('/data/tangq/SPT/gaussian_0_75_beam_npixels_'+str(real_map.shape[0])+'_'+str(real_map.shape[1])+'_.fits')

new_zp_mask_map = zp_mask_map
new_zp_mask_map[1200:1200+small,1200:1200+small] = bin_psf + zp_mask_map[1200:1200+small,1200:1200+small] 
filt_output,filt_map =  calc.call_matched_filt_point(new_zp_mask_map, cmb = False, skysigvar = np.zeros((25000)), noisepsd=noise_psd, fwhm_beam_arcmin=beamsizes[beamsize], ellmax = 25000, ell_highpass = 10, ell_lowpass = 20000,use_fft_as_noise=False, plot_map=False, return_filtered_map=True, mask = zp_mask, reso_arcmin = reso_arcmin, ngridx = zp_mask_map.shape[0],ngridy = zp_mask_map.shape[0])
filt_output1,filt_map1 =  calc.call_matched_filt_point(new_zp_mask_map, cmb = False, skysigvar = np.zeros((25000)), noisepsd=noise_psd, fwhm_beam_arcmin=beamsizes[beamsize], ellmax = 25000, ell_highpass = 10, ell_lowpass = 20000,use_fft_as_noise=False, plot_map=False, return_filtered_map=True, mask = zp_mask, reso_arcmin = reso_arcmin, ngridx = zp_mask_map.shape[0],ngridy = zp_mask_map.shape[0])


#code implicitly assumes the map has been filtered in one direction for SPT
#try filter one map
plotting.plotRadialProfile(np.fft.fftshift(m_filt[0].optfilt), center_bin=[ngridx/2,ngridy/2], resolution=(ellx[0,1]-ellx[0,0]), histogram=True, range=[0,20000],plot=True, bin_resolution_factor=0.1, figure=5,new=False)
plotting.kspaceImshow(m_filt[0].optfilt,reso_arcmin=reso_arcmin,figure=4,log=True,title='ell-space optfilt')

plotting.kspaceImshow(noisepsd,reso_arcmin=reso_arcmin,figure=4,log=False,title='noise psd in k space')

random_map = np.random.normal(0,0.1,zp_mask_map.shape)
reso = (reso_arcmin/60)*np.pi/180.
dngrid = zp_mask_map.shape[0]
dk = 1./dngrid/reso
factor = np.sqrt(dngrid**2*dk**2)
noise_psd = np.sqrt(np.abs(np.fft.ifft2(random_map)/factor))


from sptpol_software.simulation.beams import calc_gauss_Bl
spotx = 1742
spoty = 832
random_map = np.zeros(([3000,3000]))
npixels = random_map.shape
pixelsize = 15#(in arcsecs)
size =  50*pixelsize
fwhm_beam = 1.2*60 #arcsecs
sigma_x,sigma_y = fwhm_beam/np.sqrt(8*np.log(2)), fwhm_beam/np.sqrt(8*np.log(2))
A = 0.36
x = np.array([np.arange(size)]) - size/2
y = np.transpose(np.array([np.arange(size)])) - size /2
psf  = A *np.exp(-(x**2/(2*sigma_x**2) + y**2/(2*sigma_y**2)))
small = int(size/15)
bin_psf = psf.reshape((small,psf.shape[0]//small, small,psf.shape[0]//small)).mean(axis=3).mean(1)
random_map[spotx:spotx+small,spoty:spoty+small] = bin_psf
reso_rad = 0.25 * constants.DTOR / 60.
ellg,ellx,elly = makeEllGrids([npixels[1],npixels[0]],reso_rad)
ell = np.arange(70000)
bfilt = calc_gauss_Bl(ell,fwhm_beam/60)
bl2d = bfilt[(np.round(ellg)).astype(int)]
skysig = np.zeros((25000))
ellmax = len(skysig)-1
cls = readCAMBCls('/data/tnatoli/camb/planck_lensing_wp_highL_bestFit_20130627_massless0p046_massive3_lensedtotCls.dat',
as_cls = True, extrapolate = True, as_uksq = False, max_ell = ellmax-1)
skysig[cls['ell']] += cls['cl']['TT']
skysig = np.zeros((len(skysig)))
skysig[cls['ell']] = cls['cl']['TT']

filt_output,filt_map =  calc.call_matched_filt_point(random_map, cmb = False, skysigvar = np.zeros((25000)), noisepsd=np.ones((npixels))*1.e-6, \
					beam=None,trans_func_grid = bl2d, use_fft_as_noise=False, \
					plot_map=False, return_filtered_map=True, mask = np.ones((npixels)), reso_arcmin = 0.25, ngridx = npixels[0],\
					ngridy = npixels[1])

# hdu = fits.PrimaryHDU(random_map)
# hdu.writeto('/data/tangq/debug/gaussian_source_sim_map.fits')
# hdu = fits.PrimaryHDU(bl2d)
# hdu.writeto('/data/tangq/debug/gaussian_source_sim_trans_func_beam.fits')
# hdu = fits.PrimaryHDU(np.ones((npixels))*1.e-6)
# hdu.writeto('/data/tangq/debug/gaussian_source_sim_noise_psd.fits')

plt.subplot(1,2,1)
plt.imshow(random_map[spotx:spotx+small,spoty:spoty+small])
plt.colorbar()
plt.title('Original map')
plt.subplot(1,2,2)
plt.imshow(filt_map[spotx:spotx+small,spoty:spoty+small])
plt.colorbar()
plt.title('Filtered map')

plt.subplot(1,2,1)
plt.imshow(bl2d)
plt.colorbar()
plt.title('Trans func + beam')
plt.subplot(1,2,2)
plt.imshow(filt_output[0].optfilt)
plt.colorbar()
plt.title('Optimal filter')

plt.subplot(1,2,1)
plotting.plotRadialProfile(np.fft.fftshift(m_filt[0].optfilt), center_bin=[ngridx/2,ngridy/2], resolution=(ellx[0,1]-ellx[0,0]), histogram=True, range=[0,20000],plot=True, bin_resolution_factor=0.1,new=False)
plt.title('original filter')
plt.subplot(1,2,2)
plotting.plotRadialProfile(np.fft.fftshift(y), center_bin=[ngridx/2,ngridy/2], resolution=(ellx[0,1]-ellx[0,0]), histogram=True, range=[0,20000],plot=True, bin_resolution_factor=0.1,new=False)
plt.title('filter w/ Gauss filt smoothing')
#plt.imshow(y)
#plt.title('Optfilt with a Gaussian filter smoothing')
#plt.colorbar()
