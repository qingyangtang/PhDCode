import numpy as np
import matplotlib.pyplot as plt
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
import scipy.constants as const
from scipy.optimize import curve_fit
c_c = const.c
k_c = const.k
h_c = const.h
from datetime import datetime
import matplotlib.gridspec as gridspec
from scipy.interpolate import UnivariateSpline

today = datetime.today().strftime('%Y%m%d')
bands = np.asarray([250.e-6, 350.e-6, 500.e-6])
freqs = c_c/bands
Tcmb = 2.725

sp_des_dir = '/data/tangq/custom_catalogs/spitzer_des_cat.fits'
stellar_m_dir = '/data/tangq/custom_catalogs/ssdf_matched_to_des_i1_lt_20_stellar_mass.sav'

# stellar_m_dir = '/data/bleeml/stellar_mass_cats/ssdf_des_try1.fout'
# #clean_cat = od.get_Spitzer_DES_cat_with_mass(sp_des_dir,stellar_m_dir,iband_cut=23)
# cat = idl.readIDLSav(stellar_m_dir).cat
# cat_dec = np.asarray([cat[x].dec for x in range(len(cat))])
# cat_ra = np.asarray([cat[x].ra for x in range(len(cat))])
# cat_zphot = np.asarray([cat[x].z_phot for x in range(len(cat))])
# cat_sm = np.asarray([cat[x].stellar_mass for x in range(len(cat))])
# cat_smchi2 = np.asarray([cat[x].chi2_sm for x in range(len(cat))])

# mask = (cat_smchi2 < 10) & (cat_zphot >0) & np.invert(np.isnan(cat_sm))
# cat_dec = cat_dec[mask]
# cat_ra = cat_ra[mask]
# cat_zphot = cat_zphot[mask]
# cat_sm = cat_sm[mask]
# cat_smchi2 = cat_smchi2[mask]
# del cat

# clean_cat= Table([cat_ra, cat_dec, cat_zphot, cat_sm, cat_smchi2], names=('ra', 'dec', 'stellar_mass_z', 'lmass', 'sm_chi2'))
clean_cat = fits.open('/data/tangq/custom_catalogs/20210614_cat.fits')[1].data


########## getting rid of galaxies within 4 arcmin of point sources
SPIRE_ptsrc_dir = "/data/tangq/SPIRE_maps/spt_starfndr_catalog.sav"

hers_src_ra, hers_src_dec, hers_src_flux = od.get_SPIRE_ptsrcs(SPIRE_ptsrc_dir, sn_cut=40)
sptsz_src_ra, sptsz_src_dec, sptsz_src_rad = od.get_SPTSZ_ptsrcs('/data/tangq/SPT/SZ/ptsrc_config_ra23h30dec-55_surveyclusters.txt')

src_ra = np.concatenate([hers_src_ra,sptsz_src_ra])
src_dec = np.concatenate([hers_src_dec,sptsz_src_dec])
idx, d2d = t.match_src_to_catalog(clean_cat['ra'], clean_cat['dec'], src_ra, src_dec)
src_mask = d2d.arcmin > 4
clean_cat = clean_cat[src_mask]

## cleaning out area in weird patch
#mask = clean_cat['ra']<357.
#clean_cat = clean_cat[mask]
mask = (clean_cat['lmass']>8.5) & (clean_cat['lmass']<12) & (clean_cat['sm_chi2'] >0) # & (clean_cat['lssfr'] <-11)
clean_cat = clean_cat[mask]
#sm_bins = np.asarray([8.5, 10.2, 10.6, 10.9, 11.2,12.])
#z_bins = np.asarray([0., 0.66, 1.33, 2.])

sm_bins = np.asarray([9, 9.5, 10., 10.5, 11., 11.5, 12])
z_bins = np.asarray([0., 0.5, 1., 1.5, 2.])
n_zbins = len(z_bins)-1

############SPT-SZ + SPTpol

fname = '/data/tangq/SPT/SZ/products/SPTSZ_filtered_no_smooth_FTfixed_20210401'
fnamepol = '/data/tangq/SPT/SPTpol/products/SPTpol_filtered_no_smooth_20210510'
# fname = '/data/tangq/SPT/SZ/products/SPTSZ_filtered_1_7_smooth_FTfixed_20210611'
# fnamepol = '/data/tangq/SPT/SPTpol/products/SPTpol_filtered_1_7_smooth_FTfixed_20210611'
map90 = np.load(fnamepol + '_90_map.npy')
map150 = np.load(fnamepol + '_150_map.npy')
map220 = np.load(fname + '_220_map.npy')
sptsz_pixelmask = np.load(fname + '_zp_mask.npy')
sptsz_reso_arcmin = np.loadtxt(fname + '_reso_arcmin.txt')
sptsz_radec0 = np.loadtxt(fname + '_radec.txt')
sptsz_cmap = [map90, map150, map220]
sptsz_npixels = [np.asarray(map90.shape),np.asarray(map150.shape), np.asarray(map220.shape)]
sptsz_mapfreq = ["95", "150", "220"]
filt90 = np.load(fnamepol+'_90_filter.npy').item()
filt150 = np.load(fnamepol+'_150_filter.npy').item()
filt220 = np.load(fname+'_220_filter.npy').item()
# filt90 = np.load('/data/tangq/SPT/SPTpol/products/SPTpol_filtered_1_7_smooth_FTfixed_20210610'+'_150_filter.npy').item()
# filt150 = filt90
# filt220 = filt90
sptsz_filt = [filt90, filt150, filt220]
spt_K2Jy_conversion = np.zeros((len(sptsz_cmap)))
for x in range(len(sptsz_cmap)):
    spt_K2Jy_conversion[x], dummy = t.calc_conversion_factor(sptsz_mapfreq[x], sptsz_reso_arcmin[x], sptsz_filt[x])
    sptsz_cmap[x] = sptsz_cmap[x]*spt_K2Jy_conversion[x]

########## Hershel

normalize_solid_angles = False

# fname = '/data/tangq/SPIRE_maps/products/Herschel_filtered_1_7_smooth_FTfixed_20210610'
fname = '/data/tangq/SPIRE_maps/products/Herschel_filtered_no_smooth_FTfixed_20210519'
map250 = np.load(fname+'_250_map.npy')
map350 = np.load(fname+'_350_map.npy')
map500 = np.load(fname+'_500_map.npy')
hers_pixelmask = np.load(fname+'_zp_mask.npy')
hers_reso_arcmin = np.loadtxt(fname+'_reso_arcmin.txt')
hers_radec0 = np.loadtxt(fname+'_radec.txt')
hers_cmap = [map250, map350, map500]
hers_npixels = [np.asarray(map250.shape),np.asarray(map350.shape), np.asarray(map500.shape)]
hers_mapfreq = ["250", "350", "500"]
filt250 = np.load(fname+'_250_filter.npy').item()
filt350 = np.load(fname+'_350_filter.npy').item()
filt500 = np.load(fname+'_500_filter.npy').item()
# filt250 = np.load('/data/tangq/SPT/SPTpol/products/SPTpol_filtered_1_7_smooth_FTfixed_20210610'+'_150_filter.npy').item()
# filt350 = filt250
# filt500 = filt250
hers_filt = [filt250, filt350, filt500]

### making galaxy mask
gal_mask_dir = '/data/tangq/custom_catalogs/SSDF_galaxy_mask_with_DES.sav'
spt_gal_mask = idl.readIDLSav(gal_mask_dir).mask
hers_gal_mask = np.zeros((spt_gal_mask.shape))
hers_gal_mask[1:, 2:] = spt_gal_mask[:-1,:-2]

spt_gal_mask = spt_gal_mask*sptsz_pixelmask
hers_gal_mask = hers_gal_mask*hers_pixelmask
spt_gal_mask[spt_gal_mask < 0.9999] = 0
hers_gal_mask[hers_gal_mask < 0.9999] = 0
spt_gal_mask_bool = np.asarray(spt_gal_mask, dtype = bool)
hers_gal_mask_bool = np.asarray(hers_gal_mask, dtype = bool)

### getting t-sz spectrum 
hers_tsz_K2Jy_conversion = np.zeros((len(hers_cmap)))
hers_tsz_eff_freq = t.tsz_eff_freq_hers()
spt_tsz_eff_freq = np.concatenate([t.tsz_eff_freq_sptpol(), np.asarray([t.tsz_eff_freq_sptsz()[-1]])])
spt_tsz_K2Jy_conversion = np.zeros((len(sptsz_cmap)))
spt_solid_angle = np.zeros((len(sptsz_cmap)))
hers_solid_angle = np.zeros((len(hers_cmap)))
for x in range(len(sptsz_cmap)):
    hers_tsz_K2Jy_conversion[x], hers_solid_angle[x] = t.calc_conversion_factor(hers_tsz_eff_freq[x], hers_reso_arcmin[x], hers_filt[x])
    spt_tsz_K2Jy_conversion[x], spt_solid_angle[x] = t.calc_conversion_factor(spt_tsz_eff_freq[x], sptsz_reso_arcmin[x], sptsz_filt[x])

theor_tsz_in_T = t.theor_tsz_hers_spt_in_T(use_sptpol = True)
theor_tsz_Jy = np.concatenate([theor_tsz_in_T[:3]*hers_tsz_K2Jy_conversion, theor_tsz_in_T[3:]*spt_K2Jy_conversion[::-1]])

#### cleaning out any galaxies in masked out places
sptsz_ypix = []
sptsz_xpix = []
hers_ypix = []
hers_xpix = []

for i in range(3):
    sptsz_ypix_i, sptsz_xpix_i = sky.ang2Pix(np.asarray([clean_cat['ra'], clean_cat['dec']]), sptsz_radec0[i], sptsz_reso_arcmin[i], sptsz_npixels[i],proj=0)[0]
    sptsz_ypixmask, sptsz_xpixmask = sky.ang2Pix(np.asarray([clean_cat['ra'], clean_cat['dec']]), sptsz_radec0[i], sptsz_reso_arcmin[i], sptsz_npixels[i],proj=0)[1] 
    hers_ypix_i, hers_xpix_i = sky.ang2Pix(np.asarray([clean_cat['ra'], clean_cat['dec']]), hers_radec0[i], hers_reso_arcmin[i], hers_npixels[i],proj=0)[0]
    hers_ypixmask, hers_xpixmask = sky.ang2Pix(np.asarray([clean_cat['ra'], clean_cat['dec']]), hers_radec0[i], hers_reso_arcmin[i], hers_npixels[i],proj=0)[1] 
    ang2pix_mask = (sptsz_ypixmask == True) & (sptsz_xpixmask == True) & (hers_ypixmask == True) & (hers_xpixmask == True)
    clean_cat = clean_cat[ang2pix_mask]
    sptsz_ypix_i = sptsz_ypix_i[ang2pix_mask]
    sptsz_xpix_i = sptsz_xpix_i[ang2pix_mask]
    hers_ypix_i = hers_ypix_i[ang2pix_mask]
    hers_xpix_i = hers_xpix_i[ang2pix_mask]
    pix_mask = (sptsz_pixelmask[sptsz_ypix_i,sptsz_xpix_i]>0.99999999) & (hers_pixelmask[hers_ypix_i,hers_xpix_i]>0.99999999)
    clean_cat = clean_cat[pix_mask]
    sptsz_ypix_i = sptsz_ypix_i[pix_mask]
    sptsz_xpix_i = sptsz_xpix_i[pix_mask]
    hers_ypix_i = hers_ypix_i[pix_mask]
    hers_xpix_i = hers_xpix_i[pix_mask] 
    sptsz_ypix.append(sptsz_ypix_i)
    sptsz_xpix.append(sptsz_xpix_i)
    hers_ypix.append(hers_ypix_i)
    hers_xpix.append(hers_xpix_i)

######

#sm_bins = np.arange(9,11.2,0.2)
n_realizations = 100

herschel_lookup_table = np.load('/data/tangq/SPIRE_maps/Herschel_effective_bands.npy')
sptsz_lookup_table = np.load('/data/tangq/wendymaps/SPTSZ_effective_bands.npy')
sptpol_lookup_table = np.load('/data/tangq/SPT/SPTpol/SPTpol_effective_bands.npy')

hers_stack_flux = np.zeros((len(sm_bins[:-1]), n_zbins, len(hers_mapfreq)))
hers_stack_flux_err = np.zeros((len(sm_bins[:-1]), n_zbins, len(hers_mapfreq)))
spt_stack_flux = np.zeros((len(sm_bins[:-1]), n_zbins, len(sptsz_mapfreq)))
spt_stack_flux_err = np.zeros((len(sm_bins[:-1]), n_zbins, len(sptsz_mapfreq)))
spt_filtered_maps = np.zeros((len(sm_bins[:-1]), n_zbins, len(sptsz_mapfreq), sptsz_cmap[0].shape[0], sptsz_cmap[0].shape[1]))
hers_filtered_maps = np.zeros((len(sm_bins[:-1]), n_zbins, len(hers_mapfreq), hers_cmap[0].shape[0], hers_cmap[0].shape[1]))

T_set = np.arange(3, 45., 0.1)
dof = np.float(len(bands))+np.float(len(sptsz_mapfreq))-1.
num_obj_zbins = np.zeros((len(sm_bins[:-1]),n_zbins))
median_z = np.zeros((len(sm_bins[:-1]), n_zbins))
mean_z = np.zeros((len(sm_bins[:-1]), n_zbins))


for i in range(len(sm_bins)-1):
    # sm_mask_dummy = ((clean_cat['lmass'] > sm_bins[i]) & (clean_cat['lmass'] < sm_bins[i+1]))
    #z_bins = np.linspace(np.min(clean_cat['stellar_mass_z'][sm_mask_dummy]),np.min([np.max(clean_cat['stellar_mass_z'][sm_mask_dummy]),1.99])+0.01, n_zbins+1)
    for x in range(3):
        cmap = hers_cmap[x]
        ypix = hers_ypix[x]
        xpix = hers_xpix[x]
        mapfreq = hers_mapfreq[x]
        flux_zbins = np.zeros((len(z_bins[:-1])))
        error = np.zeros((len(z_bins[:-1])))
        ### creating a galaxy mask
        filtered_maps = np.zeros((len(z_bins[:-1]),cmap.shape[0],cmap.shape[1]))
        for z_i in range(len(z_bins[:-1])):
            sm_mask = (clean_cat['stellar_mass_z'] > z_bins[z_i]) & (clean_cat['stellar_mass_z'] < z_bins[z_i+1]) \
                & (clean_cat['lmass'] > sm_bins[i]) & (clean_cat['lmass'] < sm_bins[i+1])
            ypix_i = ypix[sm_mask]
            xpix_i = xpix[sm_mask]
            num_obj_zbins[i,z_i] = len(ypix_i)
            if num_obj_zbins[i,z_i] != 0:
                hitmap = np.zeros((cmap.shape),dtype=bool)
                hitmap[ypix_i,xpix_i] = True
                flux_zbins[z_i] = np.sum(cmap[hitmap])/num_obj_zbins[i,z_i]
                median_z[i,z_i] = np.median(clean_cat['stellar_mass_z'][sm_mask])
                mean_z[i,z_i] = np.mean(clean_cat['stellar_mass_z'][sm_mask])
                galaxy_pixels = cmap[ypix_i,xpix_i]
                error[z_i] = t.bootstrap(n_realizations,len(ypix_i),galaxy_pixels)
                #hers_filtered_maps[i, z_i, x, :, :] = t.filtered_hitmaps(ypix_i, xpix_i, hers_filt[x][0].trans_func_grid, hers_filt[x][0].optfilt)

        hers_stack_flux[i,:, x] = flux_zbins
        hers_stack_flux_err[i,:, x] = error

    for y in range(3):
        cmap = sptsz_cmap[y]
        ypix = sptsz_ypix[y]
        xpix = sptsz_xpix[y]
        flux_zbins = np.zeros((len(z_bins[:-1])))
        error = np.zeros((len(z_bins[:-1])))
        filtered_maps = np.zeros((len(z_bins[:-1]),cmap.shape[0],cmap.shape[1]))
        for z_i in range(len(z_bins[:-1])):
            sm_mask = (clean_cat['stellar_mass_z'] > z_bins[z_i]) & (clean_cat['stellar_mass_z'] < z_bins[z_i+1]) \
                & (clean_cat['lmass'] > sm_bins[i]) & (clean_cat['lmass'] < sm_bins[i+1])
            ypix_i = ypix[sm_mask]
            xpix_i = xpix[sm_mask]
            if num_obj_zbins[i,z_i] != 0:
                hitmap = np.zeros((cmap.shape),dtype=bool)
                hitmap[ypix_i,xpix_i] = True
                flux_zbins[z_i] = np.sum(cmap[hitmap])/num_obj_zbins[i,z_i]
                galaxy_pixels = cmap[ypix_i,xpix_i]
                error[z_i] = t.bootstrap(n_realizations,len(ypix_i),galaxy_pixels)
                #spt_filtered_maps[i, z_i, y, :,:] = t.filtered_hitmaps(ypix_i, xpix_i, sptsz_filt[y][0].trans_func_grid, sptsz_filt[y][0].optfilt)
        spt_stack_flux[i,:, y] = flux_zbins
        spt_stack_flux_err[i,:, y] = error

    print "finished stacking SM bin:" +str(i)

##### 
spt_solid_angle = np.zeros((len(sptsz_cmap)))
hers_solid_angle = np.zeros((len(sptsz_cmap)))
for x in range(len(sptsz_cmap)):
    hers_tsz_K2Jy_conversion[x], hers_solid_angle[x] = t.calc_conversion_factor(hers_tsz_eff_freq[x], hers_reso_arcmin[x], hers_filt[x])
    spt_tsz_K2Jy_conversion[x], spt_solid_angle[x] = t.calc_conversion_factor(spt_tsz_eff_freq[x], sptsz_reso_arcmin[x], sptsz_filt[x])
solid_angles = np.concatenate([hers_solid_angle, spt_solid_angle[::-1]])
theor_tsz_Jy_sr = theor_tsz_Jy/solid_angles

eff_freq_zarr = np.zeros((len(sm_bins[:-1]),n_zbins, len(sptsz_mapfreq)+len(hers_mapfreq)))
stacked_flux_zarr= np.zeros((len(sm_bins[:-1]),n_zbins, len(sptsz_mapfreq)+len(hers_mapfreq)))
stacked_flux_err_zarr = np.zeros((len(sm_bins[:-1]),n_zbins, len(sptsz_mapfreq)+len(hers_mapfreq)))
eff_freq_gc_zarr = np.zeros((n_zbins, len(sptsz_mapfreq)+len(hers_mapfreq)))
gc_flux_zarr = np.zeros((n_zbins, len(sptsz_mapfreq)+len(hers_mapfreq)))
gc_flux_err_zarr = np.zeros((n_zbins, len(sptsz_mapfreq)+len(hers_mapfreq)))
fig, axarr1 = plt.subplots(len(sm_bins[:-1]),len(z_bins[:-1]), squeeze=True,sharex=True, sharey=True,  figsize=(3*len(sm_bins[:-1]),3*len(z_bins[:-1])))
T_fits = np.zeros((len(sm_bins[:-1]),len(z_bins[:-1])))
T_high = np.zeros((len(sm_bins[:-1]),len(z_bins[:-1])))
T_low = np.zeros((len(sm_bins[:-1]),len(z_bins[:-1])))
for i in range(len(sm_bins)-1):
    chi2_arr_all = []
#    axarr1[i,0].set_ylabel(str(sm_bins[i])+r'<$\mathregular{\log}(\mathregular{L}/\mathregular{L}_\odot$)<'+str(sm_bins[i+1]),fontsize=7)#
    fig.text(0.91, (float(i)+1)/float(len(sm_bins))*0.98+0.04, str(sm_bins[i])+'-'+str(sm_bins[i+1]),rotation='vertical')
    for z_i in range(len(z_bins)-1):
        if num_obj_zbins[i,z_i] < 10:
            axarr1[i, z_i ].axis('off')
        else:
            stacked_flux = np.concatenate([hers_stack_flux[i,z_i,:], spt_stack_flux[i,z_i,:][::-1] ])
            stacked_flux_err = np.concatenate([ hers_stack_flux_err[i,z_i,:], spt_stack_flux_err[i,z_i,:][::-1] ])
            mod_stacked_flux = stacked_flux/solid_angles
            mod_stacked_flux_err = stacked_flux_err/solid_angles
            z = mean_z[i, z_i]
            eff_freq_arr = np.zeros((len(hers_mapfreq)+len(sptsz_mapfreq), len(T_set)))
            chi2_arr = np.zeros((len(T_set)))
            A_arr = np.zeros((len(T_set)))
            y_arr = np.zeros((len(T_set)))
            for n, T in enumerate(T_set):
                hers_eff_freq = t.get_eff_band(herschel_lookup_table, T, z)*1.e9
                sptsz_eff_freq = t.get_eff_band(sptsz_lookup_table, T, z)*1.e9
                sptpol_eff_freq = t.get_eff_band(sptpol_lookup_table, T, z)*1.e9
                eff_freq = np.concatenate([hers_eff_freq, np.asarray([sptsz_eff_freq[-1]]), sptpol_eff_freq[::-1]])
                eff_freq_arr[:,n] = eff_freq
                popt, pcov = curve_fit(lambda p1, p2, p3: t.mod_BB_curve_with_tsz(p1, p2, p3, T, z, theor_tsz_Jy_sr), eff_freq, mod_stacked_flux, \
                               sigma = mod_stacked_flux_err, p0 = [1.e-10, 1.e-8], \
                               maxfev=1000000)
                A_arr[n], y_arr[n] = popt
                chi2_arr[n] = np.sum((t.mod_BB_curve_with_tsz(eff_freq, popt[0],popt[1],T,z, theor_tsz_Jy_sr)-mod_stacked_flux)**2/mod_stacked_flux_err**2/dof)
            T_ind = np.where(chi2_arr == np.min(chi2_arr))
            print str(np.min(chi2_arr))
            T = T_set[T_ind[0]][0]
            A = A_arr[T_ind[0]][0]
            y = y_arr[T_ind[0]][0]
            eff_freq = eff_freq_arr[:,T_ind[0]][:,0]
            eff_freq_zarr[i,z_i,:] = eff_freq
            stacked_flux_zarr[i,z_i,:] = mod_stacked_flux
            stacked_flux_err_zarr[i,z_i,:] = mod_stacked_flux_err
            chi2_arr_all.append(chi2_arr)
            T_fits[i,z_i] = T
            T_index = T_ind[0][0]
            T_low[i,z_i] = T_set[np.where((chi2_arr[:T_index]-chi2_arr[T_index]-1) ==np.min(chi2_arr[:T_index]-chi2_arr[T_index]-1))[0][0]]
            T_high[i,z_i] = T_set[np.where((chi2_arr[T_index+1:]-chi2_arr[T_index]-1) ==np.min(chi2_arr[T_index+1:]-chi2_arr[T_index]-1))[0][0]+T_index+1]
            ### BB SED plots
            x_var = np.arange(5.e10, 1.2e12, 1.e10)
            axarr1[i,z_i].errorbar(eff_freq/1.e9, mod_stacked_flux/1.e3, yerr=mod_stacked_flux_err/1.e3, color='black',marker = 'o',ls='none', \
                label='Stacked flux')
            x = (x_var/1.e9)*1.e9*const.h/(const.k*Tcmb)
            theor_tsz_Jy_full = t.calc_theor_tsz_T(x_var/1.e9)*1.e26*(2*const.k/(const.c)**2)*((const.k*Tcmb/const.h)**2)*(x**4)*np.exp(x)/(np.exp(x)-1)**2
            axarr1[i,z_i ].plot(x_var/1.e9, t.mod_BB_curve_with_tsz(x_var, A, y, T, z,  theor_tsz_Jy_full)/1.e3,\
                color='purple', label='Fit', ls='--')
            axarr1[i,z_i].text(100, 30, r'$\mathregular{N_{gal}}$='+str(int(num_obj_zbins[i,z_i]))+'\nT='+str(T)+'K', fontsize=14)
    #       axarr1[int(z_i/4), z_i % 4].set_xlim(100., 3500.)
            #axarr1[i,z_i].set_title('z='+str(z)+',Nobj='+str(num_obj_zbins[i,z_i]))
# axarr1[i, z_i].set_ylabel('Flux')
# #   plt.xscale('log')
# axarr1[(z_i % 4)].set_xlabel('Freq (Hz)')
for z_i in range(len(z_bins)-1):
    axarr1[0,z_i].set_title(str(z_bins[z_i])+'<z<'+str(z_bins[z_i+1]))
fig.text(0.5, 0.04, 'Frequency (GHz)', ha='center', va='center',fontsize=18)
fig.text(0.09, 0.5, 'Intensity (kJy/sr)', ha='center', va='center', rotation='vertical', fontsize=18)
fig.text(0.94, 0.5, r'$\mathregular{\log}(\mathregular{M}/\mathregular{M}_\odot$)', ha='center', va='center', rotation='vertical', fontsize=18)
#plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig('/data/tangq/blackbody_plots/'+today+'SM_redshift_table_BB_plots.png')

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10,5))
colors = plt.cm.rainbow(np.linspace(0,1,len(sm_bins)-1))
for i in range(len(sm_bins)-1):
    asym_error = np.array(list(zip(T_fits[i,:]-T_low[i,:], T_high[i,:]-T_fits[i,:]))).T
    ax1.errorbar((z_bins[:-1]+z_bins[1:])/2., T_fits[i,:], yerr = asym_error,label=r'$\mathregular{\log}(\mathregular{M}/\mathregular{M}_\odot$)='+str(sm_bins[i])+'-'+str(sm_bins[i+1]), \
        ls='none', marker='o',color=colors[i],markeredgewidth=0.0)
ax1.set_xlabel('Redshift')
ax1.legend(loc='lower right',fontsize=8,frameon=False,labelspacing=.3)
colors = plt.cm.rainbow(np.linspace(0,1,len(z_bins)-1))
for z_i in range(len(z_bins)-1):
    asym_error = np.array(list(zip(T_fits[:,z_i]-T_low[:,z_i], T_high[:,z_i]-T_fits[:,z_i]))).T
    ax2.errorbar((sm_bins[:-1]+sm_bins[1:])/2., T_fits[:,z_i], yerr = asym_error,label='z='+str(z_bins[z_i])+'-'+str(z_bins[z_i+1]), \
        ls='none', marker='o',color=colors[z_i],markeredgewidth=0.0)
ax2.set_xlabel(r'$\mathregular{\log} \mathregular{M}$ $(\mathregular{M}_\odot)$')
ax2.legend(loc='lower right',fontsize=8,frameon=False,labelspacing=.3)
ax1.set_ylabel('Rest-Frame Temperature from Fit (K)')
ax1.set_ylim(12, 25)
plt.savefig('/data/tangq/blackbody_plots/'+today+'_T_trend_with_SM_z.png')


fig.suptitle('BB curvefit to galaxy stacked flux at SM=' +str((sm_bins[i]+sm_bins[i+1])/2.))
plt.close()
    #--------- for 1D plots of chi2 over T for different z ---------##

fig, axarr1 = plt.subplots(1,len(z_bins[:-1]), squeeze=True,sharex=True, sharey=False, figsize=(5*len(z_bins[:-1]),5))
for z_i in range(len(z_bins)-1):
    axarr1[z_i % 4].plot(T_set, chi2_arr_all[z_i],label=r'$\mathregular{N_{gal}}$='+str(int(num_obj_zbins[i,z_i])))
    axarr1[z_i % 4].set_title(str(z_bins[z_i])+'< z <'+str(z_bins[z_i+1]))
    axarr1[z_i % 4].text(30, 0.9*np.max(chi2_arr_all[z_i]),r'$\mathregular{N_{gal}}$='+str(int(num_obj_zbins[i,z_i])), ha='center', va='center')
fig.text(0.5, 0.02, 'Blackbody Template Temperature (K)', ha='center', va='center',fontsize=14)
fig.text(0.09, 0.5, r'reduced $\chi^2$', ha='center', va='center', rotation='vertical', fontsize=18)
#fig.suptitle('reduced chi2 to BB curvefit to galaxy AVG stacked flux at SM=' +str((sm_bins[i]+sm_bins[i+1])/2.)+',binned in redshifts')
plt.savefig('/data/tangq/blackbody_plots/'+today+'gal_stacking_SM'+str(i)+'chi2_BBfits_1D.png')
plt.close()

    fig, axarr1 = plt.subplots(1,len(z_bins[:-1]), squeeze=True,sharex=True, sharey=False, figsize=(5*len(z_bins[:-1]),5))
    for z_i in range(len(z_bins)-1):
        axarr1[z_i % 4].plot(T_set, chi2_arr_gc_all[z_i], label='z'+"{:.2f}".format(mean_z[i,z_i]))
        axarr1[z_i % 4].set_title(str(z_bins[z_i])+'< z <'+str(z_bins[z_i+1])+',Nobj=' + str(num_obj_zbins[i, z_i]))
        axarr1[z_i % 4].set_xlabel('T')
        axarr1[z_i % 4].text(0.7, 0.8,r'$\mathregular{N_{gal}}$='+str(int(num_obj_zbins[i,z_i])), ha='center', va='center', transform=ax.transAxes)
    axarr1[0].set_ylabel('chi2')
    fig.suptitle('reduced chi2 to BB curvefit to galaxy hitmap fitted flux at SM=' +str((sm_bins[i]+sm_bins[i+1])/2.)+',binned in redshifts')
    plt.savefig('/data/tangq/blackbody_plots/'+today+'gal_hitmap_fit_SM'+str(i)+'chi2_BBfits_1D.png')
    plt.close()
    #np.save('/home/tangq/chi2_array_SM'+str(i)+'.npy', chi2_arr)
    print "SM Bin " + str(i)

## making quiescent gal plot
from mpl_toolkits.axes_grid1 import make_axes_locatable
axMain = plt.subplot(111)
i = 2
z_i = 0#    axarr1[i,0].set_ylabel(str(sm_bins[i])+r'<$\mathregular{\log}(\mathregular{L}/\mathregular{L}_\odot$)<'+str(sm_bins[i+1]),fontsize=7)#
stacked_flux = np.concatenate([hers_stack_flux[i,z_i,:], spt_stack_flux[i,z_i,:][::-1] ])
stacked_flux_err = np.concatenate([ hers_stack_flux_err[i,z_i,:], spt_stack_flux_err[i,z_i,:][::-1] ])
mod_stacked_flux = stacked_flux/solid_angles
mod_stacked_flux_err = stacked_flux_err/solid_angles
axMain.errorbar(eff_freq/1.e9, mod_stacked_flux/1.e3, yerr=mod_stacked_flux_err/1.e3, color='black',marker = 'o',ls='none', label='Stacked flux')
axMain.plot(x_var/1.e9, t.mod_BB_curve_with_tsz(x_var, A, y, T, z,  theor_tsz_Jy_full)/1.e3,\
color='purple', label='Fit', ls='-',lw=2)
axMain.plot((x_var+4)/1.e9, y*theor_tsz_Jy_full/1.e3, ls='--', color = 'blue',lw = 2, label='tSZ fit')
axMain.plot(x_var/1.e9, t.mod_BB_curve_with_z(x_var, A,T, z)/1.e3,ls='--', color='red',lw = 2,label='BB fit')
axMain.set_yscale('linear')
axMain.set_ylim((-1, 3))
axMain.spines['top'].set_visible(False)
axMain.xaxis.set_ticks_position('bottom')
axMain.set_xlabel('Frequency (GHz')
axMain.legend(loc='best')
divider = make_axes_locatable(axMain)
axLin = divider.append_axes("top", size=2.0, pad=0.02, sharex=axMain)
axLin.errorbar((eff_freq/1.e9)[0:3], (mod_stacked_flux/1.e3)[0:3], yerr=(mod_stacked_flux_err/1.e3)[:3], color='black',marker = 'o',ls='none')
axLin.plot((x_var/1.e9)[40:], (t.mod_BB_curve_with_tsz(x_var, A, y, T, z,  theor_tsz_Jy_full)/1.e3)[40:],\
color='purple', ls='-', lw=2)
axLin.plot(((x_var+4)/1.e9)[40:], y*theor_tsz_Jy_full[40:]/1.e3, ls='--', color = 'blue',lw = 2)
axLin.plot((x_var/1.e9)[40:], t.mod_BB_curve_with_z(x_var, A,T, z)[40:]/1.e3,ls='--', color='red',lw = 2)
axLin.set_yscale('log')
axLin.set_ylim((3, 80))
axLin.spines['bottom'].set_visible(False)
axLin.xaxis.set_ticks_position('top')
axLin.text(100, 15, r'$\mathregular{N_{gal}}$='+str(int(num_obj_zbins[i,z_i]))+'\nT='+str(T)+'K', fontsize=14)
plt.setp(axLin.get_xticklabels(), visible=False)
plt.text(-100, 1.6, 'Intensity (kJy/sr)', ha='center', va='center', rotation='vertical', fontsize=12)
plt.savefig('/data/tangq/blackbody_plots/'+today+'_quiescent_BB.png')


sptpol150_filtmap = np.load('/data/tangq/SPT/SPTpol/products/SPTpol_filtered_no_smooth_20210510_150_map.npy')
sptpol150 = idl.readIDLSav('/data/tangq/SPT/SPTpol/ra23h_cutout/sptpol_for_amy.sav').out[1]
plt.figure(figsize=(12,12))
plt.subplot(221)
plt.imshow(sptpol150*sptsz_pixelmask*1.e3, vmax=0.0002*1.e3, vmin = -0.0002*1.e3, extent=[343.75226,352.50226+8.75,-63.7539,-46.2539])
plt.xlabel('RA (deg)')
plt.ylabel('dec (deg)')
plt.colorbar()
plt.title('SPTpol 150 raw map')
plt.subplot(222)
plt.imshow(sptpol150_filtmap*1.e3, vmax=0.00002*1.e3, vmin = -0.00002*1.e3,extent=[343.75226,352.50226+8.75,-63.7539,-46.2539])
plt.colorbar()
plt.xlabel('RA (deg)')
plt.ylabel('dec (deg)')
plt.title('SPTpol 150 filtered map')
plt.subplot(223)
ypix1 = 2331
xpix1 = 1742
ra1 = 343.75226+0.25*(1./60)*ypix1
dec1 = -46.2539-0.25*(1./60)*xpix1
plt.imshow(sptpol150[ypix1-50:ypix1+50, xpix1-50:xpix1+50]*1.e3,extent=[-50*0.25, 50*0.25,-50*0.25, 50*0.25])
plt.colorbar()
plt.title('SPTpol 150 raw map around a SMG')
plt.xlabel(r'$\Delta$ RA (arcmin)')
plt.ylabel(r'$\Delta$ dec (arcmin)')
plt.subplot(224)
plt.imshow(sptpol150_filtmap[ypix1-50:ypix1+50, xpix1-50:xpix1+50]*1.e3,extent=[-50*0.25, 50*0.25,-50*0.25, 50*0.25])
plt.colorbar()
plt.title('SPTpol 150 filtered map around a SMG')
plt.xlabel(r'$\Delta$ RA (arcmin)')
plt.ylabel(r'$\Delta$ dec (arcmin)')
plt.savefig('/data/tangq/SPTpol150_filtmap.png')

mask = sptsz_pixelmask <0.6
sptpol150[mask] = np.nan
sptpol150_filtmap[mask] = np.nan
hdu = fits.PrimaryHDU(sptpol150)
hdu.writeto('/data/tangq/SPT/SPTpol/ra23h_cutout/sptpol_for_amy1.fits')
sptpol150_filtmap = np.load('/data/tangq/SPT/SPTpol/products/SPTpol_filtered_no_smooth_20210510_150_map.npy')
hdu = fits.PrimaryHDU(sptpol150_filtmap)
hdu.writeto('/data/tangq/SPT/SPTpol/products/SPTpol_filtered_no_smooth_20210510_150_map1.fits')


for z_i in range(len(z_bins[:-1])):
    np.savetxt('/data/tangq/blackbody_plots/'+today+'_z'+str(z_i)+'_stacked_flux.txt', stacked_flux_zarr[:,z_i,:]/1.e3)
    np.savetxt('/data/tangq/blackbody_plots/'+today+'_z'+str(z_i)+'_stacked_flux_err.txt', stacked_flux_err_zarr[:,z_i,:]/1.e3)
    np.savetxt('/data/tangq/blackbody_plots/'+today+'_z'+str(z_i)+'_eff_freq.txt', eff_freq_zarr[:,z_i,:]/1.e3)


