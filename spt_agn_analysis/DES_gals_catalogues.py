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
from sptpol_software.util import fits
import sptpol_software.analysis.maps_amy as maps
import pickle
from astropy.io import fits
import sptpol_software.scratch.tnatoli.transient.calc_optfilt_oneband_2d_amy as calc
import os
from astropy.table import Table, Column

###########Getting COSMOS catalog
COSMOS_cat = fits.open('/data/tangq/COSMOS/COSMOS2015_Laigle+_v1.1.fits')
#masking out B, V, Ks, i+, photo-z, and COSMOS area flags
COSMOS_mask = ((COSMOS_cat[1].data['FLAG_COSMOS'] == 1) & (COSMOS_cat[1].data['B_FLAGS']==0) & (COSMOS_cat[1].data['PHOTOZ'] > 0) & (COSMOS_cat[1].data['V_FLAGS']==0) &
	(COSMOS_cat[1].data['ip_FLAGS']==0) & (COSMOS_cat[1].data['Ks_FLAGS']==0) & (COSMOS_cat[1].data['PHOTOZ'] < 9) & (COSMOS_cat[1].data['Ks_FLAGS']==0))

COSMOS_ra = COSMOS_cat[1].data['ALPHA_J2000'][COSMOS_mask] #in deg
COSMOS_dec = COSMOS_cat[1].data['DELTA_J2000'][COSMOS_mask] #in deg
#stellar mass, flags
COSMOS_sm = COSMOS_cat[1].data['MASS_MED'][COSMOS_mask] #log of stellar mass , based from PDF
COSMOS_smlow = COSMOS_cat[1].data['MASS_MED_MIN68'][COSMOS_mask] #lower limit of mass
COSMOS_smhigh = COSMOS_cat[1].data['MASS_MED_MAX68'][COSMOS_mask] #upper limit of mass
COSMOS_sm1 = COSMOS_cat[1].data['MASS_BEST'][COSMOS_mask] #log of stellar mass , based from 
COSMOS_z = COSMOS_cat[1].data['PHOTOZ'][COSMOS_mask]
COSMOS_zfit = COSMOS_cat[1].data['ZPDF'][COSMOS_mask] # photo-zs using galaxy templates
del COSMOS_cat, COSMOS_mask

#SPITZER data
spitzer_file = '/data/tangq/spitzer/irsa_catalog_search_results.tbl'
sp_ch11_norm = 0.610 #3.6um band in 1.4" aperture from https://irsa.ipac.caltech.edu/data/COSMOS/gator_docs/scosmos_irac_colDescriptions.html
sp_ch21_norm =  0.590 #4.5um band in 1.4" aperture, see above

f = open(spitzer_file, 'r')
lines = f.readlines()
spitzer_indices = lines[116]
spitzer_units = lines[118]
ra, dec, flag, flux_c11, flux_c21, flux_c11err, flux_c21err = [],[],[],[],[], [], []
for line in lines[120:]:
    line = line.split()
    ra.append(float(line[6]))
    dec.append(float(line[7]))
    flag.append(int(line[14]))
    flux_c11.append(float(line[15]))
    flux_c11err.append(float(line[16]))
    flux_c21.append(float(line[24]))
    flux_c21err.append(float(line[25]))
f.close()
flux_c11 = np.asarray(flux_c11)/sp_ch11_norm
flux_c21 = np.asarray(flux_c21)/sp_ch21_norm
flux_c11err = np.asarray(flux_c11err)/sp_ch11_norm
flux_c21err = np.asarray(flux_c21err)/sp_ch21_norm
mag_11 = -2.5*np.log10(flux_c11)+23.9
mag_21 = -2.5*np.log10(flux_c21)+23.9
magerr_11 = -2.5*np.log10(flux_c11err)+23.9
magerr_21 = -2.5*np.log10(flux_c21err)+23.9
spitzer_tab = Table([ra, dec, flag, mag_11,  mag_21, magerr_11, magerr_21], names=('ra', 'dec', 'flag', 'mag_11', 'mag_21','magerr_11', 'magerr_21'))
spitzer_cat = np.hstack([[ra, dec, flag, mag_11, mag_21, magerr_11, magerr_21]])
spitzer_cat = spitzer_cat.astype(float)

#COSMOS + SPITZER + DES
c = SkyCoord(ra=COSMOS_ra*u.degree, dec=COSMOS_dec*u.degree)
catalog = SkyCoord(ra=spitzer_tab['ra']*u.degree, dec=spitzer_tab['dec']*u.degree)  
idx, d2d, d3d = c.match_to_catalog_sky(catalog)
dist_cut = np.where(d2d.arcminute < (2./60.))[0]
spitzer_cosmo_cat = np.hstack([[COSMOS_ra[dist_cut], COSMOS_dec[dist_cut]]])
c = SkyCoord(ra = spitzer_cosmo_cat[0]*u.degree, dec = spitzer_cosmo_cat[1]*u.degree)
catalog = SkyCoord(ra=spitzer_tab['ra']*u.degree, dec=spitzer_tab['dec']*u.degree)  
idx, d2, d3 = coord.match_coordinates_sky(c, catalog)
sp_cosmos_tab =Table([COSMOS_ra[dist_cut], COSMOS_dec[dist_cut], spitzer_tab['flag'][idx], spitzer_tab['mag_11'][idx], 
	spitzer_tab['mag_21'][idx], spitzer_tab['magerr_11'][idx], spitzer_tab['magerr_21'][idx], COSMOS_z[dist_cut], 
	COSMOS_sm[dist_cut]],  names=('ra', 'dec', 'sp_flag','sp_mag1', 'sp_mag2', 'sp_magerr1','sp_magerr2', 'z', 'stellar_mass'))
# sp_cosmos_tab.write('spitzer_cosmos_cat.fits', format='fits')

hdulist = fits.open('/data/bleeml/for_amy/sva1_gold_r1.0_catalog.fits')
data = hdulist[1]
des_y1_ra = data.data['RA']
des_y1_dec = data.data['DEC']
des_cosmo_mask = np.where((des_y1_ra < 151.5) & (des_y1_ra > 149.) & (des_y1_dec >1.1) & (des_y1_dec <3.3))[0]
des_y1_ra = data.data['RA'][des_cosmo_mask]
des_y1_dec = data.data['DEC'][des_cosmo_mask]
des_y1_coadd = data.data['COADD_OBJECTS_ID'][des_cosmo_mask]
des_y1_flagg = data.data['FLAGS_G'][des_cosmo_mask]
des_y1_flagr = data.data['FLAGS_R'][des_cosmo_mask]
des_y1_flagi = data.data['FLAGS_I'][des_cosmo_mask]
des_y1_flagz = data.data['FLAGS_Z'][des_cosmo_mask]
des_y1_magg = data.data['MAG_AUTO_G'][des_cosmo_mask]
des_y1_magr = data.data['MAG_AUTO_R'][des_cosmo_mask]
des_y1_magi = data.data['MAG_AUTO_I'][des_cosmo_mask]
des_y1_magz = data.data['MAG_AUTO_Z'][des_cosmo_mask]
des_y1_maggerr = data.data['MAGERR_AUTO_G'][des_cosmo_mask]
des_y1_magrerr = data.data['MAGERR_AUTO_R'][des_cosmo_mask]
des_y1_magierr = data.data['MAGERR_AUTO_I'][des_cosmo_mask]
des_y1_magzerr = data.data['MAGERR_AUTO_Z'][des_cosmo_mask]
del data
del hdulist

c = SkyCoord(ra=sp_cosmos_tab['ra']*u.degree, dec = sp_cosmos_tab['dec']*u.degree)
catalog = SkyCoord(ra=des_y1_ra*u.degree, dec=des_y1_dec*u.degree)  
idx, d2d, d3d = c.match_to_catalog_sky(catalog)
dist_cut = np.where(d2d.arcminute < (2./60.))[0]
spitzer_cosmo_des_cat = np.hstack([[sp_cosmos_tab['ra'][dist_cut], sp_cosmos_tab['dec'][dist_cut]]])
c = SkyCoord(ra = spitzer_cosmo_des_cat[0]*u.degree, dec = spitzer_cosmo_des_cat[1]*u.degree)
catalog = SkyCoord(ra=des_y1_ra*u.degree, dec=des_y1_dec*u.degree)  
idx, d2, d3 = coord.match_coordinates_sky(c, catalog)

sp_cosmos_des_tab =Table([sp_cosmos_tab['ra'][dist_cut], sp_cosmos_tab['dec'][dist_cut], sp_cosmos_tab['sp_flag'][dist_cut], sp_cosmos_tab['sp_mag1'][dist_cut],
	sp_cosmos_tab['sp_mag2'][dist_cut], sp_cosmos_tab['sp_magerr1'][dist_cut], sp_cosmos_tab['sp_magerr2'][dist_cut],sp_cosmos_tab['z'][dist_cut], sp_cosmos_tab['stellar_mass'][dist_cut], 
	des_y1_magg[idx], des_y1_magr[idx], des_y1_magi[idx], des_y1_magz[idx], des_y1_maggerr[idx], des_y1_magrerr[idx], des_y1_magierr[idx], des_y1_magzerr[idx],
	des_y1_flagg[idx], des_y1_flagr[idx], des_y1_flagi[idx], des_y1_flagz[idx], des_y1_coadd[idx]],
	names = ('ra', 'dec', 'sp_flag','sp_mag1', 'sp_mag2', 'sp_mag1err', 'sp_mag2err', 'cosmos_z', 'cosmos_stellar_mass', 'des_mag_g', 'des_mag_r', 'des_mag_i', 'des_mag_z',
		'des_magerr_g', 'des_magerr_r', 'des_magerr_i','des_magerr_z', 'des_flag_g', 'des_flag_r', 'des_flag_i', 'des_flag_z', 'des_coadd_id'))
#sp_cosmos_des_tab.write('/data/tangq/custom_catalogs/spitzer_cosmos_des_cat_mag.fits', format='fits')

#adding in the photoz:
#file_y1_photoz = '/data/tangq/DES_data/mcal-y1a1-combined-griz-blind-v3-matched_BPZbase.fits'
coadd_max = max(sp_cosmos_des_tab['des_coadd_id'])
coadd_min = min(sp_cosmos_des_tab['des_coadd_id'])
file_y1_photoz = '/data/tangq/DES_data/sva1_gold_r1.0_bpz_pdf.fits'
cat_y1_photoz = fits.open(file_y1_photoz)
des_y1_coadd_id_pz = cat_y1_photoz[1].data['coadd_objects_id']
mask = ((des_y1_coadd_id_pz <= coadd_max) & (des_y1_coadd_id_pz >= coadd_min))
des_y1_coadd_id_pz = cat_y1_photoz[1].data['coadd_objects_id'][mask]
des_y1_zmean_pz = cat_y1_photoz[1].data['z_mean'][mask]
des_y1_zpeak_pz = cat_y1_photoz[1].data['z_peak'][mask]
del cat_y1_photoz

coadd_indices = []
for j in range(len(sp_cosmos_des_tab['des_coadd_id'])):
	coadd_index = np.where((des_y1_coadd_id_pz == sp_cosmos_des_tab['des_coadd_id'][j]))[0]
	coadd_indices.append(coadd_index)
des_y1_coadd_id_pz_good = des_y1_coadd_id_pz[coadd_indices][:,0]
des_y1_z_mean_pz_good = des_y1_zmean_pz[coadd_indices][:,0]
des_y1_z_peak_pz_good = des_y1_zpeak_pz[coadd_indices][:,0]

col_zmean = Column(name = 'des_z_mean', data = des_y1_z_mean_pz_good)
col_zpeak = Column(name = 'des_z_peak', data = des_y1_z_peak_pz_good)
sp_cosmos_des_tab.add_columns([col_zmean, col_zpeak])

write_file_dir = '/data/tangq/custom_catalogs/spitzer_cosmos_des_cat_mag1.fits'
if os.path.exists(write_file_dir):
  os.remove(write_file_dir)
sp_cosmos_des_tab.write(write_file_dir, format='fits')








########################## SPITZER + DES on 100d FIELD ###########

sp_ch11_norm = 0.610 #3.6um band in 1.4" aperture from https://irsa.ipac.caltech.edu/data/COSMOS/gator_docs/scosmos_irac_colDescriptions.html
sp_ch21_norm =  0.590 #4.5um band in 1.4" aperture, see above


spitzer_des_file = '/data/tangq/spitzer/spitzer_spt100deg_ch12noflag.tbl'
f = open(spitzer_des_file, 'r')
lines = f.readlines()
sp_100deg_indices = lines[307]
sp_100deg_units = lines[309]
ra, dec, flux_c11, flux_c21, flux_c11err, flux_c21err, flux_c1_flag, flux_c2_flag = [],[],[],[],[],[],[],[]
for line in lines[311:]:
    line = line.split()
    ra.append(float(line[4]))
    dec.append(float(line[5]))
    flux_c1_flag.append(int(line[16]))
    flux_c2_flag.append(int(line[17]))
    flux_c11.append(float(line[25]))
    flux_c11err.append(float(line[26]))
    flux_c21.append(float(line[35]))
    flux_c21err.append(float(line[36]))
f.close()
flux_c11 = np.asarray(flux_c11)/sp_ch11_norm
flux_c21 = np.asarray(flux_c21)/sp_ch21_norm
flux_c11err = np.asarray(flux_c11err)/sp_ch11_norm
flux_c21err = np.asarray(flux_c21err)/sp_ch21_norm
mag_11 = -2.5*np.log10(flux_c11)+23.9
mag_21 = -2.5*np.log10(flux_c21)+23.9
magerr_11 = -2.5*np.log10(flux_c11err)+23.9
magerr_21 = -2.5*np.log10(flux_c21err)+23.9
sp_100deg_tab = Table([ra, dec, flux_c1_flag, flux_c2_flag, mag_11,  mag_21, magerr_11, magerr_21, flux_c11, flux_c21, flux_c11err, flux_c21err], 
	names=('ra', 'dec', 'flux_c1_flag','flux_c2_flag', 'mag_11', 'mag_21','magerr_11', 'magerr_21', 'flux_11', 'flux_21', 'flux11_err', 'flux21_err'))
sp_100deg_cat = np.hstack([[ra, dec, mag_11,  mag_21, magerr_11, magerr_21, flux_c11, flux_c21, flux_c11err, flux_c21err]])
sp_100deg_cat = sp_100deg_cat.astype(float)

file_photoz = '/data/tangq/DES_data/spt100d_desgold_10092019.fits'
cat_photoz = fits.open(file_photoz)
med_z_pz = cat_photoz[1].data['zmean_bpz']
mc_z_pz = cat_photoz[1].data['zmc_bpz']
medz_dnf_pz = cat_photoz[1].data['zmean_dnf']
mcz_dnf_pz = cat_photoz[1].data['zmc_dnf']
ra_pz = cat_photoz[1].data['ra']
dec_pz = cat_photoz[1].data['dec']
flags_pz = cat_photoz[1].data['flags_gold']
g_pz = cat_photoz[1].data['sof_cm_mag_corrected_g']
r_pz = cat_photoz[1].data['sof_cm_mag_corrected_r']
i_pz = cat_photoz[1].data['sof_cm_mag_corrected_i']
z_pz = cat_photoz[1].data['sof_cm_mag_corrected_z']
gerr_pz = cat_photoz[1].data['sof_cm_mag_err_g']
rerr_pz = cat_photoz[1].data['sof_cm_mag_err_r']
ierr_pz = cat_photoz[1].data['sof_cm_mag_err_i']
zerr_pz = cat_photoz[1].data['sof_cm_mag_err_z']

#DES + SPIZER in 100deg field
c = SkyCoord(ra=ra_pz*u.degree, dec=dec_pz*u.degree)
catalog = SkyCoord(ra=sp_100deg_tab['ra']*u.degree, dec=sp_100deg_tab['dec']*u.degree)  
idx, d2d, d3d = c.match_to_catalog_sky(catalog)
dist_cut = np.where(d2d.arcminute < (2./60.))[0]
spitzer_des_cat = np.hstack([[ra_pz[dist_cut], dec_pz[dist_cut], med_z_pz[dist_cut]]])
c = SkyCoord(ra = spitzer_des_cat[0]*u.degree, dec = spitzer_des_cat[1]*u.degree)
catalog = SkyCoord(ra=sp_100deg_tab['ra']*u.degree, dec=sp_100deg_tab['dec']*u.degree)  
idx, d2, d3 = coord.match_coordinates_sky(c, catalog)

sp_des_tab = Table([ra_pz[dist_cut], dec_pz[dist_cut], 
	sp_100deg_tab['mag_11'][idx], sp_100deg_tab['mag_21'][idx], sp_100deg_tab['magerr_11'][idx],sp_100deg_tab['magerr_21'][idx], 
	sp_100deg_tab['flux_11'][idx],sp_100deg_tab['flux_21'][idx], sp_100deg_tab['flux11_err'][idx], sp_100deg_tab['flux21_err'][idx],
	sp_100deg_tab['flux_c1_flag'][idx], sp_100deg_tab['flux_c2_flag'][idx],
	med_z_pz[dist_cut], mc_z_pz[dist_cut],g_pz[dist_cut], r_pz[dist_cut], i_pz[dist_cut], z_pz[dist_cut], 
	gerr_pz[dist_cut], rerr_pz[dist_cut],ierr_pz[dist_cut], zerr_pz[dist_cut], medz_dnf_pz[dist_cut], mcz_dnf_pz[dist_cut], flags_pz[dist_cut]],  
	names=('ra', 'dec', 'sp_mag1', 'sp_mag2', 'sp_mag1err', 'sp_mag2err', 'sp_flux1', 'sp_flux2', 'sp_flux1err', 'sp_flux2err', 'sp_1_flag', 'sp_2_flag',
	'des_z_mean', 'des_z_mc', 'g','r','i','z', 'gerr', 'rerr', 'ierr', 'zerr', 'des_z_mean_dnf', 'des_z_mc_dnf', 'des_flags'))
write_file_dir = '/data/tangq/custom_catalogs/spitzer_des_cat.fits'
if os.path.exists(write_file_dir):
  os.remove(write_file_dir)
sp_des_tab.write(write_file_dir, format='fits')


#stellar mass catalog
sm_file = '/data/bleeml/stellar_mass_cats/ssdf_des_try1.fout'
sm_catalog = np.loadtxt(sm_file)




