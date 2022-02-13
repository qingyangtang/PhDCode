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

import scipy.constants as const
from scipy.optimize import curve_fit
import tool_func as t
import ConfigParser

SPT_SMGs = np.genfromtxt('/home/tangq/data/SPT_SMG_srcs.txt',dtype ='str')
from astropy.coordinates import Angle
SPT_SMGs[:,4][-10] = '-53:58:39.8'
SPT_SMGs[:,4][42] = '-50:30:52.5'
SPT_SMGs[:,4][45] = '-48:25:01.8'
ras = Angle(SPT_SMGs[:,3],unit=u.hour).degree
decs = Angle(SPT_SMGs[:,4],unit=u.degree).degree

herschel_config='/home/tangq/scripts/herschel_config.txt'
configParser = ConfigParser.RawConfigParser()   
configFilePath = herschel_config
configParser.read(configFilePath)
herschel_mapfreqs = ["250", "350", "500"]
map250 = idl.readIDLSav(configParser.get('map_and_ptsrc_list', 'map250'))
map350 = idl.readIDLSav(configParser.get('map_and_ptsrc_list', 'map350'))
map500 = idl.readIDLSav(configParser.get('map_and_ptsrc_list', 'map500'))
herschel_maps = [map250.values()[2], map350.values()[2],map500.values()[2]]
spt_config='/home/tangq/scripts/spt_config.txt'
configParser = ConfigParser.RawConfigParser()   
configFilePath = spt_config
configParser.read(configFilePath)
spt_mapfreqs = np.asarray(["90","150","220"])
spt_maps = idl.readIDLSav(configParser.get('map_and_files', 'map')).maps

herschel_no_smooth_fname='/data/tangq/SPIRE_maps/products/Herschel_filtered_no_smooth_20201130'
map_250_ns = np.load(herschel_no_smooth_fname+'_250_map.npy')
map_350_ns = np.load(herschel_no_smooth_fname+'_350_map.npy')
map_500_ns = np.load(herschel_no_smooth_fname+'_500_map.npy')
herschel_ns_maps = np.asarray([map_250_ns, map_350_ns, map_500_ns])
herschel_ns_radec0 = np.loadtxt(herschel_no_smooth_fname+'_radec.txt')
herschel_ns_reso_arcmin = np.loadtxt(herschel_no_smooth_fname+'_reso_arcmin.txt')
herschel_ns_mask = np.load(herschel_no_smooth_fname+'_zp_mask.npy')

spt_no_smooth_fname = '/data/tangq/SPT/SZ/products/SPTSZ_filtered_no_smooth_FTfixed_20210401'
map_90_ns = np.load(spt_no_smooth_fname+'_90_map.npy')
map_150_ns = np.load(spt_no_smooth_fname+'_150_map.npy')
map_220_ns = np.load(spt_no_smooth_fname+'_220_map.npy')
spt_ns_maps = np.asarray([map_90_ns, map_150_ns, map_220_ns])
spt_ns_radec0 = np.loadtxt(spt_no_smooth_fname+'_radec.txt')
spt_ns_reso_arcmin = np.loadtxt(spt_no_smooth_fname+'_reso_arcmin.txt')
spt_ns_mask = np.load(spt_no_smooth_fname+'_zp_mask.npy')

sptpol_no_smooth_fname = '/data/tangq/SPT/SPTpol/products/SPTpol_filtered_no_smooth_20210510'
map_90_ns_pol = np.load(sptpol_no_smooth_fname+'_90_map.npy')
map_150_ns_pol = np.load(sptpol_no_smooth_fname+'_150_map.npy')
sptpol_ns_maps = np.asarray([map_90_ns_pol, map_150_ns_pol])
sptpol_ns_radec0 = np.loadtxt(sptpol_no_smooth_fname+'_radec.txt')
sptpol_ns_reso_arcmin = np.loadtxt(sptpol_no_smooth_fname+'_reso_arcmin.txt')
sptpol_ns_mask = np.load(sptpol_no_smooth_fname+'_zp_mask.npy')

# spt_no_smooth_lind_fname = '/data/tangq/SPT/SZ/products/SPTSZ_filtered_no_smooth_lindsey_20210301'
# map_90_ns_lind = np.load(spt_no_smooth_lind_fname+'_90_map.npy')
# map_150_ns_lind = np.load(spt_no_smooth_lind_fname+'_150_map.npy')
# map_220_ns_lind = np.load(spt_no_smooth_lind_fname+'_220_map.npy')
# spt_ns_maps_lind = np.asarray([map_90_ns_lind, map_150_ns_lind, map_220_ns_lind])
# spt_ns_radec0_lind = np.loadtxt(spt_no_smooth_lind_fname+'_radec.txt')
# spt_ns_reso_arcmin_lind = np.loadtxt(spt_no_smooth_lind_fname+'_reso_arcmin.txt')
# spt_ns_mask_lind = np.load(spt_no_smooth_lind_fname+'_zp_mask.npy')

spt_90_lindsey = idl.readIDLSav('/data/tangq/debug/ra23h30dec-55_2year_90_check.sav')
spt_150_lindsey = idl.readIDLSav('/data/tangq/debug/ra23h30dec-55_2year_150_check.sav')
spt_220_lindsey = idl.readIDLSav('/data/tangq/debug/ra23h30dec-55_2year_220_check.sav')
map_90_lindsey = spt_90_lindsey.filtered_map
map_150_lindsey = spt_150_lindsey.filtered_map
map_220_lindsey = spt_220_lindsey.filtered_map
spt_lindsey_maps = np.asarray([map_90_lindsey, map_150_lindsey, map_220_lindsey])
spt_lindsey_radec0 = np.asarray([spt_90_lindsey.radec0, spt_150_lindsey.radec0, spt_220_lindsey.radec0])
spt_lindsey_reso_arcmin = np.asarray([spt_90_lindsey.reso_arcmin, spt_150_lindsey.reso_arcmin, spt_220_lindsey.reso_arcmin])

spt_3360_no_smooth_fname = '/data/tangq/SPT/SZ/products/SPTSZ_filtered_no_smooth_lindsey_20210323'
map_90_ns_3360 = np.load(spt_3360_no_smooth_fname+'_90_map.npy')
map_150_ns_3360 = np.load(spt_3360_no_smooth_fname+'_150_map.npy')
map_220_ns_3360 = np.load(spt_3360_no_smooth_fname+'_220_map.npy')
spt_ns_maps_3360 = np.asarray([map_90_ns_3360, map_150_ns_3360, map_220_ns_3360])
spt_ns_radec0_3360 = np.loadtxt(spt_3360_no_smooth_fname+'_radec.txt')
spt_ns_reso_arcmin_3360 = np.loadtxt(spt_3360_no_smooth_fname+'_reso_arcmin.txt')
spt_ns_mask_3360 = np.load(spt_3360_no_smooth_fname+'_zp_mask.npy')

herschel_ypix, herschel_xpix = sky.ang2Pix(np.asarray([ras, decs]), herschel_ns_radec0[0], herschel_ns_reso_arcmin[0], map_250_ns.shape,proj=0)[0]
id_mask = np.arange(len(ras))
mask = (herschel_ypix > 0) & (herschel_ypix < map_250_ns.shape[0]) & (herschel_xpix > 0) & (herschel_xpix < map_250_ns.shape[1])
id_mask = id_mask[mask]
herschel_ypix, herschel_xpix = sky.ang2Pix(np.asarray([ras[mask], decs[mask]]), herschel_ns_radec0[0], herschel_ns_reso_arcmin[0], map_250_ns.shape,proj=0)[0]
ypixmask, xpixmask = sky.ang2Pix(np.asarray([ras[mask], decs[mask]]), herschel_ns_radec0[0], herschel_ns_reso_arcmin[0], map_250_ns.shape,proj=0)[1] 
ang2pix_mask = (ypixmask == True) & (xpixmask == True) & (herschel_ns_mask[herschel_ypix,herschel_xpix]>0.99999999)
id_mask = id_mask[ang2pix_mask]
herschel_ypix = herschel_ypix[ang2pix_mask]
herschel_xpix = herschel_xpix[ang2pix_mask]
ra1 = ras[id_mask]
dec1 = decs[id_mask]

ra1 = ra1[1:]
dec1 = dec1[1:]
id_mask = id_mask[1:]
herschel_ypix = herschel_ypix[1:]
herschel_xpix = herschel_xpix[1:]

spt_ypix, spt_xpix = sky.ang2Pix(np.asarray([ra1, dec1]), spt_ns_radec0[0], spt_ns_reso_arcmin[0], map_90_ns.shape,proj=0)[0]
ypixmask, xpixmask = sky.ang2Pix(np.asarray([ra1, dec1]), spt_ns_radec0[0], spt_ns_reso_arcmin[0], map_90_ns.shape,proj=0)[1] 
ang2pix_mask = (ypixmask == True) & (xpixmask == True) & (spt_ns_mask[spt_ypix,spt_xpix]>0.99999999)
spt_ypix= spt_ypix[ang2pix_mask]
spt_xpix = spt_xpix[ang2pix_mask]
herschel_ypix = herschel_ypix[ang2pix_mask]
herschel_xpix = herschel_xpix[ang2pix_mask]
id_mask = id_mask[ang2pix_mask]
SPT_SMG_flux = np.genfromtxt('/home/tangq/data/SPT_SMG_srcs_flux.txt',dtype='str')
new_mask = np.zeros((len(SPT_SMGs[:,0][id_mask])),dtype=bool)
indices = []
for x in range(len(SPT_SMGs[:,0][id_mask])):
        new_mask[x] = (SPT_SMGs[:,0][id_mask][x] in SPT_SMG_flux[:,0])
        if new_mask[x] == True:
                indices.append(np.where(SPT_SMG_flux[:,0] == SPT_SMGs[:,0][id_mask][x])[0][0])
id_mask = id_mask[new_mask]
herschel_ypix = herschel_ypix[new_mask]
herschel_xpix = herschel_xpix[new_mask]
spt_ypix = spt_ypix[new_mask]
spt_xpix = spt_xpix[new_mask]
ra1 = ra1[new_mask]
dec1 = dec1[new_mask]



spt_ypix_lindsey, spt_xpix_lindsey = sky.ang2Pix(np.asarray([ra1, dec1]), spt_lindsey_radec0[0], spt_lindsey_reso_arcmin[0], spt_lindsey_maps[0].shape,proj=0)[0]

flux500 = SPT_SMG_flux[:,10][indices]
err500 = SPT_SMG_flux[:,11][indices]
flux350 = SPT_SMG_flux[:,12][indices]
err350 = SPT_SMG_flux[:,13][indices]
flux250 = SPT_SMG_flux[:,14][indices]
err250 = SPT_SMG_flux[:,15][indices]
herschel_flux = np.asarray([ flux250, flux350, flux500])

from datetime import datetime
today = datetime.today().strftime('%Y%m%d')

herschel_sig = np.zeros((len(indices), 3, 4))
herschel_err = np.zeros((len(indices), 3, 4))

#herschel_solan = np.asarray([1.25e-8, 1.86e-8, 3.75e-8])
herschel_solan = np.asarray([2.22e-8 ,          2.85e-8 , 4.62e-8 ])
herschel_100_solan = (1.08e-7/herschel_solan)
herschel_175_solan = (3.33e-7/herschel_solan)

# for i in range(len(indices)):
#         plt.figure(figsize=(20,15))
#         for j in range(3):
#                 plt.subplot(4,3,j+1)
#                 plt.imshow(herschel_maps[j][herschel_ypix[i]-20:herschel_ypix[i]+20,herschel_xpix[i]-20:herschel_xpix[i]+20]*1000)
#                 plt.colorbar()
#                 herschel_sig[i,j,0] = herschel_maps[j][herschel_ypix[i],herschel_xpix[i]]*1000
#                 rand_ypix = np.random.choice(np.arange(40),100)
#                 rand_xpix = np.random.choice(np.arange(40),100)
#                 herschel_err[i,j,0] = np.std((herschel_maps[j][herschel_ypix[i]-20:herschel_ypix[i]+20,herschel_xpix[i]-20:herschel_xpix[i]+20]*1000)[rand_ypix, rand_xpix])
#                 plt.title(herschel_mapfreqs[j] + 'um raw, %.1f' %(herschel_sig[i,j,0])+'+/- %.1f' %(herschel_err[i,j,0]))
#                 plt.subplot(4,3,3+j+1)
#                 plt.imshow(herschel_ns_maps[j][herschel_ypix[i]-20:herschel_ypix[i]+20,herschel_xpix[i]-20:herschel_xpix[i]+20]*1000)
#                 herschel_sig[i,j,1] = herschel_ns_maps[j][herschel_ypix[i],herschel_xpix[i]]*1000
#                 rand_ypix = np.random.choice(np.arange(40),100)
#                 rand_xpix = np.random.choice(np.arange(40),100)
#                 herschel_err[i,j,1] = np.std((herschel_ns_maps[j][herschel_ypix[i]-20:herschel_ypix[i]+20,herschel_xpix[i]-20:herschel_xpix[i]+20]*1000)[rand_ypix, rand_xpix])
#                 plt.colorbar()
#                 plt.title(herschel_mapfreqs[j] + 'um filt, %.1f' %(herschel_sig[i,j,1])+'+/-%.1f' %(herschel_err[i,j,1]))
#                 plt.subplot(4,3,6+j+1)
#                 plt.imshow(herschel_100_maps[j][herschel_ypix[i]-20:herschel_ypix[i]+20,herschel_xpix[i]-20:herschel_xpix[i]+20]*1000*herschel_100_solan[j])
#                 plt.colorbar()
#                 herschel_sig[i,j,2] = herschel_100_maps[j][herschel_ypix[i],herschel_xpix[i]]*1000*herschel_100_solan[j]
#                 rand_ypix = np.random.choice(np.arange(40),100)
#                 rand_xpix = np.random.choice(np.arange(40),100)
#                 herschel_err[i,j,2] = np.std((herschel_100_maps[j][herschel_ypix[i]-20:herschel_ypix[i]+20,herschel_xpix[i]-20:herschel_xpix[i]+20]*1000*herschel_100_solan[j])[rand_ypix, rand_xpix])
#                 plt.title(herschel_mapfreqs[j] + 'um 1" filt, %.1f' %(herschel_sig[i,j,2])+'+/-%.1f' %(herschel_err[i,j,2]))
#                 plt.subplot(4,3,9+j+1)
#                 plt.imshow(herschel_175_maps[j][herschel_ypix[i]-20:herschel_ypix[i]+20,herschel_xpix[i]-20:herschel_xpix[i]+20]*1000*herschel_175_solan[j])
#                 plt.colorbar()
#                 herschel_sig[i,j,3] = herschel_175_maps[j][herschel_ypix[i],herschel_xpix[i]]*1000*herschel_175_solan[j]
#                 rand_ypix = np.random.choice(np.arange(40),100)
#                 rand_xpix = np.random.choice(np.arange(40),100)
#                 herschel_err[i,j,3] = np.std((herschel_175_maps[j][herschel_ypix[i]-20:herschel_ypix[i]+20,herschel_xpix[i]-20:herschel_xpix[i]+20]*1000*herschel_175_solan[j])[rand_ypix, rand_xpix])
#                 plt.title(herschel_mapfreqs[j] + 'um 1.75" filt, %.1f' %(herschel_sig[i,j,3])+'+/-%.1f' %(herschel_err[i,j,3]))

#         plt.suptitle('Literature values, 250um: ' + str(flux250[i]) + '+/-' + err250[i] + ', 350um: ' + str(flux350[i]) + '+/-' + err350[i] +', 500um: ' + str(flux500[i]) + '+/-' + err500[i])
#         plt.savefig('/data/tangq/debug/'+today+'_'+SPT_SMGs[:,0][id_mask][i]+'_Herschel_smoothed_map_comparison_shifted_pixel.jpg')
#         plt.close()



# flux3mm = SPT_SMG_flux[:,2][indices]
# err3mm = SPT_SMG_flux[:,3][indices]
# flux2mm = SPT_SMG_flux[:,4][indices]
# err2mm = SPT_SMG_flux[:,5][indices]
# flux1mm = SPT_SMG_flux[:,6][indices]
# err1mm = SPT_SMG_flux[:,7][indices]
# spt_flux = np.asarray([ flux3mm, flux2mm, flux1mm])

spt_herschel_factor175 = np.asarray([ 86.63667295,  89.79231335,  95.52984136])*1.e6/1.e3
#spt_herschel_factor = np.asarray([ 41.14864182,  49.8005297 ,  49.30301973])*1.e6/1.e3
spt_herschel_factor100 = np.asarray([ 20.47296958,  41.35435986,  52.20319736])*1.e6/1.e3

spt_ns_filt_90 = np.load(spt_no_smooth_fname+'_90_filter.npy').item()[0]
spt_ns_filt_150 = np.load(spt_no_smooth_fname+'_150_filter.npy').item()[0]
spt_ns_filt_220 = np.load(spt_no_smooth_fname+'_220_filter.npy').item()[0]
spt_ns_filt = np.asarray([spt_ns_filt_90, spt_ns_filt_150, spt_ns_filt_220])
spt_herschel_factor = np.zeros((3))

import scipy.constants as const
Tcmb = 2.725
for j in range(len(spt_ns_filt)):
        reso = (spt_ns_reso_arcmin[j]/60)*np.pi/180.
        dngrid = spt_ns_filt[j].optfilt.shape[0]
        dk = 1./dngrid/reso
        tauprime = spt_ns_filt[j].prof2d*spt_ns_filt[j].trans_func_grid #* filt_output[0].area_eff
        solid_angle =1./ np.sum(spt_ns_filt[j].optfilt*tauprime)  / dk**2
        print str(solid_angle)
        x = float(spt_mapfreqs[j])*1.e9*const.h/(const.k*Tcmb)
        conversion = solid_angle*1.e26*(2*const.k/(const.c)**2)*((const.k*Tcmb/const.h)**2)*(x**4)*np.exp(x)/(np.exp(x)-1)**2   
        spt_herschel_factor[j] = conversion*1.e6/1.e3

sptpol_ns_filt_90 = np.load(sptpol_no_smooth_fname+'_90_filter.npy').item()[0]
sptpol_ns_filt_150 = np.load(sptpol_no_smooth_fname+'_150_filter.npy').item()[0]
sptpol_ns_filt = np.asarray([sptpol_ns_filt_90, sptpol_ns_filt_150])
sptpol_herschel_factor = np.zeros((2))
for j in range(len(sptpol_ns_filt)):
        reso = (sptpol_ns_reso_arcmin[j]/60)*np.pi/180.
        dngrid = sptpol_ns_filt[j].optfilt.shape[0]
        dk = 1./dngrid/reso
        tauprime = sptpol_ns_filt[j].prof2d*sptpol_ns_filt[j].trans_func_grid #* filt_output[0].area_eff
        solid_angle =1./ np.sum(sptpol_ns_filt[j].optfilt*tauprime)  / dk**2
        print str(solid_angle)
        x = float(spt_mapfreqs[j])*1.e9*const.h/(const.k*Tcmb)
        conversion = solid_angle*1.e26*(2*const.k/(const.c)**2)*((const.k*Tcmb/const.h)**2)*(x**4)*np.exp(x)/(np.exp(x)-1)**2   
        sptpol_herschel_factor[j] = conversion*1.e6/1.e3


x = np.asarray([90,150,220])*1.e9*const.h/(const.k*Tcmb)

solid_angle_lindsey = np.asarray([spt_90_lindsey.computearea, spt_150_lindsey.computearea, spt_220_lindsey.computearea])
conversion_lindsey = solid_angle_lindsey*1.e26*(2*const.k/(const.c)**2)*((const.k*Tcmb/const.h)**2)*(x**4)*np.exp(x)/(np.exp(x)-1)**2
spt_herschel_factor_lindsey = conversion_lindsey*1.e6/1.e3

spt_ns_filt_90_3360 = np.load(spt_3360_no_smooth_fname+'_90_filter.npy').item()[0]
spt_ns_filt_150_3360 = np.load(spt_3360_no_smooth_fname+'_150_filter.npy').item()[0]
spt_ns_filt_220_3360 = np.load(spt_3360_no_smooth_fname+'_220_filter.npy').item()[0]
spt_ns_filt_3360 = np.asarray([spt_ns_filt_90_3360, spt_ns_filt_150_3360, spt_ns_filt_220_3360])
spt_herschel_factor_3360 = np.zeros((3))

import scipy.constants as const
Tcmb = 2.725
for j in range(len(spt_ns_filt)):
        reso = (spt_ns_reso_arcmin[j]/60)*np.pi/180.
        dngrid = spt_ns_filt[j].optfilt.shape[0]
        dk = 1./dngrid/reso
        tauprime = spt_ns_filt[j].prof2d*spt_ns_filt[j].trans_func_grid #* filt_output[0].area_eff
        solid_angle =1./ np.sum(spt_ns_filt[j].optfilt*tauprime)  / dk**2
        print str(solid_angle)
        x = float(spt_mapfreqs[j])*1.e9*const.h/(const.k*Tcmb)
        conversion = solid_angle*1.e26*(2*const.k/(const.c)**2)*((const.k*Tcmb/const.h)**2)*(x**4)*np.exp(x)/(np.exp(x)-1)**2   
        spt_herschel_factor_3360[j] = conversion*1.e6/1.e3


spt_sig = np.zeros((len(indices), 3, 4))
spt_err = np.zeros((len(indices), 3, 4))

# for i in range(len(indices)):
#         plt.figure(figsize=(20,15))
#         for j in range(3):
#                 plt.subplot(4,3,j+1)
#                 plt.imshow(spt_maps[j][spt_ypix[i]-20:spt_ypix[i]+20,spt_xpix[i]-20:spt_xpix[i]+20]*spt_herschel_factor[j])
#                 plt.colorbar()
#                 spt_sig[i,j,0] = spt_maps[j][herschel_ypix[i],herschel_xpix[i]]*spt_herschel_factor[j]
#                 rand_ypix = np.random.choice(np.arange(40),100)
#                 rand_xpix = np.random.choice(np.arange(40),100)
#                 spt_err[i,j,0] = np.std((spt_maps[j][spt_ypix[i]-20:spt_ypix[i]+20,spt_xpix[i]-20:spt_xpix[i]+20]*spt_herschel_factor[j])[rand_ypix, rand_xpix])
#                 plt.title(spt_mapfreqs[j] + 'GHz raw, %.1f' %(spt_sig[i,j,0])+'+/- %.1f' %(spt_err[i,j,0]))
#                 plt.subplot(4,3,3+j+1)
#                 plt.imshow(spt_ns_maps[j][spt_ypix[i]-20:spt_ypix[i]+20,spt_xpix[i]-20:spt_xpix[i]+20]*spt_herschel_factor[j])
#                 plt.colorbar()
#                 spt_sig[i,j,1] = spt_ns_maps[j][herschel_ypix[i],herschel_xpix[i]]*spt_herschel_factor[j]
#                 rand_ypix = np.random.choice(np.arange(40),100)
#                 rand_xpix = np.random.choice(np.arange(40),100)
#                 spt_err[i,j,1] = np.std((spt_ns_maps[j][spt_ypix[i]-20:spt_ypix[i]+20,spt_xpix[i]-20:spt_xpix[i]+20]*spt_herschel_factor[j])[rand_ypix, rand_xpix])
#                 plt.title(spt_mapfreqs[j] + 'GHz filt map, %.1f' %(spt_sig[i,j,1])+'+/- %.1f' %(spt_err[i,j,1]))
#                 # plt.subplot(4,3,6+j+1)
#                 # plt.imshow(spt_100_maps[j][spt_ypix[i]-20:spt_ypix[i]+20,spt_xpix[i]-20:spt_xpix[i]+20]*spt_herschel_factor100[j])
#                 # plt.colorbar()
#                 # spt_sig[i,j,2] = spt_100_maps[j][herschel_ypix[i],herschel_xpix[i]]*spt_herschel_factor100[j]
#                 # rand_ypix = np.random.choice(np.arange(40),100)
#                 # rand_xpix = np.random.choice(np.arange(40),100)
#                 # spt_err[i,j,2] = np.std((spt_100_maps[j][spt_ypix[i]-20:spt_ypix[i]+20,spt_xpix[i]-20:spt_xpix[i]+20]*spt_herschel_factor100[j])[rand_ypix, rand_xpix])
#                 # plt.title(spt_mapfreqs[j] + 'GHz 1" filt map, %.1f' %(spt_sig[i,j,2])+'+/- %.1f' %(spt_err[i,j,2]))
#                 # plt.subplot(4,3,9+j+1)
#                 # plt.imshow(spt_175_maps[j][spt_ypix[i]-20:spt_ypix[i]+20,spt_xpix[i]-20:spt_xpix[i]+20]*spt_herschel_factor175[j])
#                 # plt.colorbar()
#                 # spt_sig[i,j,3] = spt_175_maps[j][herschel_ypix[i],herschel_xpix[i]]*spt_herschel_factor175[j]
#                 # rand_ypix = np.random.choice(np.arange(40),100)
#                 # rand_xpix = np.random.choice(np.arange(40),100)
#                 # spt_err[i,j,3] = np.std((spt_175_maps[j][spt_ypix[i]-20:spt_ypix[i]+20,spt_xpix[i]-20:spt_xpix[i]+20]*spt_herschel_factor175[j])[rand_ypix, rand_xpix])
#                 # plt.title(spt_mapfreqs[j] + 'GHz 1.75" filt map, %.1f' %(spt_sig[i,j,3])+'+/- %.1f' %(spt_err[i,j,3]))
#         plt.suptitle('Literature values, 90GHz: ' + str(flux3mm[i]) + '+/-' + err3mm[i] + ', 150GHz: ' + str(flux2mm[i]) + '+/-' + err2mm[i] +', 220GHz: ' + str(flux1mm[i]) + '+/-' + err1mm[i])
#         plt.savefig('/data/tangq/debug/'+today+'_'+SPT_SMGs[:,0][id_mask][i]+'_SPT_smoothed_map_comparison.jpg')
#         plt.close()


### check with Wendy's ptsrc fluxes
indices = np.asarray(indices)

wendy_ptsrc_list = np.loadtxt('/data/tangq/wendymaps/Everett20_SPT-SZ_PS_catalog_v3.dat',dtype='str')
wendy_list_ra = wendy_ptsrc_list[:,1].astype(float)
wendy_list_dec = wendy_ptsrc_list[:,2].astype(float)
wendy_list_90 = wendy_ptsrc_list[:,4].astype(float)
wendy_list_90err= wendy_list_90/wendy_ptsrc_list[:,3].astype(float)
wendy_list_150 = wendy_ptsrc_list[:,9].astype(float)
wendy_list_150err= wendy_list_150/wendy_ptsrc_list[:,8].astype(float)
wendy_list_220 = wendy_ptsrc_list[:,14].astype(float)
wendy_list_220err= wendy_list_220/wendy_ptsrc_list[:,13].astype(float)

idx, d2d = t.match_src_to_catalog(ra1, dec1, wendy_list_ra, wendy_list_dec)
wendy_mask = d2d.arcminute < 1.
for i in range(len(wendy_mask)):
        if (np.abs(wendy_list_90err[idx[i]]) == np.inf) or (np.abs(wendy_list_150err[idx[i]]) == np.inf) or (np.abs(wendy_list_220err[idx[i]]) == np.inf):
                wendy_mask[i] = False
indices90 = indices[wendy_mask]
ra90 = ra1[wendy_mask]
dec90 = dec1[wendy_mask]
idx = idx[wendy_mask]
spt_ypix90 = spt_ypix[wendy_mask]
spt_xpix90 = spt_xpix[wendy_mask]
spt_ypix150 = spt_ypix90
spt_xpix150 = spt_xpix90
spt_ypix220 = spt_ypix90
spt_xpix220 = spt_xpix90
spt_ypix90_lindsey = spt_ypix_lindsey[wendy_mask]
spt_xpix90_lindsey = spt_xpix_lindsey[wendy_mask]
spt_ypix150_lindsey = spt_ypix90_lindsey
spt_xpix150_lindsey = spt_xpix90_lindsey
spt_ypix220_lindsey = spt_ypix90_lindsey
spt_xpix220_lindsey = spt_xpix90_lindsey

wendy90_flux_good = wendy_list_90[idx]
wendy90_err_good = wendy_list_90err[idx]
wendy150_flux_good = wendy_list_150[idx]
wendy150_err_good = wendy_list_150err[idx]
wendy220_flux_good = wendy_list_220[idx]
wendy220_err_good = wendy_list_220err[idx]

# mapfreq = "150"
# sourcelist = np.loadtxt('/data/tangq/wendymaps/ra23h30dec-55_allyears/source_list_3sig_allyears_'+mapfreq+'.dat')
# cat_ra = sourcelist[:,3]
# cat_dec = sourcelist[:,4]
# idx, d2d = t.match_src_to_catalog(ra1, dec1, cat_ra, cat_dec)
# wendy_mask = d2d.arcminute < 1.
# indices150 = indices[wendy_mask]
# ra150 = ra1[wendy_mask]
# dec150 = dec1[wendy_mask]
# idx = idx[wendy_mask]
# spt_ypix150 = spt_ypix[wendy_mask]
# spt_xpix150 = spt_xpix[wendy_mask]
# idx, d2d = t.match_src_to_catalog(ra150, dec150, wendy_list_ra, wendy_list_dec)
# wendy_mask = d2d.arcminute < 1.
# indices150_1 = indices150[wendy_mask]
# ra150_1 = ra150[wendy_mask]
# dec150_1 = dec150[wendy_mask]
# idx = idx[wendy_mask]
# spt_ypix150 = spt_ypix150[wendy_mask]
# spt_xpix150 = spt_xpix150[wendy_mask]
# wendy150_flux_good = wendy_list_150[idx]
# wendy150_err_good = wendy_list_150err[idx]



spt90_sig = np.zeros((len(indices90),2))
spt90_err = np.zeros((len(indices90),2))
spt150_sig = np.zeros((len(indices90), 2))
spt150_err = np.zeros((len(indices90), 2))
spt220_sig = np.zeros((len(indices90), 2))
spt220_err = np.zeros((len(indices90), 2))
sptpol90_sig = np.zeros((len(indices90)))
sptpol90_err = np.zeros((len(indices90)))
sptpol150_sig = np.zeros((len(indices90)))
sptpol150_err = np.zeros((len(indices90)))

spt90_sig_lindsey = np.zeros((len(indices90)))
spt90_err_lindsey = np.zeros((len(indices90)))
spt150_sig_lindsey = np.zeros((len(indices90)))
spt150_err_lindsey = np.zeros((len(indices90)))
spt220_sig_lindsey = np.zeros((len(indices90)))
spt220_err_lindsey = np.zeros((len(indices90)))
spt90_sig_3360 = np.zeros((len(indices90)))
spt90_err_3360 = np.zeros((len(indices90)))
spt150_sig_3360 = np.zeros((len(indices90)))
spt150_err_3360 = np.zeros((len(indices90)))
spt220_sig_3360 = np.zeros((len(indices90)))
spt220_err_3360 = np.zeros((len(indices90)))

for i in np.arange(len(indices90)-1)+1:
        j = 0
        spt90_sig[i,0] = spt_maps[j][spt_ypix90[i],spt_xpix90[i]]*spt_herschel_factor[j]
        spt90_sig[i,1] = spt_ns_maps[j][spt_ypix90[i],spt_xpix90[i]]*spt_herschel_factor[j]
        sptpol90_sig[i] = sptpol_ns_maps[j][spt_ypix90[i],spt_xpix90[i]]*sptpol_herschel_factor[j]
        rand_ypix = np.random.choice(np.arange(1500,2500,1),100)
        rand_xpix = np.random.choice(np.arange(1500,2500,1),100)
        spt90_err[i,0] = np.std((spt_maps[j][rand_ypix, rand_xpix]*spt_herschel_factor[j]))
        spt90_err[i,1] = np.std((spt_ns_maps[j][rand_ypix, rand_xpix]*spt_herschel_factor[j]))
        sptpol90_err[i] = np.std((sptpol_ns_maps[j][rand_ypix, rand_xpix]*sptpol_herschel_factor[j]))

        spt90_sig_lindsey[i] = spt_lindsey_maps[j][spt_ypix90_lindsey[i],spt_xpix90_lindsey[i]]*spt_herschel_factor_lindsey[j]
        spt90_sig_3360[i] = spt_ns_maps_3360[j][spt_ypix90_lindsey[i],spt_xpix90_lindsey[i]]*spt_herschel_factor_3360[j]
        rand_ypix = np.random.choice(np.arange(1500,2500,1),100)
        rand_xpix = np.random.choice(np.arange(1500,2500,1),100)
        spt90_err_lindsey[i] = np.std((spt_lindsey_maps[j][rand_ypix, rand_xpix]*spt_herschel_factor_lindsey[j]))
        spt90_err_3360[i] = np.std((spt_ns_maps_3360[j][rand_ypix, rand_xpix]*spt_herschel_factor_3360[j]))

for i in np.arange(len(indices90)-1)+1:
        j = 1
        spt150_sig[i,0] = spt_maps[j][spt_ypix150[i],spt_xpix150[i]]*spt_herschel_factor[j]
        spt150_sig[i,1] = spt_ns_maps[j][spt_ypix150[i],spt_xpix150[i]]*spt_herschel_factor[j]
        sptpol150_sig[i] = sptpol_ns_maps[j][spt_ypix150[i],spt_xpix150[i]]*sptpol_herschel_factor[j]
        rand_ypix = np.random.choice(np.arange(1500,2500,1),100)
        rand_xpix = np.random.choice(np.arange(1500,2500,1),100)
        spt150_err[i,0] = np.std((spt_maps[j][rand_ypix, rand_xpix]*spt_herschel_factor[j]))
        spt150_err[i,1] = np.std((spt_ns_maps[j][rand_ypix, rand_xpix]*spt_herschel_factor[j]))
        sptpol150_err[i] = np.std((sptpol_ns_maps[j][rand_ypix, rand_xpix]*sptpol_herschel_factor[j]))

        spt150_sig_lindsey[i] = spt_lindsey_maps[j][spt_ypix150_lindsey[i],spt_xpix150_lindsey[i]]*spt_herschel_factor_lindsey[j]
        spt150_sig_3360[i] = spt_ns_maps_3360[j][spt_ypix150_lindsey[i],spt_xpix150_lindsey[i]]*spt_herschel_factor_3360[j]
        rand_ypix = np.random.choice(np.arange(1500,2500,1),100)
        rand_xpix = np.random.choice(np.arange(1500,2500,1),100)
        spt150_err_lindsey[i] = np.std((spt_lindsey_maps[j][rand_ypix, rand_xpix]*spt_herschel_factor_lindsey[j]))
        spt150_err_3360[i] = np.std((spt_ns_maps_3360[j][rand_ypix, rand_xpix]*spt_herschel_factor_3360[j]))

for i in np.arange(len(indices90)-1)+1:
        j = 2
        spt220_sig[i,0] = spt_maps[j][spt_ypix220[i],spt_xpix220[i]]*spt_herschel_factor[j]
        spt220_sig[i,1] = spt_ns_maps[j][spt_ypix220[i],spt_xpix220[i]]*spt_herschel_factor[j]
        rand_ypix = np.random.choice(np.arange(1500,2500,1),100)
        rand_xpix = np.random.choice(np.arange(1500,2500,1),100)
        spt220_err[i,0] = np.std((spt_maps[j][rand_ypix, rand_xpix]*spt_herschel_factor[j]))
        spt220_err[i,1] = np.std((spt_ns_maps[j][rand_ypix, rand_xpix]*spt_herschel_factor[j]))

        spt220_sig_lindsey[i] = spt_lindsey_maps[j][spt_ypix220_lindsey[i],spt_xpix220_lindsey[i]]*spt_herschel_factor_lindsey[j]
        spt220_sig_3360[i] = spt_ns_maps_3360[j][spt_ypix220_lindsey[i],spt_xpix220_lindsey[i]]*spt_herschel_factor_3360[j]
        rand_ypix = np.random.choice(np.arange(1500,2500,1),100)
        rand_xpix = np.random.choice(np.arange(1500,2500,1),100)
        spt220_err_lindsey[i] = np.std((spt_lindsey_maps[j][rand_ypix, rand_xpix]*spt_herschel_factor_lindsey[j]))
        spt220_err_3360[i] = np.std((spt_ns_maps_3360[j][rand_ypix, rand_xpix]*spt_herschel_factor_3360[j]))

plt.figure()
plt.subplot(131)
x = np.arange(len(spt90_sig[:,0])-1)
#plt.errorbar(x+0.1, spt90_sig[1:,0], yerr = spt90_err[1:,0], label='raw SPT-SZ map',ls= 'none',fmt='.')
plt.errorbar(x+0.1, spt90_sig_3360[1:], yerr = spt90_err_3360[1:], label='filt SPT-SZ map (3360pix)',ls= 'none',fmt='.')
plt.errorbar(x+0.15, spt90_sig[1:,1], yerr = spt90_err[1:,1], label='filt SPT-SZ map (4200pix)',ls= 'none',fmt='.')
plt.errorbar(x+0.2, wendy90_flux_good[1:], yerr = wendy90_err_good[1:], label='Wendy ptsrc list', ls= 'none', fmt='.')
plt.errorbar(x+0.0, spt90_sig_lindsey[1:], yerr = spt90_err_lindsey[1:], label='Lindsey filt map',ls= 'none',fmt='.')
plt.errorbar(x+0.25, sptpol90_sig[1:], yerr = sptpol90_err[1:,], label='filt SPTpol map', ls='none', fmt='.')
plt.legend(loc='best')
plt.xlabel('SPT SMGs')
plt.ylabel('flux (mJy) ' )
plt.title('90 GHz')
plt.xlim(-0.5, x[-1]+0.5)

plt.subplot(132)
x = np.arange(len(spt150_sig[:,0])-1)
plt.errorbar(x+0.1, spt150_sig_3360[1:], yerr = spt150_err_3360[1:], label='filt SPT-SZ map (3360pix)',ls= 'none',fmt='.')
plt.errorbar(x+0.15, spt150_sig[1:,1], yerr = spt150_err[1:,1], label='filt SPT-SZ map ',ls= 'none',fmt='.')
plt.errorbar(x+0.2, wendy150_flux_good[1:], yerr = wendy150_err_good[1:], label='Wendy ptsrc list', ls= 'none', fmt='.')
plt.errorbar(x+0.0, spt150_sig_lindsey[1:], yerr = spt150_err_lindsey[1:], label='Lindsey filt map',ls= 'none',fmt='.')
plt.errorbar(x+0.25, sptpol150_sig[1:], yerr = sptpol150_err[1:], label='filt SPTpol map', ls='none', fmt='.')
plt.legend(loc='best')
plt.xlabel('SPT SMGs')
plt.ylabel('flux (mJy) ' )
plt.title('150 GHz')
plt.xlim(-0.5, x[-1]+0.5)

plt.subplot(133)
x = np.arange(len(spt220_sig[:,0])-1)
plt.errorbar(x+0.1, spt220_sig[1:,0], yerr = spt220_err[1:,0], label='raw SPT-SZ map',ls= 'none',fmt='.')
plt.errorbar(x+0.15, spt220_sig[1:,1], yerr = spt220_err[1:,1], label='filt SPT-SZ map',ls= 'none',fmt='.')
plt.errorbar(x+0.2, wendy220_flux_good[1:], yerr = wendy220_err_good[1:], label='Wendy ptsrc list', ls= 'none', fmt='.')
plt.errorbar(x+0.0, spt220_sig_lindsey[1:], yerr = spt220_err_lindsey[1:], label='Lindsey filt map',ls= 'none',fmt='.')
plt.legend(loc='best')
plt.xlabel('SPT SMGs')
plt.ylabel('flux (mJy) ' )
plt.title('220 GHz')
plt.xlim(-0.5, x[-1]+0.5)

def bootstrap(n_real, sample_len, galaxy_arr):
    #returns the MAD error using the bootstrap medians
    bs_med = np.zeros((n_real))
    for j in range(n_real):
        indices = np.random.choice(sample_len, sample_len)
        bs_med[j] = np.median(galaxy_arr[indices])
    return bs_med

bs_med90 = np.median(bootstrap(10000, len(wendy90_flux_good), wendy90_flux_good/spt90_sig[:,1]))
bs_med150 = np.median(bootstrap(10000, len(wendy150_flux_good), wendy150_flux_good/spt150_sig[:,1]))
bs_med220 = np.median(bootstrap(10000, len(wendy220_flux_good), wendy220_flux_good/spt220_sig[:,1]))

print str(spt_herschel_factor)

print "90 GHz boostrap median: " + str(bs_med90)
print "150 GHz boostrap median: " + str(bs_med150)
print "150 GHz boostrap median: " + str(bs_med150)

apod_mask = fits.open(configParser.get('map_and_files', 'apod_mask'))[1].data['APOD_MASK'][0]
mapinfo = idl.readIDLSav(configParser.get('map_and_files', 'mapinfo')).field_info
radec0 = np.asarray([ [mapinfo[1]['ra0'],mapinfo[1]['dec0']],[mapinfo[1]['ra0'],mapinfo[1]['dec0']],[mapinfo[1]['ra0'],mapinfo[1]['dec0']] ])
reso_arcmin = np.asarray([mapinfo[1]['reso_arcmin'], mapinfo[1]['reso_arcmin'], mapinfo[1]['reso_arcmin']])
proj = 0
npixels = np.asarray([map0s[0].shape, map0s[1].shape, map0s[2].shape])



#### checking SPTSMG fluxes with filtered hitmaps:

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
hers_filt = [filt250, filt350, filt500]

spt_filt_map_90 = t.filtered_hitmaps([spt_ns_maps[0].shape[0]/2],[spt_ns_maps[0].shape[1]/2], spt_ns_filt_90.trans_func_grid, spt_ns_filt_90.optfilt)
spt_filt_map_150 = t.filtered_hitmaps([spt_ns_maps[0].shape[0]/2],[spt_ns_maps[0].shape[1]/2], spt_ns_filt_150.trans_func_grid, spt_ns_filt_150.optfilt)
spt_filt_map_220 = t.filtered_hitmaps([spt_ns_maps[0].shape[0]/2],[spt_ns_maps[0].shape[1]/2], spt_ns_filt_220.trans_func_grid, spt_ns_filt_220.optfilt)
hers_filt_map_250 = t.filtered_hitmaps([hers_cmap[0].shape[0]/2],[hers_cmap[0].shape[1]/2], filt250[0].trans_func_grid, filt250[0].optfilt)
hers_filt_map_350 = t.filtered_hitmaps([hers_cmap[0].shape[0]/2],[hers_cmap[0].shape[1]/2], filt350[0].trans_func_grid, filt350[0].optfilt)
hers_filt_map_500 = t.filtered_hitmaps([hers_cmap[0].shape[0]/2],[hers_cmap[0].shape[1]/2], filt500[0].trans_func_grid, filt500[0].optfilt)

def func1(X, a):
    aa = X
    return a*aa

cutout_mask = np.zeros((spt_ns_maps[0].shape),dtype =bool)
cutout_shape = np.asarray([10,10])
cutout_mask[(cutout_mask.shape[0]-cutout_shape[0])/2:(cutout_mask.shape[0]+cutout_shape[0])/2,(cutout_mask.shape[1]-cutout_shape[1])/2:(cutout_mask.shape[1]+cutout_shape[1])/2] = True

spt90_fit = np.zeros((len(spt_ypix90)))
spt150_fit = np.zeros((len(spt_ypix90)))
spt220_fit = np.zeros((len(spt_ypix90)))
hers250_fit = np.zeros((len(spt_ypix90)))
hers350_fit = np.zeros((len(spt_ypix90)))
hers500_fit = np.zeros((len(spt_ypix90)))

for i in range(len(spt_ypix90)):
    j=0
    spt_90_cutout = spt_ns_maps[j][spt_ypix90[i]-cutout_shape[0]/2:spt_ypix90[i]+cutout_shape[0]/2,spt_xpix90[i]-cutout_shape[1]/2:spt_xpix90[i]+cutout_shape[1]/2]*spt_herschel_factor[j]
    popt,pcov = curve_fit(func1, (spt_filt_map_90[cutout_mask].flatten()), spt_90_cutout.flatten())
    spt90_fit[i] = popt[0]
    j=1
    spt_150_cutout = spt_ns_maps[j][spt_ypix90[i]-cutout_shape[0]/2:spt_ypix90[i]+cutout_shape[0]/2,spt_xpix90[i]-cutout_shape[1]/2:spt_xpix90[i]+cutout_shape[1]/2]*spt_herschel_factor[j]
    popt,pcov = curve_fit(func1, (spt_filt_map_150[cutout_mask].flatten()), spt_150_cutout.flatten())
    spt150_fit[i] = popt[0]
    j=2
    spt_220_cutout = spt_ns_maps[j][spt_ypix90[i]-cutout_shape[0]/2:spt_ypix90[i]+cutout_shape[0]/2,spt_xpix90[i]-cutout_shape[1]/2:spt_xpix90[i]+cutout_shape[1]/2]*spt_herschel_factor[j]
    popt,pcov = curve_fit(func1, (spt_filt_map_220[cutout_mask].flatten()), spt_220_cutout.flatten())
    spt220_fit[i] = popt[0]

    hers_250_cutout = map250[spt_ypix90[i]+1-cutout_shape[0]/2:spt_ypix90[i]+1+cutout_shape[0]/2,spt_xpix90[i]+2-cutout_shape[1]/2:spt_xpix90[i]+2+cutout_shape[1]/2]
    popt,pcov = curve_fit(func1, (hers_filt_map_250[cutout_mask].flatten()), hers_250_cutout.flatten())
    hers250_fit[i] = popt[0]  
    hers_350_cutout = map350[spt_ypix90[i]+1-cutout_shape[0]/2:spt_ypix90[i]+1+cutout_shape[0]/2,spt_xpix90[i]+2-cutout_shape[1]/2:spt_xpix90[i]+2+cutout_shape[1]/2]
    popt,pcov = curve_fit(func1, (hers_filt_map_250[cutout_mask].flatten()), hers_350_cutout.flatten())
    hers350_fit[i] = popt[0]  
    hers_500_cutout = map500[spt_ypix90[i]+1-cutout_shape[0]/2:spt_ypix90[i]+1+cutout_shape[0]/2,spt_xpix90[i]+2-cutout_shape[1]/2:spt_xpix90[i]+2+cutout_shape[1]/2]
    popt,pcov = curve_fit(func1, (hers_filt_map_500[cutout_mask].flatten()), hers_500_cutout.flatten())
    hers500_fit[i] = popt[0]  
    
hers250_flux = np.zeros((len(spt_ypix90)))
hers350_flux = np.zeros((len(spt_ypix90)))
hers500_flux = np.zeros((len(spt_ypix90)))
hers250_err = np.zeros((len(spt_ypix90)))
hers350_err = np.zeros((len(spt_ypix90)))
hers500_err = np.zeros((len(spt_ypix90)))
for i in np.arange(len(spt_ypix90)):
        hers250_flux[i] = map250[spt_ypix90[i]+1,spt_xpix90[i]+2]
        hers350_flux[i] = map350[spt_ypix90[i]+1,spt_xpix90[i]+2]
        hers500_flux[i] = map500[spt_ypix90[i]+1,spt_xpix90[i]+2] 
        rand_ypix = np.random.choice(np.arange(1500,2500,1),100)
        rand_xpix = np.random.choice(np.arange(1500,2500,1),100)
        hers250_err[i] = np.std((map250[rand_ypix, rand_xpix])) 
        hers350_err[i] = np.std((map350[rand_ypix, rand_xpix])) 
        hers500_err[i] = np.std((map500[rand_ypix, rand_xpix])) 

plt.figure()
plt.errorbar(np.arange(len(spt_ypix90)), hers250_flux, yerr= hers250_err, color='blue', ls='none',label='250um flux')
plt.plot(np.arange(len(spt_ypix90)), hers250_fit,  marker='.', ls='none', color='blue',label='250um fit')
plt.errorbar(np.arange(len(spt_ypix90)), hers350_flux, yerr= hers250_err, color='green',ls='none',label='350um flux')
plt.plot(np.arange(len(spt_ypix90)), hers350_fit,  marker='.', ls='none', color='green',label='350um fit')
plt.errorbar(np.arange(len(spt_ypix90)), hers500_flux, yerr= hers250_err, color='orange',ls='none',label='500um flux')
plt.plot(np.arange(len(spt_ypix90)), hers500_fit,  marker='.', ls='none', color='orange',label='500um fit')
plt.xlabel('SMG')
plt.ylabel('Flux (Jy)')
plt.legend(loc='best')
plt.xlim(0.5,7.5)
plt.title('Herschel map SPT-SMG flux comparison')

plt.figure()
plt.errorbar(np.arange(len(spt_ypix90)), spt90_sig[:,1], yerr= hers250_err, color='blue', ls='none',label='90GHz flux')
plt.plot(np.arange(len(spt_ypix90)), spt90_fit,  marker='.', ls='none', color='blue',label='90GHz fit')
plt.errorbar(np.arange(len(spt_ypix90)), spt150_sig[:,1], yerr= hers250_err, color='green',ls='none',label='150GHz flux')
plt.plot(np.arange(len(spt_ypix90)), spt150_fit,  marker='.', ls='none', color='green',label='150GHz fit')
plt.errorbar(np.arange(len(spt_ypix90)), spt220_sig[:,1], yerr= hers250_err, color='orange',ls='none',label='220GHz flux')
plt.plot(np.arange(len(spt_ypix90)), spt220_fit,  marker='.', ls='none', color='orange',label='220GHz fit')
plt.xlabel('SMG')
plt.ylabel('Flux (mJy)')
plt.legend(loc='best')
plt.xlim(0.5,7.5)
plt.title('SPT map SPT-SMG flux comparison')

plt.figure()
plt.subplot(131)
i = 2
hers_250_cutout = map250[spt_ypix90[i]+1-cutout_shape[0]/2:spt_ypix90[i]+1+cutout_shape[0]/2,spt_xpix90[i]+2-cutout_shape[1]/2:spt_xpix90[i]+2+cutout_shape[1]/2]
plt.imshow(hers_250_cutout)
plt.colorbar()
plt.title('Herschel 250um map')
plt.subplot(132)
plt.imshow(hers_filt_map_250[2095:2105,2095:2105])
plt.colorbar()
plt.title('Filtered hitmap')
plt.subplot(133)
plt.imshow(hers_250_cutout-hers250_fit[i]*hers_filt_map_250[2095:2105,2095:2105])
plt.colorbar()
plt.title('real map - fit * hitmap')

