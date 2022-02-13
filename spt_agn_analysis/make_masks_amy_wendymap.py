#!/usr/bin/env python

"""
makePtSrcMask takes a point source config file and a map and makes a mask.
Created by DH, modified and added to sptpol_software by ATC.
makeApodMask takes a map and makes an apodization mask.  Very VERY
loosely based on MAKE_APOD_MASK_LPS12.pro,  WEIGHT_MAP METHOD and
spt_analsysis/powspec/maskkernel/hanningize_mask.pro from the low-el
spt-sz power specturm.  Could use some improvements, but this can 
get us started?
smoothMask is terrible and needs to be fixed.
Classes
  
Functions:
   makePtSrcMask
   makeApodMask - This function is dead and removed
   smoothMask
Non-Core Dependencies
   NumPy
   SciPy
"""

__metaclass__ = type  #Use "new-style" classes
__author__    = "Abby Crites"
__email__     = "abbycrites@gmail.com"
__version__   = "1.0"
__date__      = "2012-12-12" #Date of last modification


import numpy as np
from scipy import interpolate, weave, ndimage
#import quickpol as qp
import pickle

import sptpol_software as sps
import sptpol_software.observation.sky as sky
from sptpol_software.util import files

import pylab as pl
import pdb



def makePtSrcMask(point_source_config, field_map, reso_arcmin,dir_out = None, debug = False,
                  center = np.array([352.51485, -54.99948]), fixed_rad_deg=None,flux_thresh=None):
    '''INPUTS
           point_source_config: (string) the location of a text file 
           containing a list of point sources to mask.
           field_map: (map object) A map you wish to mask
           dir_out: if None, do nothing if string write a file in that directory
      OUTPUT
          The mask.
          Example usage:
          point_source_config = '/home/acrites/sptpol_code
          /sptpol_software/config_files/ptsrc_config_ra23h30dec-5520101118_203532.txt'
          field_map = '/data22/acrites/20121203/maps/150/left/ra23h30dec-55/
          ra23h30dec-55_map_20121025_000557_150ghz.h5'
          mask = makePtSrcMask(point_source_config, field_map)
    CHANGES
        27 Nov 2013: Copy tmap on input, to avoid altering the center. 
            Remove "no polarization maps present" message. SH
    '''

    tmap = field_map

    center = center
    #print '**WARNING: Center of map is hardcoded as [352.51485,-54.99948].'
        

    #shape=(450, 450), reso_arcmin=2., projection=5, center=[ 352.5,  -55. ])

    # map_ras  = np.zeros(tmap.shape)
    # map_dcs  = np.zeros(tmap.shape)
    # for i in xrange(0, tmap.shape[0]):
    #     for j in xrange(0, tmap.shape[1]):
    #         map_ras[i,j], map_dcs[i,j] = tmap.pix2Ang([i,j])        
    #     if i%20==0:
    #         print "Finished %d of %d." % (i, tmap.shape[0])

    # Forget that for loop noise!  This is PYTHON, baby!
    npixels = np.asarray(tmap.shape)
    map_ras, map_dcs = sky.pix2Ang((np.repeat(np.arange(tmap.shape[0]),tmap.shape[1]),
                                     np.tile(np.arange(tmap.shape[1]),tmap.shape[0])),center, reso_arcmin,npixels)
    map_ras = map_ras.reshape(tmap.shape)
    map_dcs = map_dcs.reshape(tmap.shape)

    map_xs = np.cos(map_ras*np.pi/180.)*np.cos(map_dcs*np.pi/180.)
    map_ys = np.sin(map_ras*np.pi/180.)*np.cos(map_dcs*np.pi/180.)
    map_zs = np.sin(map_dcs*np.pi/180.)

    #catalog = np.loadtxt("inputs/ptsrc_config_ra23h30dec-5520101118_203532.txt", skiprows=2)
    catalog = np.loadtxt(point_source_config, skiprows=0)


    #This is just in case you only have one point source in your file.
    use_mask = np.ones(catalog.shape[0],dtype=bool)

    cat_ras = catalog[:,3][use_mask]
    cat_dcs = catalog[:,4][use_mask]
    cat_flux = catalog[:,6][use_mask]
    cat_mds = np.zeros((len(cat_flux)))
    cat_mds[np.where(cat_flux > 100)[0]] = 8./60.
    cat_mds[np.where(cat_flux < 100)[0]] = 5./60.

    cat_xs = np.cos(cat_ras*np.pi/180.)*np.cos(cat_dcs*np.pi/180.)
    cat_ys = np.sin(cat_ras*np.pi/180.)*np.cos(cat_dcs*np.pi/180.)
    cat_zs = np.sin(cat_dcs*np.pi/180.)

    mask = np.ones( tmap.shape )

    for (cx, cy, cz, md) in zip(cat_xs, cat_ys, cat_zs, cat_mds):
        dmap = cx*map_xs + cy*map_ys + cz*map_zs
        mask[ np.where(dmap >= np.cos(md*np.pi/180.)) ] = 0.0
    dir_out = '/data/tangq/wendymap/ra23h30dec-55_allyears/150GHz_'
    
    np.save(dir_out + 'ptsrc_mask.npy', mask)

    return mask

def getPtSrcMask(point_source_config, shape, center, proj, reso_arcmin,
                 fixed_rad_deg=None,flux_thresh=None):

    map_ras, map_dcs = pix2Ang((np.repeat(np.arange(shape[0]),shape[1]),
                                     np.tile(np.arange(shape[1]),shape[0])))
    map_ras = map_ras.reshape(shape)
    map_dcs = map_dcs.reshape(shape)

    map_xs = np.cos(map_ras*np.pi/180.)*np.cos(map_dcs*np.pi/180.)
    map_ys = np.sin(map_ras*np.pi/180.)*np.cos(map_dcs*np.pi/180.)
    map_zs = np.sin(map_dcs*np.pi/180.)

    #catalog = np.loadtxt("inputs/ptsrc_config_ra23h30dec-5520101118_203532.txt", skiprows=2)
    catalog = np.loadtxt(point_source_config, skiprows=2)

    #This is just in case you only have one point source in your file.
    use_mask = np.ones(catalog.shape[0],dtype=bool)
    if flux_thresh is not None:
        try:
            cat_thresh = catalog[:,4]
            use_mask = cat_thresh > flux_thresh if flux_thresh>0 else cat_thresh < -flux_thresh

        except IndexError:
            pass
    cat_ras = catalog[:,1][use_mask]
    cat_dcs = catalog[:,2][use_mask]
    cat_mds = catalog[:,3][use_mask] if fixed_rad_deg is None else np.ones(catalog[:,3][use_mask].shape)*fixed_rad_deg

    cat_xs = np.cos(cat_ras*np.pi/180.)*np.cos(cat_dcs*np.pi/180.)
    cat_ys = np.sin(cat_ras*np.pi/180.)*np.cos(cat_dcs*np.pi/180.)
    cat_zs = np.sin(cat_dcs*np.pi/180.)

    mask = np.ones( shape )
    for (cx, cy, cz, md) in zip(cat_xs, cat_ys, cat_zs, cat_mds):
        dmap = cx*map_xs + cy*map_ys + cz*map_zs
        mask[ np.where(dmap >= np.cos(md*np.pi/180.)) ] = 0.0
    return mask

def makeApodMask(field_map,taper_pixels = 4.0,threshold=.5,fwhm_smooth_pixels=30, dir_out = None):
    '''
    sptpol_software.analysis.maps.py:makeBorderApodization is the prefered method to make
         map apodization masks. 
    Do not use this function, it is dead, use makeBorderApodization. 
    '''

    print '*****************************************************************************************'
    print ' Do not use this function, it is dead, use sptpol_software.analysis.maps.py:makeBorderApodization'
    print '*****************************************************************************************'
    

def smoothMask(mask,taper_pixels,threshold,fwhm_smooth_pixels):
    '''
      kind of follows from spt_analsysis/powspec/maskkernel/hanningize_mask.pro
      apod = hanningize_mask(apod,taper_arcmin,threshold,fwhm_smooth_arcmin)
      INPUTS
           mask: (array) the apodization mask before smoothing
           taper_pixels: (float) the edge taper in pixels
           threshold: (float) pick this as the edge
           fwhm_smooth_pixels: (float) smoothing radius
      OUTPUTS
           a smoothed apodization mask created fromt the weight mask
    '''
    
    smoothing_kernel = fwhm_smooth_pixels / 2*np.sqrt(2*np.log(2))
    mask = ndimage.gaussian_filter(mask, smoothing_kernel)
    
    #mask[mask > threshold] = 1.0
    #mask[mask <= threshold] = 0.0
    
    smoothing_kernel = taper_pixels / 2*np.sqrt(2*np.log(2))
    mask = ndimage.gaussian_filter(mask, smoothing_kernel)

    return mask