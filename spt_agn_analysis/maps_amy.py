"""
Perform analysis tasks on maps, using real or simulated data.
Classes
   MapMaker
      reconstructMap
      reconstructTemperatureMap
      reconstructTimestreamMaps
      combineBands
   MapAnalyzer
      constructEB
      constructEBV2 # obsolete, now just a pointer to constructEB
      calculateCls
Functions:
    makeBorderApodization
    makeApodizedPointSourceMask
    calculateNoise
    calculateChi
    calculateChiWithWeight
    pointSourceWienerSetup
    paintOverPointSources
    projectTFromP
    basicMapper
    matchedBeamMapper
    bin2DPowerSpectrum
Non-Core Dependencies
   NumPy
   SciPy
"""

__metaclass__ = type  #Use "new-style" classes
__version__   = "1.2"
__date__      = "2014-03-26" #Date of last modification


import pdb
import warnings, os, time, pickle
import copy
import numpy as np
import scipy as sp
import numexpr as ne
from copy import deepcopy
from glob import glob
from scipy import linalg as LA
from numpy import array, sqrt, abs, where
from scipy import weave, ndimage, stats
from scipy.weave import converters
from ..observation import sky, telescope
from .. import constants, float_type, c_float_type
from .. import util
from ..util import idl, tools, math, files
from ..util.math import d2dx2, d2dy2, d2dxdy, ddx, ddy
from ..util.tools import struct
from sptpol_software.analysis.powspec import make_masks_amy, covariance_utils
from .. import float_type, util, constants
from ..util import time as spt_time
from ..analysis import processing
from ..data import c_interface

ne.set_num_threads(int(os.environ.get('OMP_NUM_THREADS',4))) # Don't use all the processors!

##########################################################################################
def checkWeights(_map):
    """
    Takes an input PolarizedMap, looks at the weights, and
    checks the conditioning of the weight matrix.
    INPUTS
        _map: (PolarizedMap)
    """
    check_map = deepcopy(_map.getTOnly().weight)
    check_map[check_map >= 0.0] = 1.0
    for i in range(0,np.shape(_map)[0]):
        for j in range(0,np.shape(_map)[0]):
            if sp.linalg.det(_map.weight[i,j,:,:]) < 1.0e6:
                check_map[i,j] = 0.0
        print(i)

    return check_map


def checkWeightsV2(_map):
    """
    This is just a much faster version of checkWeights
    """
    w = _map.weight
    a = w[:,:,0,0]
    b = w[:,:,0,1]
    c = w[:,:,0,2]
    d = w[:,:,1,0]
    e = w[:,:,1,1]
    f = w[:,:,1,2]
    g = w[:,:,2,0]
    h = w[:,:,2,1]
    i = w[:,:,2,2]
    det = a*e*i+b*f*g+c*d*h-c*e*g-b*d*i-a*f*h
    is_it_invertible = np.abs(det) > 1e6

    return is_it_invertible

def checkWeightsV3(_map, threshold = 4.0):
    """
        This is another way to check the weights
    """
    check_map = deepcopy(_map.getTOnly().weight)
    check_map[check_map >= 0.0] = 1.0
    for i in range(0,np.shape(_map)[0]):
        for j in range(0,np.shape(_map)[0]):
            if np.sum(_map.weight[i,j,:,:]) == 0.0:
                check_map[i,j] = 0.0
                continue
            evals = np.abs(np.linalg.eigvalsh(_map.weight[i,j,:,:]))
            if np.min(evals)*threshold < np.max(evals):
                check_map[i,j] = 0.0
        print(i)

    return check_map

def checkAllBundleWeights(bundle_list):
    for f in bundle_list:
        _map = files.read(f)
        check_map = checkWeightsV3(_map)
        try:
            new_check_map *= check_map
        except:
            new_check_map = check_map

        print(f + " done")
    return new_check_map


def makeBundleBorderApod(_bun_dir, search_str = 'smap_set',
                         apod_type='cos', weight_threshold=0.3, radius_arcmin=30.,
                         zero_border_arcmin=4., smooth_weights_arcmin=10.,
                         threshold_median=False, verbose=False):
    '''
    This function looks through all bundles in a directory and creates a single
       appropriate apodization mask for all of the bundles. The one apod mask
       it returns is an intersection of all the thresholded bundle weight maps
       with the proper smoothing applied after the intersection has been
       completed.
    INPUTS
        _bun_dir: (string) The directory of the bundles that a apod mask is being made for.
        search_str ['smap_set']: (string) A string that every bundle must contain to be included
            in the apod creation process. This will almost always be 'smap_set'
        apod_type ['gaus']: (string) What type of apodization?
        weight_threshold [0.3]: (float) In fractions of the maximum weight. Set the
            mask to 0 when the weight is below this fraction of the median (or maximum) weight.
        radius_arcmin [30.]: (float) In arcminutes. Use this as the width of the function
            with which we're convolving the mask, if any.
        zero_border_arcmin [4.]: (float) In arcminutes. If nonzero, pad the border of the mask
            with zeros out to this radius.
        smooth_weights_arcmin [10.]: (float) In arcminutes. If nonzero, smooth the input weights
            with a Gaussian having this sigma before thresholding.
        threshold_median [False]: (bool) If True, threshold the weight based on the median
            of the temperature weights. If False, use the maximum. I believe this needs to be
            set to False since the thesholded weight maps will only have 2 values, 0 & 1.
        verbose [True]: (bool) Extra screen output.
    OUTPUT
        The 2D apodization mask to be used for every bundle.
    CHANGES
        21 July 2014: Swap the order of smoothing: Smooth individual bundles, not final product. SH
        25 Sep 2014: small fix to make work with T only maps. TJN
    '''
    # check and make sure the given dir include a '/' at the end
    #    if it doesn't, add one
    if not _bun_dir.endswith('/'):
            _bun_dir+='/'

    # Get all the bundles in a directory with search_str
    if search_str:
        filenames = glob(os.path.join(_bun_dir,search_str + '*.h5'))
    # get all bundles in a directory
    else:
        filenames = glob(os.path.join(_bun_dir,'*.h5'))

    filenames.sort()

    if verbose:
        print(filenames)

    for _num,_file in enumerate(filenames):
        if verbose:
            print 'Doing bundle %i of %i' %(_num,len(filenames))
        bun_now = files.read(_file)

        now_apod = makeBorderApodization(bun_now, weight_threshold = weight_threshold,
                                         radius_arcmin = 0., zero_border_arcmin = 0.,
                                         smooth_weights_arcmin = smooth_weights_arcmin,
                                         threshold_median = True,
                                         verbose = False)
        if _num == 0:
            total_apod = now_apod
        else:
            total_apod = total_apod*now_apod

    total_apod = makeBorderApodization(total_apod, weight_threshold = weight_threshold,
                                       radius_arcmin = radius_arcmin,
                                       zero_border_arcmin = zero_border_arcmin,
                                       smooth_weights_arcmin = 0.,
                                       threshold_median = threshold_median,
                                       verbose = verbose, given_weight_map = True,
                                       given_reso_arcmin = bun_now.getTOnly().reso_arcmin)


    return total_apod


def makeBorderApodization(Map, apod_type='cos', reso_arcmin = 0.25, weight_threshold=0.3, radius_arcmin=20.,
                          zero_border_arcmin=0., smooth_weights_arcmin=10.,
                          threshold_median=True, verbose=True, given_weight_map = False,
                          given_reso_arcmin = None, apod_threshold=1e-8):
    """
    Takes an input Map or PolarizedMap, looks at the temperature weights, and makes an
    apodization mask based on the weights.
    INPUTS
        _map: (Map or PolarizedMap) Only the TT weights will be used from a PolarizedMap.
        apod_type ['gaus']: (string) What type of apodization?
        weight_threshold [0.3]: (float) In fractions of the maximum weight. Set the
            mask to 0 when the weight is below this fraction of the median (or maximum) weight.
        radius_arcmin [20.]: (float) In arcminutes. Use this as the width of the function
            with which we're convolving the mask, if any.
        zero_border_arcmin [0.]: (float) In arcminutes. If nonzero, pad the border of the mask
            with zeros out to this radius.
        smooth_weights_arcmin [10.]: (float) In arcminutes. If nonzero, smooth the input weights
            with a Gaussian having this sigma before thresholding.
        threshold_median [True]: (bool) If True, threshold the weight based on the median
            of the temperature weights. If False, use the maximum.
        verbose [True]: (bool) Extra screen output.
        given_weight_map [False]: (bool) If True the function will expect to be given a weight
            map to create an apodization mask for instead of a polarized map
        given_reso_arcmin [None]: (float) The reso_arcmin for a given weight map if given_weight_map
            is True
         apod_threshold [1e-8] Make the apodization mask below this threshold 0.0,
            and make pixels in the mask this close to 1.0 exactly equal to 1.0.
            We don't want small mask values later being multiplied by crazy-high unweighted map pixels,
            which started happening with rounding issues when we switched to fftconvolve.
            Experimentally for one particular mask, the difference between fft and convolution mask
            is less than 2e-15 everywhere, but the the lowest non-zero number was 1e-6, so 1e-8 is conservative.
    OUTPUT
        The 2D apodization mask.
    EXCEPTIONS
        NotImplementedError if the user tries to create a mask with the "sin" method.
    CHANGES
        15 Oct 2013: Don't call "getTOnly" if the input is a T map. This avoids making an unnecessary copy. SH
    """

    # Create the mask, and start by masking pixels with too low a weight.
    t_map = Map
    mask = np.ones(t_map.shape, dtype=np.float)

    # If we're doing a cosine apodization, adjust the border zero padding so that the smearing
    # doesn't extend the mask into the region which is supposed to be masked out.
    if apod_type.lower().startswith('cos') and (radius_arcmin > 0):
        if verbose: print "Increasing zero_border_arcmin by %f so that the cosine smearing doesn't put us outside the boundaries." % (radius_arcmin/2.)
        zero_border_arcmin += radius_arcmin/2.

    # Add extra zeros around the border.
    if zero_border_arcmin > 0:
        if verbose: print "Padding the border with zeros to a depth of %f arcminutes (%d pixels)." % (zero_border_arcmin, int(zero_border_arcmin / reso_arcmin))
        distance_transform = ndimage.distance_transform_edt(mask, sampling=reso_arcmin)
        mask[distance_transform <= zero_border_arcmin] = 0

    # Smear the mask.
    if apod_type.lower().startswith('gaus'):
        # Smear by a Gaussian.
        if verbose: print "Smearing the mask with a Gaussian having sigma = %f arcminutes." % radius_arcmin
        mask = ndimage.gaussian_filter(mask, radius_arcmin / reso_arcmin / (2*np.sqrt(2*np.log(2)))) # Not certain we need the normalization.
    elif apod_type.lower().startswith('cos') and (radius_arcmin > 0):
        # Use a cosine function
        npix_cos = int(radius_arcmin / reso_arcmin)

        if verbose: print "Applying cosine apodization with width %f arcminutes (%d pixels). (2D convolution version)" % (radius_arcmin, npix_cos)
        # First generate a convolution kernel based on the requested radius_arcmin.
        # Start by calculating the distance of each point from the center, then
        # apply the cosine function.
        xs, ys    = np.meshgrid( np.linspace(-1., 1., npix_cos+1), np.linspace(-1., 1., npix_cos+1) )
        kern_cos = np.sqrt(xs**2 + ys**2)
        outside_circle = kern_cos>1 # Set these to zero in a moment.
        #kern_cos = np.cos(kern_cos*np.pi/2) # Wrong!
        kern_cos = 0.5 - 0.5*np.cos( (1.-kern_cos)*np.pi)
        kern_cos[outside_circle] = 0.

        # Smear the mask by the just-constructed cosine function.
        # BUT DO THE FFTCONVOLVE IF YOU DON'T WANT TO USE 100GB TO MAKE A 3 DEG WIDE, 1 ARCMIN RES MASK ON A 500d MAP!
        # But see note about apod_threshold above.
        # mask = ndimage.convolve(mask, kern_cos)  # old slow memory-intensive convolution, but gives exactly 0's and 1's.
        mask = sp.signal.fftconvolve(mask,kern_cos,'same')

        # Normalize the mask.
        mask /= mask.max()
    elif apod_type.lower().startswith('fastcos') and (radius_arcmin > 0):
        # Use a cosine function - this way is faster than the convolution, but less smooth.
        # Not advised, unless
        if verbose: print "Applying cosine apodization with width %f arcminutes. (Quick and dirty version.)" % radius_arcmin
        distance_transform = ndimage.distance_transform_edt(mask, sampling=reso_arcmin)
        wh_apply_mask = (distance_transform <= radius_arcmin)
        mask[wh_apply_mask] = 0.5 - 0.5*np.cos(distance_transform[wh_apply_mask]/radius_arcmin*np.pi)
    elif apod_type.lower().startswith('sin') and (radius_arcmin > 0):
        # 1 / (sin(theta) + epsilon) : See eq 36 of astro-ph/0511629v2
        raise NotImplementedError()
    else:
        if verbose: print "Tophat mask (no smearing)."

    if apod_threshold != 0.0:    #Nov 15 edit to try and deal with FFT apodization mask artifacts LB and JG
        mask[np.abs(mask)    < apod_threshold] = 0.0
        mask[np.abs(mask-1.) < apod_threshold] = 1.0
        # now the fftconvolve mask is identical to the old ndimage.convolve mask everywhere except the transition, where it's only off by a part in 10^15

    return mask

##########################################################################################
def makeApodizedPointSourceMask(Map, source_mask, reso_arcmin=0.25, center=np.asarray([352.50226, -55.000390]),ftype = 'tom',flux_thresh=None,apod_type='cos', radius_arcmin=20.,
                                zero_border_arcmin=0., verbose=True,
                                fixed_rad_deg=None, apod_threshold=1e-8):
    """
    this makes a smoothed point source mask
   INPUTS
          source_mask: (str) a pickle file with a source mask or a text file
          with a point source list
        source_mask [None]: If you give this option a file location containing a point source
            mask it will produce a mask with sources masked as well as the edges apodized.
            you can also give it a pointsource config file but this will take longer
        flux_thresh [None]:  The flux threshold (in mJy) above which to include sources.  This will only
                           work if the source file has a fifth column containing fluxes.  Otherwise it does nothing.
        center: [None]:   By default, this is pulled from the _map object.  If it doesn't have a center, supply a 1x2 numpy array here.
        fixed_rad_deg [None]:  This gets passed to powspec.make_masks_makePtSrcMask() to change the default radius around
                              each point source (which should be in the config file you sent)
        apod_threshold [1e-8]: As with makeBorderApodization, the fftconvolve() function can yield small machine-rounding errors
                              on masked pixels, so we use this threshold to set smaller values to identically 0.
    OUTPUT
         mask, a smoothed point source mask
    """
    t_map = Map

    # try:
    #     mask = pickle.load(open(source_mask,'r'))
    # except KeyError:
    if source_mask != None:
        try:
            point_source_config = source_mask
            if ftype == 'tom':
                mask = make_masks_amy.makePtSrcMask(point_source_config, t_map, center=center, reso_arcmin= reso_arcmin, flux_thresh=flux_thresh, fixed_rad_deg=fixed_rad_deg)
            elif ftype == 'wendy':
                mask = make_masks_amy.makePtSrcMask(point_source_config, t_map, center=center, reso_arcmin= reso_arcmin, flux_thresh=flux_thresh, fixed_rad_deg=fixed_rad_deg)
            elif ftype == 'herschel':
                mask = make_masks_amy.makePtSrcMask_herschel(point_source_config, t_map, center=center, reso_arcmin= reso_arcmin, flux_thresh=flux_thresh, fixed_rad_deg=fixed_rad_deg,proj=0)
                edgemask = np.ones((mask.shape))
                wherezero = (t_map==0)
                edgemask[wherezero] = 0
            elif ftype == 'herschel_nan':
                mask = make_masks_amy.makePtSrcMask_herschel(point_source_config, t_map, center=center, reso_arcmin= reso_arcmin, flux_thresh=flux_thresh, fixed_rad_deg=fixed_rad_deg,proj=5)
                edgemask = np.ones((mask.shape))
                wherenan = np.isnan(t_map)
                edgemask[wherenan] = 0
        except:
            print('Did not get the right inputs')
    else:
        mask = np.ones((t_map.shape))
        if ftype == 'herschel_nan': 
            edgemask = np.ones((mask.shape))
            wherenan = np.isnan(t_map)
            edgemask[wherenan] = 0
        elif ftype == 'herschel':
            edgemask = np.ones((mask.shape))
            wherezero = (t_map==0)
            edgemask[wherezero] = 0

    # This is the same thing that is used to smooth the borders in makeApodizationMask
    # If we're doing a cosine apodization, adjust the border zero padding so that the smearing
    # doesn't extend the mask into the region which is supposed to be masked out.
    if apod_type.lower().startswith('cos') and (radius_arcmin > 0):
        if verbose: print "Increasing zero_border_arcmin by %f so that the cosine smearing doesn't put us outside the boundaries." % (radius_arcmin/2.)
        zero_border_arcmin += radius_arcmin/2.

    # Add extra zeros around the border.
    if zero_border_arcmin > 0:
        if verbose: print "Padding the border with zeros to a depth of %f arcminutes (%d pixels)." % (zero_border_arcmin, int(zero_border_arcmin / reso_arcmin))
        distance_transform = ndimage.distance_transform_edt(mask, sampling=reso_arcmin)
        mask[distance_transform <= zero_border_arcmin] = 0

    # Smear the mask.
    if apod_type.lower().startswith('gaus'):
        # Smear by a Gaussian.
        if verbose: print "Smearing the mask with a Gaussian having sigma = %f arcminutes." % radius_arcmin
        mask = ndimage.gaussian_filter(mask, radius_arcmin / reso_arcmin / (2*np.sqrt(2*np.log(2)))) # Not certain we need the normalization.
    elif apod_type.lower().startswith('cos') and (radius_arcmin > 0):
        # Use a cosine function
        npix_cos = int(radius_arcmin / reso_arcmin)

        if verbose: print "Applying cosine apodization with width %f arcminutes (%d pixels). (2D convolution version)" % (radius_arcmin, npix_cos)
        # First generate a convolution kernel based on the requested radius_arcmin.
        # Start by calculating the distance of each point from the center, then
        # apply the cosine function.
        xs, ys    = np.meshgrid( np.linspace(-1., 1., npix_cos+1), np.linspace(-1., 1., npix_cos+1) )
        kern_cos = np.sqrt(xs**2 + ys**2)
        outside_circle = kern_cos>1 # Set these to zero in a moment.
        #kern_cos = np.cos(kern_cos*np.pi/2)
        kern_cos = 0.5 - 0.5*np.cos( (1.-kern_cos)*np.pi)
        kern_cos[outside_circle] = 0.

        # Smear the mask by the just-constructed cosine function.
        # mask = ndimage.convolve(mask, kern_cos)
        mask = sp.signal.fftconvolve(mask,kern_cos,'same')

        if apod_threshold != 0.0:    #Nov 15 edit to try and deal with FFT apodization mask artifacts LB and JG
            mask[np.abs(mask)    < apod_threshold] = 0.0
            mask[np.abs(mask-1.) < apod_threshold] = 1.0
        # Normalize the mask.
        mask /= mask.max()

        if ftype == 'herschel_nan' or ftype=='herschel':
            border_npix_cos = npix_cos*2.5
            xs, ys    = np.meshgrid( np.linspace(-1., 1., border_npix_cos+1), np.linspace(-1., 1., border_npix_cos+1) )
            kern_cos = np.sqrt(xs**2 + ys**2)
            outside_circle = kern_cos>1 # Set these to zero in a moment.
            #kern_cos = np.cos(kern_cos*np.pi/2)
            kern_cos = 0.5 - 0.5*np.cos( (1.-kern_cos)*np.pi)
            kern_cos[outside_circle] = 0.

            # Smear the mask by the just-constructed cosine function.
            # mask = ndimage.convolve(mask, kern_cos)
            edgemask = sp.signal.fftconvolve(edgemask,kern_cos,'same')

            if apod_threshold != 0.0:    #Nov 15 edit to try and deal with FFT apodization mask artifacts LB and JG
                edgemask[np.abs(edgemask)    < apod_threshold] = 0.0
                edgemask[np.abs(edgemask-1.) < apod_threshold] = 1.0
            # Normalize the mask.
            edgemask /= edgemask.max()
            mask = np.multiply(mask, edgemask)

    elif apod_type.lower().startswith('fastcos') and (radius_arcmin > 0):
        # Use a cosine function - this way is faster than the convolution, but less smooth.
        # Not advised, unless
        if verbose: print "Applying cosine apodization with width %f arcminutes. (Quick and dirty version.)" % radius_arcmin
        distance_transform = ndimage.distance_transform_edt(mask, sampling=reso_arcmin)
        wh_apply_mask = (distance_transform <= radius_arcmin)
        mask[wh_apply_mask] = 0.5 - 0.5*np.cos(distance_transform[wh_apply_mask]/radius_arcmin*np.pi)
    elif apod_type.lower().startswith('sin') and (radius_arcmin > 0):
        # 1 / (sin(theta) + epsilon) : See eq 36 of astro-ph/0511629v2
        raise NotImplementedError()
    else:
        if verbose: print "Tophat mask (no smearing)."

    return mask

def makePointSourceMask(Map, source_mask, reso_arcmin=0.25, center=np.asarray([352.50226, -55.000390]),ftype = 'tom',flux_thresh=None,apod_type='cos', radius_arcmin=20.,
                                zero_border_arcmin=0., verbose=True,
                                fixed_rad_deg=None, apod_threshold=1e-8):
    """
    this makes a smoothed point source mask
   INPUTS
          source_mask: (str) a pickle file with a source mask or a text file
          with a point source list
        source_mask [None]: If you give this option a file location containing a point source
            mask it will produce a mask with sources masked as well as the edges apodized.
            you can also give it a pointsource config file but this will take longer
        flux_thresh [None]:  The flux threshold (in mJy) above which to include sources.  This will only
                           work if the source file has a fifth column containing fluxes.  Otherwise it does nothing.
        center: [None]:   By default, this is pulled from the _map object.  If it doesn't have a center, supply a 1x2 numpy array here.
        fixed_rad_deg [None]:  This gets passed to powspec.make_masks_makePtSrcMask() to change the default radius around
                              each point source (which should be in the config file you sent)
        apod_threshold [1e-8]: As with makeBorderApodization, the fftconvolve() function can yield small machine-rounding errors
                              on masked pixels, so we use this threshold to set smaller values to identically 0.
    OUTPUT
         mask, a smoothed point source mask
    """
    t_map = Map

    # try:
    #     mask = pickle.load(open(source_mask,'r'))
    # except KeyError:
    if source_mask != None:
        try:
            point_source_config = source_mask
            if ftype == 'tom':
                mask = make_masks_amy.makePtSrcMask(point_source_config, t_map, center=center, reso_arcmin= reso_arcmin, flux_thresh=flux_thresh, fixed_rad_deg=fixed_rad_deg)
            elif ftype == 'wendy':
                mask = make_masks_amy.makePtSrcMask(point_source_config, t_map, center=center, reso_arcmin= reso_arcmin, flux_thresh=flux_thresh, fixed_rad_deg=fixed_rad_deg)
            elif ftype == 'herschel_nan':
                mask = make_masks_amy.makePtSrcMask_herschel(point_source_config, t_map, center=center, reso_arcmin= reso_arcmin, flux_thresh=flux_thresh, fixed_rad_deg=fixed_rad_deg,proj=5)
                edgemask = np.ones((mask.shape))
                wherenan = np.isnan(t_map)
                edgemask[wherenan] = 0
            elif ftype == 'herschel':
                mask = make_masks_amy.makePtSrcMask_herschel(point_source_config, t_map, center=center, reso_arcmin= reso_arcmin, flux_thresh=flux_thresh, fixed_rad_deg=fixed_rad_deg,proj=0)
                edgemask = np.ones((mask.shape))
                wherezero = (t_map==0)
                edgemask[wherezero] = 0
        except:
            print('Did not get the right inputs')
    else:
        mask = np.ones((t_map.shape))
        if ftype == 'herschel_nan': 
            edgemask = np.ones((mask.shape))
            wherenan = np.isnan(t_map)
            edgemask[wherenan] = 0

    # This is the same thing that is used to smooth the borders in makeApodizationMask
    # If we're doing a cosine apodization, adjust the border zero padding so that the smearing
    # doesn't extend the mask into the region which is supposed to be masked out.
    # if apod_type.lower().startswith('cos') and (radius_arcmin > 0):
    #     if verbose: print "Increasing zero_border_arcmin by %f so that the cosine smearing doesn't put us outside the boundaries." % (radius_arcmin/2.)
    #     zero_border_arcmin += radius_arcmin/2.

    # # Add extra zeros around the border.
    # if zero_border_arcmin > 0:
    #     if verbose: print "Padding the border with zeros to a depth of %f arcminutes (%d pixels)." % (zero_border_arcmin, int(zero_border_arcmin / reso_arcmin))
    #     distance_transform = ndimage.distance_transform_edt(mask, sampling=reso_arcmin)
    #     mask[distance_transform <= zero_border_arcmin] = 0

    # Smear the mask.
    if apod_type.lower().startswith('gaus'):
        # Smear by a Gaussian.
        if verbose: print "Smearing the mask with a Gaussian having sigma = %f arcminutes." % radius_arcmin
        mask = ndimage.gaussian_filter(mask, radius_arcmin / reso_arcmin / (2*np.sqrt(2*np.log(2)))) # Not certain we need the normalization.
    elif apod_type.lower().startswith('cos') and (radius_arcmin > 0):
        # Use a cosine function
        npix_cos = int(radius_arcmin / reso_arcmin)

        if verbose: print "Applying cosine apodization with width %f arcminutes (%d pixels). (2D convolution version)" % (radius_arcmin, npix_cos)
        # First generate a convolution kernel based on the requested radius_arcmin.
        # Start by calculating the distance of each point from the center, then
        # apply the cosine function.
        xs, ys    = np.meshgrid( np.linspace(-1., 1., npix_cos+1), np.linspace(-1., 1., npix_cos+1) )
        kern_cos = np.sqrt(xs**2 + ys**2)
        outside_circle = kern_cos>1 # Set these to zero in a moment.
        #kern_cos = np.cos(kern_cos*np.pi/2)
        kern_cos = 0.5 - 0.5*np.cos( (1.-kern_cos)*np.pi)
        kern_cos[outside_circle] = 0.

        # Smear the mask by the just-constructed cosine function.
        # mask = ndimage.convolve(mask, kern_cos)
        mask = sp.signal.fftconvolve(mask,kern_cos,'same')

        if apod_threshold != 0.0:    #Nov 15 edit to try and deal with FFT apodization mask artifacts LB and JG
            mask[np.abs(mask)    < apod_threshold] = 0.0
            mask[np.abs(mask-1.) < apod_threshold] = 1.0
        # Normalize the mask.
        mask /= mask.max()

        if ftype == 'herschel':
            border_npix_cos = npix_cos*2.5
            xs, ys    = np.meshgrid( np.linspace(-1., 1., border_npix_cos+1), np.linspace(-1., 1., border_npix_cos+1) )
            kern_cos = np.sqrt(xs**2 + ys**2)
            outside_circle = kern_cos>1 # Set these to zero in a moment.
            #kern_cos = np.cos(kern_cos*np.pi/2)
            kern_cos = 0.5 - 0.5*np.cos( (1.-kern_cos)*np.pi)
            kern_cos[outside_circle] = 0.

            # Smear the mask by the just-constructed cosine function.
            # mask = ndimage.convolve(mask, kern_cos)
            edgemask = sp.signal.fftconvolve(edgemask,kern_cos,'same')

            if apod_threshold != 0.0:    #Nov 15 edit to try and deal with FFT apodization mask artifacts LB and JG
                edgemask[np.abs(edgemask)    < apod_threshold] = 0.0
                edgemask[np.abs(edgemask-1.) < apod_threshold] = 1.0
            # Normalize the mask.
            edgemask /= edgemask.max()
            mask = np.multiply(mask, edgemask)

    elif apod_type.lower().startswith('fastcos') and (radius_arcmin > 0):
        # Use a cosine function - this way is faster than the convolution, but less smooth.
        # Not advised, unless
        if verbose: print "Applying cosine apodization with width %f arcminutes. (Quick and dirty version.)" % radius_arcmin
        distance_transform = ndimage.distance_transform_edt(mask, sampling=reso_arcmin)
        wh_apply_mask = (distance_transform <= radius_arcmin)
        mask[wh_apply_mask] = 0.5 - 0.5*np.cos(distance_transform[wh_apply_mask]/radius_arcmin*np.pi)
    elif apod_type.lower().startswith('sin') and (radius_arcmin > 0):
        # 1 / (sin(theta) + epsilon) : See eq 36 of astro-ph/0511629v2
        raise NotImplementedError()
    else:
        if verbose: print "Tophat mask (no smearing)."

    return mask

##########################################################################################
def makeUVMask(_map=None, apod_mask=None, fft_shape=None, reso_arcmin=None,
               smooth_pixels=4, do_polarization=True,
               threshold_max_fraction=0.3, zero_above_threshold=False,
               ellx_min=0, elly_min=0, ellxy_box=None,
               ell_min=None, ell_max=None, square_mask=False,
               plot=False):
    """
    Creates a k-space mask from an input map. For best results, input a difference
    (noise-only) map! The k-space mask is the inverse of the smoothed FT power in the map,
    for k-space pixels in which the smoothed FT power exceeds a threshold. The threshold
    is a fraction of the maximum smoothed FT power. Any pixels with power below
    the threshold will have a mask value of 1. The mask is adjusted so that
    values range from 0 to 1.
    You may also specify set regions of k-space to set to zero (if, for example,
    timestream filtering removes all modes in a given region).
    INPUTS
        _map [None]: (Map or PolarizedMap) We'll create a k-space mask based on the power in
            this map. If we aren't given a map, then this function will just make boxes
            as specified by the ell* arguments.
        apod_mask [None]: (2D array) An array with size equal to the input _map. We'll multiply
            the map by this mask before taking the FFT. If you input a map, you really
            should input a mask as well.
        fft_shape [None]: (2-tuple) If not None, then zero-pad the map to this shape
            before taking the FFT.
        reso_arcmin [None]: (float) Only used if we don't have a map as input. This
            specifies the map resolution (in arcminutes) corresponding to mask we
            wish to create.
        smooth_pixels [4]: (int) Smooth the FFT power by a Gaussian kernel with this
            standard deviation (in pixels) before using it to create a mask.
        do_polarization [True]: (bool) Only applies if _map is a PolarizedMap. If so,
            then create a k-space mask independently for each of the Q, U maps, then
            create a combined mask by taking the minimum of the two values in each pixel.
            If do_polarization is False and the input is a PolarizedMap, create
            the k-space mask based only on the temperature map.
        threshold_max_fraction [0.3]: (float) As a fraction of the maximum (smoothed)
            FT power in the map, pixels with larger power will receive a mask value
            equal to the inverse power, and pixels with smaller power will receive
            a mask value of 1.
                Set to zero to skip using power in the map FFT to define a mask.
        zero_above_threshold [False]: (bool) If True, set k-space weight to zero when the
            FT power is too large, instead of weighting.
        ellx_min [0.]: (float) Regardless of FT power in the pixel, set the mask value to
            zero for any pixels at ell_x smaller than this value.
        elly_min [0.]: (float) Regardless of FT power in the pixel, set the mask value to
            zero for any pixels at ell_y smaller than this value.
        ellxy_box [None]: (list of 4-tuples) A list of (min ell_x, max ell_x, min ell_y, max_ell_y)
            which defines regions of k-space we want to mask out. We'll set the
            final k-space mask to zero for any values of ell_x, ell_y which fall inside
            the specified region.
        ell_min [None]: (float) Regardless of FT power in the pixel, set the mask value to
            zero for any pixels at |ell| smaller than this value.
        ell_max [None]: (float) Regardless of FT power in the pixel, set the mask value to
            zero for any pixels at |ell| larger than this value.
        square_mask [False]: (bool) If True, square the k-space mask. This will downweight
            noisy regions of the map even more strongly.
        plot [False]: (bool) Plot the resulting mask on the screen.
    OUTPUT
        A 2D array of shape fft_shape, in standard FFT ordering. Apply this mask by
        multiplying it into the result of np.fft.fft2(maparray).
        (Or just pass it to any of the other analysis.maps functions as an input.)
    EXCEPTIONS
        ValueError if a requested fft_shape is smaller than the map shape.
    AUTHOR
        Stephen Hoover, 27 December 2013
    CHANGES
        17 Feb 2014: Add zero_above_threshold argument. SH
        25 Mar 2014: Add ell_min and ell_max arguments. SH
        21 Jul 2014: Modify so that this function can be run with no input map. SH
    """

    if _map is not None and isinstance(_map, sky.PolarizedMap):
        ########################################################################
        # If we got a PolarizedMap as input, we need to pick which map(s) to
        # look at when making the k-space mask.
        ########################################################################

        # Record the input arguments so we can pass them through again.
        myargs = tools.myArguments(combine_posargs=True)
        del myargs['_map']
        del myargs['plot']

        if do_polarization:
            # If we're making a k-space mask for polarization, do a separate
            # mask for Q and U, then combine them.
            k_mask_q = makeUVMask(_map['Q'], plot=False, **myargs)
            k_mask_u = makeUVMask(_map['U'], plot=False, **myargs)
            k_mask = np.min(np.dstack((k_mask_q, k_mask_u)), axis=2)
        else:
            # Else only look at the temperature map.
            k_mask = makeUVMask(_map.getTOnly(), **myargs)

    else:
        ########################################################################
        # We have a single Map object, so use the k-space power in that Map
        # to create a mask.
        ########################################################################

        if _map is not None:
            # First check any requested padding to be sure it's not too small.
            if fft_shape is not None and (np.asarray(_map.shape) > np.asarray(fft_shape)).any():
                raise ValueError("You've requested a mask shape of %s, but the input map shape is %s!"
                                 % (str(fft_shape), str(_map.shape)))

            # If we didn't get a request for padding, make the mask the same size as the map.
            if fft_shape is None:
                fft_shape = _map.shape

            reso_rad = _map.reso_rad

            # Now create the k-space mask.
            k_mask = np.ones(fft_shape)
            if threshold_max_fraction > 0:
                # Take the FFT of the map. We care most about the region with k near zero, so
                # the "fftshift" step makes sure we don't need to worry about ambiguity
                # at the array borders.
                map_fft = np.fft.fft2(_map.map*apod_mask, s=fft_shape)*_map.reso_rad**2
                map_fft = np.fft.fftshift((map_fft * map_fft.conj()).real)

                # Smooth the FFT and set the threshold for mask creation.
                smoothed_fft = sp.ndimage.gaussian_filter(map_fft, smooth_pixels, mode='wrap')
                threshold = threshold_max_fraction*smoothed_fft.max()

                # Set the k-space mask based on FFT power.
                if zero_above_threshold:
                    k_mask[ smoothed_fft > threshold ] = 0.
                else:
                    k_mask /= threshold
                    k_mask[ smoothed_fft > threshold ] = (1/smoothed_fft)[ smoothed_fft > threshold ]
                    k_mask /= k_mask.max()

                # Shift the mask back to FFT ordering.
                k_mask = np.fft.fftshift(k_mask)
        else:
            # If we don't have an input map, then the "mask" at this point is all ones.
            if fft_shape is None:
                raise ValueError("Without an input map, you must specify the mask shape!")
            if reso_arcmin is None:
                raise ValueError("Without an input map, you must specify a resolution!")
            reso_rad = np.deg2rad(reso_arcmin/60.)
            k_mask = np.ones(fft_shape)

        # Mask k-space regions based on hard ell cuts, if requested.
        ell_x, ell_y = np.meshgrid(2*np.pi*np.fft.fftfreq(fft_shape[1], reso_rad),
                                   2*np.pi*np.fft.fftfreq(fft_shape[0], reso_rad))
        ellgrid = np.sqrt(ell_x**2 + ell_y**2)
        k_mask[np.abs(ell_x)<ellx_min] = 0.
        k_mask[np.abs(ell_y)<elly_min] = 0.
        if ell_min:
            k_mask[ellgrid < ell_min] = 0.
        if ell_max:
            k_mask[ellgrid > ell_max] = 0.
        if ellxy_box:
            for min_ell_x, max_ell_x, min_ell_y, max_ell_y in ellxy_box:
                k_mask[(ell_x>min_ell_x) & (ell_x<max_ell_x) &
                       (ell_y>min_ell_y) & (ell_y<max_ell_y)] = 0.

        # If requested, square the mask.
        if square_mask:
            k_mask = k_mask**2

    # Display the results!
    if plot:
        # Calculate the extent of the k-space mask in terms of ell.
        ell_x = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(k_mask.shape[1], reso_rad))
        ell_y = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(k_mask.shape[0], reso_rad))

        # Import a couple of needed modules.
        from sptpol_software.analysis.plotting import spt_imshow
        from matplotlib import pyplot as plt

        # Plot the mask and label the plot.
        spt_imshow(np.fft.fftshift(k_mask), extent=[ell_x[0], ell_x[-1], ell_y[0], ell_y[-1]],
                   title="K-space mask")
        plt.xlabel("$\ell_x$", fontsize='large')
        plt.ylabel("$\ell_y$", fontsize='large')

    return k_mask

###################################################################
def makeKendrickWindow(shape, reso_arcmin, apod_cos_width=None, width_of_zeros=0.05):
    """
    This function will generate a cicular apodization mask and its first and
    second partial derivatives. The edge of the mask can be apodized either with
    a cosine rolldown of specified width, or a hanning window over the entire window.
    The derivatives of the cosine mask all go to zero at the edge of the mask,
    but this is not the case with the hanning window.
    The returned window will be the largest circle which can fit into
    a region with the input shape.
    Translated from the IDL make_kendrick_window.pro.
    INPUTS
        shape: (2-element list or array) The shape in pixels of the desired mask.
            The function will accept non-square shapes, but the output mask will
            still be circular!
        reso_arcmin: (float) The size of each pixel, in arcminutes.
        apod_cos_width [None]: (float) In arcminutes, this will be the width
            of the cosine rolldown on the border of the apodization mask.
            If not provided, we'll use a hanning window.
        width_of_zeros [0.05]: (float) As a fraction of the maximum radius,
            what width of zeros should we leave around the edge of the mask?
    OUTPUT
        A dictionary with the apodization mask and its first and second derivatives.
        The dictionary has the keys:
        ['W', 'dW/dx', 'dW/dy', 'd2W/dx2', 'd2W/dy2', 'd2W/dx/dy']
    EXCEPTIONS
        None
    AUTHOR
        Stephen Hoover, 8 November 2013
        (Translated from make_kendrick_window.pro by ... Tom?)
    """

    shape = np.asarray(shape) # Enforce arrayness.

    # The output will contain the window and 5 partials: d/dx, d/dy, d^2/dx^2,
    # d^2/dy^2, and d^2/dx/dy
    windows = struct({'W':np.zeros(shape),
                      'dW/dx':np.zeros(shape),
                      'dW/dy':np.zeros(shape),
                      'd2W/dx2':np.zeros(shape),
                      'd2W/dy2':np.zeros(shape),
                      'd2W/dx/dy':np.zeros(shape)}) #np.zeros(np.append(shape, 6))

    #window = np.zeros( [shape[0], shape[1]] )
    reso_rad = reso_arcmin * constants.DTOR/60 # Grid resolution in radians
    radius_rad = np.min(shape)/2. * reso_rad # Distance from window center to edges, in radians. (Note: This is a 2-element array.)
    radius_rad *= (1-width_of_zeros) # Leave a small gap at the edge of the window.


    # This creates a grid of x-y coordinates for the window, centered at the window center, and
    # in units of radians.
    xy_grid = np.mgrid[-shape[0]/2.+0.5:shape[0]/2.-0.5:shape[0]*1j , -shape[1]/2.+0.5:shape[1]/2.-0.5:shape[1]*1j]*reso_rad
    dist_grid = sqrt( xy_grid[0]**2 + xy_grid[1]**2 )

    if apod_cos_width is not None:
        # If we're given a "start_radius", assume we want a cosine apodization window.

        # Assume r_star is in arcmin.
        # All distances are in radians, so convert.
        start_radius_rad = apod_cos_width/60.*constants.DTOR

        distarg = -np.pi*(dist_grid-radius_rad)/start_radius_rad
        cos_distarg = np.cos(distarg)
        sin_distarg = np.sin(distarg)

        # On the flat top, the window is one and all derivatives are zero.
        windows['W'][dist_grid < (radius_rad-start_radius_rad)] = 1.0

        # Calculate the window and its derivatives in the apodized region.
        rd = in_rolldown = ((radius_rad-start_radius_rad) < dist_grid) & (dist_grid < radius_rad)
        windows['W'][rd] = 0.5*(1-cos_distarg[rd])
        windows['dW/dx'][rd] = -0.5*sin_distarg[rd]*(np.pi/start_radius_rad)*(xy_grid[1][rd]/dist_grid[rd])
        windows['dW/dy'][rd] = -0.5*sin_distarg[rd]*(np.pi/start_radius_rad)*(xy_grid[0][rd]/dist_grid[rd])

        windows['d2W/dx2'][rd] = 0.5*( cos_distarg[rd]*((np.pi/start_radius_rad)*(xy_grid[1][rd]/dist_grid[rd]))**2
                                       - sin_distarg[rd]*(np.pi/start_radius_rad)*(1/dist_grid[rd] - xy_grid[1][rd]**2/dist_grid[rd]**3))
        windows['d2W/dy2'][rd] = 0.5*( cos_distarg[rd]*((np.pi/start_radius_rad)*(xy_grid[0][rd]/dist_grid[rd]))**2
                                       - sin_distarg[rd]*(np.pi/start_radius_rad)*(1/dist_grid[rd] - xy_grid[0][rd]**2/dist_grid[rd]**3))

        windows['d2W/dx/dy'][rd] = 0.5*(cos_distarg[rd]*( (np.pi/start_radius_rad)**2 * (xy_grid[1][rd]*xy_grid[0][rd]/dist_grid[rd]**2) )
                                        - sin_distarg[rd]*(np.pi/start_radius_rad)*(-xy_grid[1][rd]*xy_grid[0][rd]/dist_grid[rd]**3) )

    else:
        # Without a "apod_cos_width", make a hanning window.

        distarg = np.pi*dist_grid/radius_rad
        cos_distarg = np.cos(distarg)
        sin_distarg = np.sin(distarg)

        windows['W'] = 0.5 + 0.5*cos_distarg
        windows['dW/dx'] = -0.5*(sin_distarg * np.pi/radius_rad * xy_grid[1]/dist_grid)
        windows['dW/dy'] = -0.5*(sin_distarg * np.pi/radius_rad * xy_grid[0]/dist_grid)
        windows['d2W/dx2'] = -0.5*(cos_distarg * (np.pi/radius_rad * xy_grid[1]/dist_grid)**2
                                   + sin_distarg * np.pi/radius_rad * (-xy_grid[1]**2/dist_grid**3 + 1/dist_grid))
        windows['d2W/dy2'] = -0.5*(cos_distarg * (np.pi/radius_rad * xy_grid[0]/dist_grid)**2
                                   + sin_distarg * np.pi/radius_rad * (-xy_grid[0]**2/dist_grid**3 + 1/dist_grid))
        windows['d2W/dx/dy'] = -0.5*(cos_distarg * (np.pi/radius_rad)**2 * xy_grid[0]*xy_grid[1]/dist_grid**2
                                     - sin_distarg * np.pi/radius_rad * xy_grid[0]*xy_grid[1]/dist_grid**3)

        # Zero the region outside of the mask.
        outside_mask = dist_grid > radius_rad
        for window in windows.itervalues():
            window[outside_mask] = 0.

    return windows


##########################################################################################
def calculateNoise(_map, mask=None, uv_mask=None, ell_range=[5500.,6500.],
                   quiet=False, verbose=False, return_noise=False, maps_to_cross=['T','Q','U']):
    """
    Calculates map noise in units of uK*arcminutes. The noise is taken from the
    average value of the power spectrum in a range of ells high enough such that
    they should be noise dominated.
    INPUTS
        _map: (PolarizedMap or output of calculateCls) This may either be a single PolarizedMap,
            or else the output of MapAnalyzer.calculateCls. If it's a map, then we'll run
            calculateCls on it to get TT, QQ, and UU power spectra.
            If this is the output of calculateCls and it doesn't contain QQ or UU power spectra,
            then we can only calculate temperature noise. Run calculateCls
            with maps_to_cross=['T','Q','U'] to get QQ and UU power spectra.
        mask [None]: (ndarray apodization mask) Only used if _map is a PolarizedMap.
            Applied to map when finding Cls. If None, the mask will be the default output
            of makeBorderApodization.
        uv_mask [None]: (2D ndarray) Only used if _map is a PolarizedMap. We'll
            use this as a k-space mask when counting up the power in the map.
        ell_range [[5500., 6500.]]: (2-element iterable) The ell range over which we'll
            average Cls to determine the noise level.
        quiet [False]: (bool) If True, suppress all screen output.
        verbose [False]: (bool) If True, give extra screen output.
    OUTPUT
        Average power in the ell range specified (presumed to be noise) in units of uK*arcmin.
    CHANGES
        12 Nov 2013: Add save_ffts=False, recreate_ffts=True to the calculateCls call.
        21 Dec 2013: Add "uv_mask" option. SH
        27 Dec 2013: Adjust to use the uv_mask shape in case it's different from the input map. SH
        1 Oct 2014: Small change to work with T only maps. TJN
        1 Dec 2015: Another small tweak to deal with T only LB
    """
    if isinstance(_map, dict) and 'ell' in _map and 'TT' in _map:
        # If it looks like the input is already the power spectra, then we don't need to recalculate.
        clout = _map
    else:
        if not quiet:
            print "Calculating Cls from the input map."
        anal = MapAnalyzer()
        if mask is None: mask = makeBorderApodization(_map)
        # figure out which maps to cross: full pol, or T only
        # try:
        #     _map.pol_maps # A Map object will have this method.
        #     maps_to_cross = ['T','Q','U']
        # except AttributeError:
        #     # it will go here if we are dealing with a T only map
        #     maps_to_cross = ['T']


        clout = anal.calculateCls(_map, sky_window_func=mask, uv_mask=uv_mask, maps_to_cross=maps_to_cross,
                                  fft_shape=(uv_mask.shape if uv_mask is not None else None),
                                  quiet=(not verbose), save_ffts=False, recreate_ffts=False)

    # Find the noise in the temperature power spectrum.
    tt_noise = np.degrees(np.sqrt(np.mean(clout['TT'][(ell_range[0] < clout['ell']['TT'])
                                                      & (clout['ell']['TT'] < ell_range[1])])) * 60)*1e6


    if len(maps_to_cross)==1 and maps_to_cross[0] == 'T':
        if not quiet:
            print "Noise levels between ell %d and %d:\n   T: %.2f uK*arcmin" % (ell_range[0], ell_range[1], tt_noise)
        return tt_noise, -1, -1, clout  #LB adding

    # If we have Q and U polarized power spectra, then calculate the noise in those too.
    if 'QQ' in clout and 'UU' in clout:
        qq_noise = np.degrees(np.sqrt(np.mean(clout['QQ'][(ell_range[0] < clout['ell']['QQ'])
                                                          & (clout['ell']['QQ'] < ell_range[1])])) * 60)*1e6
        uu_noise = np.degrees(np.sqrt(np.mean(clout['UU'][(ell_range[0] < clout['ell']['UU'])
                                                          & (clout['ell']['UU'] < ell_range[1])])) * 60)*1e6

        if not quiet:
            print ("Noise levels between ell %d and %d:\n   T: %.2f uK*arcmin\n   Q: %.2f uK*arcmin\n   U: %.2f uK*arcmin"
                   % (ell_range[0], ell_range[1], tt_noise, qq_noise, uu_noise))
        #return tt_noise, qq_noise, uu_noise
    else:
        if not quiet:
            print "Noise levels between ell %d and %d:\n   T: %.2f uK*arcmin" % (ell_range[0], ell_range[1], tt_noise)
        #return tt_noise

    if 'EE' in clout:
        ee_noise = np.degrees(np.sqrt(np.mean(clout['EE'][(ell_range[0] < clout['ell']['EE'])
                                                          & (clout['ell']['EE'] < ell_range[1])])) * 60)*1e6
        if not quiet:
            print "Noise levels between ell %d and %d:\n   E: %.2f uK*arcmin" % (ell_range[0], ell_range[1], ee_noise)
    if 'BB' in clout:
        bb_noise = np.degrees(np.sqrt(np.mean(clout['BB'][(ell_range[0] < clout['ell']['BB'])
                                                          & (clout['ell']['BB'] < ell_range[1])])) * 60)*1e6
        if not quiet:
            print "Noise levels between ell %d and %d:\n   B: %.2f uK*arcmin" % (ell_range[0], ell_range[1], bb_noise)

    if return_noise:
        return tt_noise, qq_noise, uu_noise, clout

    return clout

##########################################################################################
def calculateChi(q_map, u_map, reso_rad, pixel_radius=2, which_chi='B',
                 apodization_window=None):
    """
    Calculates chi_B or chi_E (The B-mode (or E-mode) map, with each
    mode multiplied by sqrt((l-1)(l)(l+1)(l+2))), given an input PolarizedMap
    object with Q and U maps.
    This function should be treating any apodizations of the Q and U maps correctly,
    but the apodization map should be chosen such that both the apodization and
    its derivative are zero at the boundaries.
    Note that any pixel within pixel_radius pixels of the boundaries is definitely
    nonsense, and it might be necessary to remove a few more pixels to get rid
    of artifacts at the edges.
    We assume a flat sky. Under this assumption, we have
    chi_B = (d_x^2 - d_y^2)U - 2(d_xy^2)Q,
    chi_E = (d_x^2 - d_y^2)Q + 2(d_xy^2)U,
    where d denotes the partial derivative. We use the method of finite differences
    to calculate the partial derivatives. The coefficients for each term have been
    chosen so as to minimize E->B leakage.
    The basis of this method is taken from Smith & Zaldarriaga 2006 (arXiv:astro-ph/0610059v1),
    equation A4. The coefficients are given in Table II from that paper.
    The correction factor which compensates for the apodization is inspired by
    section II of Zhao & Baskaran 2010 (arXiv:1005.1201v3)
    INPUTS
       q_map: (2D ndarray) Q polarization data.
       u_map: (2D ndarray) U polarization data.
       reso_rad: (float) Width of a pixel in one of the above maps in radians.
       pixel_radius [2]: (int) How many pixels out should we look when taking the
          partial derivatives? Valid choices are 1, 2, or 3. pixel_radius=1 uses
          only neighboring pixels, pixel_radius=2 uses pixels up to 2 pixels away, etc.
          Using more pixels will reduce E->B leakage. A 2-pixel radius should
          be sufficient for any analysis with a realistic level of the B-mode
          gravitational lensing signal.
       which_chi ['B']: (string) Either 'B' to indicate that we should calculate chi_B,
          or 'E', to indicate that we should calculate chi_E.
       apodization_window [None]: (2D ndarray, same size as q_map and u_map) This array
          multiplies the q_map and u_map. Any effects of the weighting should be removed
          by the corrections factors at the end, but the weighting function must
          be chosen such that both the function and its derivative are zero at the borders.
    OUTPUT
       A Map with the calculated chi_B or chi_E. The input "map" is also modified to add
       this Map, as the polarization "chiB" (or "chiE").
       Note that the pixel values on the map's border (to a depth of pixel_radius
       pixels) will be nonsense.
    AUTHOR
        Stephen Hoover, March 2013
    CHANGES
        1 Nov 2013: Rename "calculateChi". If input mask is None, then don't do any calculations that involve it. SH
        11 Nov 2013: BUGFIX: Switch minus sign on the Q/U swap when calculating chiE. SH
    """
    if pixel_radius==3:
        # Leave this enabled for testing purposes right now.
        warnings.warn("Pixel radius 3 not yet working. The E and B mode maps will be incorrect!", RuntimeWarning)

    if which_chi!='B' and which_chi!='E':
        raise ValueError('You must pick either chi_B or chi_E! Please set which_chi="B" or which_chi="E".')

    # Pixel weights from the Smith & Zaldarriaga (2006) paper (Table 2).
    if pixel_radius==1:
        w = np.array([1., 1./2, 0, 0, 0, 0])
    elif pixel_radius==2:
        w = np.array([4./3, 2./3, -1./12, -1./24, 0, 0])
    elif pixel_radius==3:
        w = np.array([806./2625, 2243./7875, 1501./10500, -239./63000, 53./2625, 907./15750])
    else:
        raise ValueError('The pixel_radius input must be one of the integers 1, 2, and 3!')
    # Divide out the resolution
    w /= reso_rad**2

    if apodization_window is not None:
        window_normalization = np.sqrt(np.sum(apodization_window**2)/np.product(apodization_window.shape))
    else:
        window_normalization = None

    # Define a helper function to simplify the notation.
    # roll( arr, [m, n] ) shifts the x-axis (index 1) of the
    # input array by m pixels and shifts the y-axis (index 0)
    # of the input array by n pixels.
    #   Note that I flip the order of the indices! This is
    # so that I can type in the equation with the x-shift first,
    # but still operate on arrays in which the x-axis is last.
    def roll(array, shift):
        out = array
        if shift[0]:
            out = np.roll( out, shift[0], axis=1 )
        if shift[1]:
            out = np.roll( out, shift[1], axis=0 )
        return out

    # Alias the array part of the Q and U Map objects.
    # Note that in the case of chi_E, the "u" and "q"
    # variables are misnamed! Everything's correct for chi_B,
    # but the naming is wrong for chi_E. This was the easiest
    # way to reuse all of the shared code.
    if which_chi=='B':
        q_noweight = np.nan_to_num(q_map)
        u_noweight = np.nan_to_num(u_map)
        if apodization_window is not None:
            q = q_noweight*apodization_window/window_normalization
            u = u_noweight*apodization_window/window_normalization
        else:
            q, u = q_noweight, u_noweight
    elif which_chi=='E':
        u_noweight = np.nan_to_num(q_map)
        q_noweight = -1*np.nan_to_num(u_map)
        if apodization_window is not None:
            u = u_noweight*apodization_window/window_normalization
            q = q_noweight*apodization_window/window_normalization
        else:
            q, u = q_noweight, u_noweight
    else:
        raise ValueError('You must pick either chi_B or chi_E! Please set which_chi="B" or which_chi="E".')

    # Equation A4. We won't bother to add in any terms with a zero weight.
    chi=( w[0]*( roll(u,[-1,0]) - roll(u,[0,-1])
                 + roll(u,[+1,0]) - roll(u,[0,+1]) )  # Alternate +/- signs, changed from paper.
          -w[1]*( roll(q,[+1,+1]) + roll(q,[-1,-1])
                  - roll(q,[-1,+1]) - roll(q,[+1,-1]) ) )
    if w[2]: # Alternate +/- signs, changed from paper.
        chi += w[2]*( roll(u,[-2,0]) - roll(u,[0,-2])
                      + roll(u,[+2,0]) - roll(u,[0,+2]) )
    if w[3]:
        chi += -w[3]*( roll(q,[+2,+2]) + roll(q,[-2,-2])
                      - roll(q,[-2,+2]) - roll(q,[+2,-2]) )
    if w[4]:
        chi += w[4]*( roll(u,[+2,-1]) + roll(u,[+1,-2]) + roll(u,[-1,+2]) + roll(u,[-2,+1])
                      - roll(u,[+2,+1]) - roll(u,[+1,+2]) - roll(u,[-1,-2]) - roll(u,[-2,-1]) )
    if w[5]:
        chi += w[5]*( roll(q,[+2,+1]) + roll(q,[+2,-1]) + roll(q,[-2,+1]) + roll(q,[-2,-1])
                      - roll(q,[+1,+2]) - roll(q,[-1,-2]) - roll(q,[+1,-2]) - roll(q,[-1,+2]) )


    if apodization_window is not None:
        # We used a non-unity weight. Compensate for that by subtracting off the counterterms.
        win = apodization_window/window_normalization # Alias for convenience. Divide off the normalization factor.
        res = reso_rad # Alias for convenience
        correction_term = ( u_noweight*(d2dx2(win,res,range=pixel_radius) - d2dy2(win,res,range=pixel_radius))
                            + 2*ddx(win,res,range=pixel_radius)*(ddx(u_noweight,res,range=pixel_radius) - ddy(q_noweight,res,range=pixel_radius))
                            - 2*ddy(win,res,range=pixel_radius)*(ddy(u_noweight,res,range=pixel_radius) + ddx(q_noweight,res,range=pixel_radius))
                            - 2*d2dxdy(win,res,range=pixel_radius)*q_noweight )
        chi -= correction_term
        chi[win != 0] /= win[win != 0]

    ## Finally, zero the pixels on the borders. Those are nonsense.
    #threshold_weight = 0.005
    #thresholded_weight = np.zeros_like(win)
    #thresholded_weight[win>threshold_weight] = 1
    #chi *= thresholded_weight
    #border_window = ndimage.minimum_filter(thresholded_weight, footprint=np.ones([2*pixel_radius+1,2*pixel_radius+1]))
    #chi *= border_window

    return chi

####################################################################################
def smithPurePseudoCellEstimator(q_map, u_map, apodization, reso_rad=None, do_e=False,
                                 qu_to_eb_angle=None, ellgrid=None, pixel_range=2):
    """
    Given an input Q and U map, this function produces the FT of the corresponding B-mode (or E-mode) map,
    modified to remove all ambiguous modes. The output FT will be free of leakage from the
    E modes (or B modes).
    This function is an implementation of Kendrick Smith's pure B-mode estimator, first presented
    in Phys Rev. D 74, 083002 (2006); arXiv:astro-ph/0511629v2 . See also http://arxiv.org/abs/astro-ph/0608662.
    The implementation in this function is translated from the IDL function
    spt_analysis/simulations/cl_flatsky_kendrick.pro (also in
    $SPTPOL_SOFTWARE/idl_functions/cl_flatsky_kendrick.pro) by Tom Crawford.
    Note that we define k-space angles differently here than in cl_flatsky_kendrick, so
    the form of the counterterms is slightly changed. We here define k-space angles
    starting from zero in the +ell_y direction and increasing toward +ell_x.
    In cl_flatsky_kendrick, angles are zero in the +ell_x direction and increase
    toward +ell_y.
    INPUTS
        q_map: (2D array or Map) The Stokes Q map.
        u_map: (2D array or Map) The Stokes U map.
        apodization: (2D array or dictionary) The apodization mask. The mask must be such that
            the mask and all derivatives go to zero at its boundaries.
            If you input a 2D array, we will take the derivatives numerically (using the
            "pixel_range" input to determine how many surrounding pixels we'll use).
            If you prefer, you may input a dictionary which contains the windown and some
            or all of its pre-computed derivatives. We'll look for the following keys:
            'W','dW/dx','dW/dy','d2W/dx2','d2W/dy2','d2W/dx/dy'. Any derivatives not present
            will be computed numerically and stored in the dictionary.
        reso_rad [None]: (float) The map resolution per pixel, in radians. If either
            "q_map" or "u_map" are Map objects, we'll use the resolution there and ignore
            this input. Otherwise, the "reso_rad" input is required.
        do_e [False]: (bool) If True, output pure E modes instead of the default pure B modes.
        qu_to_eb_angle [None]: (2D array) For computational efficiency, you may input
            a pre-computed array of k-space angles. This should contain np.arctan2(ly, lx)
            for each k-space bin. If you don't provide this input, we'll compute it here.
        ellgrid [None]: (2D array or dict) For computational efficiency, you may input
            a pre-computed array of ells. This should contain sqrt(k_x**2+k_y**2) for each
            bin in k-space. If not provided, we'll compute it here.
               You may also input a dictionary. For a dictionary input, we'll attempt
            to access the key 'inv_safe_ellgrid'.
        pixel_range [2]: (int) What pixel radius should we use for any numeric derivatives?
            There should be no reason to alter this from the defaults.
    OUTPUT
        The k-space pure B modes (or E modes, if do_e is True) corresponding to the input
        Q and U maps.
    EXECPTIONS
        ValueError if we don't get a reso_rad input and neither q_map nor u_map are Map objects.
        ValueError if the "apodization" input is a dictionary, but doesn't contain a 'W' (for "window") key.
    AUTHOR
        Stephen Hoover, 8 November 2013 (translated from the IDL cl_flatsky_kendrick.pro by Tom Crawford)
    CHANGES
        None
    """
    # If the "q_map" and "u_map" are Map objects, pull out the map data arrays.
    try:
        reso_rad = q_map.reso_rad
        q_map = q_map.map
    except AttributeError:
        pass
    try:
        reso_rad = u_map.reso_rad
        u_map = u_map.map
    except AttributeError:
        pass

    # Make sure we know what the map resolution is.
    if reso_rad is None:
        raise ValueError("I need to know the map resolution! Please give me the resolution explicitly, or input a Map object.")

    # If we want to calculate pure E modes, swap Q and U. The math is identical, with this substitution.
    if do_e: q_map, u_map = -u_map, q_map

    # Compute the apodization window derivatives, if we dont' have them already.
    if not isinstance(apodization, dict):
        apodization = {'W':apodization}
    apodization.setdefault('dW/dx', math.ddx(apodization['W'], reso_rad, pixel_range) )
    apodization.setdefault('dW/dy', math.ddy(apodization['W'], reso_rad, pixel_range) )
    apodization.setdefault('d2W/dx2', math.d2dx2(apodization['W'], reso_rad, pixel_range) )
    apodization.setdefault('d2W/dy2', math.d2dy2(apodization['W'], reso_rad, pixel_range) )
    apodization.setdefault('d2W/dx/dy', math.d2dxdy(apodization['W'], reso_rad, pixel_range) )

    # Compute k-space angles if not already provided.
    if qu_to_eb_angle is None:
        lx, ly = np.meshgrid( np.fft.fftfreq( q_map.shape[1], reso_rad )*2.*np.pi,
                              np.fft.fftfreq( q_map.shape[0], reso_rad )*2.*np.pi )
        qu_to_eb_angle = np.arctan2(lx, -ly)

    # Compute 1/ell for each point in k-space, if not already provided.
    if ellgrid is None:
        ellgrid = util.math.makeEllGrid(shape=q_map.shape, resolution=reso_rad)
        ellgrid[0,0] = 1e6 # Make safe to invert by removing central zero.
    elif isinstance(ellgrid, dict):
        ellgrid = ellgrid['inv_safe_ellgrid']

    # For convenience and readability, pre-compute the two factors of ell in our final equation.
    # The factor_1 will be attached to the first derivatives of the apodization window, and
    # factor_2 goes will multiply the second derivatives of the apodization window.
    ellgrid_factor_1 = 1. / np.sqrt( (ellgrid-1)*(ellgrid+2) )
    ellgrid_factor_2 = 1. / np.sqrt( (ellgrid-1)*ellgrid*(ellgrid+1)*(ellgrid+2) )

    # Calcluate various combinations of FTs.
    map_fft = struct()
    map_fft['Q*W'] = np.fft.fft2(q_map*apodization['W'])
    map_fft['U*W'] = np.fft.fft2(u_map*apodization['W'])
    map_fft['Q*dW/dx'] = np.fft.fft2(q_map*apodization['dW/dx'])
    map_fft['Q*dW/dy'] = np.fft.fft2(q_map*apodization['dW/dy'])
    map_fft['U*dW/dx'] = np.fft.fft2(u_map*apodization['dW/dx'])
    map_fft['U*dW/dy'] = np.fft.fft2(u_map*apodization['dW/dy'])
    map_fft['2Q*d2W/dx/dy + U*(d2W/dy2 - d2W/dx2)'] = np.fft.fft2(2.*q_map*apodization['d2W/dx/dy'] +
                                                                  u_map*(apodization['d2W/dy2']
                                                                         -apodization['d2W/dx2']))

    # Add all of that with the correct prefactors to get pure B estimates.
    # (This will actually be E if we'd set do_e=True up above.)
    map_fft['B'] = ( -np.sin(2*qu_to_eb_angle)*map_fft['Q*W'] + np.cos(2*qu_to_eb_angle)*map_fft['U*W']
                     - 2j*ellgrid_factor_1 * (np.cos(qu_to_eb_angle)*map_fft['Q*dW/dx'] - np.sin(qu_to_eb_angle)*map_fft['Q*dW/dy'] +
                                              np.cos(qu_to_eb_angle)*map_fft['U*dW/dy'] + np.sin(qu_to_eb_angle)*map_fft['U*dW/dx'])
                    - ellgrid_factor_2 * map_fft['2Q*d2W/dx/dy + U*(d2W/dy2 - d2W/dx2)'] )

    # Finally, normalize :
    map_fft['B'] /= (np.sqrt(np.sum(apodization['W']**2)/np.product(q_map.shape)))

    return map_fft['B']


####################################################################################
def pointSourceWienerSetup(inputmap,nside,nmask,file_root,noffs=100):
    """
    INPUTS:
    input map (Map or PolarizedMap)
    nside (int) - length of submap side in pixels
    nmask (int) - length of masked region side in pixels
    file_root (string) - directory and name stem for interp_matrix pickle files
    noffs (int) [100] - noffs^2 submaps used to estimate the covariance matrix,
                        numbers < 100 won't sample the covariance matrix well enough
    OUTPUT:
    Returns a dictionary of the interp_matrix pickle files saved
    Note: this works on an individual Map or PolarizedMap object
    This module generates a wiener filter C*D^(-1) = C / (C + N), where C is an
    estimate of the map signal+noise covariance matrix and in N we artificially boost
    the noise at the point source (within nmask).  This will essentially downweight
    the information at the pointsource and generate a gaussian realization
    constrained by the larger scale map correlations.
    The saved interp_matrix files are used in maps.paintOverPointSources to do the
    actual point source interpolation.
    See van Engelen, et al (arXiv:1202.0546) for more info on how this was used in
    the SPTsz lensing analysis and references.
    """

    #check to see if we've got a PolarizedMap or Map
    if hasattr(inputmap,'pol_maps'):

        #cycle through each pol_map
        filenames = {}
        for pol, thismap in inputmap.pol_maps.iteritems():

            #find inner region of map
            mapsize = np.shape(thismap)
            ilo = mapsize[0]*0.3
            ihi = mapsize[0]*0.7
            jlo = mapsize[1]*0.3
            jhi = mapsize[1]*0.7

            nimage = nside*nside
            covar = np.zeros([nimage,nimage])
            icntr = 0.

            #threshold the map
            tempmap = thismap.map.copy()
            rms = np.std(tempmap[ilo:ihi,jlo:jhi])
            high = np.where(tempmap > 7.*rms)
            if len(high[0])>0:
                tempmap[high] = 7.*rms

            #calculate average covariance matrix over noffs^2 submaps
            for ii in np.arange(noffs):
                ioff = ii*(ihi-ilo)/noffs
                for jj in np.arange(noffs):
                    joff = jj*(jhi-jlo)/noffs
                    submap = tempmap[ilo+ioff:ilo+ioff+nside, jlo+joff:jlo+joff+nside]
                    dvec = submap.reshape(1,nside*nside)[0]
                    covar += np.outer(dvec,dvec)
                    icntr += 1.

            covar /= icntr

            #generate the D matrix by down-weighting the central values of map
            mask = np.zeros([nside,nside])
            mask[nside/2-nmask/2:nside/2+nmask/2,nside/2-nmask/2:nside/2+nmask/2] = 1.
            target = np.where(mask.flatten() > 0.)[0]
            dcovar = covar.copy()
            dcovar[target,target] = np.max(covar)*1e4
            dinv = LA.pinv2(dcovar)
            interp_matrix = np.dot(covar,dinv) #wiener filter C*D^(-1)

            #and save some files
            wf = struct()
            wf.noffs = noffs
            wf.nmask = nmask
            wf.nside = nside
            wf.interp_matrix = interp_matrix
            wf.covar = covar
            fn = file_root+'wf_nside'+str(nside)+'_nmask'+str(nmask)+'_'+pol+'.pkl'
            filenames[pol] = fn
            pickle.dump(wf,open(fn,'wb'))

    else:
        #only have to run once
        #find inner region of map
        filenames = {}
        mapsize = np.shape(inputmap)
        ilo = mapsize[0]*0.3
        ihi = mapsize[0]*0.7
        jlo = mapsize[1]*0.3
        jhi = mapsize[1]*0.7

        nimage = nside*nside
        covar = np.zeros([nimage,nimage])
        icntr = 0.

        #threshold the map
        tempmap = inputmap.map.copy()
        rms = np.std(tempmap[ilo:ihi,jlo:jhi])
        high = np.where(tempmap > 7.*rms)
        if len(high[0])>0:
            tempmap[high] = 7.*rms

        #calculate average covariance matrix over noffs^2 submaps
        for ii in np.arange(noffs):
            ioff = ii*(ihi-ilo)/noffs
            for jj in np.arange(noffs):
                joff = jj*(jhi-jlo)/noffs
                submap = tempmap[ilo+ioff:ilo+ioff+nside, jlo+joff:jlo+joff+nside]
                dvec = submap.reshape(1,nside*nside)[0]
                covar += np.outer(dvec,dvec)
                icntr += 1.

        covar /= icntr

        #generate the D matrix by down-weighting the central values of map
        mask = np.zeros([nside,nside])
        mask[nside/2-nmask/2:nside/2+nmask/2,nside/2-nmask/2:nside/2+nmask/2] = 1.
        target = np.where(mask.flatten() > 0.)[0]
        dcovar = covar.copy()
        dcovar[target,target] = np.max(covar)*1e4
        dinv = LA.pinv2(dcovar)
        interp_matrix = np.dot(covar,dinv) #wiener filter C*D^(-1)

        #and save some files
        wf = struct()
        wf.noffs = noffs
        wf.nmask = nmask
        wf.nside = nside
        wf.interp_matrix = interp_matrix
        wf.covar = covar
        fn = file_root+'wf_nside'+str(nside)+'_nmask'+str(nmask)+'.pkl'
        filenames[inputmap.polarization] = fn
        pickle.dump(wf,open(fn,'wb'))


    return filenames


####################################################################################

def paintOverPointSources(inputmap,filenames,point_source_config):
    """
    INPUTS:
    inputmap (Map or PolarizedMap) - the map with the offending point sources
    filenames (dict) - dictionary with interp_matrix files indexed by polarization,
                       this is output by pointSourceWienerSetup.
    point_source_config (str) - point source config file, in the format used by SPTsz
    OUTPUT:
    None, acts directly on inputmap.
    Note you need to run pointSourceWienerSetup() to generature the interpolation
    matrix (wiener filter) used by this code.  If you want to use difference larger
    masked regions for large point sources, you will have to run pointSourceWienerSetup
    multiple times with different nmasks and input separately into this function
    with split point source config files.  If you input a PolarizedMap object, it will
    use the same point source mask for T and P.
    """

    #read in point source locations [#, ra, dec, radius]
    catalog = np.loadtxt(point_source_config, skiprows=2)
    cat_ras = catalog[:,1]
    cat_dcs = catalog[:,2]
    n_sources = len(cat_ras)

    #check if we have a PolarizedMap
    if hasattr(inputmap,'pol_maps'):

        for pol, fn in filenames.iteritems():

            #read in wiener filter matrix
            wf = pickle.load(open(fn,'rb'))

            try:
                #make sure we have the correct type of map
                rez = inputmap.pol_maps[pol].reso_arcmin
                #submap shape in degrees
                map_shape = [wf.nside*rez/60.,wf.nside*rez/60.]

            except KeyError:
                print 'Pol_map '+pol+' does not exist, continuing...'
                continue

            #cycle through all the point sources
            for ps in np.arange(n_sources):

                delta_deg = (np.array([cat_ras[ps],cat_dcs[ps]])-inputmap.center)
                submap = inputmap.pol_maps[pol].getSubmap(map_shape,center_offset=delta_deg,units='degrees')
                dvec = submap.map.reshape(1,wf.nside*wf.nside)[0]
                newvec = np.dot(wf.interp_matrix,dvec)
                newmap = newvec.reshape(wf.nside,wf.nside)
                inputmap.pol_maps[pol].setSubmap(newmap,center_offset=delta_deg,units='degrees')

    else:

        #we have a single Maps
        pol = inputmap.polarization

        #read in wiener filter matrix
        try:
            wf = pickle.load(open(filenames[pol],'rb'))
        except KeyError:
            print 'Map polarization not in filenames list, exiting...'
            raise

        rez = inputmap.reso_arcmin
        #submap shape in degrees
        map_shape = [wf.nside*rez/60.,wf.nside*rez/60.]

        #cycle through all the point sources
        for ps in np.arange(n_sources):

            delta_deg = (np.array([cat_ras[ps],cat_dcs[ps]])-inputmap.center)
            submap = inputmap.getSubmap(map_shape,center_offset=delta_deg,units='degrees')
            dvec = submap.map.reshape(1,wf.nside*wf.nside)[0]
            newvec = np.dot(wf.interp_matrix,dvec)
            newmap = newvec.reshape(wf.nside,wf.nside)
            inputmap.setSubmap(newmap,center_offset=delta_deg,units='degrees')

##########################################################################################
def projectTFromP(pol_map, qScale = 0.0115, uScale = -0.0146, redo_projection=False, Tmap=None):
    """
    Experimental function which removes a fraction of the T map from Q and U maps.
    This function only does the removal; it does not determine the correct fraction to use.
    INPUTS
        pol_map: (PolarizedMap) The map to modify.
        qScale [0.0115]: (float)
        uScale [-0.0146]: (float) Magic numbers from Ryan. True only for a particular map run,
            and would change with different polcal or different relative gain factors.
        redo_projection [False]: (bool) If True, allow the code to be run more than
            once on the same map. Later runs will start from the map as input,
            not the original map!
        Tmap [None]: The temperature map template used for the deprojection, presumably the full-coadd
            T map from the map run of which "pol_map" is a part.  For efficiency, this is just the 2d
            array, i.e. coadded_Map.pol_maps.T.map, SO BE SURE Tmap AND pol_map HAVE IDENTICAL PROPERTIES!
    OUTPUT
        Returns the input PolarizedMap. The input map will have been modified in place!
    EXCEPTIONS
        ValueError if you run this function twice on the same map and haven't set redo_projection=True.
    AUTHOR
        Stephen Hoover, 17 June 2013
    CHANGES
        5 July 2013: Changed default qScale from 0.0088 to 0.0115 and default uScale from -0.0147 to -0.0146 based on new info from Ryan. SH
        30 Sep 2015: Added Tmap argument to allow a full-depth T map to be used as the template. JTS.
    """

    # If we don't specify the T map template, just use the T map of the input pol_map object.
    if Tmap is None:
        Tmap = pol_map['T'].map

    #This recursively defines processing_applied within pol_map, which does nasty things when you try to write
    #the file to disk.  To avoid this, remove the copy of pol_map (and Tmap while we're at it... we don't need it).
    myargs = tools.myArguments(remove_self=True, remove_data=True, combine_posargs=True) # Store this for use in a couple of lines.

    del myargs['Tmap']
    del myargs['pol_map']

    if getattr(pol_map, 'tp_projection_done', False) and not redo_projection:
        raise ValueError("This map has already had T removed from the Q and U maps. I won't do it again!")


    # Project out T from P.
    pol_map['Q'].map -= (qScale*Tmap)
    pol_map['U'].map -= (uScale*Tmap)

    # Record that we've done this.
    pol_map.tp_projection_done = True
    pol_map.processing_applied['projectTFromP'] = myargs

    # Remove the E and B maps if we had any going in.
    if 'B' in pol_map.pol_maps:
        del pol_map.pol_maps['B']
        pol_map.polarizations.pop(pol_map.polarizations.index('B'))
    if 'E' in pol_map.pol_maps:
        del pol_map.pol_maps['E']
        pol_map.polarizations.pop(pol_map.polarizations.index('E'))
    if hasattr(pol_map, 'ft_maps'):
        if 'B' in pol_map.ft_maps: del pol_map.ft_maps['B']
        if 'E' in pol_map.ft_maps: del pol_map.ft_maps['E']

    return pol_map

##########################################################################################
def cleanRAStripes(_map, apod_mask=None, smooth_template=0., in_place=True, return_template=False,
                   dec_range=None, quiet=False):
    """
    This function filters scan-synchronous noise from maps. It averages together map pixels
    with the same RA, then subtracts this RA template from the map. The SPT scans in
    rows of (nearly) constant declination, so if there's any feature in the data which is
    the same in every scan, you'll see vertical stripes in the map. This function removes
    those stripes.
    INPUTS
        _map: (A Map or PolarizedMap) The map to filter. If this is a PolarizedMap,
            we'll filter independently on every polarization within the object.
        smooth_template [0.]: (float) In arcminutes of RA, we will apply a hanning
            smoothing function with this width to the RA template before subtracting
            it from the map. Values of 0, None, or False indicate no smoothing.
        apod_mask [None]: Ignore some elements of the map when averaging map pixels into
            an RA template. We'll ignore any pixel which does not have an apodization
            mask value of exactly 1. I strongly recommend that you use an apodization
            mask which blocks out crazy border values and point sources.
        in_place [True]: (bool) If True, the input object will be modified. If False,
            the input map will be unchanged, and the returned map will be a
            (filtered) copy.
        return_template [False]: (bool) If True, return the RA template which we subtracted
            from each map.
        dec_range [None]: (2-tuple) If not None, only use portions of the map with
            dec_range[0] < declination < dec_range[1]. We'll still subtract the template
            from the entire map. You can use this argument to exclude noisy pixels at the
            top and bottom of the map. (You could equivalently use the apodization mask
            for this, but the dec_range argument can be easier.)
        quiet [False]: (bool) Suppress all screen output.
    OUTPUT
        A filtered version of the input map. If in_place is True, then the output map is
        the input map. If return_template is True, the output is a tuple of
        (map, templates), where the templates output has (template value, left bin edge)
        for the RA template. If the input is a PolarizedMap, then "templates" is a dictionary
        of 2-tuples. Otherwise it is a 2-tuple.
    EXCEPTIONS
        None
    AUTHOR
        Stephen Hoover, 11 December 2013
    CHANGES
        12 Dec 2013: Remove the input map from the arguments stored in processing_applied.
            Add the smooth_template input argument.
            Increase number of histogram bins. Smooth with a proper smoothing function instead. SH
        11 Feb 2014: Ignore warnings when dividing map_ra_template_sum / map_ra_template_count. SH
    """
    # Store input arguments. Replace the mask (if any) with a sum of all mask
    # elements. We don't want to store the whole thing.
    myargs = tools.myArguments(remove_self=True, remove_data=True, combine_posargs=True) # Store this for use at the bottom of the function.
    if type(apod_mask) == str:
        apod_mask = files.read(apod_mask)
    myargs['apod_mask'] = (np.sum(apod_mask) if apod_mask is not None else None)
    del myargs['_map'] # Don't store a copy of yourself!

    # Decide if we're modifying the input map or not.
    if in_place:
        this_map = _map
    else:
        this_map = _map.copy()

    # It doesn't make sense to do this filtering on a weighted map, so remove the weight now.
    if this_map.weighted_map:
        if not quiet: print "[maps.cleanRAStripes] Removing weight from output map before cleaning."
        this_map.removeWeight(in_place=True)

    # Find the RA of every map pixel.
    ra, dec = this_map.pix2Ang(np.indices(this_map.shape))
    ra.flat[:] = tools.unwrapAz(ra.flatten())

    # Use the input mask only to determine which map elements we average together.
    # Use only portions of the mask exactly equal to 1; we assume that the other
    # mask elements represent data too noisy to want to use.
    if apod_mask is None:
        bool_mask = np.ones(this_map.shape, dtype=np.bool)
    else:
        bool_mask = (apod_mask == 1)

    # Restrict the declination range of the template, if requested.
    if dec_range is not None:
        bool_mask &= (dec_range[0] < dec) & (dec < dec_range[1])

    # Check if we're doing a PolarizedMap or a Map. If it's just
    # a Map, then wrap a dictionary around it so that we can use
    # the same code for both cases.
    try:
        map_polarizations = this_map.polarizations
        map_is_polarized = True
    except AttributeError:
        map_polarizations = ['T']
        this_map = {'T':this_map}
        map_is_polarized = False

    # Loop over every polarization in the map. Calculate and subtract a template for a change in RA
    # separately for each polarization.
    templates = struct()
    for pol in map_polarizations:
        # Average together the map values at each RA. First we find the sum of all of the
        # map values at each RA, then we count the number of occurences of each RA.
        # Their quotient is the template for the signal in the map which varies with RA only.
        map_ra_template_sum, ra_bins = np.histogram(ra[bool_mask], weights=this_map[pol].map[bool_mask],
                                                    bins=2*this_map[pol].shape[1])
        map_ra_template_count, ra_bins = np.histogram(ra[bool_mask], bins=2*this_map[pol].shape[1])

        with warnings.catch_warnings():
            # This division will probably complain about zeros in the denominator.
            # That's why it's wrapped in a nan_to_num.
            warnings.simplefilter("ignore")
            map_ra_template = np.nan_to_num(map_ra_template_sum / map_ra_template_count)

        # Smooth the template.
        if smooth_template:
            window_length = np.ceil(smooth_template/(60*(ra_bins[1]-ra_bins[0])) ) # Assume smooth_template is in arcminutes
            map_ra_template = math.smooth(map_ra_template, window_len=window_length, window='hanning')

        # The np.histogram function returns the bin numbers in such a way that the last
        # bin is closed, i.e., we could have a value in "ra" which is exactly equal to the last
        # bin number. The np.digitize function would treat that as an overflow. If we nudge
        # the last bin edge up by epsilon, such a value won't be in overflow.
        ra_bins[-1] += 1e-10*ra_bins[-1]

        # Figure out which index in the template each RA map value corresponds to.
        # For any RA values less than or greater than the template RAs, set those
        # to use the edge bin.
        i_ra_bin = np.digitize(ra.flat, ra_bins) # Note that this is a flattened array.
        i_ra_bin[i_ra_bin>len(map_ra_template)] = len(map_ra_template)
        i_ra_bin[i_ra_bin<1] = 1

        # Create the 2D version of the RA template. Note that we subtract bin number by 1
        # so that bins start with 0. (The np.digitize function uses a zero index as
        # the underflow designation.)
        map_template = map_ra_template[ i_ra_bin-1 ].reshape(ra.shape)
        this_map[pol].map -= map_template

        # Store the flat template, and the corresponding bins.
        # Remove the last bin so that the bins are the same length as the template.
        templates[pol] = (map_ra_template, ra_bins[:-1])

    # If we'd wrapped a Map in a dictionary, pull it out now.
    if not map_is_polarized:
        this_map = this_map['T']
        templates = templates['T']

    # Store a note in the map that we've done this.
    this_map.processing_applied['cleanRAStripes'] = myargs

    # Finished!
    if return_template:
        return this_map, templates
    else:
        return this_map

##########################################################################################
def setMapFTWeight(_map,
                   apod_mask=None,
                   uv_mask=None,
                   shape=None,
                   return_profiles=False,
                   verbose=True):
    """
    Calculates a single number for overall map weight, based on the average of the
    map's Fourier transform squared. You can treat this number as an estimate of the noise
    in the map, as long as the noise is well-behaved. (I would trust this overall noise
    less if the noise is very non-uniform in a map. In such a case, consider extending
    this function to use a k-space weighting in the average.
    INPUTS
        _map: (PolarizedMap) The map on which we wish to set a weight.
        apod_mask [None]: (2D array) An apodization mask for the map. If you don't
            supply one, a mask will be created for you (this is slow).
        uv_mask [None]: (2D array) If input, use this array as a mask (or weight) for the
            FT values when averaging.
        return_profiles [False]: (bool) If True, return profiles of the FFT in
            k_y, instead of an overall average.
        verbose [True]: (bool) Extra output to the screen.
    OUTPUT
        A 2-tuple with the average of FT(Q)*FT(Q).conj, and similarly for U.
        If return_profiles=True, return instead the average along axis 1 (a k_y plot).
        We will also modify map_ft_weight, map_mean_ft_q, and map_mean_ft_u attributes
        on the input map.
    EXCEPTIONS
        None
    AUTHOR
        Stephen Hoover, 9 May 2014
    CHANGES
        None
    """
    # We need a mask. Create one if we didn't have one input.
    if apod_mask is None:
        apod_mask = makeBorderApodization(_map, apod_type='cos', radius_arcmin=60., zero_border_arcmin=10)

    # Set defaults for shape and uv_mask.
    if shape is None: shape = _map.shape
    if uv_mask is None: uv_mask = np.ones(shape)

    # FFT the map, both in Q and in U.
    mapnoweight = _map.removeWeight(in_place=False)
    map_fft_q = np.fft.fft2(mapnoweight['Q'].map*apod_mask, s=shape)*_map.reso_rad**2
    map_fft_u = np.fft.fft2(mapnoweight['U'].map*apod_mask, s=shape)*_map.reso_rad**2
    map_fft_q = uv_mask*(map_fft_q * map_fft_q.conj()).real
    map_fft_u = uv_mask*(map_fft_u * map_fft_u.conj()).real

    # Multiply by 1e12 to convert from K_CMB to uK_CMB.
    unit_conv = (1e12 if  _map.units=='K_CMB' else 1.)

    q_mean = unit_conv*np.average(map_fft_q, weights=uv_mask)
    u_mean = unit_conv*np.average(map_fft_u, weights=uv_mask)
    if verbose:
        print "Mean of map FFT**2 in Q, U is %.5e, %.5e." % (q_mean, u_mean)

    # Set an attribute on the map to store the weight.
    _map.map_ft_weight = 1. / np.mean([q_mean, u_mean])
    _map.map_mean_ft_q, _map.map_mean_ft_u = q_mean, u_mean

    if return_profiles:
        return np.average(map_fft_q, axis=1, weights=uv_mask), np.average(map_fft_u, axis=1, weights=uv_mask)
    else:
        return q_mean, u_mean

##########################################################################################
def calculatePolarizationSelfCal(input_cls,
                                 expectation_cls=None,
                                 ell_range=[400,2500],
                                 covariance_range=5,
                                 approximate_cov=False,
                                 use_bb=False,
                                 quiet=False,
                                 debug=False):
    """
    This function checks for an overall focal plane rotation by using measured TB and EB spectra.
    If the averaged measured detector polarization angle is different from the true average
    detector polarization angle, then E-mode power will leak into B (and vice-versa, but B->E leakage
    is not very important). This will modify the BB spectrum, which is bad, but it will also leak
    power from TE to TB, and create a signal in EB. The TE and EB are expected to be zero
    under standard cosmology.
    This function is inspired by the prescription in
    B. Keating, M. Shimon, and A. Yadav, "Self-Calibration of CMB Polarization Experiments" (arXiv:1211.5734v1),
    and, to a lesser extent, the work in Kaufman et al., "Self-Calibration of BICEP1 Three-Year
    Data and Constraints on Astrophysical Polarization Rotation" (arXiv:1312.7877v1).
    Compared to Keating et al., this function uses a full covariance matrix. It also ignores
    sample variance since it uses TE and EE spectra measured over the same region as
    the TB and EB spectra. Compared to Kaufman et al., this function should match the
    procedure detailed in Section III.B.1, "Gaussian Bandpower Likelihood Approximation".
    INPUTS
        input_cls: (dictionary) A dictionary of C_ell values, as output by cross_spectrum_multiprocessing.
            Must contain both the average bandpowers and covariance matrices keyed as "covariance_XX",
            where "XX" is the spectrum type.
        expectation_cls [None]: (dictionary) A dictionary which contains the "TB" and "EB" keys.
            These should contain power spectra in the same binning as the input_cls.
            We will assume that the expectation_cls represent zero focal plane rotation and
            subtract them from the input_cls in each bin.
            We'll check the expectation_cls['ell'] key for "TB" and "EB" keys to verify
            that the binning is the same.
        ell_range [[400,3000]]: (2-tuple of floats) Include only bandpowers in this ell range
            in the calculations.
        covariance_range [5]: (int) We will condition the covariance matrices we find in input_cls.
            This value is how many places off the diagonal we will keep. If covariance_range<0,
            we will not condition the covariance matrices.
        approximate_cov [False]: (bool) If True, approximate the covariance used in
            the chi^2 estimator by either cov(TB) or cov(EB). These are the dominant contributions.
            The EE, TE, and BB covariances are approximately a few percent as large.
        use_bb [False]: (bool) The exact chi^2 for the EB estimator includes a
            "cls['EE'] - cls['BB']" term. The BB spectrum should be much smaller than
            the EE spectrum, so you may choose to include it or not, depending on how
            much you trust it.
        quiet [False]: (bool) Silence all screen output.
        debug [False]: (bool) If True, return a 6-tuple with individual TB and EB rotation
            estimates, errors, and chi^2 functions.
    OUTPUT
            A 2-tuple of floats, containing the maximum likelihood estimate of focal
        plane rotation, using both the measured TB and EB spectra, and the distance to
        the edge of the 68% confidence region. Both values are in degrees.
            Unless quiet=True, the function will print individual focal plane
        rotation estimates and the bounds of the 68% confidence region for each
        of the TB and EB estimators.
    EXCEPTIONS
        ValueError if we're missing data or covariances in the input_cls.
        AssertionError if the focal plane rotation angle and errors don't match up between
            an analytic calculation and a direct estimate from the chi^2 distribution.
        ValueError if we're given expectation_cls which don't match the format of the input_cls.
    AUTHOR
        Stephen Hoover, 17 January 2014
    CHANGES
        20 Jan 2014: Calculate and print to screen the chi^2/NDF and PTE for each fit.
                    Add "error_check_tolerance" parameter, and increase it a bit. SH
        28 Jan 2014: Add fit to zero power. SH
        26 Feb 2014: Add expectation_cls argument. SH
    """
    error_check_tolerance = 5e-2 # Check that the confidence regions are symmetric to this level.

    # Make sure that we have data and covariances for all of the spectra that we will use.
    if use_bb and ('BB' not in input_cls or 'covariance_BB' not in input_cls):
        raise ValueError("You want me to use the BB spectrum in my calculations, but that's not in your input!")
    for spec in ['TB','EB','EE','TE']:
        if spec not in input_cls or 'covariance_'+spec not in input_cls:
            raise ValueError("Missing data or covariances for the %s spectrum." % spec)

    # Assume we need to condition the covariance matrices. We don't want to condition
    # too far past the region we're using, so find the maximum bin to consider.
    # Go a short distance beyond the end of the ell range requested.
    max_bin = dict( [(spec, np.where(input_cls['ell'][spec] < ell_range[1]+500)[0][-1]+1)
                     for spec in input_cls['ell']])

    ##################################################################################
    # Pull out the covariance matricies for the TB and EB spectra and condition them.
    # If any diagonal elements are zero, then assign a very large value
    # to them. Finally, invert the covariance matricies.
    #
    # Note that, even if we're using measured TE and EE spectra, we neglect their
    # contribution to the covariance (unless approximate_cov is True). We can do
    # this because the covariance of TE is similar to that of TB, and EE is
    # similar to EB. Once we multiply the covariance by the small factor of
    # sin(2*angle) or 0.5*sin(4*angle), it becomes ~ a few percent of the TB or EB.
    cov = {}
    for spec in ['TB','EB','EE','TE']:
        try:
            cov[spec] = input_cls['covariance_'+spec]['total'][:max_bin[spec],:max_bin[spec]]
        except KeyError:
            cov[spec] = input_cls['covariance_'+spec]['independent'][:max_bin[spec],:max_bin[spec]]
        if covariance_range >= 0:
            cov[spec] = np.nan_to_num(covariance_utils.condition_cov_matrix(cov[spec], order=covariance_range))

    # If any diagonals are zero, give then a large covariance instead, so that we can invert
    # the matrix.
    for this_cov in cov.itervalues():
        for ii in xrange(this_cov.shape[0]):
            if this_cov[ii,ii]==0.:
                this_cov[ii,ii] = 1000*np.max(this_cov)

    # Invert!
    inv_cov = dict( [(spec, np.linalg.inv(this_cov)) for spec, this_cov in cov.iteritems()])

    ########################################################
    # Pull out the power spectrum values that we'll use.
    # We separate them to make it easy to, for example, use
    # theory values in place of some spectra in the future.
    # For the "EE", allow an option to include the BB spectrum.
    # The exact equation uses cls['EE'] - cls['BB'], but
    # we should have BB<<EE, so you may exclude BB to avoid
    # worries about extra error.
    cls = struct()
    cls['TB'] = input_cls['TB'].copy()
    cls['TE'] = input_cls['TE'].copy()
    cls['EB'] = input_cls['EB'].copy()
    if use_bb:
        cls['EE'] = input_cls['EE'] - input_cls['BB']
    else:
        cls['EE'] = input_cls['EE']

    # If we have expectation values, subtract those from the data C_ells.
    if expectation_cls is not None:
        # Assume that the expectation values represent zero focal plane rotation, and
        # subtract that off of the TB and EB spectra. Any change in the TE and EE spectra
        # should be negligible, so don't change those.
        for spec in ['TB','EB']:
            # Check the format of the expectation values.
            if (spec not in expectation_cls or
                len(expectation_cls[spec]) != len(input_cls[spec]) or
                (input_cls['ell'][spec] != expectation_cls['ell'][spec]).any()):
                raise ValueError("The expectation value for the %s spectrum doesn't match the input C_ells in some way." % spec)

            # Subtract the expectation values from the input power spectra.
            cls[spec] -= expectation_cls[spec]

    # Find the ell bins that we'll use to make this calculation.
    # Assume that all spectra have the same binning.
    ell = input_cls['ell']['EE']
    ell2use = (ell > ell_range[0]) & (ell < ell_range[1])
    bin_range = [np.where(ell2use)[0][0], np.where(ell2use)[0][-1]+1]
    ell = ell[ell2use]

    #######################################################################
    #######################################################################
    # Calculate the Gaussian chi-squared for the TB estimator.
    def chiSquaredTB(angle, approximate_cov=True):
        """
        This is the chi-squared metric for measuring focal plane
        rotation from the observed TB spectrum.
        INPUTS
            angle : (float) The focal plane rotation angle, in radians.
            approximate_cov [True]: (bool) If True, ignore the contribution
                to the covariance from the TE measurement.
        OUTPUT
            A float, the chi-squared for the fit of the measured TB spectrum
            to a model which assumes a given focal plane rotation angle.
        """
        # This is the difference between the measured TB and the
        # expected TB under the assumption of a focal plane
        # rotation of "angle".
        delta_cl = cls['TB'][ell2use] + np.sin(2*angle)*cls['TE'][ell2use]

        # Get the portion of the inverse covariance matrix which is relevant to us.
        if approximate_cov:
            this_inv_cov = inv_cov['TB'][bin_range[0]:bin_range[1], bin_range[0]:bin_range[1]]
        else:
            # Include the extra (small) contribution to the covariance from the TE measurement.
            this_inv_cov = np.linalg.inv(cov['TB'] + (np.sin(2*angle)**2)*cov['TE'])
            this_inv_cov = this_inv_cov[bin_range[0]:bin_range[1], bin_range[0]:bin_range[1]]

        # Calculate the chi-squared!
        chisq = np.dot( np.transpose(delta_cl), np.dot(this_inv_cov, delta_cl))
        return chisq
    # Calculate the Gaussian chi-squared for the EB estimator.
    def chiSquaredEB(angle, approximate_cov=True):
        """
        This is the chi-squared metric for measuring focal plane
        rotation from the observed EB spectrum.
        INPUTS
            angle : (float) The focal plane rotation angle, in radians.
            approximate_cov [True]: (bool) If True, ignore the contribution
                to the covariance from the EE and BB measurements.
        OUTPUT
            A float, the chi-squared for the fit of the measured EB spectrum
            to a model which assumes a given focal plane rotation angle.
        """
        # This is the difference between the measured TB and the
        # expected TB under the assumption of a focal plane
        # rotation of "angle".
        delta_cl = cls['EB'][ell2use] + 0.5*np.sin(4*angle)*cls['EE'][ell2use]

        # Get the portion of the inverse covariance matrix which is relevant to us.
        if approximate_cov:
            this_inv_cov = inv_cov['EB'][bin_range[0]:bin_range[1], bin_range[0]:bin_range[1]]
        else:
            # Include the extra (small) contribution to the covariance from the EE and BB measurements.
            if use_bb:
                this_inv_cov = np.linalg.inv(cov['EB'] + ((0.5*np.sin(4*angle))**2)*(cov['EE']+cov['BB']))
            else:
                this_inv_cov = np.linalg.inv(cov['EB'] + ((0.5*np.sin(4*angle))**2)*cov['EE'])
            this_inv_cov = this_inv_cov[bin_range[0]:bin_range[1], bin_range[0]:bin_range[1]]

        # Calculate the chi-squared!
        chisq = np.dot( np.transpose(delta_cl), np.dot(this_inv_cov, delta_cl))
        return chisq
    #######################################################################
    #######################################################################

    #################################################################################
    # Next, calculate intermediate terms, analagous to eq. 10. in Keating et al.
    # In the comment lines, "M" is the (Nbins x Nbins) covariance matrix, "*" is matrix
    # multiplication, the Cls are (Nbins x 1) matrices, and sums are over all bins.
    # Here I approximate the covariance matrix M as equal to cov(TB) (for the TB terms)
    # or cov(EB) (for the EB terms).

    # A_TB = Sum( Cls^TE * M^-1 * Cls^TB ) + Sum( Cls^TB * M^-1 * Cls^TE )
    a_tb = ( np.dot( np.transpose(cls['TE'][ell2use]),
                     np.dot(inv_cov['TB'][bin_range[0]:bin_range[1], bin_range[0]:bin_range[1]],
                            cls['TB'][ell2use]) )
            + np.dot( np.transpose(cls['TB'][ell2use]),
                     np.dot(inv_cov['TB'][bin_range[0]:bin_range[1], bin_range[0]:bin_range[1]],
                            cls['TE'][ell2use]) ) )
    # B_TB = 2*Sum( Cls^TE * M^-1 * Cls^TE )
    b_tb = 2*np.dot( np.transpose(cls['TE'][ell2use]),
                     np.dot(inv_cov['TB'][bin_range[0]:bin_range[1], bin_range[0]:bin_range[1]],
                            cls['TE'][ell2use]) )

    # A_EB = Sum( Cls^EE * M^-1 * Cls^EB ) + Sum( Cls^EB * M^-1 * Cls^EE )
    a_eb = ( np.dot( np.transpose(cls['EE'][ell2use]),
                     np.dot(inv_cov['EB'][bin_range[0]:bin_range[1], bin_range[0]:bin_range[1]],
                            cls['EB'][ell2use]) )
            + np.dot( np.transpose(cls['EB'][ell2use]),
                     np.dot(inv_cov['EB'][bin_range[0]:bin_range[1], bin_range[0]:bin_range[1]],
                            cls['EE'][ell2use]) ) )
    # B_EB = Sum( Cls^EE * M^-1 * Cls^EE )
    b_eb = np.dot( np.transpose(cls['EE'][ell2use]),
                   np.dot(inv_cov['EB'][bin_range[0]:bin_range[1], bin_range[0]:bin_range[1]],
                          cls['EE'][ell2use]) )


    # Now I can estimate the focal plane rotation angles.
    delta_phi_tb = -0.5*np.arcsin(a_tb / b_tb)
    delta_phi_eb = -0.25*np.arcsin(a_eb / b_eb)

    # Calculate the errors: sigma_delta**2 ~ 0.5 / ( d^2 / d delta^2 ) chi^2(delta) | delta=delta_0
    # These are the approximate bounds of the 68% confidence region.
    sigma_tb = np.sqrt(1. / (0.5*4*np.cos(2*delta_phi_tb)* b_tb ))
    sigma_eb = np.sqrt(1. / (0.5*8*np.cos(4*delta_phi_eb)* b_eb ))
    sigma_tb_deg = np.rad2deg(sigma_tb)
    sigma_eb_deg = np.rad2deg(sigma_eb)

    # If I want to be more exact with the covariances, then minimize the chi-squared functions,
    # including extra covariance from the TE and EE terms.
    if not approximate_cov:
        if debug:
            print "Ignoring the covariance of TE and EE, I calculate rotation angles of:"
            print "\tTB : %f +/- %f" % (np.rad2deg(delta_phi_tb), sigma_tb_deg)
            print "\tEB : %f +/- %f" % (np.rad2deg(delta_phi_eb), sigma_eb_deg)

        # Mimimize the chi^2 functions, using my approximate rotation angles as our starting position.
        delta_phi_tb = sp.optimize.fmin(chiSquaredTB, delta_phi_tb, args=(approximate_cov,),
                                              ftol=1e-10, disp=0)[0]
        delta_phi_eb = sp.optimize.fmin(chiSquaredEB, delta_phi_eb, args=(approximate_cov,),
                                              ftol=1e-10, disp=0)[0]
    delta_phi_tb_deg = np.rad2deg(delta_phi_tb)
    delta_phi_eb_deg = np.rad2deg(delta_phi_eb)

    ##########################################################################
    # Calculate an improved 68% confidence region by using the chi^2 functions
    # to look for the delta(chi^2)=1 points. These errors will be close to
    # the previously estimated errors. I add this step because it's easy, and
    # serves as a cross-check on the math.
    min_chi2_tb = chiSquaredTB(delta_phi_tb, approximate_cov)
    errFuncTB = lambda dphi: (chiSquaredTB(dphi+delta_phi_tb, approximate_cov) - min_chi2_tb - 1)**2
    error_tb = [sp.optimize.fmin(errFuncTB, -sigma_tb, ftol=1e-10, disp=0)[0],
                sp.optimize.fmin(errFuncTB, sigma_tb, ftol=1e-10, disp=0)[0]]
    error_tb = np.rad2deg(error_tb)

    min_chi2_eb = chiSquaredEB(delta_phi_eb, approximate_cov)
    errFuncEB = lambda dphi: (chiSquaredEB(dphi+delta_phi_eb, approximate_cov) - min_chi2_eb - 1)**2
    error_eb = [sp.optimize.fmin(errFuncEB, -sigma_eb, ftol=1e-10, disp=0)[0],
                sp.optimize.fmin(errFuncEB, sigma_eb, ftol=1e-10, disp=0)[0]]
    error_eb = np.rad2deg(error_eb)

    # The confidence region should be very nearly symmetric, so average the errors
    # on each side to make a symmetric error.
    symmetric_error_tb = np.mean(np.abs(error_tb))
    symmetric_error_eb = np.mean(np.abs(error_eb))

    # Make sure the confidence region is symmetric, and that the
    # errors I calculated in two different ways match.
    assert np.mean(error_tb)/symmetric_error_tb < error_check_tolerance, "The TB confidence region is not symmetric!"
    assert np.mean(error_eb)/symmetric_error_eb < error_check_tolerance, "The EB confidence region is not symmetric!"
    assert np.abs((symmetric_error_tb-np.rad2deg(sigma_tb))/symmetric_error_tb)<error_check_tolerance, "Something's wrong with error estimation!"
    assert np.abs((symmetric_error_eb-np.rad2deg(sigma_eb))/symmetric_error_eb)<error_check_tolerance, "Something's wrong with error estimation!"

    ################################################################################
    # Combine the focal plane rotation angles estimated from the TB and EB spectra
    # into a single estimate of focal plane rotation.
    error_comb_approx = np.sqrt(1./(1/symmetric_error_tb**2 + 1/symmetric_error_eb**2))
    delta_phi_combined_approx = ((delta_phi_tb_deg/symmetric_error_tb**2) +
                                 (delta_phi_eb_deg/symmetric_error_eb**2))*error_comb_approx**2

    # Go ahead and calculate this from the sum of the TB and EB chi^2 distributions.
    # This should be almost exactly the same. I'm doing it in an excess of caution.
    chiSquaredComb = lambda dphi, approx: chiSquaredTB(dphi, approx) + chiSquaredEB(dphi, approx)
    delta_phi_combined = sp.optimize.fmin(chiSquaredComb, np.deg2rad(delta_phi_combined_approx),
                                          args=(approximate_cov,),
                                          ftol=1e-10, disp=0)[0]
    min_chi2_comb = chiSquaredComb(delta_phi_combined, approximate_cov)
    errFuncComb = lambda dphi: (chiSquaredComb(dphi+delta_phi_combined, approximate_cov) - min_chi2_comb - 1)**2
    error_comb = [sp.optimize.fmin(errFuncComb, -np.deg2rad(error_comb_approx), ftol=1e-10, disp=0)[0],
                  sp.optimize.fmin(errFuncComb, np.deg2rad(error_comb_approx), ftol=1e-10, disp=0)[0]]
    error_combined = np.rad2deg(error_comb)
    symmetric_error_comb = np.mean(np.abs(error_combined))
    delta_phi_combined = np.rad2deg(delta_phi_combined)

    # Check the combined angle and errors.
    assert np.mean(error_comb)/symmetric_error_comb < error_check_tolerance, "The combined confidence region is not symmetric!"
    assert np.abs((symmetric_error_comb-error_comb_approx)/symmetric_error_comb)<error_check_tolerance, "Something's wrong with the combined error estimation!"
    assert np.abs((delta_phi_combined_approx-delta_phi_combined)/delta_phi_combined)<error_check_tolerance, "The combined angle estimate shifted too much when I calculated from the chi^2 function!"

    if debug:
        print "\nCombining the TB and EB estimates in the naive way, I get a focal plane rotation of:"
        print "\t%f +/- %f\n" % (delta_phi_combined_approx, error_comb_approx)

    #####################################################################
    # Use the chi-squared to calculate the goodness of fit.
    ndf = np.sum(ell2use) - 1 # Number of bins, minus one fit parameter.
    pte_tb = 1 - stats.chi2.cdf(min_chi2_tb, ndf)
    pte_eb = 1 - stats.chi2.cdf(min_chi2_eb, ndf)

    ndf_combined = 2*np.sum(ell2use) - 1 # Number of bins, minus one fit parameter.
    pte_combined = 1 - stats.chi2.cdf(min_chi2_comb, ndf_combined)

    #####################################################################
    # Also calculate how well the EB and TB spectra fit to a value of
    # zero, the expectation when we don't have any angle rotations.
    ndf_to_zero = np.sum(ell2use) # Number of bins, with no fit parameters.

    chi2_to_zero_eb = np.dot( np.transpose(cls['EB'][ell2use]),
                              np.dot(inv_cov['EB'][bin_range[0]:bin_range[1], bin_range[0]:bin_range[1]],
                                     cls['EB'][ell2use]))
    chi2_to_zero_tb = np.dot( np.transpose(cls['TB'][ell2use]),
                              np.dot(inv_cov['TB'][bin_range[0]:bin_range[1], bin_range[0]:bin_range[1]],
                                     cls['TB'][ell2use]))

    pte_to_zero_eb = 1 - stats.chi2.cdf(chi2_to_zero_eb, ndf_to_zero)
    pte_to_zero_tb = 1 - stats.chi2.cdf(chi2_to_zero_tb, ndf_to_zero)

    #####################################################################
    # Done! Print the result to the screen, and return it as a tuple.
    # Print the upper and lower bound to the 68% confidence band
    # separately as a check. We have a well-behaved Gaussian likelihood,
    # so the bounds should be symmetric.
    if not quiet:
        print "Maximum likelihood rotation angle as estimated by"
        print ("\tTB : %f - %f + %f (68%% confidence region: [%f, %f])"
               % (delta_phi_tb_deg, -error_tb[0], error_tb[1],
                  delta_phi_tb_deg+error_tb[0], delta_phi_tb_deg+error_tb[1]))
        print "\t\tChi-squared for this fit is %.2f with %d DOF. PTE = %f" % (min_chi2_tb, ndf, pte_tb)
        print ("\t\tGoodness of fit to zero TB power: Chi^2 = %.2f with %d DOF. PTE = %f"
               % (chi2_to_zero_tb, ndf_to_zero, pte_to_zero_tb))
        print ("\tEB : %f - %f + %f (68%% confidence region: [%f, %f])"
               % (delta_phi_eb_deg, -error_eb[0], error_eb[1],
                  delta_phi_eb_deg+error_eb[0], delta_phi_eb_deg+error_eb[1]))
        print "\t\tChi-squared for this fit is %.2f with %d DOF. PTE = %f" % (min_chi2_eb, ndf, pte_eb)
        print ("\t\tGoodness of fit to zero EB power: Chi^2 = %.2f with %d DOF. PTE = %f"
               % (chi2_to_zero_eb, ndf_to_zero, pte_to_zero_eb))

        print "\nThe combined estimate for the focal plane rotation angle is:"
        print ("\t%f - %f + %f (68%% confidence region: [%f, %f])"
               % (delta_phi_combined, -error_combined[0], error_combined[1],
                  delta_phi_combined+error_combined[0], delta_phi_combined+error_combined[1]))
        print ("\t\tChi-squared for the combined fit is %.2f with %d DOF. PTE = %f"
               % (min_chi2_comb, ndf_combined, pte_combined))

    # Return the results of our calculations.
    if debug:
        return (delta_phi_tb_deg, symmetric_error_tb, chiSquaredTB,
                delta_phi_eb_deg, symmetric_error_eb, chiSquaredEB)
    else:
        return delta_phi_combined, symmetric_error_comb

##########################################################################################
def calculateKeatingPolarizationSelfCal(cls, fsky=0.0023,
                                        camb_filename='/data/hoover/camb_outputs/planck_bestfit/planck_lensing_wp_highL_bestFit_lensedtotCls.dat',
                                        ell_range=[400,3000],
                                        use_theoretical_error=False,
                                        detector_noise=10e-6,
                                        beam_fwhm=3.2e-4,
                                        quiet=False):
    """
    This function is an implementation of the prescription in
    B. Keating, M. Shimon, and A. Yadav, "Self-Calibration of CMB Polarization Experiments" (arXiv:1211.5734v1).
    The function checks for an overall focal plane rotation by using measured TB and EB spectra.
    If the averaged measured detector polarization angle is different from the true average
    detector polarization angle, then E-mode power will leak into B (and vice-versa, but B->E leakage
    is not very important). This will modify the BB spectrum, which is bad, but it will also leak
    power from TE to TB, and create a signal in EB. The TE and EB are expected to be zero
    under standard cosmology.
    Differences between this function and calculatePolarizationSelfCal: This function uses
    only diagonal errors, instead of the full covariance matrix. It uses theory Cls in
    calculation of the errors, which may not be appropriate (the theory Cls have different
    binning, and don't have beam effects, mode mixing, transfer functions, etc.).
    This function may allow calculation of a focal plane rotation in situation where one
    doesn't have a full covariance matrix.
        In practice, the only significant difference between this function and
    calculatePolarizationSelfCal is in the errors, and I trust the results of
    calculatePolarizationSelfCal more.
    INPUTS
        cls: (dictionary) A dictionary of C_ell values, as output by MapAnalyzer.calculateCls.
        fsky [0.0023]: (float) The fraction of the sky observed in the data which
            produced the input C_ells. The default is 95 deg^2.
        camb_filename []: (string) Read theory values from here. We use these to calculate the
            errors on the angle calibration.
        ell_range [[400,3000]]: (2-element list) Use only ells between these two values when
            doing computations.
        use_theoretical_error [False]: (bool) If True, calculate errors on TB and EB using
            equation 7 of the paper, ignoring anything in the input "cls".
        detector_noise [10e-6]: (float or dict) Noise level of the maps which created the input Cls.
            This is used only if the input Cls don't have "sigma_XX" entries with the noise
            estimates or if use_theoretical_error is True. I think (not certain) that this should
            be in units of (map_units)-arcmin. (I.e., uK-arcmin or K-arcmin.)
               If this is a float, we'll use the same noise level for T, E, and B. If it's a dictionary,
            it must have keys "T", "E", and "B", storing the noise levels in each of those maps.
        beam_fwhm [3.2e-4]: (float) The full width at half maximum of the beam which made the
            input cls. This is used only if the input Cls don't have "sigma_XX" entries with
            the noise estimates or if use_theoretical_error is True. I'm guessing that the
            units are radians, and the default, 3.2e-4, is 1.1 arcminutes.
        quiet [False]: (bool) Suppress all screen output.
    OUTPUT
        A 4-tuple with the focal plane rotation fit from the TB spectrum, the uncertainty on the
        TB spectrum fit, the angle fit from the EB spectrum, and the uncertainty on that fit.
    EXCEPTIONS
        None
    AUTHOR
        Stephen Hoover, 27 November 2013
    CHANGES
        3 Jan 2014: Use errors from the "cls" input. SH
        17 Jan 2014: Improved treatment of errors, and better capacity for pure theory errors. SH
    """
    warnings.warn("Please use calculatePolarizationSelfCal instead, if you can. The error treatment is better there.", DeprecationWarning)

#    # First check that we have the same ell binnings for each spectrum.
#    # We can't compare C_ells directly if they have different binning,
#    # and rebinning is beyond the scope of this function.
#    for spec_ells in cls['ell'].itervalues():
#        if len(spec_ells)!=len(cls['ell']['BB']) or not (spec_ells==cls['ell']['BB']).all():
#            raise ValueError("All spectra must have the same ell binning!")

    # Find the ell bins that we'll use to make this calculation.
    ell = cls['ell']['TT']
    ell2use = (ell > ell_range[0]) & (ell < ell_range[1])
    ell = ell[ell2use]

    ############################################################################
    # Start by calculating the maximum likelihood focal plane rotation angle.
    # We get two independent estimates, one from the TB spectrum and the other
    # from the EB spectrum (each of which we expect to be zero in standard
    # LCDM cosmology).
    ############################################################################

    # Define the (squared) errors on the power spectrum. If the "cls" input comes
    # from the cross-spectrum code, it already has an estimate of the error.
    # (Look for the "error_XX" keys, and fall back on "sigma_XX".)
    # If we don't have an error estimate in the input, we'll calculate it ourselves,
    # following eq. 7.
    n_maps = np.ceil(np.sqrt(cls['n_spectra']*2))
    if 'error_TB' in cls and not use_theoretical_error:
        if not quiet:
            print "Using TB errors from input C_ells dictionary."
        delta_cl_tb_sq = cls['error_TB'][ell2use]**2
    else:
        # If the "detector_noise" is a dictionary, separate it into T and B components.
        if not quiet:
            print "Calculating my own TB errors."

        if isinstance(detector_noise, dict):
            detector_noise_t = detector_noise['T']
            detector_noise_b = detector_noise['B']
        else:
            detector_noise_t = detector_noise_b = detector_noise
        cl_tt_tot = cls['TT'][ell2use] + detector_noise_t**2 * np.exp(ell**2 * beam_fwhm**2 / 8. / np.log(2))
        cl_bb_tot = cls['BB'][ell2use] + detector_noise_b**2 * np.exp(ell**2 * beam_fwhm**2 / 8. / np.log(2))
        delta_cl_tb_sq = (cl_tt_tot*cl_bb_tot) / (2*ell + 1) / fsky

    if 'error_EB' in cls and not use_theoretical_error:
        if not quiet:
            print "Using EB errors from input C_ells dictionary."
        delta_cl_eb_sq = cls['error_EB'][ell2use]**2
    else:
        # If the "detector_noise" is a dictionary, separate it into E and B components.
        if not quiet:
            print "Calculating my own EB errors."

        if isinstance(detector_noise, dict):
            detector_noise_e = detector_noise['E']
            detector_noise_b = detector_noise['B']
        else:
            detector_noise_e = detector_noise_b = detector_noise
        cl_ee_tot = cls['EE'][ell2use] + detector_noise_e**2 * np.exp(ell**2 * beam_fwhm / 8. / np.log(2))
        cl_bb_tot = cls['BB'][ell2use] + detector_noise_b**2 * np.exp(ell**2 * beam_fwhm / 8. / np.log(2))
        delta_cl_eb_sq = (cl_ee_tot*cl_bb_tot) / (2*ell + 1) / fsky

    # Next, calculate the intermediate terms from eq. 10.
    a_tb = np.sum( (1./2.) * cls['TB'][ell2use]*cls['TE'][ell2use] / delta_cl_tb_sq )
    b_tb = np.sum( (1./2.) * cls['TE'][ell2use]*cls['TE'][ell2use] / delta_cl_tb_sq )
    a_eb = np.sum( (1./2.) * cls['EB'][ell2use]*cls['EE'][ell2use] / delta_cl_eb_sq )
    b_eb = np.sum( (1./2.) * cls['EE'][ell2use]*cls['EE'][ell2use] / delta_cl_eb_sq )

    # Now estimate the angles (eq. 9)
    delta_phi_tb = np.rad2deg(-0.5*np.arcsin(a_tb / b_tb))
    delta_phi_eb = np.rad2deg(-0.25*np.arcsin(2*a_eb / b_eb))

    # Calculate the signal-to-noise for EB and TB (eq. 11)
    snr_tb = np.sqrt( np.sum( cls['TB'][ell2use]**2 / delta_cl_tb_sq / 2. ) )
    snr_eb = np.sqrt( np.sum( cls['EB'][ell2use]**2 / delta_cl_eb_sq / 2. ) )

    # Finally, calculate the uncertainties, using a theory spectrum. (eq. 12)
    theory_cls = files.readCAMBCls(camb_filename, as_cls=True, as_uksq=False, extrapolate=False, lensed=True)
    tcls = theory_cls['cl']
    tell = theory_cls['ell']
    #tell2use = (tell > ell_range[0]) & (tell < ell_range[1])
    tell2use = np.asarray( [ (thisell in ell.astype(int)) for thisell in tell] )
    tell = tell[tell2use]

    sigma_tb = np.sqrt(1. / (4*np.sum( (1./2.) * tcls['TE'][tell2use]**2 / delta_cl_tb_sq )))
    sigma_eb = np.sqrt(1. / (4*np.sum( (1./2.) * (tcls['EE'][tell2use]-tcls['BB'][tell2use])**2 / delta_cl_eb_sq )))

    # Convert the uncertainties from radians to degrees.
    sigma_tb = np.rad2deg(sigma_tb)
    sigma_eb = np.rad2deg(sigma_eb)

    # Done! Print the result to the screen, and return it as a tuple.
    if not quiet:
        print "Maximum likelihood rotation angle as estimated by"
        print "\tTB : %f +/- %f" % (delta_phi_tb, sigma_tb)
        print "\tEB : %f +/- %f" % (delta_phi_eb, sigma_eb)
        print "Signal-to-noise ratio for EB and TB detections:"
        print "\tTB : %f" % snr_tb
        print "\tEB : %f" % snr_eb

    return delta_phi_tb, sigma_tb, delta_phi_eb, sigma_eb


##########################################################################################
##########################################################################################
class MapMaker:
    """
    This object knows something about the telescope (has a Telescope object)
    and can perform fun and productive tasks with data.
    """

    ###################################################################
    def __init__(self, telescope=None, filters=[], ell_bins={}, foregrounds=None, simulated_data=False, scanrate=85., nsqarcmin=2250000.):
        """
        "foregrounds" should be a ForegroundMaker object, used to determine the coeffients that we'll use to combine multiple bands.
        "scanrate" is in arcminutes/second
        "nsqarcmin" is in arcminutes^2
        """
        self.telescope = telescope
        self.foregrounds = foregrounds
        self.scanrate = scanrate
        self.nsqarcmin = nsqarcmin

        if simulated_data:
            # Come up with an initial value for the band combination coefficients (we'll regenerate if necessary).
            self.createBandComboCoeffs(scanrate=self.scanrate, nsqarcmin=self.nsqarcmin, foregrounds=self.foregrounds, store_result=True)

    ###################################################################
    def getReceiver(self):
        return self.telescope.receiver
    receiver=property(getReceiver)

    ###################################################################
    def createBandComboCoeffs(self, scanrate, nsqarcmin, foregrounds=None, store_result=True):
        """
        Create a set of coefficients, used to combine _maps made in different bands into
        a single map. This is currently copied straight from the IDL simulation (main procedure).
        """

        # First, check to see if we've already calculated the band combination coefficients.
        try:
            if self._band_coeffs['scanrate']==scanrate and \
               self._band_coeffs['nsqarcmin']==nsqarcmin and \
               self._band_coeffs['foregrounds']==foregrounds:
                return self._band_coeffs['coeffs']
        except AttributeError:
            pass

        # If we don't have a self.band_coeffs or any of those fields,
        # or if we fail the if statement, then continue on to recalculate the coefficients.

        bolo = self.telescope.receiver.pixels[0][0] # Assume all bolos are the same in relevant properties.
        pixnoise_1am_white, noise_mjy = {}, []
        for band in self.telescope.receiver.bands:
            try:
                total_n_bolos = sum( [ bolo.quantity for bolo in self.telescope.receiver.iterBolosInBand(band) ] )
            except AttributeError:
                warnings.warn("Using self.telescope.receiver.n_bolos as the number of bolometers.", Warning)
                total_n_bolos = self.telescope.receiver.n_bolos
            pixnoise_1am_white[band] = np.sqrt( (bolo.white_noise_level/2) / (bolo.n_live_years * total_n_bolos/nsqarcmin) )
            noise_mjy.append(idl.idlRun("mjy_to_uk(%f*1e6,freq=%f,beams=90.,/inv)" % (pixnoise_1am_white[band], float(band))))


        f_knee = self.receiver.pixels[0][0].f_knee # Grab the f_knee from one bolo, assume it's the same everywhere.
        band_num = [float(band) for band in self.telescope.receiver.bands]
        pixnoise_1am_white_array = [pixnoise_1am_white[band] for band in self.telescope.receiver.bands] # Put this into an ordered array

        # Inputs common to both the foregrounds and no foregrounds cases.
        idl_inputs={'f_knee':f_knee, 'scanrate':scanrate, 'pixnoise_1am_white':pixnoise_1am_white_array, 'bands':band_num}
        if foregrounds and not foregrounds.no_foreground_removal:
            idl_command= """
            nbands = n_elements(bands)
            covtemp = make_sptpol09_cov(f_knee,scanrate,pixnoise_1am_white, bands=bands,fluxcut=fluxcut, fcband=fcband,fac_margin=1.)
            coeffs = combine_sptmaps_byell_getcoeffs(dblarr(nbands)+1.,covtemp)
            """
            idl_inputs.update({'fluxcut':self.foregrounds.flux_cut.values()[0], 'fcband':float(self.foregrounds.flux_cut.keys()[0])})
        else:
            idl_command = """
            nbands = n_elements(bands)
            covtemp = make_sptpol09_cov(f_knee,scanrate,pixnoise_1am_white,/noforeg,bands=bands)
            coeffs = combine_sptmaps_byell_getcoeffs(dblarr(nbands)+1.,covtemp)
            """

        band_combo_coeffs = idl.idlRun(idl_command, return_name='coeffs', **idl_inputs )

        band_combo_coeffs = dict( [ (band, band_combo_coeffs[index]) for index, band in enumerate(self.receiver.bands)] )

        if store_result:
            self._band_coeffs = {}
            self._band_coeffs['scanrate']=scanrate
            self._band_coeffs['nsqarcmin']=nsqarcmin
            self._band_coeffs['foregrounds']=foregrounds
            self._band_coeffs['coeffs'] = band_combo_coeffs

        return band_combo_coeffs


    ###################################################################
    def combineBands( self, maps, scan_speed, area_arcmin, window_func=None, foregrounds=None ):
        """
        Assume that input "maps" is a dictionary of PolarizedMaps, keyed by band.
        """
        if foregrounds is None:
            foregrounds = self.foregrounds

        combined_map = sky.PolarizedMap(**maps[self.receiver.bands[0]].initArgs(data_type=complex))
        if len(maps)==1:
            combined_map = maps[maps.keys()[0]] # If there's only one band, then there's nothing to combine.
        else:
            for band in self.receiver.bands:

                ellgrid = util.math.makeEllGrid(shape=maps[band].shape, resolution=maps[band].reso_rad)

                coefficients = self.createBandComboCoeffs(scanrate=scan_speed, nsqarcmin=area_arcmin,
                                                          foregrounds=foregrounds, store_result=True)

                coefficient_array = np.zeros(ellgrid.shape)
                coefficient_array = coefficients[band][ellgrid.astype(int)]

                map_fft = maps[band].getAllFFTs( sky_window_func=window_func )

                for pol in map_fft.polarizations:
                    combined_map[pol].map += map_fft[pol].applyWindow(coefficient_array)

        return combined_map.getInverseFFT(real_part=True)


    ###################################################################
    def __getattr__(self, name):
        if name=="band_combo_coeffs":
            self.band_combo_coeffs = self.createBandComboCoeffs
            return self.band_combo_coeffs
        else:
            raise AttributeError("'MapMaker' object has no attribute '"+name+"'")

    ###################################################################
    def setMapmaker(self):
        """
        Eventually, use this to set a default set of operations for processing
        timestreams into maps. Decide which filters to use, or whatever else.
        """
        pass

    ###################################################################
    def filterTimestreams(self, timestreams, filters=['all']):
        """
        This method performs various filtering tasks. Give it
        a list or array of timestreams and a list of filters you wish to
        apply, and it will modify the timestreams in place.
        """

        pass

    ###################################################################
    def reconstructTemperatureMap(self, timestreams, input_pointings, timeit=False, good_samples=None,
                                  radec_center=None, combine_bands=False, scan_weight=None,
                                  processing_dtype=np.float64, return_dtype=np.float64,
                                  inverse_noise_weighted_map=False, stopit=False, verbose=False):
        """
        From a list or array of timestreams, this method reconstructs
        the sky that they were looking at. It assumes that each
        timestream is tagged with the ID of the bolometer that
        made it, and the MapMaker must know about that bolometer!
        This function creates temperature maps. It will ignore any polarization
        sensitivity in the bolometers.
        Note: This function uses C-code via scipy.weave to speed up loops.
        The input_pointings must be type int64, and the timestreams must
        have floats. If you get mysterious segmentation faults, check
        your input data types!
        INPUTS
            good_samples [None]: (array of bools) Array with "True" for each
                element of the input timestreams which should be used, and
                "False" for each element which should be ignored. A value
                of None (the default) will result in all samples being used.
                Can also be an array of integers specifying which indices to use.
            radec_center [None]: (2-element array or list) [RA, dec] - not used
                in this function, but set as an attribute on maps created.
            scan_weight [None]: (dictionary of 1d arrays) Input dictionary of weight arrays matching
                lengths in timestreams list.
                If not set, uses default single weight for each bolo for entire observation.
            processing_dtype [np.float64]: (dtype) Use this type of data for the processing
                (we want greater precision when summing multiple timestreams together).
            return_dtype [np.float64]: (dtype) Return the maps holding this dtype
                (we may want something smaller, to save space).
            inverse_noise_weighted_map [False]: (bool) If True, don't divide the final
                map by its weights. This will make it easier to coadd large numbers of
                maps without encountering numerical issues in small-weight pixels.
            verbose [False]: If True, output more to the screen.
        """
        mem_usage_start = tools.resident()
        if verbose: print "Memory usage going into reconstructTemperatureMap: "+str((mem_usage_start)*1e-6)+" MB."

        # First, check the "pointings" input. If it's a single PointingSequence,
        # make a note of that, and we'll construct an individual pointing for
        # each bolo during the main loop. Note that,
        # if we do construct a list from a single entry, each list element will
        # point to the same PointingSequence object.
        pointings_is_boresight = False
        if isinstance(input_pointings, telescope.PointingSequence):
            pointings = [input_pointings]
            pointings_is_boresight = True
            #pointings = self.receiver.createPointings(input_pointings)
        else:
            pointings = input_pointings
            # If the input pointings is a length 1 list, then use the same pointing object
            # for all bolometers.
            if isinstance(pointings, list) and len(pointings)==1:
                pointings = tools.struct( [(_ts.bolo_id, pointings[0]) for _ts in timestreams] )


        mem_usage_ptgs = tools.resident()
        if verbose: print "Memory change after creating pointings: "+str((mem_usage_ptgs-mem_usage_start)*1e-6)+" MB."


        # In case the input timestreams were a dict, grab the values.
        try:
            timestreams = timestreams.values()
        except AttributeError:
            pass

        # Set default RA/dec
        if radec_center is None: radec_center=np.array([0.,0.])

        good_samples = self._standardizeSampleMaskInput(timestreams, good_samples)

        # Make sure that we have the same number of pointings as timesamples!
        if len(pointings[0]) != len(timestreams[0]) and not pointings_is_boresight:
            raise ValueError('You must have one pointing for each data point in the timestreams!')

        mem_usage_good_samp = tools.resident()
        if verbose: print "Memory change after good samples mask: "+str((mem_usage_good_samp-mem_usage_ptgs)*1e-6)+" MB."

        # Get the map parameters indicated in the PointingSequence.
        # NB: We assume that each PointingSequence goes to the same map, so
        # we take all map parameters from the first object!
        map_shape = pointings[0].map_shape
        map_active_shape = pointings[0].map_active_shape
        scan_speed = pointings[0].scan_speed # Needed when we're combining different frequency bands together.
        reso_arcmin = pointings[0].pixel_resolution_arcmin

        # Start looping over bands. Only do bands for which we have timestreams.
        reconstructed_map = tools.struct()
        timestream_bands = set( [self.receiver[_ts.bolo_id].band_name for _ts in timestreams])
        for band in timestream_bands:
            if timeit: start_time = time.clock()

            summed_pixel_map = sky.Map( shape =map_shape,
                                        active_shape = map_active_shape,
                                        data_type=processing_dtype,
                                        weight=np.zeros(map_shape, dtype=processing_dtype),
                                        reso_arcmin=reso_arcmin,
                                        projection=pointings[0].proj,
                                        center=radec_center,
                                        band=band, polarization="T" )

            weight_map = summed_pixel_map.weight #pixel_weighting_map.map # For the C code
            pixel_map = summed_pixel_map.map # For the C code

            for timestream in timestreams:
                if pointings_is_boresight:
                    these_pointings = self.receiver.getBoloPointing(timestream.bolo_id, pointings[0],
                                                                    clip_out_of_bounds=False)
                else:
                    these_pointings = pointings[timestream.bolo_id]
                # Verify that the pointings are the correct data type!
                if these_pointings[0].dtype != int or these_pointings[1].dtype != int:
                    raise ValueError("The pointing inputs must be ints!")
                pt_y, pt_x = these_pointings[0], these_pointings[1] # For the C code
                max_pt_y, max_pt_x = pixel_map.shape[:2]
                n_pointings = len(these_pointings)

                # One band at a time.
                if self.receiver[timestream.bolo_id].band_name != band: continue

                # Define a couple of temporary variables to save some computation time.
                ts_data = timestream.data.astype(processing_dtype) # For use in the C code
                if ((scan_weight) and (timestream.weight != 0)):  # keep zero if global is
                    ts_weight = (scan_weight[timestream.bolo_id] * timestream.data_valid * good_samples[timestream.id].astype(processing_dtype))
                else:
                    ts_weight = (timestream.weight * timestream.data_valid * good_samples[timestream.id]).astype(processing_dtype) # For use in the C code

                # Loop over all the pointings and sort them into the proper pixels.
                # Note that we ignore any pointings which would be off the map.
                c_code = """
                #line 457 "maps.py"
                for (int ipting=0; ipting<n_pointings; ++ipting) {
                   int this_pt_x = pt_x(ipting);
                   int this_pt_y = pt_y(ipting);
                   if (this_pt_x<0 || this_pt_y<0 || this_pt_x>=max_pt_x || this_pt_y>=max_pt_y)
                     continue;
                   pixel_map(this_pt_y, this_pt_x) += ts_weight(ipting)*ts_data(ipting);
                   weight_map(this_pt_y, this_pt_x) += ts_weight(ipting);
                } //End for (loop over pointings)
                """
                weave.inline(c_code, ['n_pointings','pt_y','pt_x','pixel_map',
                                      'weight_map', 'max_pt_y', 'max_pt_x','ts_data','ts_weight'],
                             type_converters=weave.converters.blitz)

            if timeit:
                end_time = time.clock()
                print "[reconstructMap] Long loop; map timestreams onto sky: ", \
                      end_time-start_time
                start_time = time.clock()

            reconstructed_map[band] = sky.ObservedSky(shape=map_shape, active_shape=map_active_shape,
                                                      data_type=processing_dtype,
                                                      projection=pointings[0].proj, center=radec_center,
                                                      polarization='T', reso_arcmin=reso_arcmin, band=band)


            # Store map's weight and pixel values.
            reconstructed_map[band].weight = weight_map
            reconstructed_map[band].map = pixel_map
            reconstructed_map[band].weighted_map = True # Still weighted at this point.

            # Divide out the weights, if desired.
            if not inverse_noise_weighted_map:
                reconstructed_map[band].removeWeight(in_place=True)

            # Convert the map data type to the return data type.
            reconstructed_map[band].changeDType(return_dtype, weight_dtype=return_dtype)


        if combine_bands and len(self.receiver.bands)>1:
            reconstructed_map = self.combineBands( reconstructed_map, scan_speed=scan_speed,
                                                   area_arcmin=reconstructed_map[self.receiver.bands[0]].area_arcmin )
        elif len(self.receiver.bands)==1:
            reconstructed_map = reconstructed_map[self.receiver.bands[0]]
        elif len(timestreams)==1:
            # If we only mapped one timestream, then don't return a pointless structure
            # with only one band in it.
            reconstructed_map=reconstructed_map[0]

        if stopit: pdb.set_trace()

        mem_usage_end = tools.resident()
        if verbose: print "Memory used at end of reconstructTemperatureMap: "+str((mem_usage_end)*1e-6)+" MB."

        else: return reconstructed_map

    ###################################################################
    def reconstructTimestreamMaps(self, timestreams, input_pointings, *args, **kwds):
        """
        Takes a set of timestreams, and, instead of combining them into a
        single map, returns a struct of timestream_id:map pairs, where
        each map is made from the data in a single timestream.
        Temperature maps only: one bolometer will not allow us to make
        polarized maps.
        This function is a wrapper for MapMaker.reconstructTemperatureMap.
        INPUTS
            timestreams: [dictionary of Timestreams] The timestreams that we'll
                use to construct the map.
            input_pointings: The timestream pointings, to be passed along to reconstructTemperatureMap.
            *args, **kwds: Arguments to pass along reconstructTemperatureMap.
        OUTPUT
            A struct of timestream_id:map pairs.
        """
        # First, check the "pointings" input. If it's a single PointingSequence,
        # change it into a list with one entry for each timestream. Note that,
        # if we do construct a list from a single entry, each list element will
        # point to the same PointingSequence object.
        if isinstance(input_pointings, telescope.PointingSequence):
            pointings = self.receiver.createPointings(input_pointings)
        else:
            pointings = input_pointings
            # If the input pointings is a length 1 list, then use the same pointing object
            # for all bolometers.
            if isinstance(pointings, list) and len(pointings)==1:
                pointings = tools.struct( [(_ts.bolo_id, pointings[0]) for _ts in timestreams] )

        _maps = tools.struct()
        for _ts in timestreams:
            _maps[_ts.id] = self.reconstructTemperatureMap([_ts], pointings, *args, **kwds)

        else:
            return _maps




    ###################################################################
    def reconstructMap(self, timestreams, input_pointings, combine_bands=True,
                       timeit=False, good_samples=None, scan_weight=None,
                       radec_center=None, processing_dtype=np.float64,
                       return_dtype=np.float64, inverse_noise_weighted_map=False,
                       stopit=False, verbose=False):
        """
        From a list or array of timestreams, this method reconstructs
        the sky that they were looking at. It assumes that each
        timestream is tagged with the ID of the bolometer that
        made it, and the MapMaker must know about that bolometer!
        This function only creates polarized maps. It will not work with
        non-polarization-sensitive bolometers.
        Note: This function uses C-code via scipy.weave to speed up loops.
        The input_pointings must be type int64, and the timestreams must
        have floats. If you get mysterious segmentation faults, check
        your input data types!
        INPUTS
            good_samples [None]: (array of bools) Array with "True" for each
                element of the input timestreams which should be used, and
                "False" for each element which should be ignored. A value
                of None (the default) will result in all samples being used.
                Can also be an array of integers specifying which indices to use.
            radec_center [None]: (2-element array or list) [RA, dec] - not used
                in this function, but set as an attribute on maps created.
            processing_dtype [np.float64]: (dtype) Use this type of data for the processing
                (we want greater precision when summing multiple timestreams together).
            return_dtype [np.float64]: (dtype) Return the maps holding this dtype
                (we may want something smaller, to save space).
            inverse_noise_weighted_map [False]: (bool) If True, don't divide the final
                map by its weights. This will make it easier to coadd large numbers of
                maps without encountering numerical issues in small-weight pixels.
        """
        # First, check the "pointings" input. If it's a single PointingSequence,
        # make a note of that, and we'll construct an individual pointing for
        # each bolo during the main loop. Note that,
        # if we do construct a list from a single entry, each list element will
        # point to the same PointingSequence object.
        pointings_is_boresight = False
        if isinstance(input_pointings, telescope.PointingSequence):
            pointings = [input_pointings]
            pointings_is_boresight = True
            #pointings = self.receiver.createPointings(input_pointings)
        else:
            pointings = input_pointings
            # If the input pointings is a length 1 list, then use the same pointing object
            # for all bolometers.
            if isinstance(pointings, list) and len(pointings)==1:
                pointings = tools.struct( [(_ts.bolo_id, pointings[0]) for _ts in timestreams] )

        if scan_weight is not None:
            raise NotImplementedError("scan_weight not yet implemented here!")

        # In case the input timestreams were a dict, grab the values.
        try:
            timestreams = timestreams.values()
        except AttributeError:
            pass

        # Set default RA/dec
        if radec_center is None: radec_center=np.array([0.,0.])

        good_samples = self._standardizeSampleMaskInput(timestreams, good_samples)

        # Make sure that we have the same number of pointings as timesamples!
        if len(pointings[0]) != len(timestreams[0]) and not pointings_is_boresight:
            raise ValueError('You must have one pointing for each data point in the timestreams!')

        # Get the map parameters indicated in the PointingSequence.
        # NB: We assume that each PointingSequence goes to the same map, so
        # we take all map parameters from the first object!
        map_shape = pointings[0].map_shape
        map_active_shape = pointings[0].map_active_shape
        scan_speed = pointings[0].scan_speed # Needed when we're combining different frequency bands together.
        reso_arcmin = pointings[0].pixel_resolution_arcmin

        # For each pixel on the map, we want to sum the "stokes parameter matrix"
        # of each bolometer that observes that pixel (once per observation),
        # as well as the product of the "stokes coupling" and the timestream
        # data element corresponding to the pixel. We'll end up with one 3x3 matrix
        # and one 3x1 matrix in each pixel. We then invert the 3x3 stokes parameter
        # matrix and multiply it by the 3x1 matrix. The resulting 3x1 matrix is
        # our reconstructed T,Q,U values. Note that this technique assumes
        # that the noise is white (no noise correlations between pixels)!
        reconstructed_map = tools.struct()
        cumulative_weight_map = tools.struct()
        for band in self.receiver.bands:
            if timeit: start_time = time.clock()
            summed_pixel_map = sky.Map( shape = np.append(map_shape, [1,3]),
                                        active_shape = np.append(map_active_shape, [1,3]),
                                        data_type=processing_dtype,
                                        reso_arcmin=reso_arcmin,
                                        center=radec_center,
                                        projection=pointings[0].proj,
                                        band=band, polarization="TQU" )

            stokes_map = np.zeros(np.append(map_shape, [3,3]), dtype=processing_dtype)
            pixel_map = summed_pixel_map.map # For the C code

            for timestream in timestreams:
                if pointings_is_boresight:
                    these_pointings = self.receiver.getBoloPointing(timestream.bolo_id, pointings[0],
                                                                    clip_out_of_bounds=False)
                else:
                    these_pointings = pointings[timestream.bolo_id]
                # Verify that the pointings are the correct data type!
                if these_pointings[0].dtype != int or these_pointings[1].dtype != int:
                    raise ValueError("The pointing inputs must be ints!")
                pt_y, pt_x = these_pointings[0], these_pointings[1] # For the C code
                max_pt_y, max_pt_x = pixel_map.shape[:2]
                n_pointings = len(these_pointings)

                # One band at a time.
                if self.receiver[timestream.bolo_id].band_name != band: continue

                # Define a couple of temporary variables to save some computation time.
                ts_bolo_stokes_matrix = np.asarray(self.receiver[timestream.bolo_id].stokes_matrix, dtype=processing_dtype)
                ts_bolo_stokes_coupling = np.asarray(self.receiver[timestream.bolo_id].stokes_coupling, dtype=processing_dtype)
                ts_data = timestream.data.astype(processing_dtype) # For use in the C code
                ts_weight = (timestream.weight * timestream.data_valid * good_samples[timestream.id]).astype(processing_dtype) # For use in the C code

                # Loop over all the pointings and sort them into the proper pixels.
                # Note that we ignore any pointings which would be off the map.
                c_code = r"""
                #line 706 "maps.py"
                for (int ipting=0; ipting<n_pointings; ++ipting) {
                   int this_pt_x = pt_x(ipting);
                   int this_pt_y = pt_y(ipting);
                   if (this_pt_x<0 || this_pt_y<0 || this_pt_x>=max_pt_x || this_pt_y>=max_pt_y)
                      continue;
                   //printf("   this_pt_x = %d, this_pt_y = %d\n", this_pt_x, this_pt_y);
                   for (int iy=0; iy<3; ++iy) {
                      for (int ix=0; ix<3; ++ix) {
                         stokes_map(this_pt_y, this_pt_x, iy, ix) += ts_weight(ipting)*ts_bolo_stokes_matrix(iy, ix);
                         } //End for (loop over ix)
                      pixel_map(this_pt_y, this_pt_x, 0, iy) += ts_weight(ipting)*ts_data(ipting)*ts_bolo_stokes_coupling(0, iy);
                      } //End for (loop over iy)
                } //End for (loop over pointings)
                """
                weave.inline(c_code, ['n_pointings','pt_y','pt_x','stokes_map','pixel_map',
                                      'ts_bolo_stokes_matrix', 'ts_bolo_stokes_coupling',
                                      'max_pt_y', 'max_pt_x', 'ts_data', 'ts_weight'],
                             type_converters=weave.converters.blitz)

            if timeit:
                end_time = time.clock()
                print "[reconstructMap] Long loop; map timestreams onto sky: ", \
                      end_time-start_time
                start_time = time.clock()


            reconstructed_map[band] = sky.PolarizedObservedSky(shape=map_shape, active_shape=map_active_shape,
                                                               data_type=processing_dtype,
                                                               projection=pointings[0].proj, center=radec_center,
                                                               reso_arcmin=reso_arcmin, band=band)

            # Put the weights and pixel values into the reconstructed_map object.
            reconstructed_map[band]['T'].map = pixel_map[:,:,0,0]
            reconstructed_map[band]['Q'].map = pixel_map[:,:,0,1]
            reconstructed_map[band]['U'].map = pixel_map[:,:,0,2]

            reconstructed_map[band].weight = stokes_map
            reconstructed_map[band].weighted_map = True # Values still multiplied by weights for now.
            reconstructed_map[band].setMapAttr('weighted_map',True)

            # Divide out the weights, if desired.
            if not inverse_noise_weighted_map:
                reconstructed_map[band].removeWeight(in_place=True)

            if timeit:
                end_time = time.clock()
                print ("[reconstructMap] %sap-building: " % ("M" if inverse_noise_weighted_map else "Matrix inversion and m")),end_time-start_time

            # Convert the map data type to the return data type.
            reconstructed_map[band].changeDType(return_dtype, weight_dtype=return_dtype)


        if combine_bands and len(self.receiver.bands)>1:
            reconstructed_map = self.combineBands( reconstructed_map, scan_speed=scan_speed,
                                                   area_arcmin=reconstructed_map[self.receiver.bands[0]].area_arcmin )

            #ellgrid = util.makeEllGrid(shape=reconstructed_map.values()[0].shape, resolution=reconstructed_map.values()[0].reso_rad)

            #coefficients = self.createBandComboCoeffs(scanrate=scan_speed, nsqarcmin=reconstructed_map.area_arcmin,
            #                                          foregrounds=self.foregrounds, store_result=True)

            #coefficient_array = np.zeros(ellgrid.shape)
            #coefficient_array = coefficients[ellgrid.astype(int)]

            #map_fft = dict([(pol, np.fft.fft2(map[pol].map * sky_window_func)) for pol in map.polarizations])

            #reconstructed_map = reconstructed_map.values()[0]
        elif len(self.receiver.bands)==1:
            reconstructed_map = reconstructed_map[self.receiver.bands[0]]

        if stopit: pdb.set_trace()
        else: return reconstructed_map




    ###################################################################
    def _standardizeSampleMaskInput(self, timestreams, good_samples):
        """
        The mapmakers take a mask of samples as input. This function checks that input and
        outputs it in a standardized form.
        INPUTS
            timestreams : An iterable of the timestreams passed to the mapmaker.
            good_samples : Iterable of booleans (mask for timesamples) or iterable
                of integers (designating good timesamples). May also be a dictionary
                of either of those, with one entry for each timestream, keyed by channel ID.
        OUTPUT
            A dictionary, keyed by channel ID, of timesample masks for each channel.
        """

        ####################################################################
        # Check the "good_samples" input and convert to the necessary format if needed.

        # If no subset of samples is specified, use them all.
        # If a subset was specified, make sure that the format is valid.
        if good_samples is None: good_samples = np.ones(len(timestreams[0]), dtype=bool)

        def checkGoodSamples(good_samples):
            # Verify that the input is the right shape, and force it to be a boolean mask array.
            good_samples = np.asarray(good_samples)
            if good_samples.dtype != bool:
                # If the input array wasn't bools, then assume it was an index array, and use it to
                # construct an array of bools.
                good_sample_temp = np.zeros(len(timestreams[0]), dtype=bool)
                good_sample_temp[good_samples] = True
                good_samples = good_sample_temp
                #warnings.warn('The "good_samples" mask has type '+str(good_samples.dtype)+", but I'm casting it as a bool!", Warning)
                #good_samples = np.asarray(good_samples, dtype=bool)
            if len(good_samples) != len(timestreams[0]):
                raise ValueError('The "good_samples" mask must have the same length as the input timestreams!')
            return good_samples

        if isinstance(good_samples, dict):
            # Check that every timestream has a mask.
            for _ts in timestreams:
                assert _ts.id in good_samples, "Every Timestream needs a mask input!"

            for _id in good_samples:
                good_samples[_id] = checkGoodSamples(good_samples[_id])
        else:
            # We want the "good_samples" input in the form of a dictionary with a boolean
            # mask for each timestream. If we get a single array, create a dictionary that
            # uses that same mask for each timestream.
            good_samples = checkGoodSamples(good_samples)
            good_samples = dict( [ (ts.id, good_samples) for ts in timestreams])

        return good_samples

##########################################################################################
def basicMapper(data, good_bolos='flagged', reso_arcmin=2., proj=0,
                t_only=False, individual_ts_maps=False, use_boresight_pointing=False,
                use_source_offset_pointing=False,
                pixel_differenced_maps=False, use_leftgoing=None, map_center='source',
                map_shape=None, ignore_features=False, use_azel_pointing=False,
                skip_bad_scans=True, pointing_offsets_in_angle_space=True,
                remove_planet_movement=True, use_scan_weights=False,
                inverse_noise_weighted_map=False,
                timestream_filtering={}, use_c_pointing=True,
                tchunks=1,
                suppress_git_hashtag=False,
                remove_empty_maps=False,
                timeit=True, verbose=True, debug=0, stopit=False):
    """
    Take timestream data from an SPTDataReader and bin it into a map.
    INPUTS
        data: (SPTDataReader) The telescope data.
        good_bolos ['flagged']: (string or list) Which bolometers should be combined
            into the final map? This input may be a list of bolometers (either IDs or
            indices) or a string which chooses a method for picking which bolos are good.
            Valid methods include 'all', 'flagged', 'calibrator', and 'elnod'.
            See sptpol_software.analysis.processing.getGoodBolos for a full list of valid
            methods.  Note that if the input is a list of bolo IDs or indices (e.g. from an
            earlier call to getGoodBolos()), no information on which bolos were cut (regardless of
            whether return_cuts_accounting is True) will be returned.
        reso_arcmin [2.]: (float) The desired resolution of the resulting map, in arcminutes per pixel.
        proj [0]: (int or string) Which map projection should I use to turn angles on the
            curved sky into pixels on a flat map? May be an integer index, or string
            name of the projection.
        t_only [False]: (bool) If True, plot only the temperature component of the map,
            ignoring any polarization information.
        individual_ts_maps [False]: (bool) If True, don't combine timestreams into a single
            map, but rather output a struct of individual maps for each timestream.
        use_boresight_pointing [False]: (bool) If True, ignore any information we have about
            individual pixel pointing offsets, and create the maps using only the telescope pointing.
            I suggest only using this in conjunction with individual_ts_maps.
        use_source_offset_pointing [False]: (bool) If True, use the offset from the source
            (as determined from tracker registers) in a coordinate system where the source
            is rotated to be at (0,0). Uses actual telescope az/el, and geocentric source
            position. Does not include any kind of online or offline pointing model.
        pixel_differenced_maps [False]: (bool) If True, create a separate map for each pixel,
            created from the differenced timestream of the two detectors in that pixel.
            This is like the "individual_ts_maps" option, but for pixel-differenced
            timestreams instead of individual detector timestreams.
        use_leftgoing [None]: If None, use both leftgoing and rightgoing scans.
            If True, use leftgoing scans only. If False, use rightgoing scans only.
        map_center ['source']: (2-tuple or 2-element array) If not None, should be the
            RA, dec of the desired map center, in degrees. If 'source', we will
            attempt to look up the RA and dec of the observed source.
            If None, the map center will be the mean RA and mean dec of
            the telescope pointings.
        map_shape [None]: (2-tuple or 2-element array) If not None, should be the
            desired shape of the map in (degrees of delta dec, degrees of delta RA)
            (where degrees are degrees on the sky - they will be naively converted
            to pixels using the reso_arcmin value).
            IMPORTANT: This ordering is the opposite of map_center's ordering.
            The management would like to apologize for any confusion this may have caused.
        ignore_features [False]: (bool) If True, ignore any feature bits which might or
            might not be present.
        use_azel_pointing [False]: (bool) If True, use the telecope pointing in aziumth and elevation
            (taken from antenna.track_actual) rather than the default of RA and dec. Note that,
            if use_azel_pointing is True, the map_center should be given in degrees of [az, el],
            and the map_shape in degrees of [delta_el, delta_az].
        skip_bad_scans [True]: (bool) If True, check the flags on each scan, and only put
            data from the good scans in the map.
        pointing_offsets_in_angle_space [True]: (bool) Apply bolometer pixel offsets to the boresight
            RA/dec, rather than simply adding in pixel space. Calculating RA/dec of each
            detector is more resource intensive (more computing time, more memory), but more
            accurate, especially over large fields.
        remove_planet_movement [True]: (bool) If True, and if the data.header.source is a
            planet name, modify the true pointing so that the planet appears to be stationary
            in the map during the period of observation.
        use_scan_weights [False]: (bool) Generate dynamic weights for each bolo on a scan-by-scan basis.
            weights are calculated as inverse square of the RMS of that bolo for that scan.
            Then passed to the map maker of choice
        inverse_noise_weighted_map [False]: (bool) If True, don't divide the final
            map by its weights. This will make it easier to coadd large numbers of
            maps without encountering numerical issues in small-weight pixels.
        timestream_filtering [{}]: (dict) If this dictionary contains parameters, call the
            cFiltering function with those parameters.
        use_c_pointing [True]: (bool) If True, use compiled C code from sptpol_cdevel to find
            the map pixel pointing of each detector.
        tchunks [1]: (int) Number of pieces to break observation time into when correcting
            for planet movement during the observation.  Each piece is linearly interpolated over
            to find the planet movement correction.
        suppress_git_hashtag [False]: (bool) By default, we put the hashtag of the current
            git repository version in the map metadata. If this argument is True, skip
            that. It can sometimes take a second or two for this call -- suppress it
            only if you're making lots of tiny maps at once.
        remove_empty_maps [False]: (bool) If True, remove any maps with no detectors contributing
            data from the output dictionary. If the result is only one map, return that map
            by itself (not in a dictionary). The default is to return a struct with one map
            for each band present in the input Receiver object.
        timeit [True]: (bool) Output time elapsed after each step of the computation.
        verbose [True]: (bool) Show additional screen output.
        debug [0]: (int) Make some modifications to test the code. Selectable levels.
        stopit [False]: (bool) For debugging. Run a pdb.set_trace() immediately before returning.
    OUTPUT
        A PolarizedObservedSky object, (from the sptpol_software.observation.sky module)
        or an ObservedSky object (if t_only=True).
    EXCEPTIONS
        ValueError : If no good channels are found.
        ValueError : If no frames have appropriate "analyze this" feature bits.
    CHANGES
        17 Feb 2014: Store mapper function arguments in the map metadata. SH
        7 Apr 2014: Add "is_trail_field" to map metadata. SH
    """
    # Store function arguments for use in map metadata. This must go before everything else!
    mapper_name = 'basicMapper'
    myargs = tools.myArguments(remove_self=True, remove_data=True, combine_posargs=True)

    import sources

    if use_leftgoing==True:
        print "Only making maps from leftgoing scans."
    elif use_leftgoing==False and use_leftgoing is not None:
        print "Only making maps from rightgoing scans."

    # If we're making individual pixel-differenced maps, then make sure that
    # 'full_pixel' is in the good_bolos list.
    if pixel_differenced_maps:
        good_bolos = set(good_bolos) | set(['full_pixel'])

    # For testing polarized mapping with the (unpolarized) SPTsz data.
    # Add a fake polarization angle to the bolometers.
    if debug==1:
        data.rec.setAllBolos("pol_eff", 0.99)
        for i in range(960):
            if i%4==1: data.rec.bolos[i].pol_angle+=90.
            if i%4==2: data.rec.bolos[i].pol_angle+=45.
            if i%4==3: data.rec.bolos[i].pol_angle-=45.

    start_time = spt_time.now()

    # Find the list of bolometers to use for this map, if we weren't given them already.
    if timeit or verbose: print "Finding good bolometers..."
    get_good_out  = processing.getGoodBolos(data, good_bolos, reset=False, return_cuts_accounting=True, verbose=verbose)
    good_bolos = get_good_out[0]
    cuts_accounting = get_good_out[1]

    if pixel_differenced_maps:
        good_bolos = set( [ data.rec[_id].pixel_id for _id in good_bolos] )
    time_after_goodbolos = spt_time.now()
    if len(good_bolos)==0:
        raise ValueError("No good bolometers found.")
    if timeit or verbose: print ("       Finished. I found "+str(len(good_bolos))+" good "+
                                 ("pixels" if pixel_differenced_maps else "bolometers")+
                                 ". Time elapsed = "+str(time_after_goodbolos-start_time)+" seconds.")

    # Now separate out the timestreams that we want to use.
    if pixel_differenced_maps:
        good_ts = [data.pixeldata[pixel] for pixel in good_bolos]
    else:
        good_ts = [data.bolodata[bolo] for bolo in good_bolos]

    # Generate scan weights, if desired
    if use_scan_weights:
        scan_weight = struct()
        for bolo in good_bolos:
            ts_weight = np.zeros(len(data.bolodata[bolo].data))
            for scan in data.scan:
                temp_weight = 1./scan.bolo_rms[data.rec[bolo].index]**2
                if np.isfinite(temp_weight): # else leave zero
                    ts_weight[scan.scan_slice] = np.repeat(temp_weight,scan.n_samples)
            scan_weight[bolo] = ts_weight
    else:
        scan_weight = None

    #print 'returning test mode data'
    #test = {'good_ts':good_ts, 'scan_weight':scan_weight}
    #return test

    # We only want to map data points that are in scans and are marked
    # as "analyze this" in the log files. Find those samples now.
    if not ignore_features:
        frames_to_analyze = where(tools.bitOn(data.array.features, data.fbits['analyze'], only=True) |
                                  tools.bitOn(data.array.features, (data.fbits['analyze'],data.fbits['trail']), only=True))[0]
        if len(frames_to_analyze)==0:
            raise ValueError("I can't find any data with only an f0 feature bit marked. Use ignore_features=True if you're sure you want to map these data.")
    else:
        frames_to_analyze = None

    # Create "inscan_samples" to keep track of good samples apart from the scan-by-scan channel mask.
    inscan_samples = data.getIndicesInScans(frames=frames_to_analyze, use_leftgoing=use_leftgoing,
                                            good_scans_only=skip_bad_scans, dict_by_channel=False, boolean_mask=True)
    samples_to_use = data.getIndicesInScans(frames=frames_to_analyze, use_leftgoing=use_leftgoing,
                                            good_scans_only=skip_bad_scans, dict_by_channel=good_bolos, boolean_mask=True)

    # Now we want to convert our telescope pointings (in RA and dec) to pixel
    # coordinates so that we can stack up all our bolometer measurements. To do that,
    # we need the center of the map, and the dimensions of the map. Find those now.
    # We only want the map to cover pointings to samples that we're actually going
    # to use, so pull those out too.
    if use_azel_pointing:
        ptg_x = data.antenna.track_actual[0]
        ptg_y = data.antenna.track_actual[1]
    elif use_source_offset_pointing:
        print "Using angle offset from source (with no pointing model corrections) for the boresight pointing."
        ptg_y, ptg_x = sources.getBoresightOffsetFromSource(data, use_actual_pointing=True, assume_perfect_telescope=True)
    else:
        ptg_x = data.antenna.ra
        ptg_y = data.antenna.dec
        if remove_planet_movement:
            try:
                # Try to find a correction for the planet pointing. If we can't, then this isn't a planet.
                ra_corr, dec_corr = sources.interpolatePlanetRADecCorrection(data.header.source, data.telescope.location,
                                                                             data.observation.start_time,
                                                                             data.observation.stop_time,
                                                                             data.observation.n_samples,
                                                                             tchunks)
                if verbose:
                    print "Correcting for the movement of %s during the observation." % data.header.source
                    print ("   %s moved %.2f arcminutes in RA and %.2f arcminutes in dec during this observation."
                           % (data.header.source.capitalize(), (ra_corr[-1]-ra_corr[0])*60, (dec_corr[-1]-dec_corr[0])*60))

                ptg_x = ptg_x - ra_corr
                ptg_y = ptg_y - dec_corr
            except ValueError: pass

    # radec is only used to figure out map shape and center if we didn't have those supplied.
    radec = [ptg_x[inscan_samples], ptg_y[inscan_samples]]
    # If we didn't get a map_shape and/or radec_center input, try to figure out something reasonable from the data.

    if use_source_offset_pointing:
        radec_center = np.asarray([0., 0.])
        if verbose: print "Forcing the map center to be at (0,0), the tracked source location in these coordinates."
    elif map_center is None:
        if use_azel_pointing==True:
            #note that this isn't actually radec, it's just called that
            radec_center = [np.mean(radec[0]),np.mean(radec[1])]
            if verbose: print "Centering map on Az, El: %s." % str(radec_center)
        else:
            radec_center = [np.mean(radec[0]), np.mean(radec[1])]
            if verbose: print "Centering map on RA, dec: %s." % str(radec_center)
        if verbose: print "Centering map on RA, dec: %s." % str(radec_center)
    elif map_center=='source':
        # Get the source center. In case this is a planet observation, set the time of observation
        # to the middle of the observation period.
        data.telescope.location.date = data.header.start_date + (data.header.stop_date-data.header.start_date)/2
        radec_center = sources.getSourceCenter(data.header.source, data.telescope.location)
        if verbose: print "Centering map on source %s, RA, dec: %s." % (data.header.source, str(radec_center))
    else:
        radec_center=np.asarray(map_center)
        # Make sure that the radec_center is between 0 and 360.
        radec_center[0] = radec_center[0] % 360
        if verbose:
            if use_azel_pointing: print "Centering map on AZ, EL: %s." % str(radec_center)
            else: print "Centering map on RA, dec: %s." % str(radec_center)

    if map_shape is None:
        delta_ra = radec[0].max() - radec[0].min()
        delta_dec = radec[1].max() - radec[1].min()
        map_shape = np.ceil(1.0*(np.array([delta_dec, delta_ra]) + np.array([1.1,1.1]))*60/reso_arcmin)
        map_shape_deg = np.ceil(1.0*(np.array([delta_dec, delta_ra]) + np.array([1.1,1.1])))
    else:
        map_shape_deg = map_shape
        map_shape = np.ceil(np.asarray(map_shape,dtype=np.float64)*60./reso_arcmin)

    # Now that we know the shape and center point of the map, we can run the angle->pixel conversion.
    if timeit or verbose: print "Converting pointing to pixel coordinates..."
    mapinfo = c_interface.MAPINFO(proj=proj, map_center=radec_center, map_shape=map_shape_deg, reso_arcmin=reso_arcmin)
    if use_c_pointing and not use_boresight_pointing:
        try:
            pix_pointing = sky.cPointingToPixels([ptg_x,ptg_y], data, good_bolos=good_bolos, mapinfo=mapinfo,
                                                 flattened_pointing=False, in_scan_only=True,
                                                 set_scan_flags=True, set_scan_bolometer_flags=False,
                                                 quiet=(not verbose), verbose=False)
        except OSError:
            if verbose: print "Can't use C code to get detector pointings. Reverting to python pointing offset code."
            pix_pointing = sky.ang2Pix([ptg_x, ptg_y], radec_center, reso_arcmin, map_shape, round=True, proj=proj, return_validity=False)
    else:
        if pointing_offsets_in_angle_space and not use_boresight_pointing:
            pix_pointing = sky.pointingToPixels([ptg_x,ptg_y], data, mapinfo=mapinfo, good_bolos=good_bolos)
        else:
            pix_pointing = sky.ang2Pix([ptg_x, ptg_y], radec_center, reso_arcmin, map_shape, round=True, proj=proj, return_validity=False)

    time_after_ang2pix = spt_time.now()
    if timeit or verbose: print "       Finished. Time elapsed = "+str(time_after_ang2pix-start_time)+" seconds."
    if debug:
        # Debugging: To make sure that the Python ang2Pix works, run the IDL version so we can compare outputs.
        idlout = idl.idlRun("ang2pix_proj0, ra, dec, npixels, radec0, reso_arcmin, ipix, xpix=xpix, ypix=ypix", return_all=True, ra=radec[0], dec=radec[1], npixels=map_shape, radec0=radec_center, reso_arcmin=reso_arcmin)

    #########################################
    # Run timestream filtering, if requested.
    if timestream_filtering:
        processing.cFiltering(data, good_bolos, map_parameters={"proj":proj, "map_center":radec_center,
                                                                "map_shape":map_shape_deg, "reso_arcmin":reso_arcmin},
                              set_scan_flags=True, set_scan_bolometer_flags=True,
                              verbose=False, quiet=(not verbose),
                              **timestream_filtering)

    # Now we've accumulated all the necessary information to make our map. Create a MapMaker object,
    # then use it to put all of the timestream data into a map!
    map_maker = MapMaker(data.telescope)
    if individual_ts_maps or pixel_differenced_maps:
        map_making_func = map_maker.reconstructTimestreamMaps
    elif t_only:
        map_making_func = map_maker.reconstructTemperatureMap
    else:
        map_making_func = map_maker.reconstructMap

    # If the mapmaker gets a single PointingSequence object, it
    # will expand that to individual pixel pointings using the data
    # stored in each Pixel. If it receives a list, then it will take
    # that value for pointings and ignore what it knows about the pixels.
    if use_boresight_pointing:
        pointing_to_use = [pix_pointing]
    else:
        pointing_to_use = pix_pointing
        if proj!=5 and not pointing_offsets_in_angle_space:
            warnings.warn('\n   PIXEL POINTINGS OFFSETS IN PIXEL SPACE ARE ONLY TREATED CORRECTLY IN PROJ 5. YOU CHOSE PROJ '+str(proj)+"!\n", SyntaxWarning)

    start_mapmaking = spt_time.now()
    if verbose and individual_ts_maps: "Starting to make individual maps."
    elif timeit or verbose: print "Reconstructing the map."
    _map = map_making_func(good_ts, pointing_to_use, combine_bands=False, radec_center=radec_center,
                           timeit=False, good_samples=samples_to_use, scan_weight=scan_weight,
                           inverse_noise_weighted_map=inverse_noise_weighted_map,
                           stopit=stopit, verbose=bool(debug))
    if timeit: print "Took "+str(spt_time.now() - start_mapmaking)+" seconds for the mapping."

    # Add the processing_applied information to the map, and
    # see if we can put a source name in the maps.
    source_name = data.header.source
    if source_name is None:
        if max(data.observation.sources_tracked.values()) / float(data.header.n_frames) > 0.6:
            source_name = data.observation.sources_tracked.keys()[np.array(data.observation.sources_tracked.values()).argmax()]
    n_scans_used = data.nScansFromIndices(inscan_samples)

    current_git_tag = None
    if not suppress_git_hashtag:
        current_git_tag = tools.getGitCommitHashEnv('SPTPOL_SOFTWARE') # Only call this once: it takes a while!

    if isinstance(_map, dict):
        for this_band, this_map in _map.iteritems():
            this_map.processing_applied = data.header.processing_applied
            this_map.mapper_arguments = myargs
            if individual_ts_maps or pixel_differenced_maps:
                channel_ids = [this_band] # this_band is actually the bolo name for individual_ts_maps
            else:
                channel_ids = [_ts.id for _ts in good_ts if data.rec[_ts.id].band_name==this_band]
            this_map.setMapAttr('channel_ids', channel_ids)
            this_map.setMapAttr('n_channels', len(channel_ids))
            this_map.setMapAttr('name',source_name)
            this_map.setMapAttr('units',data.header.bolodata_units)
            this_map.setMapAttr('n_scans',n_scans_used)
            this_map.setMapAttr('start_date',data.header.start_date)
            this_map.setMapAttr('stop_date',data.header.stop_date)
            if use_leftgoing == True:
                this_map.setMapAttr('is_leftgoing','left')
            elif use_leftgoing == False:
                this_map.setMapAttr('is_leftgoing','right')
            else:
                this_map.setMapAttr('is_leftgoing','both')
            this_map.setMapAttr('is_trail_field',data.data['observation'].get('is_trail_field',None))
            this_map.cuts = cuts_accounting

            # Record the optics bench position. Check to be sure it's not changing (it really shouldn't).
            try:
                if not (np.diff(data.antenna.scu_benchoff, axis=1)==0.).all():
                    warnings.warn("\n----The optics bench position changed during this mapping!",RuntimeWarning)
                this_map.setMapAttr('optics_bench_offset', data.antenna.scu_benchoff[:,0])
            except AttributeError: pass
            this_map.setMapAttr('git_commit_tag_data', data.header.get('git_commit_tag',None))
            this_map.setMapAttr('git_commit_tag_map', current_git_tag)

            # Commented the flattenPol out -- Duncan recommends that this be done after removeWeight has been called.
            ## Try to remove projection distortions from the map. Will get (and catch) an
            ## exception if this isn't a PolarizedMap.
            #try:
            #    this_map.flattenPol()
            #    print "Rotated Q/U in the %s map so that the zero angle of polarization is toward the top of the map." % this_band
            #except AttributeError:
            #    pass

        # Check for and remove any empty maps.
        if remove_empty_maps:
            empty_maps = []
            for band, this_map in _map.iteritems():
                if this_map.n_channels==0:
                    empty_maps.append(band)
            for band in empty_maps:
                del _map[band]
            if len(_map)==1:
                _map = _map[0]
    else:
        _map.processing_applied = data.header.processing_applied
        _map.mapper_arguments = myargs
        _map.setMapAttr('n_channels', len(good_ts))
        _map.setMapAttr('channel_ids',[_ts.id for _ts in good_ts])
        _map.setMapAttr('name',source_name)
        _map.setMapAttr('units',data.header.bolodata_units)
        _map.setMapAttr('n_scans',n_scans_used)
        _map.setMapAttr('start_date',data.header.start_date)
        _map.setMapAttr('stop_date',data.header.stop_date)
        if use_leftgoing == True:
            _map.setMapAttr('is_leftgoing','left')
        elif use_leftgoing == False:
            _map.setMapAttr('is_leftgoing','right')
        else:
            _map.setMapAttr('is_leftgoing','both')
        _map.setMapAttr('is_trail_field',data.data['observation'].get('is_trail_field',None))
        _map.cuts = cuts_accounting

        # Record the optics bench position. Check to be sure it's not changing (it really shouldn't).
        try:
            if not (np.diff(data.antenna.scu_benchoff, axis=1)==0.).all():
                warnings.warn("\n----The optics bench position changed during this mapping!",RuntimeWarning)
            _map.setMapAttr('optics_bench_offset', data.antenna.scu_benchoff[:,0])
        except AttributeError: pass
        _map.setMapAttr('git_commit_tag_data', data.header.get('git_commit_tag',None))
        _map.setMapAttr('git_commit_tag_map', current_git_tag)

        # Commented the flattenPol out -- Duncan recommends that this be done after removeWeight has been called.
        ## Try to remove projection distortions from the map. Will get (and catch) an
        ## exception if this isn't a PolarizedMap.
        #try:
        #    _map.flattenPol()
        #    print "Rotated Q/U so that the zero angle of polarization is toward the top of the map."
        #except AttributeError:
        #    pass

    # Done! If we want extra debugging information, then return the Python and IDL pixel pointings.
    # Otherwise, it's just the map.
    if timeit or verbose: print "Finished quickmap! Total time elapsed: "+str(spt_time.now() - start_time)+" seconds."
    if stopit: pdb.set_trace()
    if debug: return _map, pix_pointing, idlout
    else: return _map
##########################################################################################
def angleGroupMapper(data, good_bolos=['flagged','has_pointing'], verbose=True,
                     quiet=False, debug=False, **mapper_kwargs):
    """
       This function creates separate maps for each angle group of detectors. An
    "angle group" is a set of detectors on the same module which all share
    the same nominal polarization angle and the same frequency band.
       This function calls basicMapper repeatedly to create the angle group maps.
    See the basicMapper doc string for the arguments available.
    INPUTS
        data: (SPTDataReader) The telescope data.
        good_bolos [['flagged', 'has_pointing']]: (string or list) Which bolometers should be combined
            into the final map? This input may be a list of bolometers (either IDs or
            indices) or a string which chooses a method for picking which bolos are good.
        verbose [True]: (bool) Extra screen output.
        quiet [False]: (bool) If True, suppress screen output.
        debug [False]: (bool) If True, will pass "True" to basicMapper's "verbose" argument.
            Produces lots of screen output!
        **mapper_kwargs: These arguments will be passed to basicMapper. See that function
            for the arguments available.
    OUTPUT
        A dictionary of Maps or PolarizedMaps, keyed by angle group name.
    EXCEPTIONS
        ValueError : If no good channels are found.
        ValueError : If no frames have appropriate "analyze this" feature bits.
    AUTHOR
        Stephen Hoover, 23 August 2013
    CHANGES
        12 Nov 2013: Add band_name to the angle group names. SH
    """
    start_time = spt_time.now()

    # We'll need the Receiver to find angle groups.
    rec = data.telescope.receiver

    # Turn the good_bolos input into a list so it's easy to combine with lists of bolos
    # in an angle group.
    good_bolos = processing.getGoodBolos(data, good_bolos, verbose=verbose)

    mapper_kwargs = mapper_kwargs.copy()
    mapper_kwargs.setdefault('inverse_noise_weighted_map', True)
    mapper_kwargs['remove_empty_maps'] = True
    mapper_kwargs['timeit'] = False

    # Make a map for each individual angle group.
    angle_group_maps = struct()
    for module in rec.modules:
        if not quiet: print "Summing detectors in module %s." % module
        pol_angles = rec.getBolos('pol_angle_nominal_deg', module_id=module, as_set=True) - set([0.])
        bands = rec.getBolos('band_name', module_id=module, as_set=True)

        # Each set of bolometers with the same nominal angle and band in this module form one angle group.
        for band_name in bands:
            for angle in pol_angles:
                if not np.isfinite(angle): continue

                bolo_ids = rec.getBolos('id', module_id=module, pol_angle_nominal_deg=angle, band_name=band_name)
                map_name = '%s_%.1f_%s' % (module, angle%180, band_name)

                # Tell the basic mapper to make a map only with our small subset of detectors.
                these_bolos = set(good_bolos) & set(bolo_ids)
                if len(these_bolos)==0: continue

                if not quiet:
                    print "\n----Making maps for angle group %s." % map_name
                this_angle_group_map = basicMapper(data, these_bolos, verbose=debug, **mapper_kwargs)
                if this_angle_group_map.n_channels > 0:
                    angle_group_maps[map_name] = this_angle_group_map

    time_finished_angle_group_maps = spt_time.now()
    if not quiet:
        print "Finished making angle group maps. It took %.2f seconds." % (time_finished_angle_group_maps-start_time).total_seconds()

    return angle_group_maps

##########################################################################################

def matchedBeamMapper(data, detector_beams= '/data/hoover/templates_zero_zero/angle_beams_from_mars.h5',
                      return_angle_group_maps=False,
                      good_bolos=['flagged','has_pointing'], reso_arcmin=2., proj=0,
                      t_only=False,
                      use_source_offset_pointing=False,
                      pixel_differenced_maps=False, use_leftgoing=None, map_center='source',
                      map_shape=None, ignore_features=False, use_azel_pointing=False,
                      skip_bad_scans=True, use_scan_weights=False,
                      inverse_noise_weighted_map=False,
                      timestream_filtering={}, use_c_pointing=True,
                      ultra_mode=False,
                      quiet=False, verbose=False, stopit=False):
    """
    TODO: When combining angle group maps into an output map, account for the different bands!
       The SPTpol detectors have differently shaped beams, depending on the detector's angle
    orientation and location on the focal plane. The shape differences are unimportant for
    temperature-only maps (they only serve to make the temperature beam a bit odder-looking),
    but can cause problems for polarization maps at small angular scales.
       This function corrects for mismatched beam shapes with the following procedure:
    First, divide the detectors which we want to use into groups based on nominal polarization
    angle and receiver module. Next, make a separate map for each group of detectors.
    Finally, for each of these angle-group maps, convolve the map with the beam of the
    complementary angle group. (The complementary group consists of the group of detectors
    from the same module but with nominal angles rotated by 90 degrees. I.e., for a group
    of "X" detectors, the complementary group is the "Y" detectors from the same pixels.)
    INPUTS
    OUTPUT
    EXCEPTIONS
        ValueError if we encounter a template with a different map resolution than its angle group map.
    AUTHOR
        Stephen Hoover, 3 July 2013
    CHANGES
        23 August 2013: Peel out the construction of angle group maps to angleGroupMapper.
        17 Feb 2014: Store mapper function arguments in the map metadata. SH
    """
    # Store function arguments for use in map metadata. This must go before everything else!
    mapper_name = 'matchedBeamMapper'
    myargs = tools.myArguments(remove_self=True, remove_data=True, combine_posargs=True)

    start_time = spt_time.now()

    # We'll feed these arguments to basicMapper when we make maps for the individual angle groups.
    mapper_kwargs = {'individual_ts_maps':False,
                     'use_boresight_pointing':False,
                     'remove_planet_movement':True,
                     'verbose':verbose,
                     'timeit':verbose,
                     'debug':0,
                     'stopit':False,
                     'remove_empty_maps':True,
                     'inverse_noise_weighted_map':True,
                     't_only':t_only, 'reso_arcmin':reso_arcmin, 'proj':proj,
                     'use_source_offset_pointing':use_source_offset_pointing,
                     'use_leftgoing':use_leftgoing, 'map_center':map_center,
                     'map_shape':map_shape, 'ignore_features':ignore_features,
                     'use_azel_pointing':use_azel_pointing,
                     'skip_bad_scans':skip_bad_scans,
                     'use_scan_weights':use_scan_weights,
                     'timestream_filtering':timestream_filtering,
                     'use_c_pointing':use_c_pointing,
                     'tchunks':1,
                     }

    angle_group_maps = angleGroupMapper(data, good_bolos, **mapper_kwargs)

    time_finished_angle_group_maps = spt_time.now()
    if not quiet:
        print "Finished making angle group maps. That step took %.2f seconds." % (time_finished_angle_group_maps-start_time).total_seconds()

    # Now convolve each map with the complementary beam. This is the beam from the detector group 90 degrees away.
    if detector_beams:
        beam_templates = files.read(detector_beams)
        for group_id, this_angle_map in angle_group_maps.iteritems():
            module_id, pol_angle = group_id.split("_")
            complement_id = '%s_%.1f' % (module_id, (float(pol_angle)+90)%180)
            print "Convolving angle group %s map with the beam from group %s." % (group_id, complement_id)

            # Grab the appropriate beam from the input beams, and make sure that
            # it has the correct resolution.
            complement_beam = beam_templates[complement_id]
            if complement_beam.reso_arcmin != this_angle_map.reso_arcmin:
                raise ValueError("I need beams with the same resolution as the maps! The maps are %f arcmin/pixel, and the beams are %f arcmin/pixel."
                                 % (this_angle_map.reso_arcmin, complement_beam.reso_arcmin))

            # We need to separately convolve each map with the complementary beam.
            # We also must convolve the weights, since we're spreading the signal from
            # the detectors around.
            if isinstance(this_angle_map, sky.PolarizedMap):
                this_angle_map['T'].map = ndimage.convolve(this_angle_map['T'].map, complement_beam.map)
                this_angle_map['Q'].map = ndimage.convolve(this_angle_map['Q'].map, complement_beam.map)
                this_angle_map['U'].map = ndimage.convolve(this_angle_map['U'].map, complement_beam.map)
                for i_x in range(3):
                    for i_y in range(3):
                        this_angle_map.weight[:,:,i_x,i_y] = ndimage.convolve(this_angle_map.weight[:,:,i_x,i_y], complement_beam.map)
            else:
                this_angle_map.map = ndimage.convolve(this_angle_map.map, complement_beam.map)
                this_angle_map.weight = ndimage.convolve(this_angle_map.weight, complement_beam.map)

            if ultra_mode:
                # Extra convolutions! Convolve with both beams from the other pixel!
                other_pixel_groups = ['%s_%.1f' % (module_id, (float(pol_angle)+45)%180), '%s_%.1f' % (module_id, (float(pol_angle)-45)%180)]
                other_pixel_beam = [beam_templates[_id] for _id in other_pixel_groups]
                for beam in other_pixel_beam:
                    if isinstance(this_angle_map, sky.PolarizedMap):
                        this_angle_map['T'].map = ndimage.convolve(this_angle_map['T'].map, beam.map)
                        this_angle_map['Q'].map = ndimage.convolve(this_angle_map['Q'].map, beam.map)
                        this_angle_map['U'].map = ndimage.convolve(this_angle_map['U'].map, beam.map)
                        for i_x in range(3):
                            for i_y in range(3):
                                this_angle_map.weight[:,:,i_x,i_y] = ndimage.convolve(this_angle_map.weight[:,:,i_x,i_y], beam.map)
                    else:
                        this_angle_map.map = ndimage.convolve(this_angle_map.map, beam.map)
                        this_angle_map.weight = ndimage.convolve(this_angle_map.weight, beam.map)


        if not quiet:
            print "Finished making the map. Total time elapsed: %.2f seconds." % (spt_time.now()-start_time).total_seconds()

    # If we only want the summed output map, combine angle group maps now.
    if not return_angle_group_maps:
        summed_map = np.sum(angle_group_maps.values())
        summed_map.setMapAttr('n_channels', np.sum([_map.n_channels for _map in angle_group_maps.itervalues()]))
        summed_map.mapper_arguments = myargs
        angle_group_maps = summed_map
        if not inverse_noise_weighted_map:
            angle_group_maps.removeWeight()
    elif not inverse_noise_weighted_map:
        warnings.warn("You requested a map with weight removed, but you also want angle group maps. I'm giving you weighted angle group maps.",
                      RuntimeWarning)
    if stopit: pdb.set_trace()
    return angle_group_maps


##########################################################################################
##########################################################################################
class MapAnalyzer:
    def __init__(self, ell_bins=None, delta_ell=None, set_special_bb_bins=False):
        """
        INPUTS
            ell_bins [None]: (dict) A dictionary, keyed by power spectrum designation
                (e.g. TT, EE), giving the bins for that power spectrum. Example format:
                ell_bins['BB'] = np.array([[0,19],[20,89],[90,160],[161,301]])
                You do not need to supply ell bins for every spectrum. We'll default to
                even spacings for any unspecified power spectra.
            delta_ell [None]: (float or dict) Width of ell bins. If a float, use that
                width for all spectra not otherwise specified in the ell_bins input.
                If a dict, should be keyed by power spectrum name. Use the specified delta_ell
                for each power spectrum. We'll use the self.default_delta_ell if there's
                any power spectra with un-specified delta_ell values.
            set_special_bb_bins [True]: (bool) If True, and we didn't get an ell_bins input,
                set a hardcoded set of BB bins. If False, use the input (or default) delta_ell.
        AUTHOR
            Stephen Hoover
        CHANGES
            24 Sept 2013: If delta_ell is a number, use it as the default_delta_ell. SH
            14 Nov 2013: Add "set_special_bb_bins" argument, so we can chose to have BB with default_delta_ell.
                Remove unused "call_idl" argument. SH
        """
        # Check the delta_ell input: it must be either a number or a dictionary.
        if delta_ell and (not np.isreal(delta_ell) and not isinstance(delta_ell, dict)):
            raise ValueError("The delta_ell input must be either a number or a dictionary of numbers!")

        self.default_delta_ell =50.
        if np.isscalar(delta_ell):
            # If we got a single delta ell value, set that as the default.
            self.default_delta_ell = delta_ell

        if delta_ell and isinstance(delta_ell, dict):
            # If instead the delta_ell was a dictionary, assume it has an entry for each spectrum.
            self.delta_ell = delta_ell
        else:
            self.delta_ell = dict()

        if ell_bins:
            self.ell_bins = ell_bins
        else:
            self.ell_bins = {}
            # Some default wider ell bins for BB.
            if set_special_bb_bins:
                self.ell_bins['BB'] = np.array([[0,19],[20,89],[90,160],[161,301],[302,401],[402,535],[536,713],[714,951],
                                                [952,1268],[1269,1691],[1692,2255],[2256,3007],[3008,4009],[4010,5236]])

        self.ell_in_bin = {}
        self.non_one_mask_sum, self.mask_sum = {}, {}

        # Set up some variables that we'll store between function calls to save uncessesary time re-generating them.
        self.reset()

    ###################################################################
    def setEllgrids(self, shape, reso_arcmin=None, ell_bins=None,
                    trim_to_shape=None, save=True):
        """
        Store pre-made ellgrids for the input map shape and resolution.
        Cuts down on unnecessary computation if the same MapAnalyzer
        object will be used to analyze many same-sized maps.
           You may save ellgrids for many different map shapes and
        resolutions. We'll use the appropriately-sized ellgrid
        when it comes time to do the work.
        INPUTS
            shape: (2-element list or array, or a Map object) If this is a
               Map or PolarizedMap, then pull out the shape and reso_rad.
               Otherwise, this should be the shape (in pixels) of the maps
               that we'll want to analyze.
            reso_arcmin [None]: (float) If "shape" isn't a Map, then you must
               separately supply the map resolution.
            ell_bins [None]: Binning for the ellgrid. See calculateCls or __init__
                for a format description. Use self.ell_bins if the input is None.
            trim_to_shape [None]: (2-element list or array) If not None, then cut
                all arrays down to this size before storing.
            save [True]: (bool) If True, save the ellgrids in this MapAnalyzer
                object. If False, only return the result. You'll usually want
                to save the result. (The exception might be if you planned to
                make ellgrids for many different resolutions, and were worried
                about memory usage.)
        OUTPUT
            None, but we'll store ellgrids in this object.
        EXCEPTIONS
            ValueError if we're missing a resolution.
        AUTHOR
            Stephen Hoover, 24 September 2013
        CHANGES
            31 October 2013, remove zero_pad_factor option. KTS
            14 Nov 2013: Delete the unused (and incorrectly named) "central_ellgrid".
                Modify so that we return the ellgrid structure, and saving is optional.
                Remove the warning that an ellgrid had already been created, and just return it. SH
            5 June 2014: Add binned ellgrids. SH
            13 June 2014: If we're storing a trimmed ellgrid, then use the trimmed shape
                as the key. Also, don't overwrite the input ell_bins. SH
            18 June 2014: Tweak again, so that this works properly with trimmed shapes
                in cross_spectra_multiprocessing.
        """
        # Make certain we have both a shape and map resolution.
        try:
            reso_arcmin = shape.reso_arcmin
            if (shape.shape == np.array([1,1])).all():
                try:
                    shape = shape.ft_maps[shape.ft_maps.keys()[0]]
                except (AttributeError, KeyError):
                    print '\n'.join(["setEllGrids(): You can't possibly mean to get an Ellgrid for a 1x1 map.",
                                     "You also don't have any FT maps, so your map shape makes no sense."])
            shape = shape.shape

        except AttributeError:
            if reso_arcmin is None:
                raise ValueError("You need to supply a resolution along with the map shape!")

        reso_rad = reso_arcmin * np.pi / 180. / 60.
        shape = np.asarray(shape).astype(int)

        # Check if we've already computed this ellgrid. If so, we can return it right away.
        if trim_to_shape is not None:
            ellgrid_key = ( tuple(trim_to_shape), reso_arcmin )
        else:
            ellgrid_key = ( tuple(shape), reso_arcmin )
        if (ellgrid_key in self.stored_ellgrids and
            (ell_bins is None or len(tools.dictDiff(self.stored_ellgrids[ellgrid_key]['ell_bins'],ell_bins))==0)):
            return self.stored_ellgrids[ellgrid_key]
        if ell_bins is None: ell_bins = self.ell_bins

        # Construct the ellgrid and inverse-safe ellgrid for the requested resolution.
        ellgrids = struct()
        ellgrids['ellgrid'] = math.makeEllGrid(shape=shape, resolution=reso_rad)
        if trim_to_shape is not None:
            ellgrids['ellgrid'] = tools.getFFTCutout(ellgrids['ellgrid'], trim_to_shape)
        ellgrids['inv_safe_ellgrid'] = ellgrids['ellgrid'].copy() #"Safe to invert", ie no zeros.
        ellgrids['inv_safe_ellgrid'][0,0] = 1e6

        # Bin the ellgrid
        binned_ell = {}
        ellgrid = ellgrids['ellgrid']
        ellgrids['ell_bins'] = ell_bins
        for xspec in ['TT','EE','BB','TE','TB','EB','BT','ET','BE']:
            if xspec not in binned_ell and xspec in self.delta_ell:
                binned_ell[xspec] = util.tools.binGrid(ellgrid, delta_bin=self.delta_ell[xspec])
            if xspec not in binned_ell:
                if xspec not in ell_bins: # Set default ell bins if they aren't provided.

                    binned_ell[xspec] = util.tools.binGrid(ellgrid, delta_bin=self.delta_ell.setdefault(xspec,self.default_delta_ell))
                    n_ell_bins = int(np.max(binned_ell[xspec]))+1
                    ell_bins[xspec] = np.asarray([ [self.delta_ell.setdefault(xspec,self.default_delta_ell)*index+1,
                                                    self.delta_ell.setdefault(xspec,self.default_delta_ell)*(index+1)]
                                                  for index in xrange(n_ell_bins)])

                elif np.isscalar(ell_bins[xspec]):
                    binned_ell[xspec] = util.tools.binGrid(ellgrid, delta_bin=ell_bins[xspec])
                    n_ell_bins = int(np.max(binned_ell[xspec]))+1
                    ell_bins[xspec] = np.asarray([ [ell_bins[xspec]*index+1, ell_bins[xspec]*(index+1)] for index in xrange(n_ell_bins)])
                else:
                    ell_bins[xspec] = np.asarray(ell_bins[xspec])
                    n_ell_bins = len(ell_bins[xspec])
                    binned_ell[xspec] = util.tools.binGrid(ellgrid, bins=ell_bins[xspec])
            elif xspec in ell_bins:
                n_ell_bins = len(ell_bins[xspec])
            else:
                n_ell_bins = int(np.max(binned_ell[xspec]))+1
                ell_bins[xspec] = np.asarray([ [self.delta_ell.setdefault(xspec,self.default_delta_ell)*index+1,
                                                self.delta_ell.setdefault(xspec,self.default_delta_ell)*(index+1)]
                                              for index in xrange(n_ell_bins)])
        ellgrids['binned_ell'] = binned_ell

        # Store the ellgrid, if we weren't told not to.
        if save:
            self.stored_ellgrids[ellgrid_key] = ellgrids

        return ellgrids

    ###################################################################
    def reset(self):
        """
	    Reset the variables stored between function calls. Will force functions to remake the relevant data.
	    These data are stored only to save unnecesary time re-generating the same data over and over again.
	    CHANGES
	        24 Sept 2013: Added "stored_ellgrids". SH
	        8 Nov 2013: Removed the 'kendrick_window' attribute, which wasn't being used. SH
	    """
        self.stored_ellgrids = struct()

    ###################################################################
    def constructEB(self, _map, apodization_window=None,
                    b_mode_method="chiB2",
                    e_mode_method="basic",
                    fp_rotation_angle=0.0,
                    fft_shape=None,
                    pre_reddening=False,
                    gaussian_smooth=None,
                    apodization_window_chi=None,
                    zero_nan=True, remake=False,
                    store_ffts=True,quiet=False,
                    debug=False):
        """
        Takes an input polarized map and uses the Q and U components to construct the
        E-mode and B-mode polarizations. Will normally save the E and B maps in the
        input map object, and will also return the pair of new maps.
        INPUTS
            _map: (PolarizedMap) Contains the Q and U polarizations which we'll convert
                to E and B.
            apodization_window [None]: (2D ndarray) Multiply the maps by this array before
                taking the FFTs.
            b_mode_method ['chiB2']: (string) How to calculate B?
                   The preferred mode is "chiB2". We want to avoid E->B leakage, so we need to
                use a pure B-mode estimator. The "chi" and "smith" modes are mathematically identical,
                but the "chi" estimator is numerically nicer. Remember to use good apdodization windows!
                'chiBX', where 'X' is 1, or 2 : Smith & Zaldarriaga chiB-method.
                    This mode uses 'maps.calculateChi'.
                'chiB-X' : The chiB method, but using partial derivatives from the math module. Identical to chiBX.
                'chi' or 'chiB' : Equivalent to 'chiB2'.
                'smith': Original Kendrick Smith pure B-mode estimator. From 2006 Phys. Rev D paper.
                    This mode uses 'maps.smithPurePseudoCellEstimator'.
                'basic' or anything else: Naive Q/U combination. Note that this will leak E->B.
                'None' or None or False : Don't make any B-mode maps.
            e_mode_method ['basic']: (string) How to calculate E?
                   The preferred mode is "basic". We don't care about B->E leakage, and the pure estimators
                tend to have increased mode mixing.
                'chiEX', where 'X' is 1 or 2. Smith & Zaldarriaga chiE-method. See 'maps.calculateChi'.
-               'chiE-X' : The chiE method, but using partial derivatives from the math module. Identical to chiEX.
                'chi' or 'chiE' : Equivalent to 'chiE2'.
                'smith': Original Kendrick Smith pure B-mode estimator (switched for E). From 2006 Phys. Rev D paper.
                    This mode uses 'maps.smithPurePseudoCellEstimator'.
                'basic' or anything else: Naive Q/U combination. Note that this will leak B->E (but you probably don't care).
                'None' or None or False : Don't make any E-mode maps.
            fp_rotation_angle: If we have measured a global focal plane rotation angle (in degrees), enter it here.  Q and U 
                maps will be rotated by this angle before E and B fft maps are calculated.  This step will happen
                after the flatten_pol angle rotation step.  fp_rotation_angle can be computed using
                analysis.maps.calculatePolarizationSelfCal(), and the sign as outputted is appropriate for this function.
            fft_shape [None]:  (2 element np.ndarray) Specify the shape of the fft map.  This should be a 2-element ndarray, [npix_x, npix_y].
                               This option is how maps are padded.
            pre_reddening [False]: (bool) Experimental and non-functional. Tilt the spectrum of the input
                Q and U maps to reduce harmful mode-mixing in the chi maps.
            gaussian_smooth [None]: (float) If not None, smooth Q and U maps with a Gaussian of this
                radius (in arcminutes) before using them for anything. Does not change input map.
            apodization_window_chi [None]: (2D ndarray) If using "chiX" methods, multiply the Q and U maps
                by this mask before creating the chi map. This map doesn't need point source masking, and
                shouldn't be particularly wide. It should be smooth on the borders (cosine apodization is good).
                It serves to let the code know what your survey boundaries look like, so it can compensate for
                the effect of the cut sky on chi-map creation. Possibly not necessary. If None, we won't
                use any mask here.
            zero_nan [True]: (bool) If True, set all values of "NaN" in the
                map to zero for processing. Will not alter input map.
            remake [False]: (bool) By default, refuse to overwrite an existing E or B map.
                If remake=True, lose all scruples about destroying existing E/B maps.
            quiet [False]: (bool) If True, suppress all screen output.
            debug [False]: (bool) Extra output for debugging purposes. Includes, e.g., storing
                auxiliary maps used in E or B mode estimation.
        OUTPUT
            2-tuple of E-mode and B-mode Map objects. These are also stored in the input map,
            if they weren't there already.
        EXCEPTIONS
            ValueError if the input map still has weights in it. (If debug=True, warn only.)
            (Formerly would raise a ValueError if the input PolarizedMap has not been flattened. Run map.flattenPol() to avoid this. Changed to warning.)
        AUTHOR
            Stephen Hoover, March 2013
        CHANGES
            23 Sept 2013: Add option to not make either E or B map if it's not needed. SH
            24 Sept 2013: Check for compatible stored ellgrids and use them if available. SH
            24 Oct 2013: Rename to "constructEB". Add "debug" argument, and stop storing "Z" map by default. SH
            1 Nov 2013: Switch use of apodization_mask and apodization_mask_chi: Now the latter is applied before chi-map construction. SH
            8 Nov 2013: Cleaned up a bit and moved original Smith pure B-mode estimator to its own function.
                We'll now get a ValueError if the input maps are still weighted. SH
            14 Nov 2013: Delete the incorrectly named "central_ellgrid", and change it to "inv_safe_ellgrid" everywhere it was used.
                Use the "setEllgrids" function instead of copying the code here.
                Allow inputs of 'chi' or 'chiE' or 'chiB' for the estimator methods. These default to 'chiX2'. SH
            22 Nov 2013: Remove probably incorrect explanation for why qu_to_eb_angle = np.arctan2(lx, -ly). SH
            27 Nov 2013: Respect "quiet" everywhere. SH
            26 Feb 2018: Adding an option to remove a global rotation from Q and U before E and B are calculated. JWH
        """
        # Check that the input map is unweighted.
        if getattr(_map, 'weighted_map', False):
            if debug:
                warnings.warn("You need to give me an unweighted map! (Run map.removeWeight().) Since it's debug mode, I'll continue anyway.", RuntimeWarning)
            else:
                raise ValueError("You need to give me an unweighted map! (Run map.removeWeight().)")

        # Check that the map Q/U angle distortions have been removed.
        if hasattr(_map, 'flattened_pol'):
            if not _map.flattened_pol:
                #raise ValueError("You must rotate Q/U angles to be flat before running this function. Run map.flattenPol.")
                warnings.warn("In principle you must rotate Q/U angles to be flat before running this function. Do this with map.flattenPol.", RuntimeWarning)
        else:
            warnings.warn("I can't tell if the Q/U angles in this map are flat! Continuing anyway (but I'm scared!).", RuntimeWarning)

        # Convert generic "chi" requests to the more specific versions.
        if e_mode_method=='chi' or e_mode_method=='chiE': e_mode_method='chiE2'
        if b_mode_method=='chi' or b_mode_method=='chiB': b_mode_method='chiB2'

        #If fp_rotation_angle is non-zero, rotate Q and U maps by the opposite angle before calculating ffts.
        if fp_rotation_angle != 0.0:
            if not quiet:
                print("Correcting for a global focal plane rotation angle %s." % str(fp_rotation_angle))
                Pmap = deepcopy(_map['Q'].map) + 1j*deepcopy(_map['U'].map)
                Pmap *= np.exp(2*1j*fp_rotation_angle*np.pi/180.)
                _map['Q'].map = deepcopy(Pmap.real)
                _map['U'].map = deepcopy(Pmap.imag)
                del Pmap

        ################################################################
        # Setup. Create a grid with the map's total ell at each
        # frequency-domain point, and initialize the output dictionaries.
        # We're also going to zero-pad the input maps (if requested).
        # (Note that the ellgrid part of the setup won't be used with
        # the chi estimators.)
        if fft_shape is None: fft_shape=np.array([ int(dim) for dim in _map.shape ])
        assert(len(fft_shape)==2)
        padded_shape = np.array(fft_shape)

        if not quiet and (padded_shape==np.asarray(_map.shape)).all():
            print("[MapAnalyzer.constructEB] Map size is %s (no zero padding applied)." % str(padded_shape))
        elif not quiet:
            print("[MapAnalyzer.constructEB] Zero-padding maps from size %s to size %s." % (str(_map.shape), str(padded_shape)))

        ellgrids = self.setEllgrids(padded_shape, _map.reso_arcmin)
        inv_safe_ellgrid = ellgrids['inv_safe_ellgrid']

        if not quiet:
            print "[MapAnalyzer.constructEB] Resolution in ell-space is %f." % (inv_safe_ellgrid[0,2]-inv_safe_ellgrid[0,1])

        # Set up the apodization windows - pad if requested, and calculate the normalization.
        if apodization_window is None:
            apodization_window = np.ones(_map.shape)
        sky_window_pad = np.zeros(padded_shape)
        sky_window_pad[:_map.shape[0],:_map.shape[1]] = apodization_window
        window_normalization = np.sqrt(np.sum(sky_window_pad**2)/np.product(sky_window_pad.shape))

        if apodization_window_chi is not None:
            apod_window_chi_pad = np.zeros(padded_shape)
            apod_window_chi_pad[:_map.shape[0],:_map.shape[1]] = apodization_window_chi
            #apodization_window_chi_normalization = np.sqrt(np.sum(apod_window_chi_pad**2)/np.product(apod_window_chi_pad.shape))
        else: apod_window_chi_pad=None

        #################################################
        # Setup for the Q and U maps. Pad if requested,
        # remove NaNs, possibly smooth by a Gaussian.
        padded_q = np.zeros(padded_shape)
        padded_q[:_map.shape[0],:_map.shape[1]] = _map['Q'].map
        padded_u = np.zeros(padded_shape)
        padded_u[:_map.shape[0],:_map.shape[1]] = _map['U'].map

        if zero_nan:
            padded_q[~np.isfinite(padded_q)]=0.
            padded_u[~np.isfinite(padded_u)]=0.

        if gaussian_smooth:
            if not quiet: print "Smoothing Q and U maps by %f arcminutes." % gaussian_smooth
            padded_q = ndimage.gaussian_filter(padded_q, gaussian_smooth / _map.reso_arcmin / (2*np.sqrt(2*np.log(2))))
            padded_u = ndimage.gaussian_filter(padded_u, gaussian_smooth / _map.reso_arcmin / (2*np.sqrt(2*np.log(2))))

        if pre_reddening:
            padded_q = np.fft.ifft2( np.fft.fft2(padded_q) / sqrt((inv_safe_ellgrid-1.)*inv_safe_ellgrid*(inv_safe_ellgrid+1.)*(inv_safe_ellgrid+2.)) ).real
            padded_u = np.fft.ifft2( np.fft.fft2(padded_u) / sqrt((inv_safe_ellgrid-1.)*inv_safe_ellgrid*(inv_safe_ellgrid+1.)*(inv_safe_ellgrid+2.)) ).real

        ####################################################################################
        # Construct k-space angles. We'll use this if we do either the "basic" construction
        # or the original Smith 2006 estimator. Note that The k-space angle is nominally
        # arctan(ly/lx) if you read any papers. We define it here as np.arctan2(lx, -ly).
        # (This makes things work, and I'm still double-checking why that's so. Perhaps
        # due to non-standard coordinate systems used in the sptpol_software? SH)
        lx, ly = np.meshgrid( np.fft.fftfreq( padded_shape[1], _map.reso_rad )*2.*np.pi,
                              np.fft.fftfreq( padded_shape[0], _map.reso_rad )*2.*np.pi )
        qu_to_eb_angle = np.arctan2(lx, -ly)

        map_fft = {}
        #############################################################################
        #############################################################################
        # Create the E-mode map.
        if ('E' not in _map.pol_maps or remake) and ((bool(e_mode_method) is True) and (e_mode_method.lower()!='none')):
            if e_mode_method.startswith('chiE'):
                # Use the chi estimator! First select between taking partial derivatives through the math
                # module, or using the calculateChi function.
                pixel_range = int(e_mode_method[4:])
                if pixel_range==0: pixel_range=-1
                if pixel_range<0:
                    # Use math functions for the partial derivatives. Should be identical to calculateChi
                    # in the case of no pre-chi apodization window and pixel_range<=2.
                    if apod_window_chi_pad is not None:
                        raise NotImplementedError("I'm not set up to use a pre-chi window with the math module functions.")
                    chiE  = ( d2dx2(padded_q, _map.reso_rad, range=abs(pixel_range))
                              - d2dy2(padded_q, _map.reso_rad, range=abs(pixel_range))
                              + 2*d2dxdy(padded_u, _map.reso_rad, range=abs(pixel_range)) )
                else:
                    # Get the chi_E.
                    chiE = calculateChi(padded_q, padded_u, _map.reso_rad, which_chi='E',
                                        pixel_radius=pixel_range,
                                        apodization_window=apod_window_chi_pad)

                # Convert chi_E into E.
                if pre_reddening:
                    map_fft['E'] = np.fft.fft2(chiE)
                    e_map_pad = chiE
                else:
                    _map.addMap('chiE',chiE[:_map.shape[0],:_map.shape[1]])
                    map_fft['E'] = ( np.fft.fft2(chiE * sky_window_pad / window_normalization)
                                     / sqrt((inv_safe_ellgrid-1.)*inv_safe_ellgrid*(inv_safe_ellgrid+1.)*(inv_safe_ellgrid+2.)))
                    #if store_ffts:_map.pol_maps['chiE'].fft = map_fft['E']

                    # Store the completed E polarization in the map object.
                    e_map_pad = np.fft.ifft2(map_fft['E']).real
                    if store_ffts and debug:
                        _map.addFTMap('weight_Z', ( np.fft.fft2(sky_window_pad / window_normalization)
                                                    / sqrt((inv_safe_ellgrid-1.)*inv_safe_ellgrid*(inv_safe_ellgrid+1.)*(inv_safe_ellgrid+2.))), active_shape=apodization_window_chi.shape)
                    if debug:
                        _map.addMap('Z', ( 1. / sqrt((inv_safe_ellgrid-1.)*inv_safe_ellgrid*(inv_safe_ellgrid+1.)*(inv_safe_ellgrid+2.))),
                                    allow_different=True)
            elif e_mode_method.startswith('smith'):
                map_fft['E'] = smithPurePseudoCellEstimator(padded_q, padded_u, sky_window_pad, reso_rad=_map.reso_rad,
                                                            do_e=True, qu_to_eb_angle=qu_to_eb_angle, ellgrid=inv_safe_ellgrid)

                #Store the completed B polarization in the map object.
                e_map_pad = np.fft.ifft2(map_fft['E']).real
            else:
                # Do the basic E-mode construction. Make sure the user understands what they're getting if we defaulted here!
                if e_mode_method != 'basic':
                    warnings.warn("I didn't recognize your requested e_mode_method, so I'm defaulting to \"basic\".", RuntimeWarning)

                if 'Q' not in map_fft or 'U' not in map_fft:
                    map_fft = {'Q':np.fft.fft2(padded_q * sky_window_pad / window_normalization),
                               'U':np.fft.fft2(padded_u * sky_window_pad / window_normalization) }
                map_fft['E'] = map_fft['Q']*np.cos(2*qu_to_eb_angle) + map_fft['U']*np.sin(2*qu_to_eb_angle)
                # Store the computed E in the map object
                e_map_pad = np.fft.ifft2(map_fft['E']).real

            e_map_pad[sky_window_pad != 0.] *= window_normalization / sky_window_pad[sky_window_pad != 0.]
            _map['E'] = e_map_pad[:_map.shape[0],:_map.shape[1]]
            if store_ffts: _map.addFTMap('E', map_fft['E'], active_shape=map_fft['E'].shape) # The entire shape is "active" in an FFT.
        elif not quiet:
            if 'E' in _map.pol_maps:
                print "E-mode map already present in this map; I will not remake it."
            else:
                print "Not making the E-mode map, as requested."

        #############################################################################
        #############################################################################
        # Now the B-mode map.
        if ('B' not in _map.pol_maps or remake) and ((bool(b_mode_method) is True) and (b_mode_method.lower()!='none')):
            if b_mode_method.startswith('chiB'):
                # Use the chi estimator! First select between taking partial derivatives through the math
                # module, or using the calculateChi function.
                pixel_range = int(b_mode_method[4:])
                if pixel_range==0: pixel_range=-1
                if pixel_range<0:
                    # Use math functions for the partial derivatives. Should be identical to calculateChi
                    # in the case of no pre-chi apodization window and pixel_range<=2.
                    if apod_window_chi_pad is not None:
                        raise NotImplementedError("I'm not set up to use a pre-chi window with the math module functions.")
                    chiB  = ( d2dx2(padded_u, _map.reso_rad, range=abs(pixel_range)) -
                              d2dy2(padded_u, _map.reso_rad, range=abs(pixel_range)) -
                              2*d2dxdy(padded_q, _map.reso_rad, range=abs(pixel_range)) )
                else:
                    # Get the chi_B.
                    chiB = calculateChi(padded_q, padded_u, _map.reso_rad, which_chi='B',
                                        pixel_radius=pixel_range,
                                        apodization_window=apod_window_chi_pad)

                # Convert chi_B into B.
                if pre_reddening:
                    map_fft['B'] = np.fft.fft2(chiB)
                    b_map_pad = chiB
                else:
                    _map.addMap('chiB',chiB[:_map.shape[0],:_map.shape[1]])
                    map_fft['B'] = ( np.fft.fft2(chiB * sky_window_pad / window_normalization)
                                     / sqrt((inv_safe_ellgrid-1.)*inv_safe_ellgrid*(inv_safe_ellgrid+1.)*(inv_safe_ellgrid+2.)))
                    # Store the completed B polarization in the map object.
                    b_map_pad = np.fft.ifft2(map_fft['B']).real
            elif b_mode_method.startswith('smith'):
                map_fft['B'] = smithPurePseudoCellEstimator(padded_q, padded_u, sky_window_pad, reso_rad=_map.reso_rad,
                                                            do_e=False, qu_to_eb_angle=qu_to_eb_angle, ellgrid=inv_safe_ellgrid)

                #Store the completed B polarization in the map object.
                b_map_pad = np.fft.ifft2(map_fft['B']).real
            else:
                # Do the basic B-mode construction. Make sure the user understands what they're getting if we defaulted here!
                if b_mode_method != 'basic':
                    warnings.warn("I didn't recognize your requested b_mode_method, so I'm defaulting to \"basic\".", RuntimeWarning)

                if 'Q' not in map_fft or 'U' not in map_fft:
                    map_fft = {'Q':np.fft.fft2(padded_q * sky_window_pad / window_normalization),
                               'U':np.fft.fft2(padded_u * sky_window_pad / window_normalization) }
                map_fft['B'] = -map_fft['Q']*np.sin(2*qu_to_eb_angle) + map_fft['U']*np.cos(2*qu_to_eb_angle) # The basic way of calculating B.
                # Store the computed B map in the map object
                b_map_pad = np.fft.ifft2(map_fft['B']).real
            b_map_pad[sky_window_pad != 0.] *= window_normalization / sky_window_pad[sky_window_pad != 0.]
            _map['B'] = b_map_pad[:_map.shape[0],:_map.shape[1]]
            if store_ffts: _map.addFTMap('B', map_fft['B'], active_shape=map_fft['B'].shape) # The entire shape is "active" in an FFT.
        elif not quiet:
            if 'B' in _map.pol_maps:
                print "B-mode map already present in this map; I will not remake it."
            else:
                print "Not making the B-mode map, as requested."

        # If we didn't calculate both the E and B maps, we can't return both.
        # Return only what we have.
        eb_maps = []
        if 'E' in _map.pol_maps:
            eb_maps.append(_map['E'])
        if 'B' in _map.pol_maps:
            eb_maps.append(_map['B'])
        return eb_maps
    constructEBV2 = constructEB # So that the old way of calling this function is still supported.

    ###################################################################
    def calculateCls(self, _map, cross_map=None,
                     sky_window_func=None,
                     ell_bins=None,
                     hanning_window=False,
                     uv_mask=None,
                     recreate_ffts=False,
                     e_mode_method="basic",
                     b_mode_method="chiB2",
                     maps_to_cross=None,
                     save_ffts=False,
                     fft_shape=None,
                     cutout_shape = None,
                     calculate_dls = False,
                     quiet=True,
                     average_equivalent_pol_cross_spectra=True,
                     return_2d=False,
                     realimag='real'):
        """
        Taken from the IDL function "cl_flatsky_kendrick" on 5 April 2011.
        Does not include 'projt' keyword or non-polarized functionality.
        INPUT:
           _map: Should be a PolarizedMap (or Map, if non-polarized) containing
               the CMB sky.
              WARNING: This method may produce incorrect output if the map's
                  active area is not square. The window calculation from the IDL code
                  does not consider non-square maps.
        OUTPUT:
        Returns a dictionary with six cross-power spectra, TT, TE, TB, EE, EB, and BB. Units are K_CMB**2.
        The input map will have E and B maps added if it did not have them already.
        The ell_bins and uv_mask dictionaries will have missing entries added with
            default values.
        The output dictionary will also have two sub-dictionaries, "ell_bins" and "ell", with
            the ell values corresponding to each Cl.
        Optional keywords:
        cross_map [None]: (PolarizedMap) If None, we'll take the "cross"-spectrum of the map
            with itself (an autospectrum). If not None, we'll cross the input map with this map.
            In outputs, the first letter of the cross-spectrum name is from "map", the second
            from "cross_map". So spectrum "EB" is map['E'] crossed with cross_map['B'], and
            spectrum "BE" is map['B'] crossed with cross_map['E'].
        sky_window_func [None]: If provided, will be multiplied by the maps in real space
                                before taking the Fourier transform.
        ell_bins:   If provided, this should be a dictionary keyed by cross-spectrum
                      ('TT', 'BB', 'TE', etc.) The output cross-spectra will be binned
                      into by these. If no ell bins are provided for a given
                      cross-spectrum, a default set of bins will be generated, based
                      on delta_ell, and stored in this dictionary. Note that the
                      ell_bins are only used if a binned_ell does not exist.
                      Ell bins input here override ell bins stored in the MapAnalyzer
                      object; normally ell bins would be supplied at the MapAnalyzer
                      construction.
        uv_mask: A dictionary keyed by cross-spectrum ('TT', 'BB', 'TE', etc.)
                     which provides weightings by mode. A provided mask will multiply
                     the cross-spectrum modes immediately prior to binning.
        hanning_window: If set to True, multiply the input sky by a Hanning window
                            before taking the FT. Only applies if you don't enter an
                            explicit apodization window.
        recreate_ffts [False]: If set to True, the method will calculate E, B and T FTs
                         (and E, B maps). Any existing FT maps will be overwritten!
        b_mode_method ['chiB2']: (string) How do we calculate the B-modes?
               Valid choices are:
               'smith2006': The original method used in this code, taken from a 2005
                            paper by Kendrick Smith.
               'chiB#': Newer method, taken from Smith & Zaldarriaga (2006).
                        Find the B-mode map by calculating the chi_B and adjusting
                        in the frequency domain for the O(ell^4) bias.
                        The '#' on the end must be 1, 2, or 3, and designates the
                        radius out to which we go when taking derivatives in pixel space.
        e_mode_method ['chiE2']: (string) How do we calculate the E-modes?
               'chiE#': Newer method, taken from Smith & Zaldarriaga (2006).
                        Find the E-mode map by calculating the chi_E and adjusting
                        in the frequency domain for the O(ell^4) bias.
                        The '#' on the end must be 1, 2, or 3, and designates the
                        radius out to which we go when taking derivatives in pixel space.
        maps_to_cross [['T','E','B']]: (list of map polarizations to combine into cross- and auto-spectra.
        save_ffts [False]: save any FTs that are made to the PolarizedMap structure
        fft_shape [None]:  (2 element np.ndarray) Specify the shape of the fft map.  This should be a 2-element ndarray, [npix_x, npix_y].
                            This option is how maps are padded.
        calculate_dls [False]: (bool) If True, the output power spectra are D_ell instead of C_ell.
            (D_ell = ell*(ell+1)/2pi * C_ell)
        average_equivalent_pol_cross_spectra [True]: (bool) If True, when we have equivalent polarization
                cross-spectra (e.g. ET and TE), average the Cls together and return only one of them.
        return_2d [False]: (bool) If True, skip binning the power spectra by ell
            and return the 2D power spectra. In this case, we won't use the uv_mask.
            The output structure will contain the ellgrid under the "ell" key, and the
            "ell_bins" key will have a dummy array.
        realimag ['real']: (string) This determines whether this function returns the real
            or imaginary part of the spectrum.  Pass 'real' to take the real part, and
            'imag' to return the imaginary part.
        EXCEPTIONS
            ValueError if the cross_map is given but is different in properties from the map, or if you
                try to use a cross_map with the IDL code.
        CHANGES
            24 Sept 2013: Check for compatible stored ellgrids and use them if available. SH
            22 Oct 2013: Add keyword "average_equivalent_pol_cross_spectra", default to True. SH
            25 Oct 2013: Change how the FT's are made, so functions from observation.sky are always called.  KTS
            29 Oct 2013: Add "recreate_ffts" argument, remove use_existing_ffts, recreate_eb. KTS
            31 Oct 2013: Correct some lingering problems from API change, and trim no-longer-used arguments. SH
            11 Nov 2013: Will now correctly FT Q and U maps. Also respect "save_ffts" option when making E and B maps. SH
            14 Nov 2013: Delete the unused (and incorrectly named) "central_ellgrid".
                Use the "setEllgrids" function instead of copying the code here. SH
            22 Nov 2013: Don't respect "save_ffts" when making E and B maps. We won't get them otherwise. SH
            27 Nov 2013: Tweak code so that we don't call constructEB if we don't need E or B maps. SH
            12 Feb 2014: Add an "is_d_ell" flag to the output dictionary. SH
            6 Mar 2014: Add the "return_2d" argument. SH
            5 June 2014: Speed optimizations-- use numexpr, use stored complex conjugates, store binned_ellgrid.
                Change order of for loops in C code (this is a big improvement!).
                Comment out mode-counting calculation of variance, since we didn't even return it.
                Remove the "do_c_code" option; the python version was not maintained and not needed. SH
            24 July 2014: When inputting a UV mask, if there's no, e.g., "ET", then use "TE". SH
            09 Sep 2014: Added 'realimag' optional input.  RK
            1  Oct 2013: small changed to work with T only maps also. TJN
            18 Feb 2016: default to maps_to_cross=['T','E','B'] even if cross_map is None.  KTS
        """
        # Set the power spectra which we'll combine if average_equivalent_pol_cross_spectra is True.
        # We will average the spectrum named by the key and the spectrum named by the value.
        # The dictionary key is the spectrum which we'll keep, and the value will be discarded after averaging.
        equivalent_ps = {'TE':'ET', 'EB':'BE', 'TB':'BT'}

        ######
        ## Note: Perhaps I can do something clever with np.bincount and/or
        ## np.digitize to make this routine more efficient?
        ######

        if maps_to_cross is None:
            try:
                _map.pol_maps      # A polarized Map object will have this method.
                if cross_map is not None: cross_map.pol_maps # A polarized Map object will have this method.

                maps_to_cross=['T','E','B']
            except AttributeError:
                maps_to_cross = ['T']

        # If we have an input "ell_bins" which is just a list, create a dictionary and assign
        # the same binning to each cross-spectrum.
        if ell_bins is not None and not isinstance(ell_bins, dict):
            ell_bins = dict([ (spec, ell_bins) for spec in [map1+map2 for map1 in maps_to_cross for map2 in maps_to_cross ]])

        # We might have gotten a dictionary for the map input, with each entry a
        # different band. If this is the case, treat each separately.
        try:
            return dict( [(band, self.calculateCls(thismap,
                                                   cross_map=(None if cross_map is None else cross_map[band]),
                                                   sky_window_func=sky_window_func,
                                                   ell_bins=ell_bins,
                                                   hanning_window=hanning_window,
                                                   uv_mask=uv_mask,
                                                   recreate_ffts=recreate_ffts,
                                                   e_mode_method=e_mode_method,
                                                   b_mode_method=b_mode_method,
                                                   maps_to_cross=maps_to_cross,
                                                   save_ffts=save_ffts,
                                                   fft_shape=fft_shape,
                                                   calculate_dls=calculate_dls,
                                                   cutout_shape=cutout_shape,
                                                   quiet=quiet,
                                                   average_equivalent_pol_cross_spectra=average_equivalent_pol_cross_spectra))
                          for band, thismap in _map.iteritems() ] )
        except AttributeError:
            # If this isn't a dictionary, pass through here and do the rest
            # of the function.
            pass

        start_time = time.clock()

        ###############################################
        # Validate inputs and options

        # Check that the input map is unweighted.
        if getattr(_map, 'weighted_map', False):
            raise ValueError("You need to give me an unweighted map! (Run map.removeWeight().)")
        if cross_map is not None and getattr(cross_map, 'weighted_map', False):
            raise ValueError("You need to give me an unweighted cross_map! (Run cross_map.removeWeight().)")

        # If we want the cross-spectrum of two maps, make sure that the second map
        # has all the same properties as the first.
        if cross_map is not None and cross_map!=_map:
            raise ValueError("Each of the two maps must have the same properties (resolution, size, etc.) if I am to find their cross-spectra.")

        ###############################################
        # Figure out what FTs we have, and what we need to make.  Standardize the fft_shape (padding).

        ffts_to_make  = []
        existing_ffts = [] # tmp variable, figure out how to remove this and clean up this code-block
        if recreate_ffts:  # make everything
            ffts_to_make = maps_to_cross
            if fft_shape is None: fft_shape = _map.shape # if padding is not specified, set shape to existing map

        else: # try to use existing fft's
            for pol_index, pol1 in enumerate(maps_to_cross):
                if pol1 in _map.ft_maps: existing_ffts.append(pol1)
                else: ffts_to_make.append(pol1)

            # Set fft_shape
            if fft_shape is None:
                if len(existing_ffts)==0: fft_shape = _map.active_shape # set shape from real-space map
                else: fft_shape = _map.ft_maps[0].shape # set shape from fft map

        # have window?
        if len(ffts_to_make) !=0 and sky_window_func is None: # the window is only used if we are making ffts here
            if hanning_window: sky_window_func = util.math.hanning(fft_shape)/np.sqrt(0.1406)
            else: sky_window_func = np.ones(fft_shape)

        ###############################################
        # If E or B FTs are missing, make them
        if (('E' in ffts_to_make) or ('B' in ffts_to_make)):
            if not quiet:
                print "Constructing E and B maps."
            self.constructEB(_map, apodization_window=sky_window_func,
                             fft_shape=fft_shape,
                             b_mode_method=b_mode_method,
                             e_mode_method=e_mode_method,
                             store_ffts=True, # Must be True, or we can't retrieve it later.
                             remake=True, quiet=quiet)
        elif not quiet and ('E' in maps_to_cross or 'B' in maps_to_cross):
            print "Using existing FTs for E and B maps."

        if cross_map is not None:
            if (('E' in ffts_to_make) or ('B' in ffts_to_make)):
                if not quiet:
                    print "Constructing E and B maps for cross_map."
                self.constructEB(cross_map, apodization_window=sky_window_func,
                                 fft_shape=fft_shape,
                                 b_mode_method=b_mode_method,
                                 e_mode_method=e_mode_method,
                                 store_ffts=True, # Must be True, or we can't retrieve it later.
                                 remake=True, quiet=quiet)
            elif not quiet and ('E' in maps_to_cross or 'B' in maps_to_cross):
                print "Using existing FTs for cross-map E and B maps."


        ###############################################
        # Output structures
        power_spectra = struct({'ell_bins':struct(),
                                'ell':struct()}) # The power_spectra struct will be used for output Cls
        #power_spectra_variance = {}

        # Grab the Fourier transforms of the input map(s).
        map_fft = struct()
        for pol in maps_to_cross:
            if pol not in ['E','B']:
                do_remake = (pol in ffts_to_make) or (recreate_ffts)
                map_fft[pol] = _map.getFTMap(return_type=pol, remake=do_remake, save_ffts=save_ffts,
                                              sky_window_func=sky_window_func, normalize_window=True,
                                              fft_shape=fft_shape)
                if (cutout_shape is not None) and not (np.array(cutout_shape) == np.array(map_fft[pol].shape)).all():
                    map_fft[pol] = (tools.getFFTCutout(map_fft[pol], cutout_shape)
                                        *np.sqrt(float(np.prod(cutout_shape))/float(np.prod(fft_shape))))

            else:
                map_fft[pol] = _map.getFTMap(return_type=pol, remake=False) # can only re-make E and B FTs with constructEB
                if (cutout_shape is not None) and not (np.array(cutout_shape) == np.array(map_fft[pol].shape)).all():
                    map_fft[pol] = (tools.getFFTCutout(map_fft[pol], cutout_shape)
                                    *np.sqrt(float(np.prod(cutout_shape))/float(np.prod(fft_shape))))


        # If we don't have a separate second map, then we're doing an autospectrum.
        # In that case, we only need to find the complex conjugate of the FTs.
        if cross_map is None:
            cross_map = _map
            autospectrum = True
        else:
            autospectrum = False
        cross_map_fft = struct()
        for pol in maps_to_cross:
            if pol not in ['E','B']:
                do_remake = (not autospectrum) and ((pol in ffts_to_make) or (recreate_ffts))
                cross_map_fft[pol] = cross_map.getFTMap(return_type=pol, remake=do_remake, save_ffts=save_ffts,
                                                        sky_window_func=sky_window_func, normalize_window=True,
                                                        fft_shape=fft_shape, conjugate=True)
                if (cutout_shape is not None) and not (np.array(cutout_shape) == np.array(cross_map_fft[pol].shape)).all():
                    cross_map_fft[pol] = (tools.getFFTCutout(cross_map_fft[pol], cutout_shape)
                                               *np.sqrt(float(np.prod(cutout_shape))/float(np.prod(fft_shape))))
            else:
                cross_map_fft[pol] = cross_map.getFTMap(return_type=pol, remake=False, conjugate=True) # can only re-make E and B FTs with constructEB
                if (cutout_shape is not None) and not (np.array(cutout_shape) == np.array(cross_map_fft[pol].shape)).all():
                    cross_map_fft[pol] = (tools.getFFTCutout(cross_map_fft[pol], cutout_shape)
                                               *np.sqrt(float(np.prod(cutout_shape))/float(np.prod(fft_shape))))


            

        ######################################################################
        # Create a grid with the map's total ell at each
        # frequency-domain point, and initialize the output dictionaries.
        try:
            if np.sum([sum(_map.pol_maps[x].padding) for x in maps_to_cross if x in _map.pol_maps]):
                _map.trimPadding() # Start by trimming the map down to its active area.
                if not autospectrum: cross_map.trimPadding()
        except AttributeError:
            # you will end up here if the map object has no field '.pol_maps', which means it is a T only map
            if sum(_map.padding):
                _map.trimPadding()
                if not autospectrum: cross_map.trimPadding()

        ellgrids = self.setEllgrids(fft_shape, _map.reso_arcmin, ell_bins=ell_bins, trim_to_shape=cutout_shape)
        ellgrid = ellgrids['ellgrid']

        timestamp1 = time.clock()
        if not quiet: print ("   Calculating Cls: Time since start: "+
                             str(timestamp1-start_time)+". Time since last checkpoint: "+
                             str(timestamp1-start_time)+".")

        # Verify that all fft's are the same shape
        f_shape = fft_shape if cutout_shape is None else cutout_shape
        if np.array([map_fft[ii].shape != tuple(f_shape) for ii in maps_to_cross]).any():
            raise ValueError("***Error in calculateCls(): Shapes of FFT maps must all be equal in order to take the cross-spectrum.")
        if not autospectrum:
            if np.array([cross_map_fft[ii].shape != tuple(f_shape) for ii in maps_to_cross]).any():
                raise ValueError("***Error in calculateCls(): Shapes of FFT cross_maps must all be equal in order to take the cross-spectrum.")

        ##############################################################
        # Finally, find the auto- and cross-correlation power spectra.
        # fft_norm = _map.reso_rad**2 / (fft_shape[0]*fft_shape[1]) # Multiply products of FFTs by this.
        fft_norm = _map.reso_rad**2 / (f_shape[0]*f_shape[1]) # Multiply products of FFTs by this.
        binned_ell = ellgrids['binned_ell'] # Holds the bin number of each ell value.
        if ell_bins is None: ell_bins=self.ell_bins.copy()
        if uv_mask is None: uv_mask={}
        map_fft_power = {}
        for pol_index, pol1 in enumerate(maps_to_cross):
            for pol_index2, pol2 in enumerate(maps_to_cross):
                # If "map" and "cross_map" are the same, then, e.g., the EB and BE cross-spectra
                # are exactly the same, so don't bother recalculating them.
                if autospectrum and pol_index2<pol_index: continue

                # Cross the FTs of the two maps. Remember that the "cross_map_fft" have already
                # had their complex conjugates taken, to speed computation in this loop.
                xspec = pol1+pol2 # Simplify things by assigning this cross-spectrum its own name.
                this_map_fft = map_fft[pol1] # Define so we can use numexpr
                this_cross_fft = cross_map_fft[pol2] # Define so we can use numexpr
                map_fft_power[xspec] = ne.evaluate("fft_norm*(this_map_fft * this_cross_fft).%s"%realimag)

                # The rest of the code in this section is binning. Skip that if we want to
                # get back the full 2D power spectrum.
                if return_2d:
                    if calculate_dls:
                        map_fft_power[xspec] *= ellgrid*(ellgrid-1)/(2.*np.pi)
                    power_spectra[xspec] = map_fft_power[xspec]
                    power_spectra['ell'][xspec] = ellgrid
                    power_spectra['ell_bins'][xspec] = np.zeros(1)
                    continue

                ###############################################
                # Set up the bins for the final power spectrum
                # Note that we should have already calculated binned_ell in the
                # self.setEllgrids call above (or retrieved a previously-calculated
                # version), so what we're doing here is finding n_ell_bins and
                # making sure that ell_bins is set.
                if xspec not in binned_ell and xspec in self.delta_ell:
                    binned_ell[xspec] = util.tools.binGrid(ellgrid, delta_bin=self.delta_ell[xspec])
                if xspec not in binned_ell:
                    if xspec not in ell_bins: # Set default ell bins if they aren't provided.
                        #ok
                        binned_ell[xspec] = util.tools.binGrid(ellgrid, delta_bin=self.delta_ell.setdefault(xspec,self.default_delta_ell))
                        n_ell_bins = int(np.max(binned_ell[xspec]))+1
                        ell_bins[xspec] = np.asarray([ [self.delta_ell.setdefault(xspec,self.default_delta_ell)*index+1,
                                                        self.delta_ell.setdefault(xspec,self.default_delta_ell)*(index+1)]
                                                      for index in xrange(n_ell_bins)])
                    elif np.isscalar(ell_bins[xspec]):
                        binned_ell[xspec] = util.tools.binGrid(ellgrid, delta_bin=ell_bins[xspec])
                        n_ell_bins = int(np.max(binned_ell[xspec]))+1
                        ell_bins[xspec] = np.asarray([ [ell_bins[xspec]*index+1, ell_bins[xspec]*(index+1)] for index in xrange(n_ell_bins)])
                    else:
                        ell_bins[xspec] = np.asarray(ell_bins[xspec])
                        n_ell_bins = len(ell_bins[xspec])
                        binned_ell[xspec] = util.tools.binGrid(ellgrid, bins=ell_bins[xspec])
                elif xspec in ell_bins:
                    n_ell_bins = len(ell_bins[xspec])
                else:
                    n_ell_bins = int(np.max(binned_ell[xspec]))+1
                    ell_bins[xspec] = np.asarray([ [self.delta_ell.setdefault(xspec,self.default_delta_ell)*index+1,
                                                    self.delta_ell.setdefault(xspec,self.default_delta_ell)*(index+1)]
                                                  for index in xrange(n_ell_bins)])

                power_spectra['ell_bins'][xspec] = ell_bins[xspec]
                power_spectra['ell'][xspec] = np.mean(ell_bins[xspec], axis=1)

                # Create default spectral masks, if needed, and total up the mask for use in normalization.
                if not isinstance(uv_mask, dict):
                    this_uv_mask = uv_mask
                elif xspec not in uv_mask:
                    if xspec[::-1] in uv_mask:
                        this_uv_mask = uv_mask[xspec[::-1]]
                    else:
                        this_uv_mask = np.ones(map_fft_power[xspec].shape)
                elif uv_mask[xspec].shape != map_fft_power[xspec].shape:
                    print "[maps.calculateCls] Error! uv_mask["+xspec+"].shape = "+str(uv_mask[xspec].shape)+", but map_fft_power["\
                             +xspec+"].shape = "+str(map_fft_power[xspec].shape)+"! Remaking the uv_mask!"
                    uv_mask[xspec] = np.ones(map_fft_power[xspec].shape)
                    this_uv_mask = uv_mask[xspec]
                else:
                    this_uv_mask = uv_mask[xspec]

                # Define some aliases to use in the weaved C loop below.
                this_mask_sum = np.zeros([n_ell_bins])
                this_power_spectrum = np.zeros([n_ell_bins])
                #this_variance = np.zeros([n_ell_bins])
                this_binned_ell = binned_ell[xspec]
                this_fft_power = map_fft_power[xspec]
                if calculate_dls: dl = 1.0
                else: dl = 0.
                c_code=r"""
                #line 4456 "maps.py"
                double pi = double(3.14159);
                double ell_factor = double(1.0);
                for (int iy=0; iy<Nellgrid[0]; ++iy) { //"y" is the first index, i.e. ellgrid.shape[0]
                   for (int ix=0; ix<Nellgrid[1]; ++ix) { //"x" is the second index, ellgrid.shape[1]
                       if ((THIS_BINNED_ELL2(iy, ix) >= n_ell_bins) || (THIS_BINNED_ELL2(iy, ix) < 0))  {
                           // Check to be sure we won't try to write outside the bounds of our arrays!
                           continue;
                           }
                        this_mask_sum[THIS_BINNED_ELL2(iy, ix)] += THIS_UV_MASK2(iy, ix);
                       if (dl != 0.0) {
                           ell_factor = (ELLGRID2(iy ,ix) * (ELLGRID2(iy ,ix) + 1.0) / ( 2.0 * pi ));
                       }
                       this_power_spectrum[THIS_BINNED_ELL2(iy, ix)] += THIS_FFT_POWER2(iy, ix) * THIS_UV_MASK2(iy, ix) * ell_factor;
                    } //End for(loop over ix)
                } //End for(loop over iy)
                for (int ibin=0; ibin<n_ell_bins; ++ibin) {
                    if (this_mask_sum[ibin]!=0) {
                       this_power_spectrum[ibin] /= this_mask_sum[ibin];
                       //this_variance[ibin] = this_power_spectrum[ibin] / (this_mask_sum[ibin]==1.? 1. : sqrt(this_mask_sum[ibin]/2.));
                       }
                } //End for (loop over ibin)
                """
                weave.inline(c_code, ['dl','ellgrid', 'this_mask_sum', 'this_binned_ell',
                                      'n_ell_bins', 'this_uv_mask','this_power_spectrum',
                                      'this_fft_power'], compiler='gcc')

                power_spectra[xspec] = this_power_spectrum.copy()
                #power_spectra_variance[xspec] = this_variance.copy()

        # If requested, combine equivalent cross spectra (e.g. "TE" and "ET").
        if average_equivalent_pol_cross_spectra:
            # Loop through the (hard-coded) list of cross-spectra to combine.
            for spec_a, spec_b in equivalent_ps.iteritems():
                # Check that we have both of the cross-spectra in the list.
                if spec_b in power_spectra and spec_a in power_spectra:
                    ## Make sure that the two spectra have the same ell binning.
                    ## I'm skipping this check in the interest of speed -- we'll get
                    ## an exception if the number of bins is different. Otherwise, we'll
                    ## be quietly wrong.
                    #if (not (power_spectra['ell'][spec_b]==power_spectra['ell'][spec_a]).all()
                    #    or not (power_spectra['ell_bins'][spec_b]==power_spectra['ell_bins'][spec_a]).all()):
                    #    warnings.warn("I can't combine the %s and %s spectra, because they don't have the same binning." % (spec_a, spec_b))
                    #    continue

                    # Average the two spectra together.
                    ps_a, ps_b = power_spectra[spec_a], power_spectra[spec_b]
                    power_spectra[spec_a] = ne.evaluate("(ps_a+ps_b) / 2.0")

                    # Remove the unneeded spectrum.
                    del(power_spectra['ell'][spec_b])
                    del(power_spectra['ell_bins'][spec_b])
                    del(power_spectra[spec_b])

        # Flag whether or not these power spectra are D_ell.
        power_spectra['is_d_ell'] = calculate_dls

        end_time = time.clock()
        if not quiet: print "   Calculating Cls: Time since last checkpoint: "+str(end_time-timestamp1)+" seconds."
        if not quiet:
            print "   Finished calculating C_ells. Total time: "+str(end_time-start_time)+" seconds."

        return power_spectra

#################################################################################################################################################
def bin2DPowerSpectrum(ps2d, ellgrid=None, uv_mask=None,
                       delta_ell=50, ell_bins=None,
                       norm_factor=1., calculate_dls=False):
    """
    Bin up a 2D power spectrum. We take a 2D power spectrum and
    azimuthally average it. We return the new 1D power spectrum.
    This is equivalent to what happens at the end of
    maps.MapAnalyzer.calculateCls, separated out here in case
    you have a 2D spectrum you want to push the rest of the way
    through.
    Note that this is the python version of the binning in
    calculateCls. It's slower than the weaved loop in that
    function. Rewrite this function to use the weave
    if speed is important to you.
    INPUTS
        ps2d: (dict or array) If an array, a 2D power spectrum.
            If a dict, it should be the output of calculateCls.
        ellgrid [None]: (array) If ps2d is a dict, we ignore this
            argument and use ps2d['ell'] instead. Otherwise we
            use this array to know the ell of each bin of the input.
        uv_mask [None]: (array) A weighting array for the azimuthal
            averaging. Defaults to all ones.
        delta_ell [50]: (float) If we don't have an explicit ell_bins
            for a given cross-spectrum, construct a binning with
            evenly spaced bins.
        ell_bins [None]: (list of 2-element lists) Binning for the
            azimuthal average. It should give both an upper and lower
            bin edge, but we'll only use the lower edges to divide bins.
        norm_factor [1.]: (float) Multiply the final power spectra
            by this number.
        calculate_dls [False]: (bool) If True, multiply the power in
            each bin by ell*(ell+1)/(2pi) before averaging.
    OUTPUT
        If the input is a dict of the form output by calculateCls, the output
        will be of the same form. If the input is an array, the output
        will be a dict with keys 'cls', 'ell', and 'ell_bins'.
    EXCEPTIONS
        ValueError if we need an ellgrid but don't have one.
        ValueError if an input ellgrid or uv_mask is the wrong shape.
    AUTHOR
        Stephen Hoover, 14 March 2014
    CHANGES
        8 May 2014: Skip spectra which don't exist. SH
    """
    #########################################################
    # First do the core of the code. This is what happens if
    # get get just a single power spectrum for input.
    output_cls = struct({'ell':struct(), 'ell_bins':struct(),
                         'is_d_ell':calculate_dls})

    if not isinstance(ps2d, dict):
        if ellgrid is None:
            raise ValueError("If you give me an array for the power spectrum, you must also give me the ellgrid!")

        # If we didn't get an input for the mask, then don't mask anything.
        if uv_mask is None:
            uv_mask = np.ones_like(ellgrid)

        if ellgrid.shape!=ps2d.shape:
            raise ValueError("The input ellgrid doesn't have the right shape!")
        if uv_mask.shape != ps2d.shape:
            raise ValueError("The input uv_mask doesn't have the right shape!")

        # Some setup: Create ell bins. If we didn't get an explicit "ell_bins"
        # input, then create evenly spaced bins from the delta_ell input.
        # Use the "one_sided_bins" array for the histogram function.
        n_ell_bins = int(np.ceil(np.max(ellgrid / delta_ell)))
        if ell_bins is None:
            ell_bins = np.asarray([ [delta_ell*index+1,
                                     delta_ell*(index+1)]
                                   for index in xrange(n_ell_bins)])
        one_sided_bins = np.append(ell_bins[:,0], [ell_bins[-1,1]])

        if calculate_dls: ell_weighting = ellgrid*(ellgrid+1)/np.pi/2.
        else: ell_weighting = 1.

        # Bin everything up. This is two steps: First find the summed
        # power*weight in each bin, then find the summed weight in each bin.
        summed_cls, bins = np.histogram(ellgrid, bins=one_sided_bins,
                                        weights=ps2d*uv_mask*ell_weighting)
        summed_weights, bins = np.histogram(ellgrid, bins=one_sided_bins,
                                            weights=uv_mask)

        # Store everything in the output structure, and return.
        output_cls['cls'] = np.nan_to_num(norm_factor * summed_cls / summed_weights)
        output_cls['ell'] = np.mean(ell_bins, axis=1)
        output_cls['ell_bins'] =  ell_bins
    else:
        #############################################################
        # If we got a dictionary for ps2d, assume it's arranged
        # like the output of calculateCls. Go through each spectrum
        # and bin each of them.

        # Start with some setup. Arrange the inputs neatly.
        default_delta_ell = 50
        if not isinstance(delta_ell, dict):
            default_delta_ell = delta_ell
            delta_ell = {}
        else:
            delta_ell = delta_ell.copy() # Don't modify the input!

        if ell_bins is None:
            ell_bins = struct()
        else:
            ell_bins = ell_bins.copy() # Don't modify the input!

        # Now go through each cross-spectrum in the input dictionary.
        for xspec in ps2d['ell']:
            if xspec not in ps2d: continue

            # Pull out the values we need: An ellgrid and the binning.
            ellgrid=ps2d['ell'][xspec]

            if uv_mask is None:
                uv_mask = np.ones_like(ellgrid)

            this_dell = delta_ell.setdefault(xspec,default_delta_ell)
            n_ell_bins = int(np.ceil(np.max(ellgrid / this_dell)))
            if xspec not in ell_bins:
                ell_bins[xspec] = np.asarray([ [this_dell*index+1,
                                                this_dell*(index+1)]
                                              for index in xrange(n_ell_bins)])

            # Re-call this function with the bare array. This will put us in
            # the first block of code.
            these_cls = bin2DPowerSpectrum(ps2d[xspec], ellgrid=ellgrid, uv_mask=uv_mask,
                                           delta_ell=this_dell, ell_bins=ell_bins[xspec],
                                           norm_factor=norm_factor,
                                           calculate_dls=calculate_dls)

            # Store everything in the output dictionary.
            output_cls['ell'][xspec] = these_cls['ell']
            output_cls['ell_bins'][xspec] =  these_cls['ell_bins']
            output_cls[xspec] = these_cls['cls']

    return output_cls

#################################################################################################################################################
def qhatUhatFromTwoMaps(polMap1, polMap2, mask,
                        uv_mask=None, laxis=350,
                        lmin=500, lmax=2500,
                        verbose=False):
    """
    Calculate the dimensionless T->P leakage coefficients, TQ/TT and TU/TT.
    You can obtain noise estimates for the input with e.g.
    maps.calculateNoise(polMap1-polMap2, mask=mask)
    INPUTS
        polMap1: (PolarizedMap) The map coadded from half of the data.
        polMap2: (PolarizedMap) The map coadded from the other half of the data.
        mask: the apodization mask for these maps.
    OPTIONAL INPUTS
        uv_mask [None]: (2D array) The uv-space mask to apply when crossing maps.  By default
            this program constructs a mask that is one everywhere except within LAXIS of the LX
            or LY axis. If you provide a uv_mask input and also set laxis>0, then the mask
            used will be the product of the uv_mask and the laxis mask.
            NB: I (RK) recommend using *only* the default uv_mask rather than
            supplying your own, because any guarantees about
            not biasing the TE spectrum are based on that default uv_mask (specifically, laxis=350).
        laxis [350]: used in constructing the default UV_MASK (see above).
        lmin [500]: the minimum ell used in estimating the TP coefficients.
        lmax [2500]: the maximum ell used in estimating the TP coefficients.
    OUTPUT
        The output is a list [scaleQhat, scaleUhat], where:
            scaleQhat - the weighted average of TQ/TT.
            scaleUhat - the weighted average of TU/TT.
    AUTHOR
        Ryan Keisler, 23 October 2013
    CHANGES
        11 Nov 2013: Changed calculateCls call to include save_ffts=False, recreate_ffts=True. Add "verbose" mode.
        27 Dec 2013: Add "uv_mask" input. SH
        05 Feb 2014: Moved lmin up to 500, made uniform ell-weighting the default, put
                     in default uv_mask that masks out regions near the LX or LY axes.  RK
        17 Feb 2014: If laxis > 0 and we have a uv_mask input, combine the two. SH
        29 Apr 2014: Implemented a better estimator.  No longer returns cl_tqtt and cl_tutt.  RK
        30 Apr 2014: Removed uniform_weight, because it's always true for the new estimator, and removed
                     several associated keywords.  RK
        11 Jun 2014: Updated 2d estimator.  RK
    """

    from sptpol_software.analysis.maps import MapAnalyzer
    from numpy import sum, ones_like, ones, pi

    # Create a map analyzer object so that we can calculate power spectra.
    map_an = MapAnalyzer()

    # Make the uv_mask.  By default this will mask out areas of
    # uv space within LAXIS of either the LX or LY axis.
    tshape = (uv_mask.shape if uv_mask is not None else polMap1['T'].map.shape)
    uv_axis_mask = np.ones(tshape)
    if laxis > 0:
        if verbose:
            print "Masking out all modes with |ell_X| < %.2f or |ell_Y| < %.2f." % (laxis, laxis)
        delta_l0 = 2.*pi/(tshape[0]*(polMap1['T'].reso_arcmin/60.*pi/180.))
        delta_l1 = 2.*pi/(tshape[1]*(polMap1['T'].reso_arcmin/60.*pi/180.))
        laxis_pix0 = int(laxis/delta_l0)
        laxis_pix1 = int(laxis/delta_l1)
        uv_axis_mask[0:laxis_pix0, :] = 0.
        uv_axis_mask[-laxis_pix0:, :] = 0.
        uv_axis_mask[:, 0:laxis_pix1] = 0.
        uv_axis_mask[:, -laxis_pix1:] = 0.
    if uv_mask is None:
        uv_mask = uv_axis_mask
    else:
        uv_mask *= uv_axis_mask

    # Take all cross-spectra.
    cl = map_an.calculateCls(polMap1, polMap2, sky_window_func=mask,
                             fft_shape=(uv_mask.shape if uv_mask is not None else None),
                             uv_mask=uv_mask, maps_to_cross=['T','Q','U'], save_ffts=False,
                             recreate_ffts=True, quiet=(not verbose), return_2d=True)

    l = cl['ell'].TT

    # We can measure TP/TT at every ell, but we'd like to return a single value.
    weight_th = uv_mask
    wh_zero = where((l<lmin) | (l>lmax))
    weight_th[wh_zero[0], wh_zero[1]]=0.
    weight_th /= sum(weight_th)
    cl_tq = 0.5*(cl['TQ'] + cl['QT'])
    cl_tu = 0.5*(cl['TU'] + cl['UT'])
    cl_tt = cl['TT']


    # Calculate and return scalePhat==weighted sum of TP/TT.
    scaleQhat = np.sum(cl_tq*weight_th)/np.sum(cl_tt*weight_th)
    scaleUhat = np.sum(cl_tu*weight_th)/np.sum(cl_tt*weight_th)

    if verbose:
        print ' '
        print 'TQ/TT = %0.4f'%scaleQhat
        print 'TU/TT = %0.4f'%scaleUhat
        print ' '

    return scaleQhat, scaleUhat


def qhatUhatFromTwoMapsRADependent(A_file,B_file, delta_RA=13., laxis=100, lmin=100,
                                   lmax=1000, radius_arcmin=30., poly_order=3):
    '''
    Calculate the T->P monopole leakage terms as functions of RA.  Calculate qScale and uScale
    over maps of strip-width delta_RA.  Then fit a polynomial of order poly_order to these fits.
    Output a map of model qScale and uScale values, instead of just one for the entire map.
    Based on code for original RA-dependent calculation of monopole leakage terms written by JT Sayre.
    Author: Jason W. Henning
    '''

    A = files.read(A_file)
    B = files.read(B_file)

    print 'Removing weights and flattening pol angles...'
    S = (A+B).removeWeight().flattenPol()
    A.removeWeight().flattenPol()
    B.removeWeight().flattenPol()

    print 'Calculating grid of RAs and Decs...'
    from sptpol_software.observation.sky import pix2Ang
    X,Y = np.meshgrid(np.arange(S.pol_maps.Q.map.shape[1]), np.arange(S.pol_maps.Q.map.shape[0]))
    RAs, Decs = pix2Ang((Y.flatten(),X.flatten()),
                        ra_dec_center=S.center, reso_arcmin=S['T'].reso_arcmin,
                        map_pixel_shape=S.shape, proj=S['T'].projection)

    RA_map = ((RAs+180)%360 -180.).reshape(S.shape)
    WMP = S.weight[:,:,0,0]

    minRA = np.amin((WMP>0)*RA_map)
    maxRA = np.amax((WMP>0)*RA_map)

    RA_breaks = np.linspace(minRA, maxRA, delta_RA)
    RAs = (RA_breaks[1:] + RA_breaks[:-1])/2.

    #Obtain qScale and uScale only over strips of RA.
    factors = []
    for RA_min,RA_max in zip(RA_breaks[:-1], RA_breaks[1:]):
        thisWT = WMP * ((RA_map>=RA_min) & (RA_map<RA_max))
        msk = makeBorderApodization(thisWT, radius_arcmin=radius_arcmin,
                                    given_weight_map=True,
                                    given_reso_arcmin=S['T'].reso_arcmin)
        qHat,uHat = qhatUhatFromTwoMaps(A,B, msk, laxis=laxis,
                                        lmin=lmin, lmax=lmax)
        factors.append([qHat,uHat])

    factors = np.array(factors)
    print 'Done calculate RA-dependent factors: ', RAs, factors

    #Now fit a low-order polynomial to the qScales and uScales.
    print 'Calculating order ', poly_order, 'poly to factors...'
    scaleFit = np.polyfit(RAs, factors, poly_order)

    #Finally, generate maps of the model value of qScale and uScale across the map region.
    print 'Generating maps of modelQ and modelU factors'
    modelQ = np.zeros(S.shape)
    modelU = np.zeros(S.shape)

    N = int(poly_order)
    while N >= 0:
        modelQ += scaleFit[poly_order - N,0]*RA_map**N
        modelU += scaleFit[poly_order - N,1]*RA_map**N
        N -= 1

    return modelQ, modelU

def getTpWeights(l, uk_arcmin_t=7.0, uk_arcmin_p=9.9,
                fwhm_arcmin=1.05, include_atmos_in_t=True,
                lmin=500, lmax=2500):
    '''
    Calculate the optimal ell weighting to calculate T->P leakage.
    INPUTS
        l: an array of multipoles at which you want the weights calculated.
    OPTIONAL INPUTS:
        lmin [300]: weights below this ell are set to zero.
        lmax [2500]: weights above this ell are set to zero.
        uk_arcmin_t [7.0]: the estimate for the T white noise level in the all-of-the-data coadd.
            The results aren't very sensitive to this.  The answer will be slightly sub-optimal,
            but unbiased.
        uk_arcmin_p [9.9]: the estimate for the Q or U white noise level in the all-of-the-data coadd.
            The results aren't very sensitive to this.  The answer will be slightly sub-optimal,
            but unbiased.
        fwhm_arcmin [1.1]: the estimate of the gaussian FWHM of the beam.  Results aren't very sensitive to this.
        include_atmos_in_t [True]: a boolean denoting whether to include a 1/f-like atmospheric noise term
            in the temperature noise model.
    OUTPUT:
        weight_out - the optimal weights for calculating the weighted mean of TP/TT across many
            multipoles.  It has same dimension as the input multipole array.
    AUTHOR
        Ryan Keisler, 23 October 2013
    '''

    from numpy import genfromtxt, pi, sqrt, exp, log, interp, where

    # Get the theory CL's.
    camb_lens='/data/hoover/camb_outputs/camb_r0p0_mnu0p0_lensedtotCls.dat'
    tmp = genfromtxt(camb_lens)
    l_camb = tmp[:,0]

    # Convert from DL to CL.
    for i in range(1,5): tmp[:,i] *= (2.*pi/l_camb/(l_camb+1.))
    cl_tt_camb = tmp[:,1]
    cl_ee_camb = tmp[:,2]
    cl_bb_camb = tmp[:,3]
    cl_te_camb = tmp[:,4]
    cl_pp_camb = cl_ee_camb + cl_bb_camb

    # Define the Gaussian beam function BL.
    fwhm_rad = fwhm_arcmin/60.*pi/180.
    sigma_rad = fwhm_rad/(2.*sqrt(log(4.)))
    bl_camb = exp(-0.5*l_camb*(l_camb+1.)*sigma_rad**2.)

    # Create the noise temperature and polarization power spectra.
    # The atmospheric temperature power (the second term below in CL_TNOISE_CAMB)
    # is taken from the 150 GHz noise power in Keisler et al. (2011).
    atmos_amp_t = 0.
    atmos_amp_p = 0.
    if include_atmos_in_t: atmos_amp = 3e-4
    cl_tnoise_camb = ((uk_arcmin_t*(1./60.*pi/180.))**2. + \
                          (atmos_amp_t)*((l_camb+1)/1000.)**(-3.1))*bl_camb**(-2.)
    cl_pnoise_camb = ((uk_arcmin_p*(1./60.*pi/180.))**2. + \
                          (atmos_amp_p)*((l_camb+1)/1000.)**(-3.1))*bl_camb**(-2.)

    # Calculate the TP variance as a function of l.
    # I think the optimal l-weighting depends on the
    # amplitude of the thing we're trying to estimate.
    # We'll stick in 1% level leakage as a placeholder.
    # The estimator is slightly, ~10% sub-optimal, which is fine.
    rough_leakage_amp = 0.01
    power = 1.0
    var_th = (1.*(abs(rough_leakage_amp)**2.)*(cl_tnoise_camb*cl_tt_camb)**power + \
                  (cl_tt_camb*cl_pp_camb)**power + \
                  (cl_tt_camb*cl_pnoise_camb)**power + \
                  (cl_tnoise_camb*cl_pp_camb)**power + \
                  (cl_tnoise_camb*cl_pnoise_camb)**power)/cl_tt_camb**(2.*power)/l_camb
    weight_th = 1./var_th

    # Interpolate onto the input multipole grid.
    weight_out = interp(l,l_camb,weight_th)
    # Zero out the weights in l<LMIN and l>LMAX.
    # The vast majority of the weight is in this region [300,2500].
    # The resulting estimator is only ~10% sub-optimal.

    wh_zero = where((l<lmin) | (l>lmax))[0]
    if len(wh_zero)>0: weight_out[wh_zero]=0.
    # Normalize the weight to have sum()=1 and return in.
    weight_out /= sum(weight_out)
    return weight_out


def tpTestScript(paper='ee'):
    '''
        This is a test script showing how to measure the TP/TT coefficients.
        AUTHOR: Ryan Keisler, 23 October 2013
    CHANGES
        11 Nov 2013: Add "flattenPol" call. Won't change the answer much here, but it's good practice. SH
        12 Nov 2013: Add a call to calculateNoise to get the map noise levels. Add "expected output". SH
        05 Feb 2014: Expected values updated for lmin=500, uniform_weights, "zeros near LX/LY axes" uv_mask. RK
        17 Feb 2014: Updated BB paper coadd path.  RK.
        29 Apr 2014: Updated expected values for new 2d estimator.  And now uses a
                     pre-calculated point-source mask for BB paper.  RK
        11 Jun 2014: Updated expected values once again, for new 2d estimator.  RK
    '''

    from sptpol_software.util import files
    from sptpol_software.analysis.maps import qhatUhatFromTwoMaps, calculateNoise
    from sptpol_software.analysis.maps import makeBorderApodization, makeApodizedPointSourceMask

    if paper=='ee':
        #expected_qhat, expected_uhat = 0.0105, -0.0137  #1d estimator
        #expected_qhat, expected_uhat = 0.0132, -0.0153  #2d estimator
        expected_qhat, expected_uhat = 0.0105, -0.0152 #2d estimator, 11jun2014

        # These coadds are from /data41/acrites/bundle_coadds/20140129/ and
        # were made using /home/rkeisler/coadd_abby_maps.py.
        coaddA = files.read('/data/rkeisler/coaddA_EE_20140129.h5').coadd1
        coaddB = files.read('/data/rkeisler/coaddB_EE_20140129.h5').coadd2
        import pickle
        mask = pickle.load(open('/data/sptdat/ptsrc_masks/mask_ra23h30dec-55_s12.pkl'))
        maskbig = np.zeros((1700,1700))
        maskbig[0:1560,0:1560]=mask
        nroll = (1700-1560)/2
        maskbig = np.roll(np.roll(maskbig,nroll,axis=0),nroll,axis=1)
        mask = maskbig
    elif paper=='bb':
        #expected_qhat, expected_uhat = 0.0045, -0.0078 #1d estimator
        #expected_qhat, expected_uhat = 0.0080, -0.0057 #2d estimator
        expected_qhat, expected_uhat = 0.0051, -0.0087 #2d estimator, 11jun2014
        coaddA = files.read('/data/sptdat/map/20140121_ra23h30dec55_bb_alt_tcmb_rotrec/bundles150/smap_weighted_halfA_test.h5')
        coaddB = files.read('/data/sptdat/map/20140121_ra23h30dec55_bb_alt_tcmb_rotrec/bundles150/smap_weighted_halfB_test.h5')
        mask10 = makeBorderApodization(coaddB, apod_type='cos', radius_arcmin=10., zero_border_arcmin=5, verbose=False)
        import pickle
        ptsrc_mask = pickle.load(open('/home/hoover/ptsrc_mask_v1_150ghz_50mJy.pkl','r'))
        mask = mask10*ptsrc_mask
    else:
        print 'Not a recognized paper.'
        return -1

    # Remove weight and flatten pol angles.
    coaddA.removeWeight()
    coaddB.removeWeight()
    coaddA.flattenPol()
    coaddB.flattenPol()

    # Calculate the TP coefficients.
    this_qhat, this_uhat = qhatUhatFromTwoMaps(coaddA, coaddB, mask)

    print ' '
    print 'TQ/TT = %0.4f'%this_qhat
    print 'TU/TT = %0.4f'%this_uhat
    print ' '
    print '(Expected output: TQ/TT = %.4f, TU/TT = %.4f.)' % (expected_qhat, expected_uhat)

    return (np.round(this_qhat,4)==expected_qhat and np.round(this_uhat,4)==expected_uhat)


def inpaintSources(polmap, inpaint_source_file=None,
                   ra_ptsrcs=None, dec_ptsrcs=None,
                   side_inpaint=6., side_value=10., maps2inpaint=None,
                   inpaint_func='median',
                   quiet=False):

    '''
    This function in-paints maps in a PolarizedMap object at the
    locations of a list of points sources.  All pixels within some
    small disc are replaced by the median value of all pixels in a
    larger annulus.
    NB: This function will operate on weighted or unweighted maps.
        You should decide whether you want to pass it weighted or
        unweighted maps.
    INPUTS
      polmap - the PolarizedMap object which you want
      (The positions of the point sources to mask can be passed as either
      a config file (inpaint_source_file) or lists or RA/DEC's (ra_ptsrc, dec_ptsrcs)
      The config file overrides the lists, but you need to provide one or the other.)
      inpaint_source_file [None] - the string of the point source config file.
          e.g. /data/sptdat/ptsrc_masks/ptsrc_config_50mJy_ra23h30dec-55_2008_20120305.txt
      ra_ptsrcs [None] - the R.A. of the point sources you want to in-paint.
      dec_ptsrcs [None] - the Dec.'s of the point sources you want to in-paint.
    OPTIONAL INPUTS
      side_inpaint - the diameter of the smaller disc around each point
                     source that is in-painted, in arcminutes [6]
      side_value - the outer diameter of the larger annulus around
                   each point source that is used to determine the
                   in-paint value, in arcminutes [10].  Inner diameter
                   is side_inpaint.
      maps2inpaint - the list of maps in polmap that you want to in-paint,
             e.g. ['T','Q','U'].  Defaults to [None], which means *all*
             maps are in-painted.
      inpaint_func - 'median' or 'mean': which function you want to use to
             determine the value of the inpainted disc.
    created 29Jul2014, RK
    HISTORY
    10 Sep 2014 - made about 10X faster.  RK
    05 Oct 2014 - switched to using mean instead of median when calculating
                  replacement value.  Makes this a linear operation.  RK
    08 Oct 2014 - switched back to median!  A weighted mean is probably the
                  way to go.  RK
    15 Dec 2015 - added inpaint_func option.  JTS
    16 Feb 2016 - changed the "mean" option to do a weighted average. JTS
    '''

    if polmap.weighted_map:
        print "inpaintSources() should be done on un-weighted maps.  No sources were masked."
        return

    if maps2inpaint is None: maps2inpaint=polmap.polarizations
    if inpaint_source_file is not None:
	tmp = np.loadtxt(inpaint_source_file, skiprows=2)
	ra_ptsrcs = tmp[:,1]
	dec_ptsrcs = tmp[:,2]
    else:
        if (ra_ptsrcs is None) or (dec_ptsrcs is None):
	    print 'No point source positions were passed.  No sources were masked.'
	    return

    nx, ny = polmap.shape
    rpix_small = np.int(0.5*side_inpaint/polmap.reso_arcmin)
    rpix_big = np.int(0.5*side_value/polmap.reso_arcmin)

    nside_template = 2*rpix_big + 1
    xtmp, ytmp = x2d_y2d_from_shape((nside_template, nside_template), zero_mean=True)
    rtmp = np.sqrt(xtmp**2. + ytmp**2.)
    wh_small_orig = np.where(rtmp<=rpix_small)
    wh_small_orig = (xtmp[wh_small_orig], ytmp[wh_small_orig])
    wh_big_orig = np.where((rtmp>rpix_small)&(rtmp<=rpix_big))
    wh_big_orig = (xtmp[wh_big_orig], ytmp[wh_big_orig])

    # Get the pixel indices of these point sources.
    # ang2pix method of polmap object
    xy = [polmap.ang2Pix([this_ra, this_dec]) for this_ra, this_dec in zip(ra_ptsrcs, dec_ptsrcs)]
    inpaint_funcs = {'median':np.median,
                     'mean':  np.average}
    func = inpaint_funcs[inpaint_func]
    inpaint_kwargs = {}
    # Loop over the point sources.
    map_types = ['T','Q','U']
    for ii, (this_x, this_y) in enumerate(xy):
        wh_small = (wh_small_orig[0]+this_x, wh_small_orig[1]+this_y)
        wh_big = (wh_big_orig[0]+this_x, wh_big_orig[1]+this_y)

        # Throw out points beyond the edge of the map.
        wh=np.where((wh_small[0]>=0)&(wh_small[0]<nx)&(wh_small[1]>=0)&(wh_small[1]<ny))[0]
        wh_small = (wh_small[0][wh], wh_small[1][wh])

        wh=np.where((wh_big[0]>=0)&(wh_big[0]<nx)&(wh_big[1]>=0)&(wh_big[1]<ny))[0]
        wh_big = (wh_big[0][wh], wh_big[1][wh])

        if not (len(wh_small[0]) and len(wh_big[0]) and len(wh_small[1]) and len(wh_big[1])) :
            if not quiet:
                print "(%i, %i) is not in the active map region.  Skipping."%(ra_ptsrcs[ii], dec_ptsrcs[ii])
            continue

        if (inpaint_func=='mean') and (polmap.weight[:,:,0,0][wh_big] == 0).all():
            if not quiet:
                print "(%i, %i) is in the zero-weight map region. Skipping."%(ra_ptsrcs[ii], dec_ptsrcs[ii])
            continue


        # Replace the smaller disc with the mean
        # of the larger annulus for each map.
        for k in maps2inpaint:
            if inpaint_func == 'mean':
                j = map_types.index(k)
                inpaint_kwargs['weights'] = polmap.weight[wh_big[0],wh_big[1],j,j]
            this_interp = func(polmap[k].map[wh_big], **inpaint_kwargs)
            polmap[k].map[wh_small] = this_interp

def x2d_y2d_from_shape(shape, zero_mean=False):
    x2d = np.repeat(np.arange(shape[0]),shape[1]).reshape(shape)
    y2d = np.tile(np.arange(shape[1]),shape[0]).reshape(shape)
    if zero_mean:
        # New cast rule in numpy force me to convert mean to integer. Assert I am not doing anything wrong
        assert (x2d.mean()).is_integer()
        assert (y2d.mean()).is_integer()
        x2d = x2d - np.int(x2d.mean())
        y2d = y2d - np.int(y2d.mean())
    return x2d, y2d
