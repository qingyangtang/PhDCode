"""
Classes and functions relating directly to the sky. Contains, for example, Map objects
(which hold data organized by location on the sky) and functions to convert
between different sky projections.

Classes
    Map
    PolarizedMap
    TrueSky
    ObservedSky
    PolarizedTrueSky
    PolarizedObservedSky
    SkyStream
    PolarizedSkyStream

Functions
    pix2Ang
    ang2Pix

Non-Core Dependencies
   NumPy
   SciPy
"""

__metaclass__ = type  # Use "new-style" classes
__author__ = "Stephen Hoover"
__email__ = "hoover@kicp.uchicago.edu"
__version__ = "1.08"
__date__ = "2013-09-30"  # Date of last modification

import warnings
import pdb
import unittest
import os
import re
import numexpr as ne
import numpy as np
import scipy as sp
from scipy import interpolate, weave, ndimage
from numpy import sin, cos, sqrt
from glob import glob
import matplotlib.pyplot as plt
import matplotlib
import telescope
import receiver
from .. import constants
from ..constants import DTOR, RTOD
from .. import float_type
from ..util import tools, math, fits, hdf, time
from ..util.tools import struct
from ..data import c_interface
from copy import deepcopy
from mpl_toolkits.axes_grid1 import make_axes_locatable

ne.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', 4)))  # Don't use all the processors!

tqu = ['T', 'Q', 'U']
teb = ['T', 'E', 'B']
# pol_index = {'T':0, 'Q':1, 'U':2, 'I':0, 'V':3, 'T':0, 'E':1, 'B':2,
# 'I':0} # For translating letters into array indices

proj_name_to_index = {'Sanson-Flamsteed': 0, 'CAR': 1, 'SIN': 2, 'Healpix': 3,
                      'Sterographic': 4, 'Lambert': 5, 'CAR00': 7, 'rot00': 8, 'BICEP': 9}
# Lower-case version for comparisons.
__proj_name_to_index = dict([(key.lower(), value) for key, value in proj_name_to_index.iteritems()])

# Projections tested in testAllProj:
proj_index_in_idl = [0, 1, 2, 4, 5]
proj_to_test = [0, 1, 2, 4, 5, 7]

##########################################################################################


def read(filename, file_type=None, verbose=True):
    """
    Read out a Map or PolarizedMap which has been written to an output file.

    INPUTS
        filename : (string or list of strings) The filename(s) to read. May be given
            as a fully-specified filename, a string parseable by glob, or a list which
            may contain both kinds of strings.

        file_type [None]: (string) If the "type" input is "FITS" or "HDF5"
            (case insensitive), then we read the file as being in that format,
            regardless of extension.

        verbose [True]: (bool) Extra screen output.

    OUTPUT
        A Map or PolarizedMap object.
    """
    # If the filename is a string, proceed. Otherwise, treat it as a list of files, and read each separately.
    try:
        filename + ''
        # In case the file name is a string with wildcards, feed it through glob.
        globized_name = glob(filename)
        if len(globized_name) == 1:
            # The Map class's read function will return a PolarizedMap if that's what the file contains.
            return Map.read(globized_name[0], file_type=file_type)
        else:
            return read(globized_name, file_type=file_type, verbose=verbose)
    except TypeError:
        if verbose:
            print "Reading %d files." % len(filename)
        return map(lambda x: read(x, file_type=file_type), filename)

##########################################################################################


def projNameToIndex(proj):
    """
    Converts a proj name to the corresponding index. Lets us use more descriptive function calls.

    Supported projections are:

     0, "Sanson[-Flamsteed]":  Sanson-Flamsteed projection (x = ra*cos(dec), y = dec)
     1, "CAR":  CAR projection (x = ra, y = dec)
     2, "SIN":  SIN projection
     3, "Healpix":  Healpix (not a projection at all, but pixels on the sphere) [NOT IMPLEMENTED]
     4, "Stereo[graphic]":  stereographic projection [SAME AS proj5??]
     5, "Lambert":  Lambert azimuthal equal-area projection
     7, "CAR00": CAR projection (x = ra, y = dec), if the field is first
         rotated so that the RA, dec center is 0,0.
     8, "rot00"
     9, "BICEP":  ra,dec but pixel size in ra is set to be square at the mean dec: roughly width reso/cos(mean_dec),
                  but rounded to make integer pixels for ra -55->+55 and dec -70->-45
                  Basically http://en.wikipedia.org/wiki/Equirectangular_projection, with phi1~middle_dec (but rounded)
    INPUTS
        proj: (string, int): Returns ints directly, or converts strings to a corresponding proj index.

    OUTPUT
        An integer proj index.
    """
    # In case 'proj' is a complete name or index, return right away.
    try:
        proj = __proj_name_to_index.get(proj.lower(), proj)
    except AttributeError:
        pass

    if proj in proj_name_to_index.values():
        return proj

    # If 'proj' isn't a complete name, look for acceptable partial names.
    proj = str(proj).lower()
    if proj.startswith('sanson') or proj == 'sf':
        proj = 0
    elif proj == 'car00':
        proj = 7
    elif proj.startswith('car'):
        proj = 1
    elif proj.startswith('sin'):
        proj = 2
    elif proj.startswith('healpix'):
        proj = 3
    elif proj.startswith('stereo'):
        proj = 4
    elif proj.startswith('lambert'):
        proj = 5
    elif proj.startswith('bicep'):
        proj = 9
    else:
        raise ValueError('I don\'t know what projection "' + proj + '" is!')

    return proj

##########################################################################################


def pix2Ang(pixel_coords, ra_dec_center, reso_arcmin, map_pixel_shape, proj=0, wrap=True):
    """
    Supported projections are:

     0:  Sanson-Flamsteed projection (x = ra*cos(dec), y = dec)
     1:  CAR projection (x = ra, y = dec)
     2:  SIN projection
     3:  Healpix (not a projection at all, but pixels on the sphere) [NOT IMPLEMENTED]
     4:  stereographic projection
     5:  Oblique Lambert azimuthal equal-area projection  (ref p. 185, Snyder, J. P. 1987, Map Projections-A Working Manual (Washington, DC: U.S. Geological Survey))
     7:  CAR projection, with map rotated so that the center is at RA, dec = (0, 0)
     8:  "rot00"
     9:  BICEP projection
    INPUTS
        pixel_coords : (2-element tuple of arrays) A tuple or list of arrays, [y_coord, x_coord].
            Note the order! The first element is the "y", the mostly-dec coordinate, and the
            second element is the "x", the mostly-RA coordinate.

        ra_dec_center : (2-element array) The [RA, declination] of the center of the map.

        reso_arcmin : (float) The width of each pixel in arcminutes, assumed to be the same for
            both x and y directions.

        map_pixel_shape : (2-element array) The height and width of the map, in pixels.

        proj [0]: (int or string) Which map projection should I use to turn pixel coordinates
            on a flat map into angles on the curved sky? May be an integer index, or string
            name of the projection.

    OUTPUT
        (ra, dec): A 2-tuple of arrays. 'ra' and 'dec' are each arrays with the same number of elements as
        the arrays in the 'pixel_coords' input. 'ra' is the right ascension in degrees
        (wrapped to the range [0, 360) if wrap==True), and 'dec' is the declination in degrees.
        returns the ra,dec of the CENTER of the pixels
    """

    # Convert the "proj" input to an index.
    proj = projNameToIndex(proj)

    # Break out the x and y coordinates, and cast them as floats.
    pixel_coords = (y_coord, x_coord) = pixel_coords[0].astype(float_type), pixel_coords[1].astype(float_type)
    n_pixels = map_pixel_shape.astype(float_type)

    # shift to the center of the pixel, subtract off npix/2 to center around 0, then convert to degrees
    y_coord = (y_coord + 0.5 - 0.5 * n_pixels[0]) * reso_arcmin / 60
    x_coord = (x_coord + 0.5 - 0.5 * n_pixels[1]) * reso_arcmin / 60

    ra_dec_center_rad = ra_dec_center * DTOR  # Convert to radians

    if proj == 0:
        dec = ra_dec_center[1] - y_coord
        ra = x_coord / np.cos(dec * DTOR) + ra_dec_center[0]
    elif proj == 1:
        dec = ra_dec_center[1] - y_coord
        ra = x_coord + ra_dec_center[0]
    elif proj == 2:
        rho = sqrt(x_coord**2 + y_coord**2) * DTOR
        c = np.arcsin(rho)
        phi_temp = np.arcsin(cos(c) * sin(ra_dec_center_rad[1]) - DTOR * y_coord * cos(ra_dec_center_rad[1]))
        lambda_temp = RTOD * np.arctan2(x_coord * DTOR * sin(c),
                                        rho * cos(ra_dec_center_rad[1]) * cos(c) + y_coord * DTOR * sin(ra_dec_center_rad[1]) * sin(c))
        bad_rho = rho < 1e-8
        phi_temp[bad_rho] = ra_dec_center_rad[1]
        lambda_temp[bad_rho] = 0.

        dec = phi_temp * RTOD
        ra = ra_dec_center[0] + lambda_temp
    elif proj == 5 or proj == 4:
        rho = sqrt(x_coord**2 + y_coord**2) * DTOR
        if proj == 5:
            c = 2 * np.arcsin(rho / 2)
        if proj == 4:
            c = 2 * np.arctan(rho / 2)
        phi_temp = np.arcsin(cos(c) * sin(ra_dec_center_rad[1]) -
                             DTOR * y_coord * sin(c) / rho * cos(ra_dec_center_rad[1]))
        lambda_temp = RTOD * np.arctan2(x_coord * DTOR * sin(c),
                                        rho * cos(ra_dec_center_rad[1]) * cos(c) + y_coord * DTOR * sin(ra_dec_center_rad[1]) * sin(c))
        bad_rho = rho < 1e-8
        phi_temp[bad_rho] = ra_dec_center_rad[1]
        lambda_temp[bad_rho] = 0.

        dec = phi_temp * RTOD
        ra = ra_dec_center[0] + lambda_temp
    elif proj == 7:
        ra, dec = applyPointingOffset(ra_dec_center, [x_coord, y_coord], offset_units='degrees', as_azel=False)
    elif proj == 9:
        # BICEP  ra_dec_center=(0.0, -57.5)
        ny, nx = n_pixels
        # nx, ny = (236, 100) # for BICEP
        pixsize = reso_arcmin / 60.  # pixsize = 0.25 for BICEP  ->  reso_arcmin=15.
        sy = pixsize  # dec spacing in degrees
        asx = pixsize / cos(ra_dec_center_rad[1])  # approximate spacing in ra
        sx = np.round(asx * nx) / nx  # round so an integer number of degrees fit in the given number of pixels.
        dec = ra_dec_center[1] - y_coord
        ra = x_coord * sx / sy + ra_dec_center[0]  # scale x_coord to make it square in center (undo ang2Pix)
    else:
        raise ValueError("I don't know what to do with proj " + str(proj) + ".")

    if wrap:
        # Wrap RA values to the [0, 360) range.
        tools.wrapAz(ra)
    else:
        # make sure branch cut for ra is opposite map center
        too_low = np.where(ra - ra_dec_center[0] < -180.)
        ra[too_low] += 360.
        too_high = np.where(ra - ra_dec_center[0] >= 180.)
        ra[too_high] -= 360.

    return ra, dec
##########################################################################################


def ang2Pix(ra_dec, ra_dec_center, reso_arcmin, map_pixel_shape, proj=0,
            round=True, bin_center_zero=True, return_validity=True, use_c_code=False):
    """
    Supported projections are:

     0:  Sanson-Flamsteed projection (x = ra*cos(dec), y = dec)
     1:  CAR projection (x = ra, y = dec)
     2:  SIN projection
     3:  Healpix (not a projection at all, but pixels on the sphere) [NOT IMPLEMENTED]
     4:  stereographic projection
     5:  Oblique Lambert azimuthal equal-area projection (ref p. 185, Snyder, J. P. 1987, Map Projections-A Working Manual (Washington, DC: U.S. Geological Survey))
     7:  CAR projection, with map rotated so that the center is at RA, dec = (0, 0)
     8:  "rot00"
     9:  BICEP projection

    INPUTS
        ra_dec : (2-element tuple of arrays) A tuple or list of arrays, [ra, dec]. Can also
            be a PointingSequence object. In degrees.

        ra_dec_center : (2-element array) The [RA, declination] of the center of the map. In degrees.

        reso_arcmin : (float) The width of each pixel in arcminutes, assumed to be the same for
            both x and y directions.

        map_pixel_shape : (2-element array) The height and width of the map, in pixels.  (n_pix_y, n_pix_x)

        proj [0]: (int or string) Which map projection should I use to turn angles on the
            curved sky into pixels on a flat map? May be an integer index, or string
            name of the projection.

        round [True]: (bool)

        bin_center_zero [True]: (bool)  only applies if round==False.  If bin_center_zero==True, shift output by 0.5

        return_validity [True]: (bool) If True, return a 2-tuple stating which pixel coordinates are good.
            If False, return only the PointingSequence object.

        use_c_code [False]: (bool) If True, use compiled C code to find the pointing. ~10-20% faster than
            the Python implementation.

    OUTPUT
        A PointingSequence object and a 2-tuple. The PointingSequence has pixel coordinates, in
        the form (y_coord, x_coord), where each element is an array of pixel indices.
        The second output tuple designates which pixel coordinates are "good", i.e., within
        bounds. It is of the form (good_ys, good_xs), where each element is
        an array of booleans.

        Note:  y_coord is more like elevation than dec: index zero corresponds to the least negative dec
               (closest to horizon at south pole)

     EXAMPLE to test 360 wrap:
        from sptpol_software.observation import sky
        ra =  np.array([-1.,0.,1., 89.,90.,91., 179.,180.,181., 269.,270.,271., 359.,360.,361.])
        dec = np.array([0]*len(ra))  # equator
        sky.ang2Pix([ra, dec], ra_dec_center=[0,    0], reso_arcmin=60., map_pixel_shape=(360, 360), return_validity = True, use_c_code = False, proj = 1)
        sky.ang2Pix([ra, dec], ra_dec_center=[180., 0], reso_arcmin=60., map_pixel_shape=(360, 360), return_validity = True, use_c_code = False, proj = 1)
        sky.ang2Pix([ra, dec], ra_dec_center=[180., 0], reso_arcmin=60., map_pixel_shape=(360, 360), return_validity = True, use_c_code = False, proj = 0)
        dec = np.array([-60]*len(ra))  # -60 (so cos = 0.5)
        sky.ang2Pix([ra, dec], ra_dec_center=[180., -60], reso_arcmin=60., map_pixel_shape=(360, 360), return_validity = True, use_c_code = False, proj = 1)
        sky.ang2Pix([ra, dec], ra_dec_center=[180., -60], reso_arcmin=60., map_pixel_shape=(360, 360), return_validity = True, use_c_code = False, proj = 0)  # y_coord's get squished
        sky.ang2Pix([ra, dec], ra_dec_center=[0., -60], reso_arcmin=60., map_pixel_shape=(360, 360), return_validity = True, use_c_code = False, proj = 5)  # proj5 not great for all-sky maps
        sky.ang2Pix([360-0.01, -40], ra_dec_center=[15.02715, -35.03219], reso_arcmin=60., map_pixel_shape=(360, 360), return_validity = True, use_c_code = False, proj = 0)
    """
    # If the input is just a single RA, dec, convert it to an array of arrays, as this function expects.
    if np.size(np.asarray(ra_dec)) == 2:
        ra_dec = np.asarray([[ra_dec[0]], [ra_dec[1]]])

    # Convert the "proj" input to an index.
    proj = projNameToIndex(proj)

    assert type(bin_center_zero) == bool

    # Check if we can access the C code before launching into the calculation.
    if use_c_code:
        try:
            clibs = c_interface.initializeCLibs()
        except OSError, err:
            warnings.warn("Failed to load sptpol_cdevel python library. Reverting to Python ang2Pix.", RuntimeWarning)
            use_c_code = False

    n_pixels = np.asarray(map_pixel_shape, dtype=float_type)  # used below if return_validity for both C and python

    if use_c_code:
        # Set up some of the variables needed for the C code.
        C = c_interface.C
        data_type = c_interface.typeindex if round else np.float64
        x_pixel_coord = np.empty(len(ra_dec[0]), dtype=data_type)
        y_pixel_coord = np.empty(len(ra_dec[1]), dtype=data_type)
        ra = np.ascontiguousarray(ra_dec[0], dtype=np.float64)
        dec = np.ascontiguousarray(ra_dec[1], dtype=np.float64)
        map_shape_deg = np.ascontiguousarray(map_pixel_shape) * reso_arcmin / 60.
        mapinfo = c_interface.MAPINFO(proj=proj, map_center=ra_dec_center,
                                      map_shape=map_shape_deg, reso_arcmin=reso_arcmin)

        # Call the appropriate function (it differs by data type returned).
        if round:
            clibs.ang2Pix(C.byref(mapinfo),
                          ra.ctypes.data_as(C.POINTER(C.c_double)),
                          dec.ctypes.data_as(C.POINTER(C.c_double)),
                          len(x_pixel_coord),
                          x_pixel_coord.ctypes.data_as(C.POINTER(c_interface.typeindex)),
                          y_pixel_coord.ctypes.data_as(C.POINTER(c_interface.typeindex)))
        else:
            clibs.ang2Pix_nonrounded(C.byref(mapinfo),
                                     ra.ctypes.data_as(C.POINTER(C.c_double)),
                                     dec.ctypes.data_as(C.POINTER(C.c_double)),
                                     len(x_pixel_coord),
                                     x_pixel_coord.ctypes.data_as(C.POINTER(C.c_double)),
                                     y_pixel_coord.ctypes.data_as(C.POINTER(C.c_double)),
                                     bin_center_zero)
    else:
        # Unpack some variables into forms that we'll use in this function.
        ra = ra_dec[0].copy()  # copy because we unwrap in place below
        elev = -ra_dec[1]  # assume geographic south pole
        ra0, dec0 = ra_dec_center
        reso_rad = reso_arcmin / 60. * DTOR

        # Wrap ra around ra0.
        # After this, all values of ra are within 180 degrees of ra0.
        # and below, (phi-phi0) will be in [-pi,+pi).
        # phi and phi0 always get subtracted --
        # they appear together as (phi-phi0) because of rotational invariance.
        # And we want to handle arbitrarily big maps centered around any ra0.
        tools.wrapAzAround(ra, ra0, modify_in_place=True)

        min_pos = -0.5 * n_pixels * reso_rad  # minimum position (left or bottom edge) in radians
        min_pos_y, min_pos_x = min_pos  # gets overwritten in BICEP projection since x and y resolutions are different
        phi, phi0 = ra * DTOR, ra0 * DTOR  # ra and ra_center in radians
        theta = (90 - elev) * DTOR  # theta polar angle in radians
        theta0 = (90 + dec0) * DTOR

        # First, get an x and y position in radians centered around zero
        # This is position relative to the map center
        if proj == 0:
            x_pos = ne.evaluate("(phi-phi0)*cos(elev*DTOR)")
            y_pos = ne.evaluate("(elev + dec0)*DTOR")
        elif proj == 1:
            x_pos = ne.evaluate("phi-phi0")
            y_pos = ne.evaluate("(elev + dec0)*DTOR")
        elif proj == 2:
            x_pos = ne.evaluate("sin(phi-phi0)*sin(theta)")
            y_pos = ne.evaluate("-sin(theta)*cos(theta0)*cos(phi-phi0) + sin(theta0)*cos(theta)")
        elif proj == 4:
            k = ne.evaluate("2/(1+cos(theta0)*cos(theta) + sin(theta0)*sin(theta)*cos(phi-phi0))")
            x_pos = ne.evaluate("k * sin(theta)*sin(phi-phi0)")
            y_pos = ne.evaluate("k * (-sin(theta)*cos(theta0)*cos(phi-phi0)+sin(theta0)*cos(theta))")
        elif proj == 5:
            k = ne.evaluate("sqrt(2/(1+cos(theta0)*cos(theta)+sin(theta0)*sin(theta)*cos(phi-phi0)))")
            x_pos = ne.evaluate("k * sin(theta)*sin(phi-phi0)")
            y_pos = ne.evaluate("k * (-sin(theta)*cos(theta0)*cos(phi-phi0)+sin(theta0)*cos(theta))")
        elif proj == 7:
            y_pos, x_pos = calculatePointingOffset([ra0, dec0], [ra, ra_dec[1]], offset_units='radians', as_azel=False)
        elif proj == 8:
            x_pos, y_pos = rotateToZeroZero(np.array([ra, ra_dec[1]]) *
                                            constants.RTOD, np.array([ra0, dec0]) * constants.RTOD)
            x_pos *= constants.DTOR
            y_pos *= constants.DTOR
        elif proj == 9:
            # BICEP default of 0.25 deg pixels, ra -55->+55, and dec -70->-45, gives a 236 by 100 pixel map.
            ny, nx = n_pixels
            # nx, ny = (236, 100) # for BICEP
            pixsize = reso_arcmin / 60.  # pixsize = 0.25 for BICEP  ->  reso_arcmin=15.
            sy = pixsize  # dec spacing in degrees
            asx = pixsize / cos(dec0 * DTOR)  # approximate spacing in ra
            # ra spacing in degrees.  round so an integer number of degrees fit in the given number of pixels.
            sx = np.round(asx * nx) / nx
            # sx  = 0.4661016949152542  # for BICEP
            # multiplying by the ratio sy/sx makes the ra pixels square-on-sky at dec0
            x_pos = ne.evaluate("(phi-phi0)*sy/sx")
            y_pos = ne.evaluate("(elev + dec0)*DTOR")
            min_pos_x = -0.5 * nx * sy * DTOR  # OVERWRITE with new, scaled position.
            # Note this is scaled by sy! It is no longer the 'minimum ra in radians', but done this way
            # so we can subtract it from x_pos and divide by reso_rad to get pixel number.
        else:
            raise ValueError("I don't know what to do with proj " + str(proj))

        # Then shift the radians position and divide by the resolution
        if round:
            # without the floor function, int conversion just chops off fractional part.
            # slightly out-of-range negative values will get incorrectly put in pixel 0,
            #  and the validity checks below will never find them.
            x_pixel_coord = np.asarray(np.floor((x_pos - min_pos_x) / reso_rad), dtype=int)
            y_pixel_coord = np.asarray(np.floor((y_pos - min_pos_y) / reso_rad), dtype=int)
        else:
            x_pixel_coord = ne.evaluate("((x_pos - min_pos_x)/reso_rad) - bin_center_zero*0.5")
            y_pixel_coord = ne.evaluate("((y_pos - min_pos_y)/reso_rad) - bin_center_zero*0.5")

    pixel_coords = (y_pixel_coord, x_pixel_coord)
    if return_validity:
        n_pix_y, n_pix_x = n_pixels
        pixel_y_is_good = ne.evaluate("(y_pixel_coord>=0) & (y_pixel_coord<n_pix_y)")
        pixel_x_is_good = ne.evaluate("(x_pixel_coord>=0) & (x_pixel_coord<n_pix_x)")

        return telescope.PointingSequence(pixel_coords, map_shape=map_pixel_shape, proj=proj,
                                          pixel_resolution_arcmin=reso_arcmin), (pixel_y_is_good, pixel_x_is_good)
    else:
        return telescope.PointingSequence(pixel_coords, map_shape=map_pixel_shape, proj=proj,
                                          pixel_resolution_arcmin=reso_arcmin)

##########################################################################################


def ang2PixV3(ra_dec, ra_dec_center, reso_arcmin, map_pixel_shape, proj=0,
              round=True, bin_center_zero=True, return_validity=True):
    """
    Never used in code, as far as grep can tell. -- Jason G 2014-06-10

    Supported projections are:

     0:  Sanson-Flamsteed projection (x = ra*cos(dec), y = dec)
     1:  CAR projection (x = ra, y = dec)
     2:  SIN projection
     3:  Healpix (not a projection at all, but pixels on the sphere) [NOT IMPLEMENTED]
     4:  stereographic projection
     5:  Oblique Lambert azimuthal equal-area projection

    INPUTS
        ra_dec : (2-element tuple of arrays) A tuple or list of arrays, [ra, dec]. Can also
            be a PointingSequence object. In degrees.

        ra_dec_center : (2-element array) The [RA, declination] of the center of the map. In degrees.

        reso_arcmin : (float) The width of each pixel in arcminutes, assumed to be the same for
            both x and y directions.

        map_pixel_shape : (2-element array) The height and width of the map, in pixels.

        proj [0]: (int or string) Which map projection should I use to turn angles on the
            curved sky into pixels on a flat map? May be an integer index, or string
            name of the projection.

        round [True]: (bool)

        bin_center_zero [True]: (bool)

    OUTPUT
        A PointingSequence object and a 2-tuple. The PointingSequence has pixel coordinates, in
        the form (y_coord, x_coord), where each element is an array of pixel indices.
        The second output tuple designates which pixel coordinates are "good", i.e., within
        bounds. It is of the form (good_ys, good_xs), where each element is
        an array of booleans.
    """
    # Convert the "proj" input to an index.
    proj = projNameToIndex(proj)

    assert type(bin_center_zero) == bool
    assert ra_dec.ndim == 2, "This won't work unless ra_dec has two dimensions!"

    # Unpack some variables into forms that we'll use in this function.
    ra, elev = ra_dec[0].copy(), -ra_dec[1].copy()
    ra0, dec0 = ra_dec_center
    n_pixels = np.asarray(map_pixel_shape, dtype=float_type)
    reso_rad = reso_arcmin / 60. * DTOR

    # Unwrap ra, and apply the same correction to ra0 as well.
    ra0 = tools.unwrapAz(ra, modify_in_place=True, extra_data=[ra0])[0]

    min_pos = -0.5 * n_pixels * reso_rad
    min_pos_y, min_pos_x = min_pos
    phi, phi0 = ra * DTOR, ra0 * DTOR
    theta = (90 - elev) * DTOR
    theta0 = (90 + dec0) * DTOR

    if proj == 0:
        x_pos = ne.evaluate("(phi-phi0)*cos((elev+dec0)*DTOR)")
        y_pos = ne.evaluate("(elev + dec0)*DTOR")
    elif proj == 1:
        x_pos = ne.evaluate("phi-phi0")
        y_pos = ne.evaluate("(elev + dec0)*DTOR")
    elif proj == 2:
        x_pos = ne.evaluate("sin(phi-phi0)*sin(theta)")
        y_pos = ne.evaluate("-sin(theta)*cos(theta0)*cos(phi-phi0) + sin(theta0)*cos(theta)")
    elif proj == 4:
        k = ne.evaluate("2/(1+cos(theta0)*cos(theta) + sin(theta0)*sin(theta)*cos(phi-phi0))")
        x_pos = ne.evaluate("k * (sin(theta)*sin(phi-phi0))")
        y_pos = ne.evaluate("-k * (sin(theta)*cos(theta0)*cos(phi-phi0)+sin(theta0)*cos(theta))")
    elif proj == 5:
        k = ne.evaluate("sqrt(2/(1+cos(theta0)*cos(theta)+sin(theta0)*sin(theta)*cos(phi-phi0)))")
        #x_pos = ne.evaluate("k * (sin(theta)*sin(phi-phi0))")
        #y_pos = ne.evaluate("-k * (sin(theta)*cos(theta0)*cos(phi-phi0)+sin(theta0)*cos(theta))")

        x_pos = ne.evaluate("((k * sin(theta)*sin(phi-phi0)) - min_pos_x) / reso_rad")
        y_pos = ne.evaluate(
            "((-k * sin(theta)*cos(theta0)*cos(phi-phi0)+sin(theta0)*cos(theta)) - min_pos_y) / reso_rad")

        #x_pos = np.empty_like(theta)
        #y_pos = np.empty_like(theta)
        c_code = r"""
        #line 468 "sky.py"

        double cos_theta0 = cos(theta0);
        double sin_theta0 = sin(theta0);
        double k, sin_theta, cos_theta, cos_phi, ksin_theta;
        double cphi0 = double(phi0);
        for (int i=0; i<Ntheta[0]; ++i) {
            sin_theta = sin(theta[i]);
            cos_theta = cos(theta[i]);
            cos_phi = cos(phi[i]-cphi0);
            k = sqrt(2./(1. + cos_theta0*cos_theta + sin_theta0*sin_theta*cos_phi));
            ksin_theta = k*sin_theta;
            x_pos[i] = ksin_theta*sin(phi[i]-cphi0);
            y_pos[i] = -ksin_theta*cos_theta0*cos_phi + sin_theta0*cos_theta;
        }
        """
        #weave.inline(c_code, ['theta', 'theta0', 'phi', 'phi0', 'x_pos', 'y_pos'])
    else:
        raise ValueError("I don't know what to do with proj " + str(proj))

    if round:
        x_pixel_coord = np.asarray((x_pos - min_pos[1]) / reso_rad, dtype=int)
        y_pixel_coord = np.asarray((y_pos - min_pos[0]) / reso_rad, dtype=int)
    elif bin_center_zero:
        #x_pixel_coord = ne.evaluate("((x_pos - min_pos_x)/reso_rad) - bin_center_zero*0.5")
        #y_pixel_coord = ne.evaluate("((y_pos - min_pos_y)/reso_rad) - bin_center_zero*0.5")
        x_pixel_coord = ne.evaluate("x_pos - 0.5")
        y_pixel_coord = ne.evaluate("y_pos - 0.5")

    pixel_coords = (y_pixel_coord, x_pixel_coord)
    if return_validity:
        n_pix_y, n_pix_x = n_pixels
        pixel_y_is_good = ne.evaluate("(y_pixel_coord>=0) & (y_pixel_coord<n_pix_y)")
        pixel_x_is_good = ne.evaluate("(x_pixel_coord>=0) & (x_pixel_coord<n_pix_x)")

        return telescope.PointingSequence(pixel_coords, map_shape=map_pixel_shape, proj=proj,
                                          pixel_resolution_arcmin=reso_arcmin), (pixel_y_is_good, pixel_x_is_good)
    else:
        return telescope.PointingSequence(pixel_coords, map_shape=map_pixel_shape, proj=proj,
                                          pixel_resolution_arcmin=reso_arcmin)
##########################################################################################


def testPix2AngVsIDL(test_proj, tolerance_arcsec=0.1, test_pix=None, test_shape=None,
                     test_map_center=None, test_resoarcmin=0.25, verbose=False):
    """
    Compare the output of pix2Ang with the equivalent IDL procedures. Raise an exception if
    the outputs are too different from each other.

    INPUTS
        test_proj: (int or str) Either the integer index of the projection to test, or its string designator.

        tolerance_arcsec [0.1]: (float) In arcseconds. The difference between the
            RA/dec from pix2Ang and the RA/dec from IDL's pix2ang must be less
            than this in both RA and dec.

        test_pix [None]: (2-element tuple of arrays) The input pixel coordinates.
            A tuple or list of arrays, [y_coord, x_coord].
            Note the order! The first element is the "y", the mostly-dec coordinate, and the
            second element is the "x", the mostly-RA coordinate.
            Defaults to np.array([[523.], [1723.]]).

        test_shape [None]: (2-element array) The map shape, in pixels_y, pixels_x.
            Defaults to np.array([3600, 3600]).

        test_map_center [None]: (2-element array) The [RA, declination] of the center of the map.

        test_resoarcmin [0.25]: (float) The width of each pixel in arcminutes, assumed to be the same for
            both x and y directions.

        verbose [False]: (bool) If True, output details of the test, regardless of the result.

    OUTPUT
        True if successful. Raises a RuntimeError if failed.

    EXCEPTIONS
        RuntimeError if the error in either direction is greater than the input tolerance.
    """
    from sptpol_software.util import idl

    # Set defaults if inputs weren't provided.
    if test_pix is None:
        test_pix = np.array([[523.], [1723.]])
    if test_shape is None:
        test_shape = np.array([3600, 3600])
    if test_map_center is None:
        test_map_center = np.array([352.5,  -55.])

    # Find IDL's RA and dec.
    idlout = idl.idlRun("pix2ang_proj%d, npixels, radec0, reso_arcmin, ra, dec, xpix=xpix, ypix=ypix" % test_proj, return_all=True,
                        xpix=test_pix[1], ypix=test_pix[0], npixels=test_shape,
                        radec0=test_map_center, reso_arcmin=test_resoarcmin)

    # Now run the Python code.
    radec = pix2Ang(test_pix, test_map_center, test_resoarcmin, test_shape, proj=test_proj)

    # Compare the answers, and complain if they're too different.
    ra_error = radec[0] - idlout['ra']
    dec_error = radec[1] - idlout['dec']
    if (np.abs(ra_error * 3600) > tolerance_arcsec).any() or (np.abs(dec_error * 3600) > tolerance_arcsec).any():
        raise RuntimeError("proj %d: max ra_error = %f arcseconds, max dec_error = %f arcseconds" %
                           (test_proj, ra_error.max() * 3600, dec_error.max() * 3600))
    elif verbose:
        print "Success! proj %d: max ra_error = %f arcseconds, max dec_error = %f arcseconds" % (test_proj, ra_error.max() * 3600, dec_error.max() * 3600)

    return True
##########################################################################################


def testAng2PixVsIDL(test_proj, tolerance_arcsec=0.1, test_radec=None, test_shape=None,
                     test_map_center=None, test_resoarcmin=0.25, verbose=False):
    """
    Compare the output of ang2Pix with the equivalent IDL procedures. Raise an exception if
    the outputs are too different from each other.

    INPUTS
        test_proj: (int or str) Either the integer index of the projection to test, or its string designator.

        tolerance_arcsec [0.1]: (float) In arcseconds. The difference between the
            pixel coordinate from ang2Pix and the RA/dec from IDL's ang2pix must be less
            than this in both x and y.
            (The difference in pixel coordinates will be transformed to arcseconds using
            the test_resoarcmin.)

        test_radec [None] : (2-element tuple of arrays) A tuple or list of arrays,
            [ra, dec]. Can also be a PointingSequence object. In degrees.
            Defaults to np.array([[352.00738525], [-49.68125153]]) if not provided.

        test_shape [None]: (2-element array) The map shape, in pixels_y, pixels_x.
            Defaults to np.array([3600, 3600]).

        test_map_center [None]: (2-element array) The [RA, declination] of the center of the map.

        test_resoarcmin [0.25]: (float) The width of each pixel in arcminutes, assumed to be the same for
            both x and y directions.

        verbose [False]: (bool) If True, output details of the test, regardless of the result.

    OUTPUT
        True if successful. Raises a RuntimeError if failed.

    EXCEPTIONS
        RuntimeError if the error in either direction is greater than the input tolerance.
    """
    from sptpol_software.util import idl

    # Set defaults if inputs weren't provided.
    if test_radec is None:
        test_radec = np.array([[352.00738525], [-49.68125153]])
    if test_shape is None:
        test_shape = np.array([3600, 3600])
    if test_map_center is None:
        test_map_center = np.array([352.5,  -55.])

    # Find IDL's pixel coordinates. Note that the IDL ang2pix_proj1 doesn't take the "/noround"
    # argument, so don't use it there. The output of that function is still unrounded pixel coordinates.
    idlout = idl.idlRun("ang2pix_proj%d, ra, dec, npixels, radec0, reso_arcmin, ipix, xpix=xpix, ypix=ypix%s" % (test_proj, ', /noround' if test_proj != 1 else ''),
                        return_all=True,
                        ra=test_radec[0], dec=test_radec[1], npixels=test_shape,
                        radec0=test_map_center, reso_arcmin=test_resoarcmin)

    # Now run the Python code.
    pixout = ang2Pix(test_radec, test_map_center, test_resoarcmin, test_shape,
                     proj=test_proj, round=False, return_validity=False)

    # Compare the answers and complain if they're too different.
    x_error = pixout[1] - idlout['xpix']
    y_error = pixout[0] - idlout['ypix']

    if (np.abs(x_error * test_resoarcmin * 60) > tolerance_arcsec).any() or (np.abs(y_error * test_resoarcmin * 60) > tolerance_arcsec).any():
        raise RuntimeError("proj %d: max x_error = %f pixels, %f arcseconds, max y_error = %f pixels, %f arcseconds" %
                           (test_proj, x_error.max(), x_error.max() * test_resoarcmin * 60, y_error.max(), y_error.max() * test_resoarcmin * 60))
    elif verbose:
        print("Success! proj %d: max x_error = %f pixels, %f arcseconds, max y_error = %f pixels, %f arcseconds" %
              (test_proj, x_error.max(), x_error.max() * test_resoarcmin * 60, y_error.max(), y_error.max() * test_resoarcmin * 60))

    return True
##########################################################################################


def testCircularPixelMappings(test_proj, tolerance_arcsec=0.1, test_pix=None, test_shape=None,
                              test_map_center=None, test_resoarcmin=0.25, verbose=False):
    """
    At a bare minimum, when the RA and dec calculated from pix2Ang are provided to ang2Pix,
    the result should be the same as the input pixel coordinate, up to float precision errors.
    This function tests that.

    INPUTS
        test_proj: (int or str) Either the integer index of the projection to test, or its string designator.

        tolerance_arcsec [0.1]: (float) The difference between the input pixel and the
            pixel output by ang2Pix must be less than this in both x and y.
            (The difference in pixel coordinates will be transformed to arcseconds using
            the test_resoarcmin.)

        test_pix [None]: (2-element tuple of arrays) The input pixel coordinates.
            A tuple or list of arrays, [y_coord, x_coord].
            Note the order! The first element is the "y", the mostly-dec coordinate, and the
            second element is the "x", the mostly-RA coordinate.
            Defaults to np.array([[523.], [1723.]]).

        test_shape [None]: (2-element array) The map shape, in pixels_y, pixels_x.
            Defaults to np.array([3600, 3600]).

        test_map_center [None]: (2-element array) The [RA, declination] of the center of the map.

        test_resoarcmin [0.25]: (float) The width of each pixel in arcminutes, assumed to be the same for
            both x and y directions.

        verbose [False]: (bool) If True, output details of the test, regardless of the result.

    OUTPUT
        True if successful. Raises a RuntimeError if failed.

    EXCEPTIONS
        RuntimeError if the error in either direction is greater than the input tolerance.
    """
    # Set defaults if inputs weren't provided.
    if test_pix is None:
        test_pix = np.array([[523.], [1723.]])
    if test_shape is None:
        test_shape = np.array([3600, 3600])
    if test_map_center is None:
        test_map_center = np.array([352.5,  -55.])

    # Run pix2Ang.
    radec = pix2Ang(test_pix, test_map_center, test_resoarcmin, test_shape, proj=test_proj)

    # Put the results ot pix2Ang into ang2Pix.
    yout, xout = ang2Pix(radec, test_map_center, test_resoarcmin, test_shape,
                         proj=test_proj, round=False, return_validity=False)

    # Compute the difference between the input to pix2Ang and the output of ang2Pix.
    if len(yout) == 1:
        yout = yout[0]
    if len(xout) == 1:
        xout = xout[0]
    error = (test_pix - np.array([yout, xout]))
    error_arcsec = error * test_resoarcmin * 60

    # Check for too-large errors.
    if (np.abs(error_arcsec) > tolerance_arcsec).any():
        raise RuntimeError("proj %d: max x_error = %f pixels, %f arcseconds, max y_error = %f pixels, %f arcseconds" %
                           (test_proj, error[1].max(), error_arcsec[1].max(), error[0].max(), error_arcsec[0].max()))
    elif verbose:
        print("Success! proj %d: max x_error = %f pixels, %f arcseconds, max y_error = %f pixels, %f arcseconds" %
              (test_proj, error[1].max(), error_arcsec[1].max(), error[0].max(), error_arcsec[0].max()))

    return True
##########################################################################################


def testAllProj(run_idl_tests=False, verbose=False, quiet=False):
    """
    Test pix2Ang and ang2Pix for each projection type. There are two kinds of tests:
    first, check that we can go in a loop from pix2Ang back through ang2Pix and end up where
    we started. Next, check the outputs of pix2Ang and ang2Pix against the
    equivalent IDL procedures.

    This function uses lists of projection indices defined at the top of the module to know
    which projections to test.

    INPUTS
        run_idl_tests [False]: (bool) If True, check IDL outputs versus Python outputs.
            If False, we'll only run the circularity check.

        verbose [False]: (bool) Output details from every single test?

        quiet [False]: (bool) Progress indicator and overall report. If True, you'll get
            no screen output.

    OUTPUT
        True if all tests passed. False if at least one test failed.

    EXCEPTIONS
        None
    """
    # Keep track of how many times we pass or fail tests.
    passcount = 0
    failcount = 0

    # Run the circularity tests.
    if not quiet:
        print "Testing pix2Ang -> ang2Pix loop."
    for proj in proj_to_test:
        try:
            testCircularPixelMappings(proj, verbose=verbose)
            print "...proj%d passed!" % proj
            passcount += 1
        except RuntimeError, err:
            print "...proj%d FAILED!: %s" % (proj, str(err))
            failcount += 1

    # If requested, check Python code versus IDL.
    if run_idl_tests:
        # First check pix2Ang
        if not quiet:
            print "Comparing Python pix2Ang with IDL pix2ang."
        for proj in proj_index_in_idl:
            try:
                testPix2AngVsIDL(proj, verbose=verbose)
                print "...proj%d passed!" % proj
                passcount += 1
            except RuntimeError, err:
                print "...proj%d FAILED!: %s" % (proj, str(err))
                failcount += 1

        # Now check ang2Pix.
        if not quiet:
            print "Comparing Python ang2Pix with IDL ang2Pix."
        for proj in proj_index_in_idl:
            try:
                testAng2PixVsIDL(proj, verbose=verbose)
                print "...proj%d passed!" % proj
                passcount += 1
            except RuntimeError, err:
                print "...proj%d FAILED!: %s" % (proj, str(err))
                failcount += 1

    # Print results to screen and exit.
    if not quiet:
        print "Finished testing Python pix2Ang and ang2Pix functions."
        print "   Passed %d tests." % passcount
        print "   Failed %d tests." % failcount

    return (failcount == 0)
##########################################################################################


def testPointingOffsetVsIDL(boresight_pointing, detector_pointing, as_azel=False, tolerance_arcsec=0.1, verbose=True):
    """
    Compare outputs of calculatePointingOffset with the equivalent IDL procedure, rotate_boresight_to_zero_zero.pro.

    INPUTS
        boresight_pointing: (2-element iterable) Telescope boresight pointing, (RA, dec), in degrees.
            The RA and dec may be arrays.

        detector_pointing: (2-element iterable) A different pointing, here named under the assumption
            that it's the pointing of an individual detector element, (RA, dec), in degrees.
            The RA and dec may be arrays.

        as_azel [False]: (bool) If True, assume the input boresight is [boresight_el, boresight_az],
            and the input offset pointing is [shifted_el, shifted_az].
            Output is [azimuth_offset, elevation_offset].

        tolerance_arcsec [0.1]: (float) The difference between the input pixel and the
            pixel output by ang2Pix must be less than this in both x and y.
            (The difference in pixel coordinates will be transformed to arcseconds using
            the test_resoarcmin.)

        verbose [False]: (bool) If True, output details of the test, regardless of the result.

    OUTPUT
        True if successful. Raises a RuntimeError if failed.

    EXCEPTIONS
        RuntimeError if the error in either direction is greater than the input tolerance.
    """
    from sptpol_software.util import idl

    # First get the Python version.
    python_offset = calculatePointingOffset(boresight_pointing, detector_pointing,
                                            offset_units='degrees', as_azel=as_azel)

    ###################################
    # That was easy, no? Now try IDL.

    # Unpack the inputs.
    if as_azel:
        # "Az/el" means that things are ordered like [elevation, azimuth].
        # bs = boresight, det = detector
        bs_ra, bs_dec = boresight_pointing[1], boresight_pointing[0]
        det_ra, det_dec = detector_pointing[1], detector_pointing[0]
    else:
        # Else it's [RA, declination] ordering.
        # bs = boresight, det = detector
        bs_ra, bs_dec = boresight_pointing[0], boresight_pointing[1]
        det_ra, det_dec = detector_pointing[0], detector_pointing[1]

    idlout = idl.idlRun("rotate_boresight_to_zero_zero, bs_ra, bs_dec, det_ra, det_dec, offset_az, offset_el",
                        return_all=True, bs_ra=bs_ra, bs_dec=bs_dec, det_ra=det_ra, det_dec=det_dec)

    idl_offset = np.array([-idlout['offset_el'], idlout['offset_az']])
    offset_error = python_offset.squeeze() - idl_offset.squeeze()

    if (np.abs(offset_error) * 3600 > tolerance_arcsec).any():
        raise RuntimeError("boresight = [%s,%s], ra/dec = [%s,%s], python offset = %s, IDL offset = %s, error = %s arcseconds" %
                           (str(bs_ra), str(bs_dec), str(det_ra), str(det_dec), str(python_offset), str(idl_offset), str(offset_error * 3600)))
    elif verbose:
        print "Success! Difference between offsets is %s arcseconds." % str(offset_error * 3600)

    return True

##########################################################################################


def applyPointingOffset(boresight_pointing, offset, offset_units='arcmin', as_azel=False):
    """
    Apply a detector pointing offset to a telescope boresight pointing. This function assumes the input boresight pointing
    is in degrees, and the input offset is either degrees or arcminutes (specified by the "offset_units" argument).
    The offset should be in degrees on the sky, with the x-offset measured on a great circle passing horizontally
    through the center of the receiver's field of view, and the y-offset measured on a vertical great circle.

    Adapted from the IDL spt_analysis/pointing/apply_pixel_offset_scan.pro.

    INPUTS
        boresight_pointing: Assumed to be 2-element iterable of iterables, ordered as [boresight_ra, boresight_dec].
            Assumed to be in degrees.

        offset: (2-element iterable) The pointing offset of this pixel, in [offset_ra, offset_dec].

        offset_units ['arcmin']: (string) Either 'arcmin', 'degrees', or 'radians'.

        as_azel [False]: (bool) If True, assume the input boresight is [boresight_el, boresight_az],
            and the input offset is [azimuth_offset, elevation_offset]. Output [shifted_el, shifted_az].

    OUTPUT
        An ndarray with the [shifted_ra, shifted_dec] in degrees.
    """
    # Unpack the inputs.
    if as_azel:
        # "Az/el" means that things are ordered like [elevation, azimuth].
        boresight_ra, boresight_dec = boresight_pointing[1] * constants.DTOR, boresight_pointing[0] * constants.DTOR
        shift_ra, shift_dec = offset[1] * constants.DTOR, offset[0] * constants.DTOR
    else:
        # Else it's [RA, declination] ordering.

        # Multiply the declination offset by -1: Assume we're in the southern hemisphere
        # (negative dec), so the receiver is "upside down". Positive elevation offset is
        # negative declination offset, and vice-versa.
        boresight_ra, boresight_dec = boresight_pointing[0] * constants.DTOR, boresight_pointing[1] * constants.DTOR
        shift_ra, shift_dec = offset[0] * constants.DTOR, -offset[1] * constants.DTOR

    RTOD = constants.RTOD
    if offset_units is not None:
        if offset_units.lower().startswith('arcmin'):
            shift_ra /= 60.
            shift_dec /= 60.
        elif offset_units.lower().startswith('rad'):
            RTOD = 1.
        elif not offset_units.lower().startswith("deg"):
            raise ValueError("I don't recognize offset_units=%s." % offset_units)

    shifted_dec = ne.evaluate(
        'RTOD*arcsin( sin(boresight_dec)*cos(shift_dec)*cos(shift_ra) + cos(boresight_dec)*sin(shift_dec))')
    shifted_ra = ne.evaluate(('-RTOD*arctan2( ( -sin(boresight_ra)*cos(boresight_dec)*cos(shift_ra) - cos(boresight_ra)*sin(shift_ra) + sin(boresight_ra)*sin(boresight_dec)*tan(shift_dec)),'
                              + '(cos(boresight_ra)*cos(boresight_dec)*cos(shift_ra) - sin(boresight_ra)*sin(shift_ra) - cos(boresight_ra)*sin(boresight_dec)*tan(shift_dec)) )'))

    if as_azel:
        return np.array([shifted_dec, shifted_ra])
    else:
        return np.array([shifted_ra, shifted_dec])

##########################################################################################


def calculatePointingOffset(boresight_pointing, detector_pointing, offset_units='arcmin', as_azel=False):
    """
    Starting from a telescope boresight pointing and an offset pointing, calculate the offset
    angles which transform between them. The offset angles will be given in terms of
    degrees on the sky, and are related in the following way, where R_i(x) is a rotation about
    axis i by x degrees, "bs" refers to boresight pointings, and "det" are detector (offset) pointings:

    R_z(-RA_bs)*R_y(dec_bs)*R_z(-X_off)*R_y(Y_off) * (1,0,0) = R_z(-RA_det)*R_y(dec_det) * (1,0,0)

    INPUTS
        boresight_pointing: (2-element iterable) Telescope boresight pointing, (RA, dec), in degrees.
            The RA and dec may be arrays.

        detector_pointing: (2-element iterable) A different pointing, here named under the assumption
            that it's the pointing of an individual detector element, (RA, dec), in degrees.
            The RA and dec may be arrays.

        offset_units ['arcmin']: (string) Sets units in which the offset will be returned.
            Either 'arcmin' or 'degrees'.

        as_azel [False]: (bool) If True, assume the input boresight is [boresight_el, boresight_az],
            and the input offset pointing is [shifted_el, shifted_az].
            Output is [azimuth_offset, elevation_offset].

    OUTPUT
        The angular difference between the boresight pointing and detector pointing.
        An ndarray with the [shift_y, shift_x] in degrees or arcminutes (units depend on offset_units input).
        The offsets output by this function can be used with sources.applyPointingOffset
        to recover the detector pointing, given a boresight pointing.
    """
    # Unpack the inputs.
    if as_azel:
        # "Az/el" means that things are ordered like [elevation, azimuth].
        # bs = boresight, det = detector
        bs_ra, bs_dec = boresight_pointing[1] * constants.DTOR, boresight_pointing[0] * constants.DTOR
        det_ra, det_dec = detector_pointing[1] * constants.DTOR, detector_pointing[0] * constants.DTOR
    else:
        # Else it's [RA, declination] ordering.
        # bs = boresight, det = detector
        bs_ra, bs_dec = boresight_pointing[0] * constants.DTOR, boresight_pointing[1] * constants.DTOR
        det_ra, det_dec = detector_pointing[0] * constants.DTOR, detector_pointing[1] * constants.DTOR

    RTOD = constants.RTOD
    shift_x = ne.evaluate(
        'RTOD*arctan2(cos(det_dec)*sin(det_ra-bs_ra), cos(bs_dec)*cos(det_dec)*cos(det_ra-bs_ra) + sin(bs_dec)*sin(det_dec))')
    shift_y = ne.evaluate('RTOD*arcsin(-sin(bs_dec)*cos(det_dec)*cos(det_ra-bs_ra) + cos(bs_dec)*sin(det_dec))')

    if offset_units is not None:
        if offset_units.lower().startswith('arcmin'):
            shift_x *= 60.
            shift_y *= 60.
        elif offset_units.lower().startswith('rad'):
            shift_x *= constants.DTOR
            shift_y *= constants.DTOR
        elif not offset_units.lower().startswith("deg"):
            raise ValueError("I don't recognize offset_units=%s." % offset_units)

    return np.array([-shift_y, shift_x])


def rotateToZeroZero(ra_dec, ra_dec_center, input_units='degrees'):
    """
    Not fully vetted. Take caution!

    INPUTS
        ra_dec : A two-element iterable, each element of which is a list of coordinates.

        ra_dec_center : A two-element iterable.

    OUTPUT
        The [ra, dec] coordinates corresponding to the input ra_dec list once it has been
        rotated such that the ra_dec_center is at [0,0].
    """

    ra_dec = np.array(ra_dec) * constants.DTOR
    ra_dec_center = np.array(ra_dec_center) * constants.DTOR

    ra_rotation = np.matrix([[cos(-ra_dec_center[0]), -sin(-ra_dec_center[0]), 0],
                             [sin(-ra_dec_center[0]), cos(-ra_dec_center[0]), 0],
                             [0, 0, 1]])
    dec_rotation = np.matrix([[cos(-ra_dec_center[1]), 0, -sin(-ra_dec_center[1])],
                              [0, 1, 0],
                              [sin(-ra_dec_center[1]), 0, cos(-ra_dec_center[1])]])

    sph_coords = [cos(ra_dec[0]) * cos(ra_dec[1]), cos(ra_dec[0]) * sin(ra_dec[1]), sin(ra_dec[0])]
    sph_coords = map(np.matrix, zip(sph_coords[0], sph_coords[1], sph_coords[2]))

    sph_coords_out = np.array(map(lambda x: dec_rotation * ra_rotation * x.T, sph_coords))
    dec_out = np.arcsin(sph_coords_out[:, 2, 0])
    ra_out = np.arcsin(sph_coords_out[:, 1, 0] / np.cos(dec_out))
    ra_dec_out = [ra_out * constants.RTOD, dec_out * constants.RTOD]
    return ra_dec_out

##########################################################################################


def pointingToPixels(radec_ptg, receiver, mapinfo=None, radec_center=None,
                     reso_arcmin=None, map_shape=None, proj=None, good_bolos=None):
    """
    Converts a boresight RA, dec pointing into map pixel for each channel.

    INPUTS
        radec_ptg: (2-element iterable of iterables) Stored as [ptg_ra, ptg_dec], in degrees.

        receiver: (Receiver or SPTDataReader object) A Receiver with the pointing offsets of each detector.
            May also be an SPTDataReader object, in which case we'll take the Receiver from there.

        mapinfo [None]: An object with the attributes "map_center", "reso_arcmin", "map_shape_pix", and "proj",
            each as defined for the arguments of the similar name in this function.
            (Like the c_interface.MAPINFO class.) Either mapinfo
            must be input, or all four of those arguments must be provided.
            If mapinfo is provided, it overrides the other arguments.

        radec_center: (2-element iterable) The center of the map in degrees [RA, dec].

        reso_arcmin: (float) Arcminutes per side of a map pixel.

        map_shape: (2-element iterable of ints) The map shape in pixels [pixels_dec, pixels_ra] NOTE ORDER.

        proj: (int) Map projection type.

        good_bolos [None]: (iterable of detector IDs) A list of detector IDs with each channel you want pointing for.

    OUTPUT
        A dictionary of PointingSequence objects, one per channel, with the pixel pointing at each timestep.
    """
    # Check if the input "receiver" is actually something which holds a Receiver.
    try:
        receiver.rec.devices.bolometers
        receiver = receiver.rec
    except AttributeError:
        pass

    # Find the map parameter inputs.
    if mapinfo is not None:
        radec_center = mapinfo.map_center
        reso_arcmin = mapinfo.reso_arcmin
        map_shape = mapinfo.map_shape_pix
        proj = mapinfo.proj
    elif (radec_center is None) or (reso_arcmin is None) or (map_shape is None) or (proj is None):
        raise ValueError("You must provide all four map parameters in some form!")

    if good_bolos is None:
        good_bolos = receiver.devices.bolometers.keys()

    pix_pointing = struct()
    for bolo_id in good_bolos:
        # Must flip the pointing offsets around to get everything straight. Boresight is [ra, dec],
        # we want output in [ra, dec], but pointing offsets are stored in [el, az].
        shifted_pointing = applyPointingOffset(radec_ptg, offset=receiver[bolo_id].pointing_offset[::-1],
                                               offset_units=receiver[bolo_id].offset_units, as_azel=False)
        # Adjust RAs to lie in [0,360] so that ang2Pix doesn't get confused.
        shifted_pointing[0][shifted_pointing[0] < 0] += 360
        pix_pointing[bolo_id] = ang2Pix(shifted_pointing, radec_center, reso_arcmin,
                                        map_shape, round=True, proj=proj, return_validity=False)

    return pix_pointing

##########################################################################################


def cPointingToPixels(radec_ptg, data, good_bolos=None, mapinfo=None, map_parameters={},
                      flattened_pointing=False, in_scan_only=True,
                      set_scan_flags=True, set_scan_bolometer_flags=True,
                      quiet=False, verbose=True):
    """
    Converts a boresight RA, dec pointing into map pixel for each channel. This function uses compiled
    C code from the sptpol_cdevel package.

    INPUTS
        radec_ptg: (2-element iterable of iterables) Stored as [ptg_ra, ptg_dec], in degrees.
            If None, then we'll take RA and dec straight from the SPTDataReader object.

        data: (SPTDataReader or MapIDF object) The data.

        good_bolos [None]: (iterable of detector IDs) A list of detector IDs with each channel you want pointing for.

        mapinfo: (c_interface.MAPINFO object) Contains map parameters. If None, a new one will be
            initialized from the map_parameters.

        map_parameters [{}]: (dict) Used only if mapinfo is None.
           Defaults to a ra23h30dec-55 field with 0.2 arcminute pixels. Valid arguments are
          "proj", "map_center", "map_shape", and "reso_arcmin". (See data.c_interface.MAPINFO.)

        flattened_pointing [False]: (bool) If True, the output will be 1D indices into a flattened map array.

        in_scan_only [True]: (bool) If True, we'll only calculate pointing inside marked scans.
            If False, we'll get pointing info for the entire timestream.

       set_scan_flags [True]: (bool) If True, then any "this scan is bad" flags set on a scan will
            cause pointing for that scan not to be calculated. Uses data.cuts.scan_bad_flags to determine which
            flags denote bad scans. If False, we will find pointing for all scans regardless of flag status.

       set_scan_bolometer_flags [True]: (bool) If True, then any "this channel is bad in this scan" flags
            set on a scan will cause that channel to not get pointing data for that scan.
            If False, we will find pointing for all channels in each scan regardless of flag status.

       verbose [False]: Extra output to screen.

        quiet [False]: Suppress screen output.

    OUTPUT
        A dictionary of telescope.PointingSequence objects, or, if flattened_pointing is True, a dictionary
        of 1D arrays with indices into a flattened map array.
    """
    #from sptpol_software.observation.telescope import PointingSequence
    C = c_interface.C
    utcnow = time.utcnow

    start_time = utcnow()

    # Get a list of timestreams to find pointing for.
    # if verbose: print "Finding good bolometers."
    #good_bolos = getGoodBolos(data, good_bolos=good_bolo_method, verbose=verbose)
    # if verbose: print "I found "+str(len(good_bolos))+" good bolos."
    if good_bolos is None:
        good_bolos = data.rec.devices.bolometers.keys()

    if not quiet:
        print "Beginning pointing calculations with the C pipeline."

    # Grab a reference to the C code.
    clibs = c_interface.initializeCLibs()

    # Convert the input data object into something that can be passed to a C function.
    # Control the flags (which will be read by the C filtering function to determine what
    # gets filtered) with the extra arguments to the SPT_DATA init.
    cdata = c_interface.SPT_DATA(data, channels_to_use=good_bolos,
                                 set_scan_flags=set_scan_flags,
                                 set_scan_bolometer_flags=set_scan_bolometer_flags)

    # If we got a new input for the RA & dec, put that in the data object now.
    if radec_ptg is not None:
        ra, dec = radec_ptg
        if ra.dtype != np.float64 or dec.dtype != np.float64:
            raise ValueError("The RA and/or dec have the wrong float precision! I expect float64s.")
        cdata.antenna0_ra_c[0] = C.c_size_t(len(ra))
        cdata.antenna0_ra = ra.ctypes.data_as(C.POINTER(C.c_double))
        cdata.antenna0_dec_c[0] = C.c_size_t(len(dec))
        cdata.antenna0_dec = dec.ctypes.data_as(C.POINTER(C.c_double))

    # Set up arrays to store the output from the C code. This is sensitive - you /must/ have
    # the correct array sizes and data types here! Getting something wrong will at best
    # generate segmentation faults. (The C code will error check length via pix_pointing_c.
    # You still need to get it right.)
    n_good_bolos = len(np.where((cdata.bolometer_flags_array | cdata.timestream_flags_array) == 0)[0])
    npts_total = cdata.antenna0_ra_c[0]
    len_pointing_array = n_good_bolos * npts_total
    pix_pointing_x = np.ascontiguousarray(np.zeros(len_pointing_array, dtype=np.uintc))
    pix_pointing_y = np.ascontiguousarray(np.zeros(len_pointing_array, dtype=np.uintc))
    pix_pointing_c = C.c_size_t(len_pointing_array)
    i_bolo_used = np.ascontiguousarray(np.zeros(cdata.observation_bolo_x_deg_nom_c[0], dtype=np.intc)) - 1

    # If we didn't get map parameters input, generate them now.
    if mapinfo is None:
        mapinfo = c_interface.MAPINFO(**map_parameters)

    setup_finished = utcnow()
    if verbose:
        print "Took %.2f seconds to be ready to call the C-code." % (utcnow() - start_time).total_seconds()

    # Call the C code!
    status = clibs.calculate_detector_pointing(C.byref(cdata), C.byref(mapinfo), in_scan_only,
                                               flattened_pointing, -1,
                                               i_bolo_used.ctypes.data_as(C.POINTER(C.c_int)),
                                               pix_pointing_x.ctypes.data_as(C.POINTER(c_interface.typeindex)),
                                               pix_pointing_y.ctypes.data_as(C.POINTER(c_interface.typeindex)),
                                               C.byref(pix_pointing_c),
                                               verbose)

    if status:
        raise RuntimeError("C pointing code exited with status %d!" % status)

    if not quiet:
        print "C pointing completed. %.2f seconds to run the C code." % (utcnow() - setup_finished).total_seconds()

    # Convert the single array output into a struct with one array or PointingSequence for each detector.
    det_ptg = struct()
    for i_bolo_c, i_bolo_data in enumerate(i_bolo_used):
        if i_bolo_data < 0:
            # A value of -1 in this array means that the detector wasn't used.
            continue
        bolo_id = data.observation.bolo_id_ordered[i_bolo_data]
        if flattened_pointing:
            det_ptg[bolo_id] = pix_pointing_x[i_bolo_c * npts_total:(i_bolo_c + 1) * npts_total]
        else:
            det_ptg[bolo_id] = telescope.PointingSequence([pix_pointing_y[i_bolo_c * npts_total:(i_bolo_c + 1) * npts_total].astype(int),
                                                           pix_pointing_x[i_bolo_c * npts_total:(i_bolo_c + 1) * npts_total].astype(int)],
                                                          proj=mapinfo.proj,
                                                          map_shape=np.array([mapinfo.n2, mapinfo.n1]),
                                                          pixel_resolution_arcmin=mapinfo.reso_arcmin)

    if not quiet:
        print "Successfully finished pointing calculations. %.2f seconds elapsed." % (utcnow() - start_time).total_seconds()

    return det_ptg

##########################################################################################


def gaussianSmooth(mapin, gaussian_fwhm, in_place=False, submaps_to_use=None):
    """
    This function takes either a Map or Polarized map, smooths it by a Gaussian kernel with
    supplied full-width-half-max and returns a new, smoothed map. Uses the
    scipy.ndimage.gaussian_filter function for all of the hard work.
    (This is a wrapper for applyFunctionToMap.)

    INPUTS
        mapin: (Map or PolarizedMap) The map to smooth.

        gaussian_fwhm: (float) In arcminutes, the full-width-half-max of the Gaussian
            smoothing kernel.

        in_place [False]: (bool) If True, modify the input map in place.

        submaps_to_use [None]: (list of strings) Used for PolarizedMap input only.
            If None, then smooth all maps in the input PolarizedMap. Otherwise,
            smooth only the maps in this list. Delete maps not in this list.

    OUTPUT
        A Map or PolarizedMap, corresponding to the input type.

    EXCEPTIONS
        ValueError if the "mapin" input is not a valid type.

    AUTHOR
        Stephen Hoover, 30 September 2013

    CHANGES
    """
    # First calculate the size of the Gaussian kernel in pixels of the map.
    pixels_per_unit = 1. / mapin.reso_arcmin  # Pixels per arcminute.
    # Convert FWHM (arcminutes) to sigma (pixels).
    gaussian_sigma = gaussian_fwhm * pixels_per_unit / (2 * np.sqrt(2 * np.log(2)))

    # This will apply ndimage.gaussian_filter(<map array>, gaussian_sigma) to
    # each map-like array (maps, submaps, weights) in the input "mapin".
    return applyFunctionToMap(ndimage.gaussian_filter, mapin, gaussian_sigma,
                              in_place=in_place, submaps_to_use=submaps_to_use)

##########################################################################################


def convolveMap(mapin, convolution_kernel, in_place=False, submaps_to_use=None):
    """
    This function takes either a Map or Polarized map, convolves it with an input kernel
    and returns a new, smoothed map. Uses scipy.ndimage.convolve to do the work.
    (This is a wrapper for applyFunctionToMap.)

    INPUTS
        mapin: (Map or PolarizedMap) The map to convolve.

        convolution_kernel: (np.ndarray) An array containing the map with which
            we'll convolve the input map.

        in_place [False]: (bool) If True, modify the input map in place.

        submaps_to_use [None]: (list of strings) Used for PolarizedMap input only.
            If None, then convolve all maps in the input PolarizedMap. Otherwise,
            convolve only the maps in this list. Delete maps not in this list.

    OUTPUT
        A Map or PolarizedMap, corresponding to the input type.

    EXCEPTIONS
        ValueError if the "mapin" input is not a valid type.

    AUTHOR
        Stephen Hoover, 30 September 2013

    CHANGES
    """
    # This will apply ndimage.gaussian_filter(<map array>, gaussian_sigma) to
    # each map-like array (maps, submaps, weights) in the input "mapin".
    return applyFunctionToMap(ndimage.convolve, mapin, convolution_kernel,
                              in_place=in_place, submaps_to_use=submaps_to_use)


##########################################################################################
def applyFunctionToMap(function, mapin, *args, **kwargs):
    """
    This function takes either a Map or Polarized map and applies a specified function to
    each of the maps and weights in the object. See e.g. the "gaussianSmooth" function in
    this module for an example of use.

    >>> smoothed_map = applyFunctionToMap(scipy.ndimage.gaussian_filter, mymap, 4.0 * mymap.reso_arcmin/60 / (2*np.sqrt(2*np.log(2))))

    INPUTS
        function: (function) Apply this function to the maps. We will call the function
            with a 2D array in the first position (either a map or a weight map) and
            the extra args and kwargs following.

        mapin: (Map or PolarizedMap) The map to do stuff to.

        in_place [False]: (bool) If True, modify the input map in place.
            (NOTE: This argument will be extracted from "kwargs".)

        submaps_to_use [None]: (list of strings) Used for PolarizedMap input only.
            If None, then convolve all maps in the input PolarizedMap. Otherwise,
            convolve only the maps in this list. Delete maps not in this list.
            (NOTE: This argument will be extracted from "kwargs".)

        *args, **kwargs: Pass these to the supplied function.
            Note that keyword arguments "in_place" and "submaps_to_use"
            will be used inside this function, and not passed to the "function".

    OUTPUT
        A Map or PolarizedMap, corresponding to the input type.

    EXCEPTIONS
        ValueError if the "mapin" input is not a valid type.

    AUTHOR
        Stephen Hoover, 30 September 2013

    CHANGES
    """
    # Check the "mapin" input to make sure we can use it.
    if not isinstance(mapin, (Map, PolarizedMap)):
        raise ValueError("You must give me either a Map or PolarizedMap object.")

    # Pull the keyword arguments out of the "kwargs" argument.
    # (We can't put them in the function definition because we also
    #  want arbitrary positional arguments, and those have to come
    #  after any named arguments.)
    in_place = kwargs.pop('in_place', False)
    submaps_to_use = kwargs.pop('submaps_to_use', None)

    # Create the map to return (or get a pointer to the input
    # map, if we want to modify it).
    smooth_map = (mapin if in_place else mapin.copy())

    if isinstance(mapin, Map):
        # Map is easy: we only need to change one map and one weight.
        smooth_map.map = function(smooth_map.map, *args, **kwargs)
        if smooth_map.weight is not None:
            smooth_map.weight = function(smooth_map.weight, *args, **kwargs)

    elif isinstance(mapin, PolarizedMap):
        # For a PolarizedMap, do every submap in turn (unless we have
        # a specified sublist), then do each of the 9 weight maps separately.
        if submaps_to_use is None:
            submaps_to_use = smooth_map.pol_maps.keys()

        # First do the submaps.
        for submap_name in submaps_to_use:
            submap = smooth_map.pol_maps[submap_name]
            submap.map = function(submap.map, *args, **kwargs)

        # Now do the weight array.
        if smooth_map.weight is not None:
            for i_x in range(3):
                for i_y in range(3):
                    smooth_map.weight[:, :, i_x, i_y] = function(smooth_map.weight[:, :, i_x, i_y], *args, **kwargs)

        # Delete un-modified submaps, so that the object is in a consistent state.
        for submap_name in smooth_map.pol_maps.keys():
            if submap_name not in submaps_to_use:
                del smooth_map.pol_maps[submap_name]

    return smooth_map

##########################################################################################


def filterMap(mapin, uv_filter, apod_mask=None, remove_apod_mask=False,
              save_complex64=False):
    """
    Takes as input a PolarizedMap. Applies the specified filter in k-space,
    then returns a PolarizedMap with the result.

    INPUTS
        mapin: (PolarizedMap) The input map. It will not be modified.

        uv_filter: (2D array) Must be the same size as the map. We'll multiply
            the result of FFTing the map by this array, then FFT back.
            Note that this uv_filter array should be in the conventional FFT
            ordering.

        apod_mask [None]: (2D array) If not None, then apodize the input map by
            this mask before FFTing. You should use a good apodization!

        remove_apod_mask [False]: (bool) If this is true it will divide out the filtered
            map by the apod_mask to remove it. This is actually a pretty dicey thing
            to do since apod masks usually have zeros in them, so be careful and
            make sure you actually want to do this.

        save_complex64 [False]: The default output of np.fft.fft2 is a complex128 array.
            If True, instead return a complex64 array to save memory.  The differences
            are small: (32bit - 64bit)/64bit is ~ few x 10^-7 at most in power
            (np.abs(fft**2)) and less in the real and imaginary components of the FFT itself.

    OUTPUT
        A PolarizedMap in which the T, Q, and U maps have been filtered in k-space
        by the input uv_filter.

    EXCEPTIONS
        ValueError if mapin isn't a PolarizedMap.

    AUTHOR
        Stephen Hoover, 6 December 2013

    CHANGES
        Added 'remove_apod_mask' option since dividing out the apod mask leaves a bunch
             of infinite values. TJN
    """
    if not isinstance(mapin, PolarizedMap):
        raise ValueError("I only know how to filter a PolarizedMap.")

    # Start by copying the input map. That way we don't alter the input, and
    # we keep all of the metadata.
    filtered_map = mapin.copy()

    # Create a default "mask" of all ones if we didn't get an input mask,
    # and calculate the mask normalization.
    if apod_mask is None:
        apod_mask = np.ones(filtered_map.shape)
    window_norm_factor = np.sqrt(np.sum(apod_mask**2) / np.product(apod_mask.shape))

    # Go through the T, Q, U maps and filter them.
    for pol in ['T', 'Q', 'U']:
        # Create the Fourier-space maps.
        filtered_map.getFTMap(return_type=pol, remake=True, save_ffts=True,
                              sky_window_func=apod_mask, normalize_window=True,
                              save_complex64=save_complex64)

        # Apply the k-space filter.
        filtered_map.ft_maps[pol] *= uv_filter

        # Transform back to real space, and remove the apodization mask.
        filtered_map[pol].map = np.fft.ifft2(filtered_map.ft_maps[pol].map).real
#        filtered_map[pol].map = np.fft.ifft2(filtered_map.ft_maps[pol].map).real / (apod_mask/window_norm_factor)

        # if you chose to remove the apod mask after going back to real space,
        #   this generally messes up a lot of stuff though, since the apod_mask probably has zeros in it
        if remove_apod_mask:
            print 'WARNING-------- You are dividing out the apod mask -- This is DANGEROUS!!!--'
            filtered_map[pol].map /= (apod_mask / window_norm_factor)

    # We only filtered the T, Q, and U maps. If there's any other
    # maps, remove them. The user can create them again if necessary.
    for pol in filtered_map.pol_maps.keys():
        if pol not in ['T', 'Q', 'U']:
            del filtered_map.pol_maps[pol]
    for pol in filtered_map.ft_maps.keys():
        if pol not in ['T', 'Q', 'U']:
            del filtered_map.ft_maps[pol]

    return filtered_map

##########################################################################################


def zoomMap(_map, zoom_factor, spline_order=3):
    """
    This function increases or decreases the number of pixels in a map, attempting to keep the image the same.
    It is a wrapper for the scipy.ndimage.zoom function.

    INPUTS
        _map: (Map object) The Map to upsample or downsample.

        zoom_factor: Increase or decrease the spacing of pixels by this amount. A number
            greater than one upsamples the map resulting in a smaller pixel size, and
            a number less than one downsamples the map, reducing the number of pixels.
            The shape of the output map will be zoom_factor*_map.shape.

        spline_order [3]: (integer) We will interpolate between pixels with a spline function
            of this order. Must be in the range 0-5.

    OUTPUT
        A Map object which is a copy of the input Map object, except that it has the
        shape zoom_factor*_map.shape and new_map.reso_arcmin = _map.reso_arcmin/zoom_factor.
        The Map will cover the same area of sky, but with a different pixel resolution.

    EXCEPTIONS
        ValueError if the first argument is anything other than a Map.

    AUTHOR
        Stephen Hoover, 18 October 2013

    CHANGES
        31 Oct 2013 : Not every Map has the lower_index_ attributes. Put that in a try/except statement.
    """
    myargs = tools.myArguments(remove_self=True, remove_data=True,
                               combine_posargs=True)  # Store input arguments for use at the bottom of the function.
    del(myargs['_map'])

    if not isinstance(_map, Map):
        raise ValueError("This function only works on Map objects!")

    # Create the new, soon-to-be-upsampled map.
    new_map = _map.copy()

    # Do the upsampling.
    new_map.map = ndimage.zoom(_map.map, zoom=zoom_factor, order=spline_order, mode='nearest')
    new_map.weight = ndimage.zoom(_map.weight, zoom=zoom_factor, order=spline_order, mode='nearest')

    # Adjust metadata to reflect the new map.
    new_map.reso_arcmin = _map.reso_arcmin / zoom_factor
    #new_map.shape = (_map.shape*upsample_factor).astype(_map.shape.dtype)
    new_map.active_shape = (_map.active_shape * zoom_factor).astype(_map.active_shape.dtype)
    new_map.padding = (_map.padding * zoom_factor).astype(_map.padding.dtype)
    try:
        new_map.lower_index_x *= zoom_factor
        new_map.lower_index_y *= zoom_factor
    except AttributeError:
        pass

    # Note the fact that we've applied this function.
    if 'upsampleMap' not in new_map.processing_applied:
        new_map.processing_applied['upsampleMap'] = myargs
    else:
        # We want to record each instance in which we called this function, so
        # figure out what we need to use as a key name to record this call.
        n_prev = len([key for key in new_map.processing_applied.keys() if key.startswith('upsampleMap')])
        new_map.processing_applied['upsampleMap%d' % (n_prev + 1)] = myargs

    return new_map

##########################################################################################
##########################################################################################


class SkyStream(receiver.Timestream):

    @classmethod
    def asSkyStream(cls, input, **kwds):
        if type(input) == SkyStream:
            return input
        else:
            return SkyStream(input, **kwds)

    def __init__(self, data, polarization='none', **kwds):
        self.polarization = polarization
        super(SkyStream, self).__init__(data, **kwds)

##########################################################################################
##########################################################################################


class PolarizedSkyStream:

    def __init__(self, skystreams, polarizations=None, **kwds):

        as_skystreams = [SkyStream.asSkyStream(stream, **kwds) for stream in skystreams]
        if polarizations is None:
            polarizations = [stream.polarization for stream in skystreams]
        else:
            for stream, this_pol in zip(as_skystreams, polarizations):
                stream.polarization = this_pol
        self.skystreams = dict([(pol, stream) for pol, stream in zip(polarizations, as_skystreams)])
        self.polarizations = polarizations

    def __getitem__(self, key):
        return self.skystreams[key]

    def __setitem__(self, key, value):
        self.skystreams[key] = value

    def __getattr__(self, name):
        if name == 'streams':
            return self.skystreams
        else:
            try:
                return self.skystreams[self.polarizations[0]].__getattribute__(name)
            except AttributeError, err:
                raise err


##########################################################################################
##########################################################################################
class PolarizedMap:

    @classmethod
    def read(cls, filename, file_type=None):
        """
        Checks the filename extension and calls the appropriate reader.
        If the "file_type" input is "FITS" or "HDF5" (case insensitive), then
        we read the file as being in that format, regardless of extension.
        """
        # If we got an input for 'file_type', check that it makes sense.
        if file_type is not None and not (file_type.lower().startswith('h') or file_type.lower().startswith('f')):
            raise ValueError("Sorry, I don't recognize the \"%s\" file file_type." % file_type)
        elif file_type is None:
            file_type = filename.split('.')[-1]

        # Case insensitive
        file_type = file_type.lower()

        if file_type.lower().startswith('h'):
            new_map = cls.readFromHDF5(filename)
        elif file_type.lower().startswith('f'):
            new_map = cls.readFromFITS(filename)
        else:
            raise ValueError("I don't recognize filename \"%s\"!" % filename)

        new_map.from_filename = filename

        return new_map

    @classmethod
    def readFromFITS(cls, filename):
        """
        Read a PolarizedMap object in from an SPT-formatted FITS file. Assume that the
        map is the only thing stored in the given FITS file.
        """
        pol_maps = fits.readSptFits(filename)
        weight = pol_maps.pop('weight', [None])[0]
        processing_applied = pol_maps.pop('processing_applied', tools.struct())
        for key, value in pol_maps.items():
            if key != key.upper():
                pol_maps[key.upper()] = value
                del(pol_maps[key])
        new_map = cls(maps=pol_maps, weight=weight)

        # Convert processing_applied arrays to structs:
        try:
            processing_applied.dtype  # Check for arrays by calling on 'dtype'.
            processing_applied = tools.struct([(_proc, tools.struct()) for _proc in processing_applied])
        except AttributeError:
            pass
        new_map.processing_applied = processing_applied

        # Store the filename in the map.
        new_map.from_filename = filename

        return new_map

    @classmethod
    def readFromHDF5(cls, filename, timeit=False):
        """
        Read a PolarizedMap object in from an SPT-formatted HDF5 file. Assume that the
        map is the only thing stored in the given file.

        Equivalent to
            table.readSptHDF5(filename)[0]
        """
        new_map = hdf.readSptHDF5(filename, timeit=timeit)

        # The file might be read out either as a Map (or PolarizedMap) object, or
        # as a struct with one entry containing a Map. Either way, we just
        # want the object.
#        if isinstance(new_map, dict):
#            new_map = new_map[0]

        # Store the filename in the map.
#        new_map.from_filename = filename

        return new_map

    @classmethod
    def readFromHDF5Stripped(cls, filename, timeit=False):
        """
        Read (most of) a PolarizedMap object from an HDF5 file. This function assumes that the HDF5 file contains
        a PolarizedMap, it assumes a certain layout of the file (valid as of 18 Dec 2012), and it ignores most
        of the meta data. The purpose of this function is to minimize readout time by only reading necessary fields.

        It doesn't actually save too much time. In tests on an 1800x1800 pixel field, the stripped reader
        was 0.1 to 0.15 seconds faster than the generic reader.

        Be careful about using this function unless you know exactly what you're reading - lack of metadata
        may mask important differences between maps.

        INPUTS
            filename: (string) Fully-specified location of an HDF5 file containing a PolarizedMap.

            timeit [False]: (bool) If True, output total time taken in readout.

        OUTPUT
            A PolarizedMap object, missing much of its metadata.
        """
        if timeit:
            start_time = time.utcnow()

        with hdf.tb.openFile(filename, 'r') as fileh:
            new_map = cls(maps={'T': fileh.root.contents.pol_maps.T.map.read(),
                                'Q': fileh.root.contents.pol_maps.Q.map.read(),
                                'U': fileh.root.contents.pol_maps.U.map.read()},
                          weight=fileh.root.contents.weight.read(),
                          weighted_map=fileh.root.contents.weighted_map.read(),
                          flattened_pol=fileh.root.contents.flattened_pol.read(),
                          from_filename=filename)
            new_map.setMapAttr('center', fileh.root.contents.pol_maps.T.center.read())
            new_map.setMapAttr('reso_arcmin', fileh.root.contents.pol_maps.T.reso_arcmin.read())
            new_map.setMapAttr('projection', fileh.root.contents.pol_maps.T.projection.read())
            new_map.setMapAttr('name', fileh.root.contents.pol_maps.T.name.read())
            new_map.setMapAttr('band', fileh.root.contents.pol_maps.T.band.read())
            new_map.setMapAttr('is_leftgoing', fileh.root.contents.pol_maps.T.is_leftgoing.read())

            new_map.from_filename = filename

        if timeit:
            print "Readout from %s took %.2f seconds." % (filename, (time.utcnow() - start_time).total_seconds())

        return new_map

    ########################################################################################################
    def __init__(self, maps=None, shape=None, polarizations=None, weight=None, processing_applied=None,
                 from_filename=None, weighted_map=False, flattened_pol=False, extra_info={}, **map_args):
        """

        INPUTS

            weight [None]: If provided, this should be an array with a 3x3 matrix
                for each pixel of the map.

        """
        if 'polarization' in map_args:
            del(map_args['polarization'])

        # Anything in 'extra_info' gets stored as an attribute.
        self.__dict__.update(extra_info)

        # Special case: We'll get "from_filename" in maps which have been read in
        # from files. Only store it if it exists.
        if from_filename is not None:
            self.from_filename = from_filename
            
        if polarizations is None and maps is None:
            polarizations = list(tqu)  # Make sure to /copy/ the "tqu" list!

        if not hasattr(self, 'observation_dates'):
            try:
                self.observation_dates = [re.search('\d{8}_\d{6}',self.from_filename).group()]
            except:
                self.observation_dates = []
        try:
            self.polarizations = list(polarizations)
        except TypeError:
            self.polarizations = polarizations

        if maps is not None:
            # Check if the input 'maps' is a dictionary. If so, assume that the keys
            # are the proper polarizations.
            try:
                self.polarizations = maps.keys()  # Will fail if this isn't a dictionary.
            except AttributeError:
                maps = tools.struct(zip(self.polarizations, maps))

            # I need to be sure that the input is a number of maps equal to the
            # number of polarizations, with each map of identical shape.
            ##assert len(maps)==n_tqu, "I need "+str(n_tqu)+" maps, one for each polarization component!"
            assert len(maps) == len(self.polarizations)

            # Check that each map is the same size.
            try:
                for _map in maps.itervalues():
                    assert _map == maps[self.polarizations[0]]
            except ValueError:
                # We'll get this exception if "maps" is a dictionary of arrays, and not of Maps.
                for pol in maps:
                    maps[pol] = Map(map=maps[pol], polarization=pol, **map_args)

            self.pol_maps = tools.struct(maps)
        elif shape is not None:
            self.pol_maps = tools.struct([(pol, Map(shape=shape, polarization=pol, **map_args))
                                          for pol in self.polarizations])
        else:
            raise TypeError("Object PolarizedMap requires either a set of input maps or a shape for blank maps.")

        if weight is not None:
            assert (weight.shape == np.append(self.shape, [3, 3])).all(
            ), "The provided weight must have a 3x3 matrix for every map pixel!"
        self.weight = weight

        if processing_applied is None:
            processing_applied = tools.struct()
        self.processing_applied = processing_applied
        self.weighted_map = weighted_map
        self.flattened_pol = flattened_pol

        # Create a space to store FFTs of maps.
        self.ft_maps = struct()

    ########################################################################################################
    def initArgs(self, noshape=False, **new_args):
        """
        Outputs a dictionary of arguments which we can use to make a new PolarizedMap object
        equivalent to this one. Example of intended use:
           new_map = PolarizedMap( maps=[new_t, new_q, new_u], polarizations=['T','Q','U'], **old_map.initArgs() )

        INPUTS
            noshape [False]: (bool) Exclude the shape arguments (shape, active_shape, and padding)
                from the output dictionary.

            **new_args : Any additional arguments supplied to this function will be passed
                directly into the output, overriding any arguments with the same
                name which might otherwise be there.

        OUTPUT
            A dictionary of arguments which can be passed directly into PolarizedMap's init function.
        """
        if 'T' in self.pol_maps:
            args = self.pol_maps['T'].initArgs(noshape=noshape)
        else:
            args = self.pol_maps[0].initArgs(noshape=noshape)
        args.update(self.__dict__.copy())
        del args['pol_maps']
        del args['polarizations']
        args.update(new_args)
        return args

    ########################################################################################################
    def changeDType(self, new_dtype, weight_dtype=None):
        """
        We may wish to switch the level of precision with which our maps are stored. This function can do that.

        INPUTS
            new_type: (dtype) The type of data (e.g. np.float32, np.float64) which we wish to store in our map.
                If None, then the map data will not be altered (if we only want to change the weight array).

            weight_dtype [None]: (dtype) The type of data (e.g. np.float32, np.float64) which we wish to store
                in our weight array. If None, then the weight array will not be altered.

        OUTPUT
            None, but the dtype of the map and/or weight arrays will be changed.
        """
        if new_dtype is not None:
            for _map in self.pol_maps.itervalues():
                _map.changeDType(new_dtype, weight_dtype=weight_dtype)
        if weight_dtype is not None:
            self.weight = self.weight.astype(weight_dtype)

    ########################################################################################################

    def degradedMap(self, dfac):
        """
        Make a new PolarizedMap object with resolution reduced by a factor of dfac.
        This is done by breaking the map up into blocks of (dfac x dfac) pixels and then
        making a weighted sum over each block.
        """

        assert(np.mod(self.shape[0], dfac) == 0)
        assert(np.mod(self.shape[1], dfac) == 0)

        dmap = type(self)(maps=[np.zeros(self.shape / dfac), np.zeros(self.shape / dfac), np.zeros(self.shape / dfac)],
                          polarizations=['T', 'Q', 'U'],
                          **self.initArgs(weight=np.zeros((self.shape[0] / dfac, self.shape[1] / dfac, 3, 3)),
                                          shape=self.shape / dfac,
                                          active_shape=self.active_shape / dfac,
                                          reso_arcmin=self.reso_arcmin * dfac,
                                          weighted_map=True))

        if self.weighted_map == False:
            tmap = self['I'].map * self.weight[:, :, 0, 0] + self['Q'].map * \
                self.weight[:, :, 0, 1] + self['U'].map * self.weight[:, :, 0, 2]
            qmap = self['I'].map * self.weight[:, :, 1, 0] + self['Q'].map * \
                self.weight[:, :, 1, 1] + self['U'].map * self.weight[:, :, 1, 2]
            umap = self['I'].map * self.weight[:, :, 2, 0] + self['Q'].map * \
                self.weight[:, :, 2, 1] + self['U'].map * self.weight[:, :, 2, 2]
        else:
            [tmap, qmap, umap] = [self['I'].map[:, :], self['Q'].map[:, :], self['U'].map[:, :]]

        for i in xrange(0, dfac):
            for j in xrange(0, dfac):
                dmap += type(dmap)(maps=[tmap[i::dfac, j::dfac], qmap[i::dfac, j::dfac], umap[i::dfac, j::dfac]],
                                   polarizations=['T', 'Q', 'U'],
                                   **dmap.initArgs(weight=self.weight[i::dfac, j::dfac, :, :]))

        if self.weighted_map == False:
            dmap = dmap.removeWeight()

        return dmap

    ########################################################################################################
    def zeroPaddedMap(self, padding_factor):
        """
        Make a new PolarizedMap object, this one padded by the requested factor.
        """

        if padding_factor < 1:
            raise ValueError("Padding means incrasing the size of the maps!")

        if padding_factor == 1:
            print('padding factor =1, nothing to be done')
            return self

        # type(self)(maps=dict([(pol,self[pol].copy()) for pol in self.polarizations]))
##        new_maps = dict([(pol,self[pol].copy()) for pol in self.polarizations])
        padded_shape = [int(padding_factor * dim) for dim in self.shape]
        # new_map = type(self)(shape=padded_shape)
        new_map = {}

        for pol in self.polarizations:
            _map = self[pol].map
            padded_map = np.zeros(padded_shape)
            padded_map[:self.shape[0], :self.shape[1]] = _map
            new_map[pol] = padded_map

        # zero padding the wright
        new_weights = np.zeros((int(padding_factor * dim), int(padding_factor * dim), 3, 3))

        new_weights[:self.shape[0], :self.shape[1], :, :] = self.weight
        return type(self)(maps=new_map, weight=new_weights)

    ########################################################################################################
    def setMapAttr(self, name, value):
        for _map in self.pol_maps.values():
            setattr(_map, name, value)

    ########################################################################################################
    def getAllFFTs(self, sky_window_func=None, return_maps=None,
                   save_complex64=False):
        """
        Return a dictionary of FT maps of all types in this object.
        This is currently only called from analysis.maps.combineBands()
        """
        if sky_window_func is None:
            sky_window_func = np.ones(self.shape, dtype=float_type)
        pols = self.polarizations if return_maps is None else return_maps
        return type(self)(maps=dict([(pol, self.getFTMap(return_type=pol, sky_window_func=sky_window_func, save_complex64=save_complex64))
                                     for pol in pols]),
                          **self.initArgs())

    ########################################################################################################
    def getFTMap(self, return_type='T',
                 remake=False, save_ffts=True, conjugate=False,
                 sky_window_func=None, normalize_window=True, fft_shape=None,
                 save_complex64=False,
                 quiet=True):
        """
        Returns a PolarizedMap object constructed so that it only has FTs, no realspace data.
        It's constructed in such a way that the pol_maps struct is the same as the ft_maps struct.

        We always take existing FTs from self.ft_maps if they exist. By default, we only use existing
        FTs. If return_maps is not None, we create any missing maps, using the input sky_window_func.

        INPUTS
            return_type ['T']: (string) the polarization map type to return.
                Should be one of: 'T', 'Q', 'U', 'E', 'B'

        OPTIONS IF RE-MAKING FFTS:
            remake [False]: (bool) If True, recreate the FTs (even if they already exist).
               Note, you can only remake T.  E and B will raise an exception, because they must
               be made with the constructEB() function.

            save_ffts [True]: (bool) If True, save the ffts to this self.ft_maps.  If the FTs already existed,
               this option will over-write the old ones.

            conjugate [False]: (bool) Save time by using more space. Conjugation takes a surprisingly
                long amount of time. If you call conjugate=True with save_ffts=True, we'll
                store conjugated FTs in self.ft_maps_conj.

            sky_window_func [None]: (2d ndarray) If we need to take new FTs, apply this
                window function to each map before FTing.

            fft_shape [None]: (2-element list) x, y shape of output fft.
                The real-space map will be padded to a size of (fft_shape[0], fft_shape[1]).

            save_complex64 [False]: The default output of np.fft.fft2 is a complex128 array.  If True,
                instead return a complex64 array to save memory.  The differences are small: (32bit - 64bit)/64bit
                is ~ few x 10^-7 at most in power (np.abs(fft**2)) and less in the real and imaginary components
                of the FFT itself.

            normalize_window [True]: (bool) If True, apply a normalization factor to the input
                window function so that the power in the resulting FT doesn't change.

        OUTPUT
            A fft map (2d complex ndarray)

        EXCEPTIONS
            ValueError if this object has no pre-existing FTs and return_maps=None.
            ValueError if the input mask is too large for the padded array size.

        AUTHOR
            Kyle Story, 25 October 2013

        CHANGES
            3 June 2014: Add the capacity to store the complex conjugate of the map, to speed up cross-spectra. SH
        """
        assert(type(return_type) == str), "Argument return_type must be of string type.  Given type was %r" % type(return_type)
        if return_type not in ['T', 'Q', 'U', 'E', 'B']:
            raise ValueError("Bad return_type: " + return_type +
                             ".  return_type must be a string of the following: 'T','Q','U','E','B'")

        if not remake:
            if self.ft_maps.has_key(return_type):
                if conjugate:
                    # If we want the complex conjugate of the map, see if we've already stored it.
                    # If we haven't, then compute it, store (if requested), and return.
                    try:
                        return self.ft_maps[return_type].map_conj
                    except AttributeError:
                        map_conj = np.conjugate(self.ft_maps[return_type].map)
                        if save_ffts:
                            self.ft_maps[return_type].map_conj = map_conj
                        return map_conj
                else:
                    return self.ft_maps[return_type].map
            else:
                if return_type in ['T', 'Q', 'U']:
                    remake = True  # if T FTs don't exist, re-make them
                else:
                    warnings.warn("You asked for " + return_type +
                                  " type FT maps, but they do not exist.  E and B FT maps should only be made with constructEB.  Returning -1", RuntimeWarning)
                    return -1

        # Make new FT maps
        if remake:
            if not quiet:
                print("getFTMap(): Remaking " + return_type + " type FT map.")
            if return_type not in ['T', 'Q', 'U']:
                raise ValueError(
                    "In this function, you can only remake T, Q, or U ffts.  E and B must be made with the constructEB() function.")

            # Put the map inside a new array, adding padding if requested.
            _map = self.pol_maps[return_type]
            if fft_shape is None:
                fft_shape = _map.shape
            assert(len(fft_shape) == 2)
            if (fft_shape < _map.shape).any():
                raise ValueError("***Error: Requested fft_shape must be >= size of existing maps.  Requested fft_shape is ",
                                 fft_shape, ", while existing map shape is ", _map.shape)

            padded_shape = np.array(fft_shape)

            padded_map = np.zeros(padded_shape, dtype=_map.map.dtype)
            padded_map[:_map.shape[0], :_map.shape[1]] = _map.map

            if sky_window_func is None:
                sky_window_func = np.ones(padded_shape, dtype=float_type)
            elif (sky_window_func.shape != padded_shape).any():
                # Make sure that the window function is compatible with the padding,
                # and pad it with zeros if necesary.
                if (sky_window_func.shape > padded_shape).any():
                    raise ValueError("The input mask is larger than the padded map size!")

                padded_window = np.zeros(padded_shape, dtype=_map.map.dtype)
                padded_window[:sky_window_func.shape[0], :sky_window_func.shape[1]] = sky_window_func
                sky_window_func = padded_window

            window_norm_factor = 1.
            if normalize_window:
                window_norm_factor = np.sqrt(np.sum(sky_window_func**2) / np.product(sky_window_func.shape))

            # Take the FFT and attach it to this map. If we padded, set the entire
            # area of the new map as "active" -- FTs don't have blank areas around
            # the edges.
            fft_map = np.fft.fft2(padded_map * sky_window_func / window_norm_factor)

            # Save space by using complex64?
            if save_complex64:
                fft_map = np.complex64(fft_map)

            if save_ffts:
                self.addFTMap(return_type, fft_map, active_shape=padded_shape)

            # Take (and store) the complex conjugate if requested.
            if conjugate:
                fft_map = np.conjugate(fft_map)
                if save_ffts:
                    self.ft_maps[return_type].map_conj = fft_map

            return fft_map

    ########################################################################################################
    def getFTOnlyMap(self, return_maps=None, sky_window_func=None,
                     normalize_window=True, remake=False, fft_shape=None,
                     save_complex64=False):
        """
        Returns a PolarizedMap object constructed so that it only has FTs, no realspace data.
        It's constructed in such a way that the pol_maps struct is the same as the ft_maps struct.

        We always take existing FTs from self.ft_maps if they exist. By default, we only use existing
        FTs. If return_maps is not None, we create any missing maps, using the input sky_window_func.

        INPUTS
            return_maps [None]: (Iterable of strings) A list of polarizations which we want in the
                new FT-only map. If None, we'll use all of the existing FTs, and nothing else.

            sky_window_func [None]: (2d ndarray) If we need to take new FTs, apply this
                window function to each map before FTing.

            normalize_window [True]: (bool) If True, apply a normalization factor to the input
                window function so that the power in the resulting FT doesn't change.

            fft_shape [None]: (2-element list) x, y shape of output fft.
                The real-space map will be padded to a size of (fft_shape[0], fft_shape[1]).

            save_complex64 [False]: The default output of np.fft.fft2 is a complex128 array.  If True,
                instead return a complex64 array to save memory.  The differences are small: (32bit - 64bit)/64bit
                is ~ few x 10^-7 at most in power (np.abs(fft**2)) and less in the real and imaginary components
                of the FFT itself.

            remake [False]: (bool) If True, recreate the FTs even if they already exist. In this case,
                we will overwrite the existing FTs with the new ones.

        OUTPUT
            A new PolarizedMap object, with the specified FTs in both the pol_maps and ft_maps structs.
            If we created new FTs along the way, they will also be stored in this object's ft_maps struct.

        EXCEPTIONS
            ValueError if this object has no pre-existing FTs and return_maps=None.
            ValueError if the input mask is too large for the padded array size.

        AUTHOR
            Stephen Hoover, 20 June 2013
        """
        if len(self.ft_maps) == 0 and return_maps is None:
            raise ValueError("I have no FTs to put into a FT-only map!")

        # Create any new maps needed.
        if return_maps is not None:
            for map_pol in return_maps:
                if map_pol not in self.ft_maps or remake:
                    self.getFTMap(return_type=map_pol, remake=True, sky_window_func=sky_window_func,
                                  normalize_window=normalize_window, fft_shape=fft_shape, save_complex64=save_complex64)

        # Create the new PolarizedMap, and connect the ft_maps struct to the pol_maps struct.
        assert(fft_shape is None or len(fft_shape) == 2)
        if (fft_shape is not None) and ((fft_shape != self.shape).any()):
            # We need to add any requested padding to the weights.
            padded_shape = np.array(fft_shape)

            if self.weight is not None:
                padded_weight = np.zeros(np.append(padded_shape, [3, 3]), dtype=self.weight.dtype)
                padded_weight[:self.shape[0], :self.shape[1], :, :] = self.weight
            else:
                padded_weight = None
            new_ft_map = type(self)(maps=dict([(pol, _map) for pol, _map in self.ft_maps.iteritems() if pol in return_maps]),
                                    **self.initArgs(weight=padded_weight))
        else:
            new_ft_map = type(self)(maps=dict([(pol, _map) for pol, _map in self.ft_maps.iteritems() if pol in return_maps]),
                                    **self.initArgs())
        new_ft_map.ft_maps = new_ft_map.pol_maps

        return new_ft_map

    ########################################################################################################
    def getInverseFFT(self, k_window_func=None, real_part=False, force_type=None,
                      save_complex64=False):
        if k_window_func is None:
            k_window_func = np.ones(self.shape, dtype=float_type)

        args = self.initArgs()
        if save_complex64:
            maps = dict([(pol, np.complex64(np.fft.ifft2(self.pol_maps[pol].map * k_window_func)))
                         for pol in self.polarizations])
        else:
            maps = dict([(pol, np.fft.ifft2(self.pol_maps[pol].map * k_window_func)) for pol in self.polarizations])
        if real_part:
            for pol in maps:
                maps[pol] = np.real(maps[pol])
            if 'data_type' in args:
                del(args['data_type'])
        if force_type:
            for pol in maps:
                maps[pol] = maps[pol].astype(force_type)
            if 'data_type' in args:
                args['data_type'] = force_type

        return type(self)(maps=maps, **args)

# if force_type is not None:
# maps=dict([(pol, (np.fft.ifft2(self.pol_maps[pol].map * k_window_func)).astype(force_type))
# for pol in self.polarizations])
# else:
# maps=dict([(pol, np.fft.ifft2(self.pol_maps[pol].map * k_window_func))
# for pol in self.polarizations])
# return PolarizedMap(maps=dict([(pol, (np.fft.ifft2(self.pol_maps[pol].map * k_window_func)).astype(force_type))
# for pol in self.polarizations]),
# **self.initArgs() )
# else:
# return PolarizedMap(maps=dict([(pol, np.fft.ifft2(self.pol_maps[pol].map * k_window_func))
# for pol in self.polarizations]),
# **self.initArgs() )

    ########################################################################################################
    def trimPadding(self):
        """
        Modifies the object in place to remove the padding around the edges of the map.
        """
        for map in self.pol_maps.itervalues():
            map.trimPadding()

    ########################################################################################################
    def getActiveMap(self):
        return type(self)(maps=dict([(pol, self.pol_maps[pol].active_map) for pol in self.polarizations]), polarizations=self.polarizations, **self.initArgs())
    active_map = property(getActiveMap)
    active_maps = property(getActiveMap)

    ########################################################################################################
    def getSubmap(self, map_shape, center_offset=[0., 0.], units='degrees'):
        """
        Returns a cutout of this map. The cutout has the same resolution and must be a subset
        of this map.

        INPUTS
            map_shape: [2 element iterable] The shape of the new map, in [Y, X] or [elevation, azimuth].

            center_offset [[0.,0.]]: [2 element iterable] How far from the center of this map should the
                cutout's map be? When in degrees or arcminutes, this is [RA, dec]. When in pixels, it's [el, az].

            units ['degrees']: (string) The units of both center_offset and map_shape (must the the same
                units for both). Valid choices are "degrees", "arcminutes" and "pixels".

        OUTPUT
            A new map with the specified dimensions and center.
        """
        # Convert inputs to pixels.
        if units.lower().startswith('deg'):
            # Convert inputs from degrees to number of pixels.
            map_radius = (np.asarray(map_shape) / 2. * 60 / self.reso_arcmin).astype(int)
            pix_center = self.ang2Pix(self.center + center_offset, round=True)
        elif units.lower().startswith('arcmin'):
            # Convert inputs from arcminutes to number of pixels.
            map_radius = (np.asarray(map_shape) / 2. / self.reso_arcmin).astype(int)
            center_offset = np.asarray(center_offset) / 60.
            pix_center = self.ang2Pix(self.center + center_offset, round=True)
        elif units.lower().startswith('pix'):
            # Inputs are already number of pixels. Just make sure that they're integer
            # arrays and change shape to half-shape.
            map_radius = (np.asarray(map_shape) / 2.).astype(int)
            center_offset = np.asarray(center_offset, dtype=int)
            pix_center = (np.asarray(self.shape) / 2. + center_offset).astype(int)
        else:
            raise ValueError("I don't recognize \"%s\" units!" % units)

        # Define the region of this map that gets used in the sub-map.
        submap_cutout = [slice(np.max([pix_center[0] - map_radius[0], 0]), np.min([pix_center[0] + map_radius[0], self.shape[0] - 1])),
                         slice(np.max([pix_center[1] - map_radius[1], 0]), np.min([pix_center[1] + map_radius[1], self.shape[1] - 1]))]

        # Get the center of the submap from the center of the pixels we actually used.
        #new_center = self.pix2Ang( map(lambda x: (x.start+x.stop)/2., submap_cutout) )

        return type(self)(maps=dict([(pol, self.pol_maps[pol].getSubmap(map_shape, center_offset, units)) for pol in self.polarizations]),
                          polarizations=self.polarizations, **self.initArgs(noshape=True, weight=(self.weight[submap_cutout] if self.weight is not None else None)))

    ########################################################################################################
    def getStokesMatrixContents(self):
        """
        Returns this PolarizedMap's contents as a single array with a 1x3 matrix of [T, Q, U] values in each pixel.

        INPUTS
            None

        OUTPUT
            An np.ndarray with shape [n_y, n_x, 1, 3], which contains the values from the T map
            in the [:,:,0,0] elements, the Q map in the [:,:,0,1] elements, and the U map
            in the [:,:,0,2] elements.
               The returned array is a copy of the map contents: changing it will not change this object.

        EXCEPTIONS
            None

        AUTHOR
            Stephen Hoover
        """

        matrix_map = np.zeros(np.append(self.shape, [1, 3]), dtype=self.pol_maps['T'].map.dtype)
        matrix_map[:, :, 0, 0] = self.pol_maps['T'].map[:, :]
        matrix_map[:, :, 0, 1] = self.pol_maps['Q'].map[:, :]
        matrix_map[:, :, 0, 2] = self.pol_maps['U'].map[:, :]

        return matrix_map

    ########################################################################################################
    def getWeightedFlag(self):
        return self.__dict__.get('weighted_map', False)

    def setWeightedFlag(self, flag):
        self.__dict__['weighted_map'] = flag
        self.setMapAttr('weighted_map', flag)
    weighted_map = property(getWeightedFlag, setWeightedFlag)
    ########################################################################################################

    def addMap(self, pol, map, allow_different=False):
        """Adds a new map, of polarization pol, to this object. If map is a Map object,
        add it directly. If map is an array or list, create a new Map object for it.

        INPUTS
            pol: (string) The "polarization" (or other name designator) of the new Map.

            map: (ndarray or Map object) The map to add. If it's an array, we'll put it
                into a new Map object.

            allow_different [False]: (bool) By default, we require new maps to be compatible
                with existing maps (same shape, resolution, etc.). If allow_different=True,
                we won't enforce that rule. USE WITH CARE!

        OUTPUT
            None, but this object will be modified.

        AUTHOR
            Stephen Hoover

        MODIFICATIONS
            7 August 2013: Add "allow_different" argument, improve docstring. SH
        """
        try:
            map.getResoRad()  # A Map object will have this method.
        except AttributeError:
            map = Map(map=map, **self.pol_maps[self.polarizations[0]].initArgs(polarization=pol))

        # Make sure that the input value matches the existing maps. If not,
        # raise an exception.
        if not allow_different:
            assert map == self.pol_maps[self.polarizations[0]
                                        ], "Input map must have the same properties as the existing maps!"

        self.pol_maps[pol] = map
        if pol not in self.polarizations:
            self.polarizations.append(pol)
    setMap = addMap  # Define "setMap" as an alias for "addMap".

    ########################################################################################################
    def addFTMap(self, pol, map, **new_args):
        """Adds a new map, of polarization pol, to this object. The input "map" is assumed to represent
        the FT of an existing map, and we store it in the self.ft_maps struct. If map is a Map object,
        add it directly. If map is an array or list, create a new Map object for it.

        INPUTS
            pol: (string) The polarization of the input map.

            map: (Map or array) The map information.

            **new_args: Any extra arguments, set as attributes of the newly-created map object.

        OUTPUT
            None, but this object is modified.

        EXCEPTIONS
            None

        AUTHOR
            Stephen Hoover, 11 June 2013
        """
        try:
            map.getResoRad()  # A Map object will have this method.
        except AttributeError:
            map = Map(map=map, **self.pol_maps[self.polarizations[0]].initArgs(polarization=pol, **new_args))

        if not hasattr(self, 'ft_maps'):
            self.ft_maps = struct()

        self.ft_maps[pol] = map

    ########################################################################################################
    def convertQUToLTheta(self, recreate=False):
        """
        Uses Q and U maps present in this object to construct L (amplitude of linearly polarized component)
        and theta (linear polarization angle).

        INPUTS
            recreate [False]: (bool) What do we do if the "L" and/or "theta" maps already exist?
                If False, do not overwrite any existing maps. If True, procede normally, overwriting existing map(s).

        OUTPUT
            None, but sets the "L" and "theta" maps on this object.
        """
        complex_L = self.pol_maps['Q'].map + (1j) * self.pol_maps['U'].map
        if 'L' in self.polarizations:
            warnings.warn("This map already has an 'L' sub-map!" +
                          (" Overwriting!" if recreate else " Skipping!"), RuntimeWarning)
        if 'L' not in self.polarizations or recreate:
            self.addMap('L', np.abs(complex_L))
        if 'theta' in self.polarizations:
            warnings.warn("This map already has a 'theta' sub-map!" +
                          (" Overwriting!" if recreate else " Skipping!"), RuntimeWarning)
        if 'theta' not in self.polarizations or recreate:
            self.addMap('theta', 0.5 * np.angle(complex_L, deg=True))
            self.pol_maps['theta'].units = "Degrees"

    ########################################################################################################
    def calculatePolFrac(self, recreate=False, clean=True):
        """
        Adds a Map of the polarization fraction abs(L/T) to this PolarizedMap, under the name "polfrac".

        INPUTS
            recreate [False]: (bool) What do we do if the "L" and/or "theta" maps already exist?
                If False, do not overwrite any existing maps. If True, procede normally, overwriting existing map(s).

            clean [True]: (bool) Set to zero pixels where the polarization fraction is unphysical.

        OUTPUT
            None, but sets the "polfrac" map on this object.
        """
        if 'polfrac' not in self.polarizations or recreate:
            self.addMap('polfrac', np.abs(self['L'].map / self['T'].map))
            self.pol_maps['polfrac'].units = "Polarization Fraction"
            if clean:
                self.pol_maps['polfrac'].map[self.pol_maps['polfrac'].map > 1.] = 0.
                self.pol_maps['polfrac'].map[self.pol_maps['polfrac'].map < 0.] = 0.

    ########################################################################################################
    def drawImage(self, polarizations=['T', 'Q', 'U'], units='arcminutes',
                  title=None, gaussian_smooth=0, weight=False,
                  reverse=False, normalize_pol=False,
                  vmin=None, vmax=None, pmin=None, pmax=None,
                  log=False, bw=True, radec=False, extent=None, mask=None, show_plot=True,
                  figure=None, layout='horizontal', subplot_positions=None, colormap='jet', imshow_param={}):
        """
        Draw this PolarizedMap on the screen.

        INPUTS
            polarizations [['T','Q','U']]: (list of strings) The polarizations to plot.

            units ['pixels']: (string) 'pixels', 'degrees' or 'arcminutes'

            title [None]: (string) A title for the plot.

            gaussian_smooth [0]: (int) The FWHM of the gaussian measured in _units_ to smooth with before plotting.  0 means don't smooth.

            weight [False]: (bool) If True, plot the map weight instead of the map itself.  Many of the other options no longer apply.

            reverse [False]: (bool) If True, plot the RA / azimuth reversed, to match standard astonomical images.

            normalize_pol [False]: (bool) If True, plot Q/I and U/I instead of Q and U.

            vmin [None]: (float) Vertical scale (color scale) minimum for temperature. Ignored if a 'norm' argument is given within imshow_param.

            vmax [None]: (float) Vertical scale (color scale) maximum for temperature. Ignored if a 'norm' argument is given within imshow_param.

            pmin [None]: (float) Vertical scale (color scale) minimum for polarization. Ignored if a 'norm' argument is given within imshow_param.

            pmax [None]: (float) Vertical scale (color scale) maximum for polarization. Ignored if a 'norm' argument is given within imshow_param.

            log [False]: (bool) If True, plot the colors on a logarithmic scale. Ignored if a 'norm' argument is given.

            bw [True]: (bool) If True, use a greyscale and no sub-pixel interpolation. Tijmen thinks this makes most maps easier to interpret.

            radec [False]: (bool) If True, axes are not centered on zero, but centered on actual ra,dec center of map.  Forces units to be 'degrees'

            extent [None]: (None or tuple) (ramin, ramax, decmin, decmax) in degrees.  Useful to specify real ra rather than ra "on the sky"

            mask [None]: (2D array with same size as map) Apply this mask to the map before plotting, if not None.
                (Does not modify the contents of the Map.)

            figure [None]: (int) Use the figure window number given to plot. Eg fig.number for fig=figure().  Will clear figure before plotting.

            layout ['horizontal']: Put the polarizations horizontally.  'vertical' is also valid.  Ignored if weight==True

            subplot_positions [None]:  If you have your own (large) figure that includes both left and right or different frequencies,
                                       and want to specify where the (usually) three subplots go,
                                       this should be a list of subplot positions.
                                       For example, [(1,3,1), (1,3,2), (1,3,3)] would duplicate horizontal layout for three polarizations.

            colormap ['jet']: If bw=False, this is the colormap that will be used to plot the image.  This is broken out of the {'cmap':xxxx} option
                of imshow_param for convenience.

            imshow_param [{}]: Input parameters to be passed to matplotlib.pyplot.imshow.  Same for all polarizations.
            show_plot ['True']: Show the plot on the screen (or not)

        OUTPUT
            None (draws to screen)

        EXCEPTIONS
            None

        AUTHOR
            Stephen Hoover

        CHANGES
            16 Sept 2013 : Make the gaussian_smooth input in units of FWHM of the Gaussian. SH
            22 Nov 2013 : Switch variable name "ax" back to "axes". SH
            2 Dec 2013 :   Cleaned up colorbar location/labeling, weight maps of different map combos work, imshow_param is now an input dict. JTS
            31 Dec 2013 : Let color scale on weight maps float. SH
            12 May 2014 : add option for a mask.  KTS
        """
        # Make sure that the "polarizations" input is an iterable.
        polarizations = np.array([polarizations]).flatten()

        if bw:
            imshow_param['cmap'] = 'bone'
        else:
            imshow_param['cmap'] = colormap

        # Don't let the plotter interpolate between pixels unless we explicitly ask it to.
        imshow_param.setdefault('interpolation', 'none')

        if radec:
            units = 'degrees'

        # Label the axes by degrees or arcmin from the center of the map
        if units.lower().startswith('deg') or units.lower().startswith('arcmin'):
            _shape = self.active_shape * self.reso_arcmin / 60.  # in degrees
            if not extent:
                if radec == True:
                    # the xticks and yticks are centered around self.center
                    extent = np.array([self.center[0] - _shape[1] / 2., self.center[0] + _shape[1] /
                                       2., self.center[1] - _shape[0] / 2., self.center[1] + _shape[0] / 2.])
                else:
                    # the xticks and yticks are centered around (0,0)
                    extent = np.array([-_shape[1] / 2., _shape[1] / 2., -_shape[0] / 2., _shape[0] / 2.])
            if units.lower().startswith('deg'):
                unit_name = "Degrees"
                self._pixels_per_unit = 1. / self.reso_arcmin * 60
            else:
                unit_name = "Arcminutes"
                extent *= 60
                self._pixels_per_unit = 1. / self.reso_arcmin
        else:
            unit_name = "Pixels"
            extent = None
            self._pixels_per_unit = 1.

        if reverse and extent is not None:
            extent = [extent[1], extent[0], extent[2], extent[3]]

        # Create the window and plot the image. Use the specified figure window if given
        if figure:
            fig = plt.figure(figure)
            if subplot_positions is None:
                fig.clear()  # make sure the figure is clear, since the window could have already been in use
        else:
            fig = plt.figure()

        if weight:
            pols = ['T', 'Q', 'U']
            # if not (polarizations==np.array(['T','Q','U'])).all():
            # raise ValueError("I don't know what to do for a non-standard list of
            # polarizations, %s!" % str(polarizations))tt
            for i_pol_y, pol_y in enumerate(polarizations):
                for i_pol_x, pol_x in enumerate(polarizations):
                    wh_y, wh_x = pols.index(pol_y), pols.index(pol_x)
                    this_submap = self.weight[:, :, wh_y, wh_x]
                    if log:
                        this_submap = np.abs(this_submap)
                    axes = fig.add_subplot(len(polarizations), len(polarizations),
                                           i_pol_y * len(polarizations) + i_pol_x + 1)

                    # Apply a mask function, if desired.
#                    if mask is not None: map_to_plot = map_to_plot*mask
                    if mask is not None:
                        this_submap = this_submap * mask

                    if reverse:
                        # Flip the map around (on the x-axis) if requested.
                        this_submap = this_submap[:, ::-1]

                    # assuming input was in FWHM of the given unit
                    gaussian_sigma = gaussian_smooth * self._pixels_per_unit / (2 * np.sqrt(2 * np.log(2)))

                    # If you want to force the same
                    image = axes.imshow(ndimage.gaussian_filter(
                        this_submap, gaussian_sigma), extent=extent, **imshow_param)
                    if i_pol_x == 0:
                        axes.set_ylabel(unit_name)
                    if i_pol_y == len(polarizations) - 1:
                        axes.set_xlabel(unit_name)
                    divider = make_axes_locatable(axes)
                    cb_ax = divider.append_axes("right", size='5%', pad=0.05)
                    cb = plt.colorbar(image, cax=cb_ax, ax=axes)
                    #cb = plt.colorbar()
                    if title is None:
                        axes.set_title(pol_y + pol_x + "-Polarization Weights")
                    else:
                        axes.set_title(title)
            fig.set_size_inches(4.2 * len(polarizations), 3.5 * len(polarizations), forward=True)
        else:
            # polarization data, not weights
            for i_pol, pol in enumerate(polarizations):
                this_submap = self.pol_maps[pol]
                if normalize_pol and pol != 'T':
                    t_map = self.pol_maps['T'].map
                    # for maps with lots of zero padding around them, this is artificialy low.
                    t_map_stddev = np.std(t_map)
                    wh_small_t = (np.abs(t_map) < 2 * t_map_stddev)
                    map_to_plot = this_submap.map / t_map
                    map_to_plot[wh_small_t] = 0.
                else:
                    map_to_plot = this_submap.map

                if subplot_positions:
                    axes = fig.add_subplot(*subplot_positions[i_pol])
                elif layout == 'horizontal':
                    axes = fig.add_subplot(1, len(polarizations), i_pol + 1)
                elif layout == 'vertical':
                    axes = fig.add_subplot(len(polarizations), 1, i_pol + 1)
                else:
                    raise ValueError(
                        'PolarizedObservedSky.drawImage needs either subplot_positions or layout horizontal or vertical')
                #self.pol_maps[pol].drawImage(units=units, title=None, **imshow_param)

                # Apply a mask function, if desired.
                if mask is not None:
                    map_to_plot = map_to_plot * mask

                if reverse:
                    # Flip the map around (on the x-axis) if requested.
                    map_to_plot = map_to_plot[:, ::-1]

                this_min = vmin if pol == 'T' else pmin  # could still be None for now
                this_max = vmax if pol == 'T' else pmax  # could still be None for now

                if 'norm' in imshow_param:
                    normalization = imshow_param.pop('norm')
                elif log:
                    normalization = matplotlib.colors.LogNorm(vmin=this_min, vmax=this_max)
                else:
                    normalization = matplotlib.colors.Normalize(vmin=this_min, vmax=this_max)

                # Determine actual min/max of (possibly masked) map.  This is only used for colobar stuff.
                if this_min is None:
                    this_min = map_to_plot.min()
                if this_max is None:
                    this_max = map_to_plot.max()

                # The "astype" call is because imshow doesn't deal well with np.float64 types. This
                # ensures that we'll get something we can plot, even if it's an np.float64.
                # Convert from FWHM to sigma.
                gaussian_sigma = gaussian_smooth * self._pixels_per_unit / (2 * np.sqrt(2 * np.log(2)))
                image = axes.imshow(ndimage.gaussian_filter(map_to_plot, gaussian_sigma).astype(
                    np.float32), extent=extent, norm=normalization, **imshow_param)
                axes.set_ylabel(unit_name)
                axes.set_xlabel(unit_name)
                if radec == True:
                    axes.set_ylabel('Declination (deg)')
                    axes.set_xlabel('Right Ascension (deg)')
                divider = make_axes_locatable(axes)
                cb_ax = divider.append_axes("right", size='5%', pad=0.05)
                cb = plt.colorbar(image, cax=cb_ax, ax=axes, norm=normalization)
                if this_submap.units is not None:
                    # Automatically set the color bar label and ticks to reasonable units (nK, uK, mK).
                    unit_scale = min(np.ceil(np.log10(np.abs(np.asarray([this_min, this_max]))) / -3.))
                    if log:
                        unit_scale = 0  # Don't do anything fancy for log
                    u_labs = {3: 'n', 2: '$\mu$', 1: 'm'}  # nK, uK, mK
                    cbar_prefix = u_labs.setdefault(unit_scale, '')
                    fact = 10**(3 * unit_scale) if cbar_prefix else 1.0
                    # Set the colorbar label
                    cb.ax.set_ylabel(cbar_prefix + this_submap.units, rotation='vertical')
                    if not log:
                        # Rescale colorbar yticklabels by fact.  But their text has Unicode U+2212 as the "Minus Sign".  Also, keep one decimal place.
                        # print [x.get_text() for x in cb.ax.get_yticklabels()]
                        # Note: for nK (pmin=-1e-7, pmax=1e-7), this doesn't actually work! If we
                        # did nothing, there would be a little 1e-7 on top of the colorbar and I
                        # can't find how to access this.
                        cb.ax.set_yticklabels([int(float(x.get_text().replace(u'\u2212', '-')) *
                                                   fact * 10.) / 10. for x in cb.ax.get_yticklabels()])
                        # Note that cb.ax.get_yticks() returns a list of numbers from 0.0 to 1.0,
                        # so it's not easy to do this without going through the strings.
                if title is None:
                    title = this_submap.name if this_submap.name is not None else ""
                axes.set_title(this_submap.polarization + "-Polarization Map" + (" of %s" % title))
            if layout == 'horizontal':
                fig.set_size_inches(4.2 * len(polarizations), 3.5, forward=True)
            elif layout == 'vertical':
                fig.set_size_inches(7.5, 4.0 * len(polarizations), forward=True)  # vertical layout means maps are wide
                fig.subplots_adjust(left=0.10, right=0.88, top=0.95, bottom=0.05, wspace=0.00, hspace=0.20)
        fig.tight_layout()
        plt.sca(axes)
        if show_plot:
            plt.show()
            plt.draw()

    plot = drawImage  # Alias the drawImage function.
    image = property(drawImage)

    ########################################################################################################
    def __getitem__(self, key):
        # First, check if we're trying to get a sub-map. If we're given a
        # letter or single integer, try to fetch the corresponding map.
        try:
            if key.upper() == 'I':  # Treat 'I' as 'T'
                return self.pol_maps['T']

            # If someone asks for 'L' or 'theta', and the map doesn't yet exist, create it.
            if (key.upper() in ['L', 'THETA'] and
                    (key.upper() not in self.pol_maps and key.lower() not in self.pol_maps)):
                self.convertQUToLTheta()

            # If someone asks for the polarization fraction, make sure it exists.
            if key.lower() == 'polfrac' and key.lower() not in self.pol_maps and key.upper() not in self.pol_maps:
                self.calculatePolFrac()

            # Try the input polarization. If that doesn't exist, then
            # check if someone was lazy and input a lower-case version
            # of an existing polarization.
            try:
                return self.pol_maps[key]
            except KeyError, err:
                if key.upper() in self.pol_maps:
                    return self.pol_maps[key.upper()]
                elif key.lower() in self.pol_maps:
                    return self.pol_maps[key.lower()]
                else:
                    raise err
        except AttributeError:
            # If someone input a non-string, then try to treat it as a pixel.
            return dict([(pol, self[pol][key]) for pol in self.polarizations])

    ########################################################################################################
    def __delitem__(self, key):
        if key in self.polarizations:
            self.polarizations.remove(key)
            del self.pol_maps[key]
        else:
            raise KeyError("Polarization " + key + " not in this PolarizedMap object!")

    ########################################################################################################
    def writeToFITS(self, filename, overwrite=False):
        """
        Write a representation of this object to an SPT-formatted FITS file.
        """
        fits_struct = self.__fitsformat__()
        fits.writeSptFits(filename, overwrite, **fits_struct)

    ########################################################################################################
    def __fitsformat__(self):
        """
        Returns a copy of this object, formatted for output to a FITS file.
        """
        fits_struct = dict([(_pol, _map.__fitsformat__()) for _pol, _map in self.pol_maps.iteritems()])
        fits_struct['weight'] = {'weight': self.weight}
        fits_struct['processing_applied'] = self.processing_applied
        # try:
        #    fits_struct['processing_applied'] = {'processing_applied':np.array(self.processing_applied.keys())}
        # except AttributeError:
        #    pass
        return fits_struct

    ########################################################################################################
    def writeToHDF5(self, filename, overwrite=False, use_compression=False):
        """
        Write a representation of this object to an SPT-formatted HDF5 file.
        """
        hdf.writeSptHDF5(filename, contents=self, overwrite=overwrite,
                         write_objects=True, use_compression=use_compression)

    ########################################################################################################
    def __hdf5format__(self, full_object=True):
        """
        Outputs a dictionary of this class, suitable for packing into an HDF5 file.

        INPUTS
            full_object [True]: (bool) If True, output all attributes of this object, so
                that it can be reassembled again after readout. If False, output only the data.

        OUTPUT
            A dictionary of this PolarizedMap's contents, suitable for writing to disk.
        """
        if full_object:
            hdf_data_out = {'__class__': tools.className(self), }
            hdf_data_out.update(self.__dict__)
        else:
            hdf_data_out = dict([(_pol, _map.map) for _pol, _map in self.pol_maps.iteritems()])

        return hdf_data_out

    ########################################################################################################
    @classmethod
    def __unpack_hdf5__(cls, data):
        """
        Recreate a PolarizedMap object from data packed into an HDF5-formatted file.
        """
        new_map = cls(maps=data.pop('pol_maps'), weight=data.pop('weight', None))
        new_map.__dict__.update(data)
        if 'weighted_map' in data:
            new_map.weighted_map = data['weighted_map']
        return new_map

    ########################################################################################################
    def getValue(self, point, polarizations):
        """ Returns a tuple of polarization values at the point 'point' corresponding to the input 'pol'. """
        return tuple([self.pol_maps[pol][point] for pol in polarizations])

    ########################################################################################################
    def setTQU(self, pixel, value):
        """
        For faster access when repeatedly setting the T, Q, U values in particular pixels.
        "value" is assumed to be some 3-element list-like object, where value[0] is 'T',
        value[1] is 'Q', and value[2] is 'U'.
        """
        self.pol_maps['T'].map[pixel] = value[0]
        self.pol_maps['Q'].map[pixel] = value[1]
        self.pol_maps['U'].map[pixel] = value[2]

    ########################################################################################################
    def getTOnly(self):
        """
        Returns a new Map object with only temperature information. Extracts weights from the [0,0] element
        of the weight matrix.
        """
        return Map(map=self.pol_maps['T'].map.copy(), **self.initArgs(polarization='T', weight=np.array(self.weight[:, :, 0, 0]),
                                                                      _map_weight=getattr(self, '_map_weight', 1.0)))

    ########################################################################################################
    def __setitem__(self, key, value):
        try:
            # Check if the input key is string-like. If so, it designates an entire map. Replace
            # the current map of that polarization (or add it if it's not there).
            key + ''

            self.addMap(key, value)

        except TypeError:
            # If the key wasn't a string, then assume we're entering a single pixel, either
            # as a dictionary or as an ordered iterable.
            try:
                for pol in self.polarizations:
                    self[pol][key] = value[pol]
            except (TypeError, ValueError):
                for index, pol in enumerate(self.polarizations):
                    if index >= len(value):
                        break
                    self[pol][key] = value[index]

    ########################################################################################################
    def __getattr__(self, name):
        """
        CHANGES
            7 Feb 2014: Add (commented out) warnings if we're disguising the fact that an attribute
                doesn't exist on this map. (Probably not necessary to have these, is it? They're distracting.) SH
        """
        if name == 'maps' or name == 'map':
            return self.pol_maps
        elif name == 'pol_maps':  # Prevent infinite recursion!
            raise AttributeError("'PolarizedMap' object has no attribute '" + name + "'")
        elif name == 'polarizations':  # Prevent infinite recursion!
            raise AttributeError("'PolarizedMap' object has no attribute '" + name + "'")
        else:
            try:
                if 'T' in self.polarizations:
                    # if name[0]!='_':
                    #    warnings.warn("I'm giving you the \"%s\" attribute from my T map." % name, RuntimeWarning)
                    return self.pol_maps['T'].__getattribute__(name)
                else:
                    # if name[0]!='_':
                    #    warnings.warn("I'm giving you the \"%s\" attribute from my %s map." % (name, self.polarizations[0]),
                    #                  RuntimeWarning)
                    return self.pol_maps[self.polarizations[0]].__getattribute__(name)
            except AttributeError, err:
                raise err

    ########################################################################################################
    # Finish this!
    def skyStream(self, pointings, polarization='all'):
        """
        Uses an input PointingSequence to return a timestream of values in the pixels observed.
        """
        # Make sure that the polarization input is an iterable.
        try:
            polarization = list(polarization)
        except TypeError:
            polarization = [polarization]

        # Allow the user to enter somthing like -1 to automatically select all polarizations.
        if polarization[0] < 0 or polarization == list('all'):
            polarization = self.polarizations

        ##         skystreams = []
        # for index, pol in enumerate(polarization):
        ##             if pol in tqu_index: pol=tqu_index[pol]
        ##             skystreams.append( self[pol].skyStream(pointings) )

        return PolarizedSkyStream([self[pol].skyStream(pointings) for pol in polarization], polarizations=polarization)
        # return np.array([dict([(pol,self[pol][pointing]) for pol in polarization]) for pointing in pointings])
        # return dict([(pol,self[pol].skyStream(pointings)) for pol in polarization])

    ########################################################################################################
    def iterSkyStream(self, pointings, polarization='all'):
        """
        Create a generator which will return elements pointed to by an input PointingSequence (or equivalent).
        """
        # Make sure that the polarization input is an iterable.
        try:
            polarization = list(polarization)
        except TypeError:
            polarization = [polarization]

        # Allow the user to enter somthing like -1 to automatically select all polarizations.
        if polarization[0] < 0 or polarization == list('all'):
            polarization = self.polarizations

        return dict([(pol, self[pol].iterSkyStream(pointings)) for pol in polarization])
        # for pointing in pointings:
        #    yield self.map[pointing]

    ########################################################################################################
    def setMapWeight(self, weight):
        """
        This function sets a weight for the map as a whole. It does this by multiplying the pixel weights
        by the input "weight" number. If a map-wide weight has already been set, it's divided out
        before multiplying by the new weight.

        INPUTS
            weight: (float) A new global map weight.

        OUTPUT
            None, but the contents of this object are modified.
        """
        # Multiply the pixel weights by the new map weight, dividing out any old map weight.
        self.weight *= weight / self.map_weight

        # If the pixel weights are included in the map values, then also multiply that.
        if self.weighted_map:
            for _map in self.pol_maps.itervalues():
                _map *= weight / self.map_weight

        # Store the new map weight.
        self._map_weight = weight

    def getMapWeight(self):
        """
        Returns the self.map_weight.
        """
        return getattr(self, '_map_weight', 1.0)
    map_weight = property(getMapWeight, setMapWeight)

    ########################################################################################################
    def removeWeight(self, in_place=True, det_cut_val=0):
        """
        If the pixels of this map are value*weight, instead of just the value, this function will
        remove the weights to give a PolarizedMap with unweighted pixels.

        INPUTS
            in_place [True]: (bool) If True, this object's contents will be modified. Otherwise,
                the unweighted map will be a copy of this one, and this object will be unchanged.
            det_cut_val [0]: (float) If the determinant of the weight pixel is less than this value,
                treat it as a T-only pixel, dividing T by the TT weight and setting the Q and U result to 0.
        OUTPUT
            The unweighted PolarizedMap
        """
        if in_place:
            new_map = self
        else:
            new_map = self.copy()

        # If this map is already unweighted, we have nothing to do.
        if not getattr(self, 'weighted_map', False):
            return new_map

        # Define some variables needed for the upcoming block of C-code.
        recon_t = new_map['T'].map
        recon_q = new_map['Q'].map
        recon_u = new_map['U'].map
        ny, nx = new_map.active_shape[0], new_map.active_shape[1]
        lower_left_y, lower_left_x = new_map.padding
        stokes_map = new_map.weight
        pixel_map = new_map.getStokesMatrixContents()  # weighted map, shape (ny, nx, 1, 3) where last index is T,Q,U

        # Increment a counter if we run into a pixel with zero determinant. Wrapping the integer
        # in an array is the easiest way for me to pass data back and forth to the C code.
        zero_det = np.zeros(1, dtype=int)

        det_cutoff = np.zeros(1, dtype='float64') + det_cut_val

        # Zero the map contents before we start.
        recon_t[:, :] = 0.
        recon_q[:, :] = 0.
        recon_u[:, :] = 0.

        c_code = """
        #line 2222 "sky.py"

        // The inverse of A is the transpose of the cofactor matrix times the reciprocal of the determinant of A
        long double cof[3][3];  // the transpose of the cofactor matrix (the [1][0] element is *negative* determinant of the the 2x2 matrix with the 0th row and 1st column crossed out)
        int y_start = lower_left_y;
        int x_start = lower_left_x;
        int y_end = int(ny)+y_start;
        int x_end = int(nx)+x_start;
        for (int iy=y_start; iy<y_end; ++iy) {
           for (int ix=x_start; ix<x_end; ++ix) {
              //First, find the inverse of the stokes matrix.
              cof[0][0] = stokes_map(iy,ix,1,1)*stokes_map(iy,ix,2,2)-stokes_map(iy,ix,1,2)*stokes_map(iy,ix,2,1);
              cof[1][0] = stokes_map(iy,ix,1,2)*stokes_map(iy,ix,2,0)-stokes_map(iy,ix,1,0)*stokes_map(iy,ix,2,2);
              cof[2][0] = stokes_map(iy,ix,1,0)*stokes_map(iy,ix,2,1)-stokes_map(iy,ix,1,1)*stokes_map(iy,ix,2,0);
              long double det = fabs(stokes_map(iy,ix,0,0)*cof[0][0] + stokes_map(iy,ix,0,1)*cof[1][0] + stokes_map(iy,ix,0,2)*cof[2][0]);
              if (fabs(det) <= det_cutoff(0) && pixel_map(iy, ix, 0, 0)!=0) {
                zero_det(0)+=1;
                if (stokes_map(iy,ix,0,0)>0)  // if TT weight is non-zero, treat this as a T-only pixel for unweighting. Leave Q and U zero.
                   recon_t(iy, ix) += pixel_map(iy, ix, 0, 0) / stokes_map(iy,ix,0,0);
                continue;
                }
              else if (det==0) {
                  continue;
              }

              cof[0][0] /= det;
              cof[0][1] = (stokes_map(iy,ix,0,2)*stokes_map(iy,ix,2,1)-stokes_map(iy,ix,0,1)*stokes_map(iy,ix,2,2))/det;
              cof[0][2] = (stokes_map(iy,ix,0,1)*stokes_map(iy,ix,1,2)-stokes_map(iy,ix,0,2)*stokes_map(iy,ix,1,1))/det;
              cof[1][0] /= det;
              cof[1][1] = (stokes_map(iy,ix,0,0)*stokes_map(iy,ix,2,2)-stokes_map(iy,ix,0,2)*stokes_map(iy,ix,2,0))/det;
              cof[1][2] = (stokes_map(iy,ix,0,2)*stokes_map(iy,ix,1,0)-stokes_map(iy,ix,0,0)*stokes_map(iy,ix,1,2))/det;
              cof[2][0] /= det;
              cof[2][1] = (stokes_map(iy,ix,0,1)*stokes_map(iy,ix,2,0)-stokes_map(iy,ix,0,0)*stokes_map(iy,ix,2,1))/det;
              cof[2][2] = (stokes_map(iy,ix,0,0)*stokes_map(iy,ix,1,1)-stokes_map(iy,ix,0,1)*stokes_map(iy,ix,1,0))/det;

              for (int ipix=0; ipix<3; ++ipix) {
                 recon_t(iy, ix) += cof[0][ipix]*pixel_map(iy, ix, 0, ipix);
                 recon_q(iy, ix) += cof[1][ipix]*pixel_map(iy, ix, 0, ipix);
                 recon_u(iy, ix) += cof[2][ipix]*pixel_map(iy, ix, 0, ipix);
                 }
              } //end for (loop over ix)
        } //end for (loop over iy)
        """
        weave.inline(c_code, ['ny', 'nx', 'stokes_map', 'pixel_map', 'zero_det',
                              'recon_t', 'recon_q', 'recon_u', 'lower_left_x', 'lower_left_y', 'det_cutoff'],
                     type_converters=weave.converters.blitz)

        zero_det = zero_det[0]  # Strip the number out of the array.
        #if zero_det: warnings.warn('Enountered '+str(zero_det)+' pixels with zero-determinant matrices!', Warning)
        # Every polarized map has a few of these on the edges of coverage. Stop warning about them!

        new_map.weighted_map = False
        new_map.setMapAttr('weighted_map', False)

        return new_map
    removePixelWeight = removeWeight  # Define a better-named alias for removeWeight

    ########################################################################################################
    def addPixelWeight(self, in_place=True):
        """
        The opposite of removeWeight, this function returns a map in which the value in each
        map pixel is map*weight.

        INPUTS
            in_place [True]: (bool) If True, this object's contents will be modified. Otherwise,
                the unweighted map will be a copy of this one, and this object will be unchanged.

        OUTPUT
            The pixel-weighted PolarizedMap

        EXCEPTIONS
            None

        AUTHOR
            Stephen Hoover, 31 July 2013
        """
        if in_place:
            new_map = self
        else:
            new_map = self.copy()

        # If this map is already unweighted, we have nothing to do.
        if getattr(self, 'weighted_map', False):
            return new_map

        # Define some variables needed for the upcoming block of C-code.
        recon_t = new_map['T'].map
        recon_q = new_map['Q'].map
        recon_u = new_map['U'].map
        ny, nx = new_map.active_shape[0], new_map.active_shape[1]
        lower_left_y, lower_left_x = new_map.padding
        stokes_map = new_map.weight
        pixel_map = new_map.getStokesMatrixContents()

        # Increment a counter if we run into a pixel with zero determinant. Wrapping the integer
        # in an array is the easiest way for me to pass data back and forth to the C code.
        zero_det = np.zeros(1, dtype=int)

        # Zero the map contents before we start.
        recon_t[:, :] = 0.
        recon_q[:, :] = 0.
        recon_u[:, :] = 0.

        c_code = """
        #line 2323 "sky.py"

        int y_start = lower_left_y;
        int x_start = lower_left_x;
        int y_end = int(ny)+y_start;
        int x_end = int(nx)+x_start;
        for (int iy=y_start; iy<y_end; ++iy) {
           for (int ix=x_start; ix<x_end; ++ix) {
              for (int ipix=0; ipix<3; ++ipix) {
                 recon_t(iy, ix) += stokes_map(iy,ix,0,ipix)*pixel_map(iy, ix, 0, ipix);
                 recon_q(iy, ix) += stokes_map(iy,ix,1,ipix)*pixel_map(iy, ix, 0, ipix);
                 recon_u(iy, ix) += stokes_map(iy,ix,2,ipix)*pixel_map(iy, ix, 0, ipix);
                 }
              } //end for (loop over ix)
        } //end for (loop over iy)
        """
        weave.inline(c_code, ['ny', 'nx', 'stokes_map', 'pixel_map', 'zero_det',
                              'recon_t', 'recon_q', 'recon_u', 'lower_left_x', 'lower_left_y'],
                     type_converters=weave.converters.blitz)

        zero_det = zero_det[0]  # Strip the number out of the array.
        if zero_det:
            warnings.warn('Enountered ' + str(zero_det) + ' pixels with zero-determinant matrices!', Warning)

        new_map.weighted_map = True
        new_map.setMapAttr('weighted_map', True)

        return new_map

    ########################################################################################################
    def flattenPol(self, in_place=True, allow_weighted=False, verbose=False, debug=False):
        """
        We measure Q,U along the e_{\theta}, e_{\phi} coordinate directions, so but for Fourier
        transforms we want to have Q,U measured along e_{x}, e_{y} of our projected patch.
        This function will rotate Q,U from the e_{\theta}, e_{\phi} basis to e_{x}, e_{y}.

        INPUTS
            in_place [True]: (bool) If True, this object's contents will be modified. Otherwise,
                the rotated map will be a copy of this one, and this object will be unchanged.

            allow_weighted [False]: (bool) We should flatten polarization angles after the weights
                have been removed. Set this to True to let us do it anyway.

            verbose [False]: (bool) If True, print status messages to screen. For debug.

        OUTPUT
            The flattened PolarizedMap

        EXCEPTIONS
            ValueError if this map hasn't had removeWeight run yet.
        """
        if getattr(self, 'weighted_map', True) and not allow_weighted:
            raise ValueError(
                "flattenPol must be called on maps after the weight has been removed. Call removeWeight() first.")

        if in_place:
            new_map = self
        else:
            new_map = self.copy()

        # If this map is already flattened, we have nothing to do.
        if getattr(self, 'flattened_pol', False):
            if verbose:
                print "This PolarizedMap has already been flattened. No need to do anything else."
            return new_map

        # Note that "ys" is the RA coordinate (which is the second value in map.shape)
        ys, xs = np.meshgrid(np.arange(0, new_map.shape[1]), np.arange(0, new_map.shape[0]))
        # pix2Ang needs to ge the "Dec" coordinate first, but gives back RA first
        ras, dcs = new_map.pix2Ang(np.vstack([xs.flatten(), ys.flatten()]))

        # avoid branch cut at ra=0
        ras += 180. - new_map.center[0]
        ras = np.mod(ras, 360.)
        assert(np.max(ras) < 350.)  # sanity test that no pixels are close to branch cut.
        assert(np.min(ras) > 10.)

        ras = ras.reshape(new_map.shape)
        dcs = dcs.reshape(new_map.shape)

        thts = np.pi / 2. - dcs * np.pi / 180.
        phis = ras * np.pi / 180.
        del ras, dcs

        thts_grd_x, thts_grd_y = np.gradient(thts)
        phis_grd_x, phis_grd_y = np.gradient(phis)

        sky = (new_map['Q'].map + 1.j * new_map['U'].map)
        sky *= np.exp(-1.j * (np.arctan2(thts_grd_y, thts_grd_x) + np.arctan2(-phis_grd_x, phis_grd_y)))

        new_map['Q'].map = sky.real
        new_map['U'].map = sky.imag

        new_map.flattened_pol = True

        if verbose:
            print "Finished adjusting the Q and U components of this map."

        if not debug:
            return new_map
        else:
            angles = {'thts_grd_x': thts_grd_x,
                      'thts_grd_y': thts_grd_y,
                      'phis_grd_x': phis_grd_x,
                      'phis_grd_y': phis_grd_y}
            return new_map, angles

    ########################################################################################################
    def subtract(self, other, noise_estimation=True, sub_with_weight=False):
        """
        Subtract two polarized maps. Unlike with addition, where we take a weighted average of the two maps,
        subtraction with default options is simply (map1 - map2)/2, for each of T, Q, U.
        This can be overriden by setting sub_with_weight to True.
        In either case, the weight of the resulting
        map is set to be the sum of the weights of the two input maps.

        If the input maps are weighted, we must divide out the weights before subtracting.
        Otherwise coherent (CMB) signals will not necessarily subtract out.
        (The input maps will not be modified.)

        If you want a "first half minus second half" noise estimate (or left-right, etc.),
        the right thing to do is coadd each half *with weights*,
        then unweight the two half-season maps maps,
        and finally call first.subtract(second, noise_estimation=True, sub_with_weight=False)
        This will properly subtract out coherent (CMB) signals,
        and normalize the result to reflect the noise present in the coadd of the two halves.

        INPUTS
            other: (PolarizedMap) The map which you wish to subtract from this one.

            noise_estimation [True]: (bool) If True, divide the difference map by 2.
                This gives an estimate of the noise in the corresponding sum (i.e. weighted coadd) map.
                When True, you can think of this as assigning each unweighted map a fake weight of 1,
                doing a weighted-subtraction, and then unweighting (by the summed weight of 2).
                If True, the result for two independent noise maps will have sqrt(2) *less* noise.
                If you set this equal to False, the result will have sqrt(2) *more* noise than each half-map.

            sub_with_weight [False]: (bool) If True, do subtraction with weights, i.e.
                (weight1 * map1 - weight2 * map2) / (weight1 + weight2).
                This argument calls the "coadd" function to do the work.
                The "noise_estimation" argument will be ignored if "sub_with_weight" is True.
                WARNING: There is *no* good reason to subtract with weight.
                         This will *not* subtract coherent (CMB) signals.

            skip_check [False]: (bool) If True, skip checking whether the two maps have the same parameters.

        OUTPUT
            A PolarizedMap which is the difference of this map minus the other map.
            The resulting map will be unweighted, regardless of whether the input maps were weighted.

        EXCEPTIONS
            ValueError if something important (shape, center, weighted-ness, etc.) is different about the two input maps.

        AUTHOR
            Stephen Hoover, 24 May 2013
        """
        if sub_with_weight:
            print 'WARNING: There is *no* good reason to subtract with weight. This will *not* subtract coherent (CMB) signals.'
            return self.coadd(other, subtract=True, sub_with_weight=sub_with_weight)

        if other != self:
            raise ValueError("I can only add two maps if they have the same parameters!")

        # If the input maps have not had their weights divided out, we need to get unweighted
        # maps for the computation.
        if getattr(other, 'weighted_map', False) or getattr(self, 'weighted_map', False):
            this_map = self.removeWeight(in_place=False)
            other_map = other.removeWeight(in_place=False)
        else:
            this_map = self
            other_map = other

        # Do we divide by two or not?
        normalization_factor = (0.5 if noise_estimation else 1.0)

        # Subtract the T, Q, U maps from each other.
        # The weight of the difference map will be the sum of the two input map weights.
        if ((self.weight is None) or (other.weight is None)):
           summed_weight = None
        else:
            summed_weight = self.weight + other.weight 
        difference_map = type(self)(maps=[normalization_factor * (this_map.pol_maps[pol].map - other_map.pol_maps[pol].map) for pol in this_map.pol_maps],
                                    polarizations=[pol for pol in this_map.pol_maps],
                                    **self.initArgs(weight=summed_weight, weighted_map=False))

        if (hasattr(self, "observation_dates") and hasattr(other, "observation_dates")):
            difference_map.observation_dates = list(set(self.observation_dates)| set(other.observation_dates))
        return difference_map


    difference = subtract  # Alias the "subtract" function.

    ########################################################################################################
    def coadd(self, other, subtract=False, sub_with_weight=False, processing_dtype=np.float64):
        """
        Sum two polarized maps together, along with their weights.
        For a polarized map, each map pixel is a 3x1 (T,Q,U) matrix and
        each pixel weight is a 3x3 matrix.

        Takes the weights into account in both cases:
        Handles weighted maps by simply summing the weighted values:
            map1 + map2
        Handles unweighted maps by taking a weighted average:
            (weight1 * map1 + weight2 * map2) / (weight1 + weight2).

        In both cases, the 3x3 weight matrix is simply summed for each pixel.

        INPUTS
            other : (PolarizedMap) The map which we will add to this one. Must have the same parameters
                (same center, same resolution, etc.).

            subtract [False]: (bool) If True, call the "subtract" function instead unless sub_with_weight is True.
                This is here for backwards compatibility.

            sub_with_weight [False]: (bool) Do subtraction with weights,
                simply flipping the sign in either of the two expressions above.
                If sub_with_weight is True, the state of the "subtract" argument
                will be ignored (we will subtract even if "subtract" is False.)
                (This is in "coadd" because it's just a matter of
                flipping a plus to a minus in the code.)

            processing_dtype [np.float64]: (dtype) Use this type of data for the processing
                (we want greater precision when doing the arithmetic). Used only if these
                aren't weighted maps, and we need to multiply values by weights before
                adding them together.

        OUTPUT
            A new PolarizedMap object (or object of the same type as 'self', for subclasses).

        EXCEPTIONS
            ValueError if the two maps have different parameters (e.g. center or reso_arcmin).
        """
        if other != self:
            # We end up here, for example, if one is a weighted_map and the other is not.
            raise ValueError("I can only add two maps if they have the same parameters!")

        if subtract and not sub_with_weight:
            # "Normal", i.e. map1-map2, subtraction goes through the "subtract" function.
            return self.subtract(other)
        elif sub_with_weight:
            # If we want to "subtract with weights", then we're subtracting.
            subtract = True

        # If the input maps have not had their weights divided out, keep that status for the coadded map
        # (and don't multiply by the weights before coadding, obviously).
        if getattr(other, 'weighted_map', False) or getattr(self, 'weighted_map', False):
            divide_weights = False
        else:
            divide_weights = True

        # Addition of weights works the same whether they've been divided out of the map data or not.
        this_weight = self.weight
        other_weight = other.weight

        if divide_weights:
            # To be extra-careful, we'll change the data type to something higher precision before
            # we do all of this matrix math.
            return_dtype = self.pol_maps[0].map.dtype  # Output a PolarizedMap with the same data type.

            summed_weight = this_weight.astype(processing_dtype) + other_weight.astype(processing_dtype)

            #this_iqu_map = np.empty(np.append(self.map.shape, [3,1]), dtype=self.map.dtype)
            this_i, this_q, this_u = self['I'].map, self['Q'].map, self['U'].map
            map_sign = -1.0 if subtract else 1.0
            other_i, other_q, other_u = map_sign * other['I'].map, map_sign * other['Q'].map, map_sign * other['U'].map
            #other_weight = ( -other.weight if subtract else other.weight )
            #other_iqu_map = np.empty(np.append(self.map.shape, [3,1]), dtype=self.map.dtype)
            summed_map = np.zeros(np.append(this_i.shape, [1, 3]), dtype=processing_dtype)

            # Keep track of the number of times we get a zero determinant in a pixel with non-zero sum.
            zero_det = np.array([0])

            n_pix_y, n_pix_x = this_i.shape

            c_code = """
            #line 2551 "sky.py"
            long double cof[3][3]; // A temporary variable
            long double summed_pixel[3];

            for (int iy=0; iy<int(n_pix_y); ++iy) {
                for (int ix=0; ix<int(n_pix_x); ++ix) {
                    for (int i=0; i<3; ++i) {

                        summed_pixel[i] = ( this_weight(iy,ix,i,0)*this_i(iy,ix) +
                                                this_weight(iy,ix,i,1)*this_q(iy,ix) +
                                                this_weight(iy,ix,i,2)*this_u(iy,ix)    ) +
                                              ( other_weight(iy,ix,i,0)*other_i(iy,ix) +
                                                other_weight(iy,ix,i,1)*other_q(iy,ix) +
                                                other_weight(iy,ix,i,2)*other_u(iy,ix)  );

                    } // end for (loop over i 0 to 3)

                    // Take the inverse of the weight matrix in this pixel.
                    cof[0][0] = summed_weight(iy,ix,1,1)*summed_weight(iy,ix,2,2)-summed_weight(iy,ix,1,2)*summed_weight(iy,ix,2,1);
                    cof[1][0] = summed_weight(iy,ix,1,2)*summed_weight(iy,ix,2,0)-summed_weight(iy,ix,1,0)*summed_weight(iy,ix,2,2);
                    cof[2][0] = summed_weight(iy,ix,1,0)*summed_weight(iy,ix,2,1)-summed_weight(iy,ix,1,1)*summed_weight(iy,ix,2,0);
                    double det = fabs(summed_weight(iy,ix,0,0)*cof[0][0] + summed_weight(iy,ix,0,1)*cof[1][0] + summed_weight(iy,ix,0,2)*cof[2][0]);
                    if (det==0 && summed_pixel[0]!=0) {
                      zero_det(0)+=1;
                      if (summed_weight(iy,ix,0,0)>0)
                         summed_map(iy, ix,0,0) += summed_pixel[0] / summed_weight(iy,ix,0,0);
                      continue;
                      }
                    else if (det==0) {
                        continue;
                    }

                    cof[0][0] /= det;
                    cof[0][1] = (summed_weight(iy,ix,0,2)*summed_weight(iy,ix,2,1)-summed_weight(iy,ix,0,1)*summed_weight(iy,ix,2,2))/det;
                    cof[0][2] = (summed_weight(iy,ix,0,1)*summed_weight(iy,ix,1,2)-summed_weight(iy,ix,0,2)*summed_weight(iy,ix,1,1))/det;
                    cof[1][0] /= det;
                    cof[1][1] = (summed_weight(iy,ix,0,0)*summed_weight(iy,ix,2,2)-summed_weight(iy,ix,0,2)*summed_weight(iy,ix,2,0))/det;
                    cof[1][2] = (summed_weight(iy,ix,0,2)*summed_weight(iy,ix,1,0)-summed_weight(iy,ix,0,0)*summed_weight(iy,ix,1,2))/det;
                    cof[2][0] /= det;
                    cof[2][1] = (summed_weight(iy,ix,0,1)*summed_weight(iy,ix,2,0)-summed_weight(iy,ix,0,0)*summed_weight(iy,ix,2,1))/det;
                    cof[2][2] = (summed_weight(iy,ix,0,0)*summed_weight(iy,ix,1,1)-summed_weight(iy,ix,0,1)*summed_weight(iy,ix,1,0))/det;

                    // Multiply the inverse weight matrix by the sum of the weighted pixels.
                    for (int i=0; i<3; ++i) {
                        for (int j=0; j<3; ++j) {
                            summed_map(iy,ix,0,j) += cof[j][i]*summed_pixel[i];
                        }
                    }

                } // end for (loop over ix)
            } // end for (loop over iy)
            """
            weave.inline(c_code, ['this_i', 'this_q', 'this_u', 'other_i',
                                  'other_q', 'other_u', 'this_weight', 'other_weight',
                                  'summed_map', 'summed_weight', 'zero_det',
                                  'n_pix_y', 'n_pix_x'],
                         type_converters=weave.converters.blitz)

            coadded_map = type(self)(maps=[summed_map[..., ..., 0, 0], summed_map[..., ..., 0, 1], summed_map[..., ..., 0, 2]],
                                     polarizations=['T', 'Q', 'U'],
                                     **self.initArgs(weight=summed_weight))
            coadded_map.changeDType(return_dtype, weight_dtype=return_dtype)
        else:
            summed_weight = this_weight + other_weight
            map_sign = -1.0 if subtract else 1.0

            coadded_map = type(self)(maps=[self.pol_maps['T'].map + map_sign * other.pol_maps['T'].map,
                                           self.pol_maps['Q'].map + map_sign * other.pol_maps['Q'].map,
                                           self.pol_maps['U'].map + map_sign * other.pol_maps['U'].map],
                                     polarizations=['T', 'Q', 'U'],
                                     **self.initArgs(weight=summed_weight))

        if hasattr(self, "observation_dates") and hasattr(other, "observation_dates"):
            coadded_map.observation_dates = list(set(self.observation_dates) | set(other.observation_dates))

        return coadded_map

    ########################################################################################################
    def thresholdMap(self, threshold=1):
        """
        For all maps in self.pol_maps: makes any point in
        the map where abs(map) > threshold equal to zero.

        INPUTS
            threshold[1]: Any points within the map above this point will be
                          set to zero.

        OUTPUT
            Replaces all maps in map.pol_maps with thresholded maps.
        """
        for _map in self.pol_maps.keys():
            for _i in range(0, len(self.pol_maps[_map][:, 0])):
                _row = self.pol_maps[_map][_i]
                to_zero_inds = np.where(np.abs(_row) - threshold > 0)
                _row[to_zero_inds] = 0
                self.pol_maps[_map][_i] = _row

    ########################################################################################################
    def copy(self):
        copy_map = type(self)(maps=dict([(pol, self[pol].copy()) for pol in self.polarizations]),
                              weight=(self.weight.copy() if self.weight is not None else None),
                              processing_applied=self.processing_applied.copy(),
                              from_filename=getattr(self, 'from_filename', None), weighted_map=self.weighted_map,
                              flattened_pol=self.flattened_pol)
        copy_map.ft_maps = struct([(pol, self.ft_maps[pol].copy()) for pol in self.ft_maps])
        return copy_map

    ########################################################################################################
    def whatsDifferent(self, other, verbose=True):
        """
        Compares two maps to see which important properties are different. This is the comparision
        which is run before adding two maps. If this function reports anything different, the maps
        mayn't be added.

        INPUTS
            other: (PolarizedMap) The map we're comparing this one with.

            verbose [True]: (bool) If True, print details to the screen about what's different.

        OUTPUT
            A list of names of attributes which differ between the two maps.

        EXCEPTIONS
            None

        AUTHOR
            Stephen Hoover, 24 May 2013

        CHANGES
            7 Feb 2014: Print name of sub-Maps to screen as we check them. Change "verbose" default to True. SH
        """
        different_attributes = []

        # Make sure that the two maps have the same set of sub-maps.
        if set(self.polarizations) != set(other.polarizations):
            different_attributes.append('polarizations')
            if verbose:
                print "This map has polarizations=%s, and the other map has %s." % (str(self.polarizations), str(other.polarizations))

        # Check each of the sub-maps one by one.
        for pol in self.polarizations:
            if verbose:
                print "\tChecking attributes in %s sub-Map." % pol
            different_attributes += self[pol].whatsDifferent(other[pol], verbose=verbose)

        # Check whether the polarization angles have been rotated to flatten the polarization projection.
        if getattr(self, 'flattened_pol', False) != getattr(other, 'flattened_pol', False):
            different_attributes.append('flattened_pol')
            if verbose:
                print "This map is%s flattened, and the other map is%s." % (('' if getattr(self, 'flattened_pol', False) else ' not'),
                                                                            ('' if getattr(other, 'flattened_pol', False) else ' not'))
        # Check if the two maps are both weighted.
        if getattr(self, 'weighted_map', False) != getattr(other, 'weighted_map', False):
            different_attributes.append('weighted_map')
            if verbose:
                print "This map is%s weighted, and the other map is%s." % (('' if getattr(self, 'weighted_map', False) else ' not'),
                                                                           ('' if getattr(other, 'weighted_map', False) else ' not'))

        return different_attributes

    ########################################################################################################
    def __iter__(self):
        pointings = telescope.PointingSequence.pointToAll(shape=self.shape)
        return self.iterSkyStream(pointings)

    ########################################################################################################
    def __eq__(self, other):
        return (set(self.polarizations) == set(other.polarizations) and
                np.array([self[pol] == other[pol] for pol in self.polarizations]).all() and
                getattr(self, 'flattened_pol', False) == getattr(other, 'flattened_pol', False) and
                getattr(self, 'weighted_map', False) == getattr(other, 'weighted_map', False))

    ########################################################################################################
    def __ne__(self, other):
        return not self.__eq__(other)

    ########################################################################################################
    # Define mathematical operations: add, subtract, or multiply maps pixel-by-pixel, but only if the maps have the same parameters.
    # Use the definitions of addition and subtraction defined in the Map class.
    def __add__(self, other):
        return self.coadd(other)
        # if isinstance(other, PolarizedMap):
        ##    assert other==self, "I can only add two maps if they have the same parameters!"
        # return type(self)(maps=dict([(pol,self[pol]+other[pol]) for pol in self.polarizations]))
        # else:
        # return type(self)(maps=dict([(pol,Map( map=(self[pol]+other[pol]),
        # **self[pol].initArgs() )) for pol in self.polarizations]))

    def __radd__(self, other):
        return other.coadd(self)
        # if isinstance(other, PolarizedMap):
        ##    assert other==self, "I can only add two maps if they have the same parameters!"
        # return type(self)(maps=dict([(pol,other[pol]+self[pol]) for pol in self.polarizations]))
        # else:
        # return type(self)(maps=dict([(pol,Map( map=(other[pol]+self[pol]),
        # **self[pol].initArgs() )) for pol in self.polarizations]))

    def __iadd__(self, other):
        new_map = self.coadd(other)
        self.pol_maps = new_map.pol_maps
        self.polarizations = new_map.polarizations
        self.weight = new_map.weight
        # if isinstance(other, PolarizedMap):
        ##    assert other==self, "I can only add two maps if they have the same parameters!"
        #for pol in self.polarizations: self[pol] += other[pol]
        # else:
        ##    for pol in self.polarizations: self[pol] += other[pol]
        return self

    def __sub__(self, other):
        return self.subtract(other)
        # if isinstance(other, PolarizedMap):
        ##    assert other==self, "I can only subtract two maps if they have the same parameters!"
        # return type(self)(maps=dict([(pol,self[pol]-other[pol]) for pol in self.polarizations]))
        # else:
        # return type(self)(maps=dict([(pol,Map( map=(self[pol]-other[pol]),
        # **self[pol].initArgs() )) for pol in self.polarizations]))

    def __rsub__(self, other):
        return other.subtract(self)
        # if isinstance(other, PolarizedMap):
        ##    assert other==self, "I can only subtract two maps if they have the same parameters!"
        # return type(self)(maps=dict([(pol,other[pol]-self[pol]) for pol in self.polarizations]))
        # else:
        # return type(self)(maps=dict([(pol,Map( map=(other[pol]-self[pol]),
        # **self[pol].initArgs() )) for pol in self.polarizations]))

    def __isub__(self, other):
        new_map = self.subtract(other)
        self.pol_maps = new_map.pol_maps
        self.polarizations = new_map.polarizations
        self.weight = new_map.weight
        # if isinstance(other, PolarizedMap):
        ##    assert other==self, "I can only subtract two maps if they have the same parameters!"
        #for pol in self.polarizations: self[pol] -= other[pol]
        # else:
        ##    for pol in self.polarizations: self[pol] += other[pol]
        return self

    def __mul__(self, other):
        # if isinstance(other, PolarizedMap):
        #    assert other==self, "I can only multiply two maps if they have the same parameters!"
        try:
            return type(self)(maps=dict([(pol, self[pol] * other[pol]) for pol in self.polarizations]), **self.initArgs())
        except (TypeError, ValueError, KeyError, IndexError):
            return type(self)(maps=dict([(pol, self[pol] * other) for pol in self.polarizations]), **self.initArgs())
        # else:
        #    try: return type(self)(maps=dict([(pol,Map( map=(self[pol]*other[pol]), **self[pol].initArgs() )) for pol in self.polarizations]))
        # except (TypeError, ValueError, KeyError): return
        # type(self)(maps=dict([(pol,Map( map=(self[pol]*other),
        # **self[pol].initArgs() )) for pol in self.polarizations]))

    def __rmul__(self, other):
        # if isinstance(other, PolarizedMap):
        #    assert other==self, "I can only multiply two maps if they have the same parameters!"
        #    return type(self)(maps=dict([(pol,Map( map=(other[pol]*self[pol]), **self[pol].initArgs() )) for pol in self.polarizations]))
        # else:
        try:
            return type(self)(maps=dict([(pol, other[pol] * self[pol]) for pol in self.polarizations]), **self.initArgs())
        except (TypeError, ValueError, KeyError, IndexError):
            return type(self)(maps=dict([(pol, other * self[pol].map) for pol in self.polarizations]), **self.initArgs())

    def __imul__(self, other):
        # if isinstance(other, PolarizedMap):
        #    assert other==self, "I can only multiply two maps if they have the same parameters!"
        #    for pol in self.polarizations: self[pol].map *= other[pol].map
        # else:
        try:
            for pol in self.polarizations:
                self[pol] *= other[pol]
        except (TypeError, ValueError, KeyError, IndexError):
            for pol in self.polarizations:
                self[pol] *= other
        return self

    def __abs__(self):
        return type(self)(maps=dict([(pol, Map(map=np.abs(self[pol].map), **self[pol].initArgs())) for pol in self.polarizations]))


##########################################################################################
##########################################################################################
# Map still needs some work to resolve use of padding. (IS THIS STILL TRUE? - SH)
class Map:
    """
    A Map holds a two-dimensional data set. This object is intended to hold a
    map of data on the sky. It holds one frequency and one polarization of data.
    A map knows what its resolution is, where it's centered on the sky, and
    what kind of data (band, polarization, float vs int) it holds.

    Note that each data point of a Map may itself be an array.

    A Map behaves like the array data which it holds. Math and data member access
    functions act directly on the data array. Equality operators on pairs of Maps
    test whether they have the same size and resolution.

    See PolarizedMap for an object which bundles multiple Maps together into
    a fully polarized description of the sky in one band.

    Attributes which may be included in a Map:
        band : (str) In GHz, the frequency band used when making this map.
        name : (str) A name of this map, typically the source or field being observed.
        polarization : (str) Single letter, typically "T", "Q", or "U" specifying which Stokes parameter is in the map.
        projection : (int) An integer denoting the projection that this map is using. See sky.ang2Pix or sky.pix2Ang's documention for more information.
        reso_arcmin : (float) In arcminutes, the width of a pixel.
        units : (str) The units of this map.
        n_channels : (int) If present, the number of data channels used to make this map.
        n_scans : (int) If present, the number of individual scans included, at least in part, in this map.
        weight : (2D array) The weight of each pixel in this map.
        center : (2-element array of floats) In degrees, the [RA, dec] of the map's center.
        active_shape : (2-element array of floats) In pixels, the shape of the map, excluding padding.
        shape : (2-element array of floats) In pixels, the shape of the map, including padding.
        padding : (2-element array of floats) In pixels, the width of padding on the sides of the map.
        processing_applied : (struct) If present, lists processing applied to the data reader used to make this map.
        map : (2D array) The data!
    """

    @classmethod
    def read(cls, filename, file_type=None):
        """
        Checks the filename extension and calls the appropriate reader.
        If the "file_type" input is "FITS" or "HDF5" (case insensitive), then
        we read the file as being in that format, regardless of extension.
        """
        # If we got an input for 'file_type', check that it makes sense.
        if file_type is not None and not (file_type.lower().startswith('h') or file_type.lower().startswith('f')):
            raise ValueError("Sorry, I don't recognize the \"%s\" file file_type." % file_type)
        elif file_type is None:
            file_type = filename.split('.')[-1]

        # Case insensitive
        file_type = file_type.lower()

        if file_type.lower().startswith('h'):
            return cls.readFromHDF5(filename)
        elif file_type.lower().startswith('f'):
            return cls.readFromFITS(filename)

        raise ValueError("I don't recognize filename \"%s\"!" % filename)

    @classmethod
    def readFromFITS(cls, filename):
        """
        Read a Map object in from an SPT-formatted FITS file. The
        map must be the only thing stored in the given FITS file.
        """
        fits_in = fits.readSptFits(filename)
        # Maybe this is a PolarizedMap?
        if len(fits_in) >= 3 and 't' in fits_in and 'q' in fits_in and 'u' in fits_in:
            return PolarizedMap.readFromFITS(filename)
        elif len(fits_in) != 1:
            raise ValueError('File ' + filename + ' has more than one object in it!')
        else:
            try:
                processing_applied = fits_in[0].processing_applied
            except AttributeError:
                processing_applied = tools.struct()

            # Convert processing_applied arrays to structs:
            try:
                processing_applied.dtype  # Check for arrays by calling on 'dtype'.
                processing_applied = tools.struct([(_proc, tools.struct()) for _proc in processing_applied])
            except AttributeError:
                pass
            fits_in[0].processing_applied = processing_applied

            # Store the filename in the map.
            fits_in[0].from_filename = filename

            return fits_in[0]

    @classmethod
    def readFromHDF5(cls, filename):
        """
        Read a Map object in from an SPT-formatted HDF5 file. Assume that the
        map is the only thing stored in the given file.

        Equivalent to
            hdf.readSptHDF5(filename)[0]
        """
        new_map = hdf.readSptHDF5(filename)

        # The file might be read out either as a Map (or PolarizedMap) object, or
        # as a struct with one entry containing a Map. Either way, we just
        # want the object.
        if isinstance(new_map, dict):
            new_map = new_map[0]

        # Store the filename in the map.
        new_map.from_filename = filename

        return new_map

    def __init__(self, shape=None, active_shape=None, padding=None, map=None, weight=None,
                 projection=5, reso_arcmin=1., band="unknown", polarization="T", units=None, n_scans=None,
                 center=(0, 0), name=None, data_type=float_type, processing_applied=None, **extra_info):
        """
        Initialize a Map either with an array of data, or with a shape in pixels for a blank map.
        At least one of "shape", "active_shape", and "map" must be supplied.
        If a "map" input is supplied, its dimensions will take precedence over any "shape" inputs.
        ("shape" may still be used to define the padding width, however.)

        INPUTS
            shape [None]: (array) The total shape of the map, in terms of pixels.
                Order: Note (ny,nx) is in "array order", not "image order".
                Must have at least two elements, but may have more.

            active_shape [None]: (array) The shape, in pixels, of the map area which holds real data.
                If None, it will be defined based on a supplied shape and padding.

            padding [None]: (array) The width around the edges of the map which are set aside for
                padding (padding can be used, for example, to hold fake data satisfying periodic
                boundary conditions). If None, it will be defined based on a supplied "active_shape"
                input, or default to (0,0).

            map [None]: (array) The array of data. If not supplied, the Map object will be
                initialized with an array of zeros in the shape specified by other inputs.

            weight [None]: (array) An array with the same size as map. This array holds the
                weight of each pixel in the map, used for coadding maps together. If not
                supplied, every element will be set to 1.

            projection [5]: (int) Not currently used. Intended to be the map projection used in this Map.

            reso_arcmin [1.]: (float) The width of each array pixel, in arcminutes. We
                assume that the pixels are squares.

            band ['unknown']: (string) Specifies the band in which this map's data were
                taken. Typically '150' or '95'.

            polarization ['none']: (string) Specifies the polarization state of the data
                contained in this Map. Typically 'T', 'Q', 'U', 'E', or 'B'.

            units [None]: (string) The units of the timestreams which made this Map.

            n_scans [None]: (int) If supplied, the number of scans in the observation which made this Map.

            center [(0,0)]: (array, list, or tuple with 2 elements) Coordinates of this
                map's center, in degrees of (RA, declination).

            name [None]: (string) Optional. If provided, 'name' should be some sort of identifier for this map.

            data_type [float_type]: (type) Only used if a "map" input is not supplied. This is
                the type of data in the initial array of zeros that we create.

            processing_applied [None]: (dict) Defaults to an empty struct. Represents functions and
                arguments applied to the data which went into this map.

            **extra_info : Any extra keywords supplied will be stored as attributes of the Map object.
        """
        self.__dict__.update(extra_info)
        self.projection = projection
        self.reso_arcmin = reso_arcmin
        self.band = band
        self.polarization = polarization
        self.center = np.asarray(center)
        self.name = name
        self.units = units
        self.n_scans = n_scans

        if padding is None:
            if active_shape is not None and shape is not None:
                padding = (np.asarray(shape)[:2] - np.asarray(active_shape)[:2]) / 2
            else:
                padding = np.array([0, 0])
        self.padding = np.asarray(padding)

        if shape is None and active_shape is not None:
            shape = np.array(active_shape) + 2 * np.array(padding)

        if map is None:
            assert len(shape) >= 2, "The shape input must be the [x,y] shape of the map!"
            self.map = np.zeros(shape, dtype=data_type)
        else:
            assert map.ndim >= 2, "The map must have at least 2 dimensions!"
            self.map = map

        if weight is None:
            self.weight = None
        elif np.isscalar(weight):
            self.weight = np.zeros(self.map.shape, float_type) + weight
        else:
            assert weight.shape == self.map.shape, "The weight array must be the same shape as the map!"
            self.weight = weight

        if active_shape is None:
            active_shape = self.shape - 2 * self.padding
        self.active_shape = np.array(active_shape)

        if processing_applied is None:
            processing_applied = tools.struct()
        self.processing_applied = processing_applied

        # Create a space to store FFTs of maps.
        self.ft_maps = struct()

    def pixelInterp(self, pix_y, pix_x, out_of_bounds=0.):
        """
        Returns a linear interpolation of the map value at the input point(s) pix_y, pix_x.

        INPUTS
            pix_y, pix_x: (floats or NumPy arrays) Fractional pixel locations at which to
                interpolate. If interpolating at multiple points, pix_y and pix_x should
                be ordered NumPy arrays of the y and x pixel coordinates.

        out_of_bounds [0.]: (float) Give this value to the interpolated function at any
            pixel coordinates outside the bound of the array.

        OUTPUT
            The value of the map, linearly interpolated to the requested fractional pixel positions.
        """
#        c_interp = math.interp2D(self.map, pix_y, pix_x, out_of_bounds=out_of_bounds)
#        py_interp = math.interp2DPython(self.map, pix_y, pix_x, out_of_bounds=out_of_bounds)
#
#        if (c_interp != py_interp).any():
#            print "Difference found!!!"
#            pdb.set_trace()
#        return py_interp

        return math.interp2D(self.map, pix_y, pix_x, out_of_bounds=out_of_bounds)
#
#        floor_x = np.floor(pix_x)
#        floor_y = np.floor(pix_y)
#        ceil_x = np.ceil(pix_x)
#        ceil_y = np.ceil(pix_y)
#
#        t = (pix_x - floor_x)/(ceil_x - floor_x)
#        u = (pix_y - floor_y)/(ceil_y - floor_y)
#
#        interp_val = ( (1-t)*(1-u)*self.map[(floor_y.astype(int),floor_x.astype(int))]
#                       + t*(1-u)*self.map[(ceil_y.astype(int), floor_x.astype(int))]
#                       + t*u*self.map[(ceil_y.astype(int), ceil_x.astype(int))]
#                       +(1-t)*u*self.map[(floor_y.astype(int), ceil_x.astype(int))] )
#        return interp_val

    def pix2Ang(self, pixel_yx, wrap=True):
        """
        Returns the RA, dec of a pixel in this map.

        INPUTS
            pixel_yx: (2-tuple) The [y, x] coordinate of the pixel for which you want the RA, dec.
                "y" is the vertical, mostly-dec coordinate, and "x" is the horizontal, mostly-RA coordinate.

        OUTPUT
            A NumPy array with the [RA, dec] of the input pixel.
        """
        pix_array = np.asarray([[pixel_yx[0]], [pixel_yx[1]]])
        angle = pix2Ang(pix_array, self.center, self.reso_arcmin, self.shape, self.projection, wrap)
        return np.asarray([angle[0][0], angle[1][0]])

    def ang2Pix(self, pixel_radec, return_validity=False, **kwds):
        """
        Returns the (y, x) coordinate in this map which corresponds to a given RA, dec.

        INPUTS
            pixel_yx: (2-tuple) The [RA, dec] coordinate for which you wish to find the pixel coordinate.

            return_validity [False]: (bool) If True, return the pix_is_good array from ang2Pix, which
                specifies if each pixel is actually inside the map.

        OUTPUT
            A NumPy array with the [y, x] pixel coordinate corresponding to the input RA, dec.
        """
        pointing_sequence = ang2Pix(pixel_radec, self.center, self.reso_arcmin, self.shape,
                                    self.projection, return_validity=return_validity, **kwds)
        if np.size(np.asarray(pixel_radec)) == 2 and not return_validity:
            # If we only input one coordinate, return a single 2-element array instead of a PointingSequence.
            return np.asarray([pointing_sequence[0][0], pointing_sequence[1][0]])
        else:
            return pointing_sequence

#        ang_array = np.asarray( [[pixel_radec[0]], [pixel_radec[1]]] )
#        pix_coord, pix_is_good = ang2Pix(ang_array, self.center, self.reso_arcmin,
#                                         self.shape, self.projection, **kwds)
#        if return_validity:
#            return np.asarray( [pix_coord[0][0], pix_coord[1][0]] ), pix_is_good
#        else:
#            return np.asarray( [pix_coord[0][0], pix_coord[1][0]] )

    def initArgs(self, noshape=False, **new_args):
        """
        Outputs a dictionary of arguments which we can use to make a new Map object
        equivalent to this one. Example of intended use:
           new_map = Map( map=new_map_array, **old_map.initArgs() )

        INPUTS
            noshape [False]: (bool) Exclude the shape arguments (shape, active_shape, and padding)
                from the output dictionary.

            **new_args : Any additional arguments supplied to this function will be passed
                directly into the output, overriding any arguments with the same
                name which might otherwise be there.

        OUTPUT
            A dictionary of arguments which can be passed directly into Map's init function.
        """
        args = self.__dict__.copy()
        args.update({'data_type': self.map.dtype})
        del args['map']
        if noshape:
            del args['padding']
            del args['active_shape']
        else:
            args.update({'shape': self.shape})
        args.update(**new_args)
        return args

    ########################################################################################################
    def zeroPaddedMap(self, padding_factor):
        """
        Make a new PolarizedMap object, this one padded by the requested factor.
        """

        print('in zero map')

        if padding_factor < 1:
            raise ValueError("Padding means incrasing the size of the maps!")

        if padding_factor == 1:
            print('padding factor =1, nothing to be done')
            return self

        self.copy()
        padded_shape = [int(padding_factor * dim) for dim in self.shape]
        _map = self.map
        padded_map = np.zeros(padded_shape)
        padded_map[:self.shape[0], :self.shape[1]] = _map



        # zero padding the wright
        new_weights = np.zeros((int(padding_factor * dim), int(padding_factor * dim), 3, 3))

        if new_weights is not None:
            new_weights = np.zeros_like(padded_map)
            new_weights[:self.shape[0], :self.shape[1]] = self.weight

        # print('new_weight', new_weights.shape,padded_shape,padded_map.shape)

        return type(self)(map=padded_map, shape=padded_shape, polarization=self.polarization, weight=new_weights)

    def changeDType(self, new_dtype, weight_dtype=None):
        """
        We may wish to switch the level of precision with which our maps are stored. This function can do that.

        INPUTS
            new_type: (dtype) The type of data (e.g. np.float32, np.float64) which we wish to store in our map.
                If None, then the map data will not be altered (if we only want to change the weight array).

            weight_dtype [None]: (dtype) The type of data (e.g. np.float32, np.float64) which we wish to store
                in our weight array. If None, then the weight array will not be altered.

        OUTPUT
            None, but the dtype of the map and/or weight arrays will be changed.
        """
        if new_dtype is not None:
            self.map = self.map.astype(new_dtype)
        if weight_dtype is not None and self.weight is not None:
            self.weight = self.weight.astype(weight_dtype)

    def trimPadding(self):
        """
        Modifies the Map in place to remove the "padding" around the edges.

        INPUTS
            None

        OUTPUT
            None (This object is modified in place.)
        """
        self.map = self.map[self.padding[0]:self.padding[0] + self.active_shape[0],
                            self.padding[1]:self.padding[1] + self.active_shape[1]]
        if self.weight is not None:
            self.weight = self.weight[self.padding[0]:self.padding[0] + self.active_shape[0],
                                      self.padding[1]:self.padding[1] + self.active_shape[1]]
        self.padding = np.array([0, 0])

    def definePadding(self, padding):
        """
        Designates a part of the Map as "padding". The map contents will not be changed.
        Any previous setting of this Map's padding will be discarded.

        INPUTS
            padding: (array) The width around the map's borders which will be designated "padding".
                May be greater or less than the previous value of padding.

        OUTPUT
            None
        """
        assert (self.shape - 2 * np.asarray(padding) > 2).all(), "The Map has shape " + \
            str(self.shape) + ", but you're trying to designate " + str(padding) + " as padding!"

        self.padding = np.asarray(padding)

        self.active_shape = self.shape - 2 * self.padding

    # def setPadding(self, padding):
    #    """
    #    Modifies the Map in place to add blank padding around the edges. If any padding previously existed, it will be discarded (along with its data).
    #    """
    #    former_padding = self.padding
    #    self.padding = np.asarray(padding)
    #    self.shape = np.array(active_shape) + 2*np.array(padding)

    def applyWindow(self, window):
        """
        Equivalent to the multiplication operator.

        INPUTS
            window: (array) Must have the same dimensions as this Map's data.

        OUTPUT
            A new map, equivalent to this one, but with data equal to this
            Map's data multiplied by the input window function.
        """
        return self.map * window

    def getShape(self):
        """
        The "shape" is the first two dimensions of the map, the x,y coordinates.
        """
        return np.array(self.map.shape[0:2])
    shape = property(getShape)

    def getResoRad(self):
        """
        OUTPUT
            Returns the map resolution in radians.
        """
        return self.reso_arcmin * constants.DTOR / 60.
    reso_rad = property(getResoRad)

    def flatMap(self):
        """
        An iterator into the flattened map. Will only flatten the map over
        the first two indices; the returned array will have a shape of length
        len(map.shape) - 1

        OUTPUT
            A generator which cycles through the x, y pixels in this Map object.
        """
        for y in xrange(self.shape[0]):
            for x in xrange(self.shape[1]):
                yield self.map[y, x]
    flat = property(flatMap)

    def skyStream(self, pointings):
        """
        Uses an input PointingSequence to return a timestream of pixels observed.

        INPUTS
            pointings: (PointingSequence) Stream of pointings which we wish to extract
                from this Map.

        OUTPUT
            A SkyStream of the data in this Map.
        """
        return SkyStream(data=self.map[pointings.coords], polarization=self.polarization,
                         timestep=self.reso_arcmin / pointings.scan_speed)

    def iterSkyStream(self, pointings):
        """
        Create a generator which will return elements pointed to by an input
        PointingSequence (or equivalent).

        INPUTS
            pointings: (PointingSequence or other iterable of x,y coordinates) Stream of pointings
                which we wish to extract from this Map.

        OUTPUT
            A generator which cycles through values of map pixels in the order specified
            by the input "pointings".
        """
        for pointing in pointings:
            yield self.map[pointing]

    def getActiveMap(self):
        """
        Returns a new Map, containing the entire active area of this Map and no padding.

        INPUTS
            None

        OUTPUT
            A Map object, equivalent to this one, but with no padding.
        """
        return Map(map=self.map[self.padding[0]:self.padding[0] + self.active_shape[0],
                                self.padding[1]:self.padding[1] + self.active_shape[1]],
                   **self.initArgs(noshape=True,
                                   weight=(self.weight[self.padding[0]:self.padding[0] + self.active_shape[0],
                                                       self.padding[1]:self.padding[1] + self.active_shape[1]]
                                           if self.weight is not None
                                           else None)))

    active_map = property(getActiveMap)

    def getSubmap(self, map_shape, center_offset=[0., 0.], units='degrees'):
        """
        Returns a cutout of this map. The cutout has the same resolution and must be a subset
        of this map.

        INPUTS
            map_shape: [2 element iterable] The shape of the new map, in [Y, X] or [elevation, azimuth].

            center_offset [[0.,0.]]: [2 element iterable] How far from the center of this map should the
                cutout's map be? When in degrees or arcminutes, this is [RA, dec]. When in pixels, it's [el, az].

            units ['degrees']: (string) The units of both center_offset and map_shape (must the the same
                units for both). Valid choices are "degrees", "arcminutes" and "pixels".

        OUTPUT
            A new map with the specified dimensions and center.
        """
        # Convert inputs to pixels.
        if units.lower().startswith('deg'):
            # Convert inputs from degrees to number of pixels.
            map_radius = (np.asarray(map_shape) / 2. * 60 / self.reso_arcmin).astype(int)
            pix_center = self.ang2Pix(self.center + center_offset, round=True)
        elif units.lower().startswith('arcmin'):
            # Convert inputs from arcminutes to number of pixels.
            map_radius = (np.asarray(map_shape) / 2. / self.reso_arcmin).astype(int)
            center_offset = np.asarray(center_offset) / 60.
            pix_center = self.ang2Pix(self.center + center_offset, round=True)
        elif units.lower().startswith('pix'):
            # Inputs are already number of pixels. Just make sure that they're integer
            # arrays and change shape to half-shape.
            map_radius = (np.asarray(map_shape) / 2.).astype(int)
            center_offset = np.asarray(center_offset, dtype=int)
            pix_center = (np.asarray(self.shape) / 2. + center_offset).astype(int)
        else:
            raise ValueError("I don't recognize \"%s\" units!" % units)

        # Define the region of this map that gets used in the sub-map.
        submap_cutout = [slice(np.max([pix_center[0] - map_radius[0], 0]), np.min([pix_center[0] + map_radius[0], self.shape[0] - 1])),
                         slice(np.max([pix_center[1] - map_radius[1], 0]), np.min([pix_center[1] + map_radius[1], self.shape[1] - 1]))]

        # Get the center of the submap from the center of the pixels we actually used.
        new_center = self.pix2Ang(map(lambda x: (x.start + x.stop) / 2., submap_cutout))

        return type(self)(map=self.map[submap_cutout].copy(), **self.initArgs(noshape=True, center=new_center, weight=(self.weight[submap_cutout] if self.weight is not None else None)))

    def setSubmap(self, submap, center_offset=[0., 0.], units='degrees'):
        """
        Replaces a subsection of this map with the input submap.

        INPUTS

            submap: (2d numpy array) The submap you want to stick into the main map.

            center_offset [[0.,0.]]: [2 element iterable] How far from the center of this map should the
                cutout's map be? When in degrees or arcminutes, this is [RA, dec]. When in pixels, it's [el, az].

            units ['degrees']: (string) The units of both center_offset. Valid choices are "degrees", "arcminutes" and "pixels".

        OUTPUT
            None.  Acts on self.
        """

        # half-shape of submap
        map_radius = (np.asarray(np.shape(submap)) / 2.).astype(int)

        # Convert inputs to pixels.
        if units.lower().startswith('deg'):
            # Convert inputs from degrees to number of pixels.
            pix_center = self.ang2Pix(self.center + center_offset, round=True)
        elif units.lower().startswith('arcmin'):
            # Convert inputs from arcminutes to number of pixels.
            center_offset = np.asarray(center_offset) / 60.
            pix_center = self.ang2Pix(self.center + center_offset, round=True)
        elif units.lower().startswith('pix'):
            # Inputs are already number of pixels. Just make sure that they're integer
            # arrays and change shape to half-shape.
            center_offset = np.asarray(center_offset, dtype=int)
            pix_center = (np.asarray(self.shape) / 2. + center_offset).astype(int)
        else:
            raise ValueError("I don't recognize \"%s\" units!" % units)

        # Define the region of this map that gets used in the sub-map.
        submap_cutout = [slice(np.max([pix_center[0] - map_radius[0], 0]), np.min([pix_center[0] + map_radius[0], self.shape[0] - 1])),
                         slice(np.max([pix_center[1] - map_radius[1], 0]), np.min([pix_center[1] + map_radius[1], self.shape[1] - 1]))]

        # stick submap into self.map
        self.map[submap_cutout] = submap

    def getArea(self, square_arcminutes=True):
        """
        Returns the Map's active area.

        INPUTS
            square_arcminutes [True]: (bool) If True, the output has units of square arcminutes.
                If False, the output has units of square degrees.

        OUTPUT
            The area in the active portion of the map, in square degrees or square arcminutes.
        """
        if square_arcminutes:
            return self.active_shape[0] * self.active_shape[1] * self.reso_arcmin**2
        else:
            return self.active_shape[0] * self.active_shape[1] * self.reso_arcmin**2 / 3600.  # 3600 converts square-arcminutes to sq-deg
    area = property(getArea)
    area_arcmin = property(getArea)

    def getAreaDeg(self):
        """
        Equivalent to self.getArea(square_arcminutes=False).
        """
        return self.getArea(square_arcminutes=False)
    area_deg = property(getAreaDeg)

    def iterPoints(self):
        """
        Iterates over points in the map. Guaranteed to hit each point in the active area once
        and only once. Equivalent to self.flatMap(), except that points in
        the padded area are excluded.

        INPUTS
            None

        OUTPUT
            A generator which cycles through values of map pixels in the active area.
        """
        for y in xrange(self.padding[0], self.padding[0] + self.active_shape[0]):
            for x in xrange(self.padding[1], self.padding[1] + self.active_shape[1]):
                yield self.map[y, x]
    iter_points = property(iterPoints)

    def iterEnumPoints(self):
        """
        As iterPoints, but also returns the coordinates for each point.

        INPUTS
            None

        OUTPUT
            A generator which cycles through values of map pixels in the active area.
            The generator returns a tuple of (ycoord, xcoord, value).
        """
        for y in xrange(self.padding[0], self.padding[0] + self.active_shape[0]):
            for x in xrange(self.padding[1], self.padding[1] + self.active_shape[1]):
                yield y, x, self.map[y, x]
    iter_enum_points = property(iterEnumPoints)

    def thresholdMap(self, threshold=1):
        """
        Make any point in the map where abs(map) > threshold zero.

        INPUTS
            threshold[1]: Any points within the map above this point will be
                          set to zero.

        OUTPUT
            Replaces map with thresholded maps.
        """

        for _i in range(0, len(self.map[:, 0])):
            _row = self.map[_i]
            to_zero_inds = np.where(np.abs(_row) - threshold > 0)
            _row[to_zero_inds] = 0
            self.map[_i] = _row

    def drawFullImage(self, **additional_args):
        """
        Draw the entire Map on the screen. Wrapper for Map.drawImage with the full_map=True option.

        INPUTS
            **additional_args : Arguments to be passed to drawImage.

        OUTPUT
            None (draws to screen)
        """
        return self.drawImage(full_map=True, **additional_args)
    full_image = property(drawFullImage)

    def drawImage(self, units='arcminutes', title=None, gaussian_smooth=0, weight=False,
                  reverse=False, full_map=False, vmin=None, vmax=None, log=False, uk=False,
                  bw=True, figure=None, mask=None, subplot=111, colormap='jet', **imshow_param):
        """
        Draw the active area of the Map on the screen.

        INPUTS
            units ['pixels']: (string) 'pixels', 'degrees', 'arcminutes', or 'radec'. (RA/dec might not
                be accurate, depending on the map projection.)

            title [None]: (string) A title for the plot.

            gaussian_smooth [0]: (int) The FWHM of gaussian to smooth with before plotting.  0 means don't smooth.

            weight [False]: (bool) If True, plot the map weight instead of the map itself.

            reverse [False]: (bool) If True, plot the RA / azimuth reversed, to match standard astonomical images.

            full_map [False]: (bool) If True, plot the full map (including padding) instead of just the active area.

            vmin [None]: (float) Vertical scale (color scale) minimum. Ignored if a 'norm' argument is given.

            vmax [None]: (float) Vertical scale (color scale) maximum. Ignored if a 'norm' argument is given.

            log [False]: (bool) If True, plot the colors on a logarithmic scale. Ignored if a 'norm' argument is given.

            uk [False]: (bool) If True, plot the temperature scale in uK instead of K. Has no effect if
                the Map's units are not set to "K_CMB".

            bw [True]: (bool) If True, use a greyscale and no sub-pixel interpolation. Tijmen thinks this makes most maps easier to interpret.

            figure [None]: (int) Specifies which figure to use, clears figure before use.

            mask [None]: (2D array with same size as map) Apply this mask to the map before plotting, if not None.
                (Does not modify the contents of the Map.)

            subplot [111]: (int,int,int) or (3 consectutive numbers) draws the plot in the subplot given.

            colormap ['jet']: If bw=False, this is the colormap that will be used to plot the image.  This is broken out of the {'cmap':xxxx} option
                of imshow_param for convenience.

            **imshow_param: Extra input parameters will be passed to matplotlib.pyplot.imshow.

        OUTPUT
            None (draws to screen)

        CHANGES
            2 Dec 2013 :   Cleaned up colorbar location/labeling, weight maps of different map combos work, imshow_param is now an input dict. JTS
            13 Sept 2013: Make "gaussian_smooth" in terms of FWHM of a Gaussian. SH
            22 Nov 2013 : Switch variable name "ax" back to "axes". SH
            3 Mar 2014: Add "uk" input argument. SH
            9 Jul 2014: Remove "radec" as a separate argument, and allow 'radec' as input for 'units'. SH
        """
        if 'norm' in imshow_param:
            normalization = imshow_param.pop('norm')
        elif log:
            normalization = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            normalization = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

        # Add black & white (bw) option.
        if bw:
            imshow_param['cmap'] = 'bone'
        else:
            imshow_param['cmap'] = colormap

        # Don't let the plotter interpolate between pixels unless we explicitly ask it to.
        imshow_param.setdefault('interpolation', 'none')

        # Label the axes by degrees from the center of the map.
        if units.lower().startswith('deg') or units.lower().startswith('arcmin') or units.lower().startswith('ra'):
            _shape = self.active_shape * self.reso_arcmin / 60.
            extent = np.array([-_shape[1] / 2., _shape[1] / 2., -_shape[0] / 2., _shape[0] / 2.])
            if units.lower().startswith('deg'):
                unit_name = "Degrees"
                self._pixels_per_unit = 1. / self.reso_arcmin * 60
            elif units.lower().startswith('arcmin'):
                unit_name = "Arcminutes"
                extent *= 60
                self._pixels_per_unit = 1. / self.reso_arcmin
            elif units.lower().startswith('ra'):
                # Thinking of "RA/dec" units here. This won't be accurate on large
                # maps unless they're in proj 1.
                unit_name = 'Degrees'
                self._pixels_per_unit = 1. / self.reso_arcmin * 60
                extent = np.array([self.center[0] - _shape[1] / 2., self.center[0] + _shape[1] / 2.,
                                   self.center[1] - _shape[0] / 2., self.center[1] + _shape[0] / 2.])
            else:
                # We should never get here.
                raise ValueError("I don't recognize units of %s ." % units)
        else:
            unit_name = "Pixels"
            extent = None
            self._pixels_per_unit = 1.
        map_to_plot = self.map if full_map else self.active_map.map

        if weight:
            if self.weight is None:
                raise ValueError("I can't plot this Map's weight, because it doesn't have any!")
            map_to_plot = self.weight

        if reverse:
            # Flip the map around (on the x-axis) if requested.
            map_to_plot = map_to_plot[:, ::-1]
            if extent is not None:
                extent = [extent[1], extent[0], extent[2], extent[3]]

        # Take the absolute value if we want to plot the log.
        if log:
            map_to_plot = np.abs(map_to_plot)

        # Apply a mask function, if desired.
        if mask is not None:
            map_to_plot = map_to_plot * mask

        # Convert from K_CMB to uK if requested.
        map_units = self.units  # Alias this because we might change it.
        if uk:
            if map_units == 'K_CMB':
                map_to_plot = 1e6 * map_to_plot  # Don't use *= because we don't want to risk altering the map in place.
                map_units = '$\mu$K_CMB'

        # Create the window and plot the image. Use specified figure window if given.
        if isinstance(figure, plt.Figure):
            fig = figure
            if subplot != 111:
                plt.clf()
        elif figure:
            fig = plt.figure(figure)
            if subplot == 111:
                plt.clf()
        else:
            fig = plt.figure()
            plt.clf()
        try:
            axes = fig.add_subplot(subplot)
        except TypeError:
            axes = fig.add_subplot(*subplot)

        # The "astype" call is because imshow doesn't deal well with np.float64 types. This
        # ensures that we'll get something we can plot, even if it's an np.float64.
        # Convert from FWHM to sigma.
        gaussian_sigma = gaussian_smooth * self._pixels_per_unit / (2 * np.sqrt(2 * np.log(2)))
        image = axes.imshow(ndimage.gaussian_filter(map_to_plot, gaussian_sigma).astype(
            np.float32), extent=extent, norm=normalization, **imshow_param)
        divider = make_axes_locatable(axes)
        cb_ax = divider.append_axes("right", size='5%', pad=0.05)
        cb = plt.colorbar(image, cax=cb_ax, ax=axes)
        cb.set_label("Weight" if weight else map_units)
        if units.lower().startswith('ra'):
            axes.set_ylabel('Declination (deg)')
            axes.set_xlabel('Right Ascension (deg)')
        else:
            axes.set_ylabel(unit_name)
            axes.set_xlabel(unit_name)

        if title is None and weight:
            axes.set_title("Weights in the " + self.polarization + "-Polarization Map" +
                           (" of %s" % self.name if self.name is not None else ""))
        elif title is None:
            axes.set_title(self.polarization + "-Polarization Map" + (" of %s" %
                                                                      self.name if self.name is not None else ""))
        else:
            axes.set_title(title)

        plt.sca(axes)
        return axes
    plot = drawImage  # Alias the drawImage function.
    image = property(drawImage)

    def copy(self):
        """
        Returns a deep copy of this Map object.
        """
        return type(self)(map=self.map.copy(), **self.initArgs(weight=(self.weight.copy() if self.weight is not None else None)))

    def getTOnly(self):
        """
        Returns a deep copy of this object. Having this function in the Map lets you call
        getTOnly without needing to know if an object is a Map or PolarizedMap.
        """
        return self.copy()

    def writeToFITS(self, filename, overwrite=False):
        """
        Write a representation of this object to an SPT-formatted FITS file.
        """
        fits_struct = self.__fitsformat__()
        fits.writeSptFits(filename, overwrite, map=fits_struct)

    def writeToHDF5(self, filename, overwrite=False, use_compression=False):
        """
        Write a representation of this object to an SPT-formatted HDF5 file.
        """
        hdf.writeSptHDF5(filename, contents=self, overwrite=overwrite,
                         write_objects=True, use_compression=use_compression)

    def __fitsformat__(self):
        """
        Returns a copy of this object, formatted for output to a FITS file.
        """
        fits_struct = {'class__': tools.className(self), 'map': self.map.copy()}
        fits_struct.update(self.initArgs(noshape=False, weight=(
            self.weight.copy() if self.weight is not None else None)))
        fits_struct['data_type'] = str(fits_struct['data_type'])
        return fits_struct

    def __hdf5format__(self, full_object=True):
        """
        Outputs a dictionary of this class, suitable for packing into an HDF5 file.

        INPUTS
            full_object [True]: (bool) If True, output all attributes of this object, so
                that it can be reassembled again after readout. If False, output only the data.

        OUTPUT
            A dictionary of this Map's contents, suitable for writing to disk.
        """
        if full_object:
            hdf_data_out = {'__class__': tools.className(self), 'map': self.map.copy()}
            hdf_data_out.update(self.initArgs(noshape=False, weight=(
                self.weight.copy() if self.weight is not None else None)))
            hdf_data_out['data_type'] = str(hdf_data_out['data_type'])
        else:
            hdf_data_out = self.map

        return hdf_data_out

    @classmethod
    def __unpack_hdf5__(cls, data):
        """
        Recreate a Map object from data packed into an HDF5-formatted file.
        """
        new_map = cls(**data)
        return new_map

    def __iter__(self):
        return self.iter_points

    def getPixel(self, key):
        return self.map[key]

    def setPixel(self, key, value):
        self.map[key] = value

    def setInputType(self, map_units):
        """
        We can switch back and forth between different units for accessing data.
        This is done by changing the __getitem__ and __setitem__ functions themselves.
        Note that this change affects all instances of Map objects!
        """
        if map_units.lower() == 'pixel':
            Map.__getitem__ = Map.getPixel
            Map.__setitem__ = Map.setPixel
            Map.map_units = 'pixel'
            print "Set pixel mode"
        else:
            warnings.warn("I don't recognize map units of " + map_units, SyntaxWarning)
    #input_type = Map.access_mode

    ########################################################################################################
    def setMapWeight(self, weight):
        """
        This function sets a weight for the map as a whole. It does this by multiplying the pixel weights
        by the input "weight" number. If a map-wide weight has already been set, it's divided out
        before multiplying by the new weight.

        INPUTS
            weight: (float) A new global map weight.

        OUTPUT
            None, but the contents of this object are modified.

        EXCEPTIONS
            ValueError if this Map doesn't have a weight.
        """
        if self.weight is None:
            raise ValueError(
                "This Map has no weight of its own. (Is it a submap of a PolarizedMap? Call PolarizedMap.getTOnly().)")

        # Multiply the pixel weights by the new map weight, dividing out any old map weight.
        self.weight *= weight / self.map_weight

        # If the pixel weights are included in the map values, then also multiply that.
        if self.weighted_map:
            self.map *= weight / self.map_weight

        # Store the new map weight.
        self._map_weight = weight

    def getMapWeight(self):
        """
        Returns the self.map_weight.
        """
        return getattr(self, '_map_weight', 1.0)
    map_weight = property(getMapWeight, setMapWeight)

    ########################################################################################################
    def removeWeight(self, in_place=True):
        """
        If the pixels of this map are value*weight, instead of just the value, this function will
        remove the weights to give a Map with unweighted pixels.

        INPUTS
            in_place [True]: (bool) If True, this object's contents will be modified. Otherwise,
                the unweighted map will be a copy of this one, and this object will be unchanged.

        OUTPUT
            The unweighted Map
        """
        if in_place:
            new_map = self
        else:
            new_map = self.copy()

        # If this map is already unweighted, we have nothing to do.
        if not getattr(self, 'weighted_map', False):
            return new_map

        # Define some variables needed for the upcoming block of C-code.
        recon_map = new_map.map
        pixel_map = new_map.map.copy()
        weight_map = new_map.weight
        ny, nx = new_map.active_shape[0], new_map.active_shape[1]
        lower_left_y, lower_left_x = new_map.padding

        recon_map[:, :] = 0.
        c_code = """
        #line 3761 "maps.py"

        int y_start = lower_left_y;
        int x_start = lower_left_x;
        int y_end = int(ny)+y_start;
        int x_end = int(nx)+x_start;
        for (int iy=y_start; iy<y_end; ++iy) {
           for (int ix=x_start; ix<x_end; ++ix) {
                if (weight_map(iy,ix)!=0) {
                   recon_map(iy, ix) += pixel_map(iy, ix) / weight_map(iy,ix);
                } //end if (make sure that weights are non-zero)
           } //end for (loop over ix)
        } //end for (loop over iy)
        """
        weave.inline(c_code, ['ny', 'nx', 'pixel_map', 'weight_map', 'recon_map', 'lower_left_x', 'lower_left_y'],
                     type_converters=weave.converters.blitz)

        new_map.weighted_map = False

        return new_map
    removePixelWeight = removeWeight  # Define a better-named alias for removeWeight

    ########################################################################################################
    def addPixelWeight(self, in_place=True):
        """
        The opposite of removeWeight, this function returns a map in which the value in each
        map pixel is map*weight.

        INPUTS
            in_place [True]: (bool) If True, this object's contents will be modified. Otherwise,
                the unweighted map will be a copy of this one, and this object will be unchanged.

        OUTPUT
            The pixel-weighted Map

        EXCEPTIONS
            None

        AUTHOR
            Stephen Hoover, 31 July 2013
        """
        if in_place:
            new_map = self
        else:
            new_map = self.copy()

        # If this map is already pixel-weighted, we have nothing to do.
        if getattr(self, 'weighted_map', False):
            return new_map

        # Define some variables needed for the upcoming block of C-code.
        recon_map = new_map.map
        pixel_map = new_map.map.copy()
        weight_map = new_map.weight
        ny, nx = new_map.active_shape[0], new_map.active_shape[1]
        lower_left_y, lower_left_x = new_map.padding

        recon_map[:, :] = 0.
        c_code = """
        #line 3819 "maps.py"

        int y_start = lower_left_y;
        int x_start = lower_left_x;
        int y_end = int(ny)+y_start;
        int x_end = int(nx)+x_start;
        for (int iy=y_start; iy<y_end; ++iy) {
           for (int ix=x_start; ix<x_end; ++ix) {
                if (weight_map(iy,ix)!=0) {
                   recon_map(iy, ix) += pixel_map(iy, ix) * weight_map(iy,ix);
                } //end if (make sure that weights are non-zero)
           } //end for (loop over ix)
        } //end for (loop over iy)
        """
        weave.inline(c_code, ['ny', 'nx', 'pixel_map', 'weight_map', 'recon_map', 'lower_left_x', 'lower_left_y'],
                     type_converters=weave.converters.blitz)

        new_map.weighted_map = True

        return new_map

    ########################################################################################################
    def subtract(self, other, noise_estimation=True, sub_with_weight=False, skip_check=False):
        """
        Subtract two unpolarized maps. Unlike with addition, where we take a weighted average of the two maps,
        subtraction with default options is simply (map1 - map2)/2 .
        This can be overriden by setting sub_with_weight to True.
        In either case, the weight of the resulting
        map is set to be the sum of the weights of the two input maps.

        If the input maps are weighted, we must divide out the weights before subtracting.
        Otherwise coherent (CMB) signals will not necessarily subtract out.
        (The input maps will not be modified.)

        If you want a "first half minus second half" noise estimate (or left-right, etc.),
        the right thing to do is coadd each half *with weights*,
        then unweight the two half-season maps maps,
        and finally call first.subtract(second, noise_estimation=True, sub_with_weight=False)
        This will properly subtract out coherent (CMB) signals,
        and normalize the result to reflect the noise present in the coadd of the two halves.

        INPUTS
            other: (Map) The map which you wish to subtract from this one.

            noise_estimation [True]: (bool) If True, divide the difference map by 2.
                This gives an estimate of the noise in the corresponding sum (i.e. weighted coadd) map.
                When True, you can think of this as assigning each unweighted map a fake weight of 1,
                doing a weighted-subtraction, and then unweighting (by the summed weight of 2).
                If True, the result for two independent noise maps will have sqrt(2) *less* noise.
                If you set this equal to False, the result will have sqrt(2) *more* noise than each half-map.

            sub_with_weight [False]: (bool) If True, do subtraction with weights, i.e.
                (weight1 * map1 - weight2 * map2) / (weight1 + weight2).
                This argument calls the "coadd" function to do the work.
                The "noise_estimation" argument will be ignored if "sub_with_weight" is True.
                WARNING: There is *no* good reason to subtract with weight.
                         This will *not* subtract coherent (CMB) signals.

            skip_check [False]: (bool) If True, skip checking whether the two maps have the same parameters.

        OUTPUT
            A Map which is the difference of this map minus the other map.
            The resulting map will be weighted or unweighted, depending on whether the input Maps were weighted.

        EXCEPTIONS
            ValueError if something important (shape, center, weighted-ness, etc.) is different about the two input maps.

        AUTHOR
            Stephen Hoover, 24 May 2013

        CHANGES
            11 October 2013: Added "skip_check" input. SH
        """
        if sub_with_weight:
            return self.subtractWeighted(other)

        if other != self and not skip_check:
            raise ValueError("I can only add two maps if they have the same parameters!")

        # Do we divide by two or not?
        normalization_factor = (0.5 if noise_estimation else 1.0)

        if self.weight is None or other.weight is None:
            difference_map = self.map - other.map
            return type(self)(map=normalization_factor * difference_map, **self.initArgs())
        else:
            summed_weights = self.weight + other.weight
            if getattr(self, 'weighted_map', False):
                # If these maps have weights multiplied in, we actually need to remove them before we can subtract.
                difference_map = np.zeros_like(self.map)
                nonzero_weight = ((other.weight != 0) & (self.weight != 0))
                difference_map[nonzero_weight] = self.map[nonzero_weight] / \
                    self.weight[nonzero_weight] - other.map[nonzero_weight] / other.weight[nonzero_weight]

                # Multiply back in the weights.
                difference_map *= summed_weights
            else:
                difference_map = self.map - other.map
                # Set to zero any pixel not represented in both maps.
                difference_map[(other.weight == 0) | (self.weight == 0)] = 0.

            return type(self)(map=normalization_factor * difference_map, **self.initArgs(weight=summed_weights))
    difference = subtract  # Alias the "subtract" function.

    ########################################################################################################
    def subtractWeighted(self, other):
        """
        Subtract two polarized maps, but use weights. In this case, subtraction is
        (weight1*map1 - weight2*map2)/(weight1+weight2). The weight of the resulting
        map is set to be the sum of the weights of the two input maps.

        INPUTS
            other: (Map) The map which you wish to subtract from this one.

        OUTPUT
            A Map which is the difference of this map and the other map.

        EXCEPTIONS
            ValueError if something important (shape, center, weighted-ness, etc.) is different about the two input maps.
            ValueError if one of both of the maps don't have a weight array.

        AUTHOR
            Stephen Hoover, 24 May 2013
        """
        if other != self:
            raise ValueError("I can only add two maps if they have the same parameters!")

        if self.weight is None or other.weight is None:
            raise ValueError("I can't subtract maps unless I know their weights")

        summed_weights = self.weight + other.weight
        if getattr(self, 'weighted_map', False):
            # If the maps are already weighted, this is easy.
            summed_maps = self.map - other.map
        else:
            # If the maps aren't weighted, then we need to multiply the weights in before subtracting.
            summed_maps = deepcopy(self.map)
            summed_maps[other.weight > 0] = (self.map[other.weight > 0] * self.weight[other.weight > 0] -
                                             other.map[other.weight > 0] * other.weight[other.weight > 0]) / summed_weights[other.weight > 0]

        return type(self)(map=summed_maps, **self.initArgs(weight=(summed_weights)))

    ########################################################################################################
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
    ########################################################################################################
    def getFTMap(self, return_type='T',
                 remake=False, save_ffts=True, conjugate=False,
                 sky_window_func=None, normalize_window=True, fft_shape=None,
                 save_complex64=False,
                 quiet=True):
        """
        Returns a Map object constructed so that it only has a T map FT, no realspace data.
        It's constructed in such a way that the map struct is and the same as the ft_maps struct.

        We always take existing FTs from self.ft_maps if they exist. By default, we only use existing
        FTs. If return_maps is not None, we create any missing maps, using the input sky_window_func.

        INPUTS
            return_type ['T']: (string) the polarization map type to return.
                Should be one of: 'T', 'Q', 'U', 'E', 'B'
                *************** 'T' is the only valid option for this ***************


        OPTIONS IF RE-MAKING FFTS:
            remake [False]: (bool) If True, recreate the FTs (even if they already exist).
               Note, you can only remake T.  E and B will raise an exception, because they must
               be made with the constructEB() function.

            save_ffts [True]: (bool) If True, save the ffts to this self.ft_maps.  If the FTs already existed,
               this option will over-write the old ones.

            conjugate [False]: (bool) Save time by using more space. Conjugation takes a surprisingly
                long amount of time. If you call conjugate=True with save_ffts=True, we'll
                store conjugated FTs in self.ft_maps_conj.

            sky_window_func [None]: (2d ndarray) If we need to take new FTs, apply this
                window function to each map before FTing.

            fft_shape [None]: (2-element list) x, y shape of output fft.
                The real-space map will be padded to a size of (fft_shape[0], fft_shape[1]).

            save_complex64 [False]: The default output of np.fft.fft2 is a complex128 array.  If True,
                instead return a complex64 array to save memory.  The differences are small: (32bit - 64bit)/64bit
                is ~ few x 10^-7 at most in power (np.abs(fft**2)) and less in the real and imaginary components
                of the FFT itself.

            normalize_window [True]: (bool) If True, apply a normalization factor to the input
                window function so that the power in the resulting FT doesn't change.

        OUTPUT
            A fft map (2d complex ndarray)

        EXCEPTIONS
            ValueError if this object has no pre-existing FTs and return_maps=None.
            ValueError if the input mask is too large for the padded array size.

        AUTHOR
            Kyle Story, 25 October 2013

        CHANGES
            3 June 2014: Add the capacity to store the complex conjugate of the map, to speed up cross-spectra. SH
            30 Sept 2014: Hijacked this code from the polarized map class and repurposed it for the Map class, so some of the
                            tenses of comments will not be right since we are only dealing with T (no T,Q, & U). TJN
        """
        assert(type(return_type) == str), "Argument return_type must be of string type.  Given type was %r" % type(return_type)
        if return_type not in ['T']:
            raise ValueError("Bad return_type: " + return_type +
                             ".  return_type can only be 'T' since this is a T only map")

        if not remake:
            # try:
            #     self.ft_maps
            # except AttributeError:
            #     warnings.warn("The FFT maps do not exist, please rerun getFTMap with remake=True.", RuntimeWarning)
            #     return -1

            if self.ft_maps.has_key(return_type):
                if conjugate:
                    # If we want the complex conjugate of the map, see if we've already stored it.
                    # If we haven't, then compute it, store (if requested), and return.
                    try:
                        return self.ft_maps[return_type].map_conj
                    except AttributeError:
                        map_conj = np.conjugate(self.ft_maps[return_type].map)
                        if save_ffts:
                            self.ft_maps[return_type].map_conj = map_conj
                        return map_conj
                else:
                    return self.ft_maps[return_type].map
            else:
                if return_type in ['T']:
                    remake = True  # if T FTs don't exist, re-make them
                else:
                    warnings.warn("You asked for " + return_type +
                                  " type FT maps, but they do not exist.", RuntimeWarning)
                    return -1

        # Make new FT maps
        if remake:
            if not quiet:
                print("getFTMap(): Remaking " + return_type + " type FT map.")
            if return_type not in ['T']:
                raise ValueError("In this function, you can only remake a T fft since this is a T-only map.")

            # Put the map inside a new array, adding padding if requested.
            _map = self.getTOnly()
            if fft_shape is None:
                fft_shape = _map.shape
            assert(len(fft_shape) == 2)
            if (fft_shape < _map.shape).any():
                raise ValueError("***Error: Requested fft_shape must be >= size of existing maps.  Requested fft_shape is ",
                                 fft_shape, ", while existing map shape is ", _map.shape)

            padded_shape = np.array(fft_shape)

            padded_map = np.zeros(padded_shape, dtype=_map.map.dtype)
            padded_map[:_map.shape[0], :_map.shape[1]] = _map.map

            if sky_window_func is None:
                sky_window_func = np.ones(padded_shape, dtype=float_type)
            elif (sky_window_func.shape != padded_shape).any():
                # Make sure that the window function is compatible with the padding,
                # and pad it with zeros if necesary.
                if (sky_window_func.shape > padded_shape).any():
                    raise ValueError("The input mask is larger than the padded map size!")

                padded_window = np.zeros(padded_shape, dtype=_map.map.dtype)
                padded_window[:sky_window_func.shape[0], :sky_window_func.shape[1]] = sky_window_func
                sky_window_func = padded_window

            window_norm_factor = 1.
            if normalize_window:
                window_norm_factor = np.sqrt(np.sum(sky_window_func**2) / np.product(sky_window_func.shape))

            # Take the FFT and attach it to this map. If we padded, set the entire
            # area of the new map as "active" -- FTs don't have blank areas around
            # the edges.
            fft_map = np.fft.fft2(padded_map * sky_window_func / window_norm_factor)

            # Save space by using complex64?
            if save_complex64:
                fft_map = np.complex64(fft_map)

            if save_ffts:
                self.addFTMap(return_type, fft_map, active_shape=padded_shape)

            # Take (and store) the complex conjugate if requested.
            if conjugate:
                fft_map = np.conjugate(fft_map)
                if save_ffts:
                    self.ft_maps[return_type].map_conj = fft_map

            return fft_map

    ########################################################################################################
    def getFTOnlyMap(self, return_maps=None, sky_window_func=None,
                     normalize_window=True, remake=False, fft_shape=None,
                     save_complex64=False):
        """
        Returns a PolarizedMap object constructed so that it only has FTs, no realspace data.
        It's constructed in such a way that the pol_maps struct is the same as the ft_maps struct.

        We always take existing FTs from self.ft_maps if they exist. By default, we only use existing
        FTs. If return_maps is not None, we create any missing maps, using the input sky_window_func.

        INPUTS
            return_maps [None]: (Iterable of strings) A list of polarizations which we want in the
                new FT-only map. If None, we'll use all of the existing FTs, and nothing else.

            sky_window_func [None]: (2d ndarray) If we need to take new FTs, apply this
                window function to each map before FTing.

            normalize_window [True]: (bool) If True, apply a normalization factor to the input
                window function so that the power in the resulting FT doesn't change.

            fft_shape [None]: (2-element list) x, y shape of output fft.
                The real-space map will be padded to a size of (fft_shape[0], fft_shape[1]).

            save_complex64 [False]: The default output of np.fft.fft2 is a complex128 array.  If True,
                instead return a complex64 array to save memory.  The differences are small: (32bit - 64bit)/64bit
                is ~ few x 10^-7 at most in power (np.abs(fft**2)) and less in the real and imaginary components
                of the FFT itself.

            remake [False]: (bool) If True, recreate the FTs even if they already exist. In this case,
                we will overwrite the existing FTs with the new ones.

        OUTPUT
            A new PolarizedMap object, with the specified FTs in both the pol_maps and ft_maps structs.
            If we created new FTs along the way, they will also be stored in this object's ft_maps struct.

        EXCEPTIONS
            ValueError if this object has no pre-existing FTs and return_maps=None.
            ValueError if the input mask is too large for the padded array size.

        AUTHOR
            Stephen Hoover, 20 June 2013
        """
        if len(self.ft_maps) == 0 and return_maps is None:
            raise ValueError("I have no FTs to put into a FT-only map!")

        # Create any new maps needed.
        if return_maps is not None:
            for map_pol in return_maps:
                if map_pol not in self.ft_maps or remake:
                    self.getFTMap(return_type=map_pol, remake=True, sky_window_func=sky_window_func,
                                  normalize_window=normalize_window, fft_shape=fft_shape, save_complex64=save_complex64)

        # Create the new PolarizedMap, and connect the ft_maps struct to the pol_maps struct.
        assert(fft_shape is None or len(fft_shape) == 2)
        if (fft_shape is not None) and ((fft_shape != self.shape).any()):
            # We need to add any requested padding to the weights.
            padded_shape = np.array(fft_shape)

            if self.weight is not None:
                padded_weight = np.zeros(np.append(padded_shape, [3, 3]), dtype=self.weight.dtype)
                padded_weight[:self.shape[0], :self.shape[1], :, :] = self.weight
            else:
                padded_weight = None
            new_ft_map = type(self)(maps=dict([(pol, _map) for pol, _map in self.ft_maps.iteritems() if pol in return_maps]),
                                    **self.initArgs(weight=padded_weight))
        else:
            new_ft_map = type(self)(maps=dict([(pol, _map) for pol, _map in self.ft_maps.iteritems() if pol in return_maps]),
                                    **self.initArgs())
        new_ft_map.ft_maps = new_ft_map.pol_maps

        return new_ft_map

    ########################################################################################################
    def getInverseFFT(self, k_window_func=None, real_part=False, force_type=None, save_complex64=False):
        if k_window_func is None:
            k_window_func = np.ones(self.shape, dtype=float_type)

        args = self.initArgs()
        if save_complex64:
            maps = dict([(pol, np.complex64(np.fft.ifft2(self.pol_maps[pol].map * k_window_func)))
                         for pol in self.polarizations])
        else:
            maps = dict([(pol, np.fft.ifft2(self.pol_maps[pol].map * k_window_func)) for pol in self.polarizations])
        if real_part:
            for pol in maps:
                maps[pol] = np.real(maps[pol])
            if 'data_type' in args:
                del(args['data_type'])
        if force_type:
            for pol in maps:
                maps[pol] = maps[pol].astype(force_type)
            if 'data_type' in args:
                args['data_type'] = force_type

        return type(self)(maps=maps, **args)

# if force_type is not None:
# maps=dict([(pol, (np.fft.ifft2(self.pol_maps[pol].map * k_window_func)).astype(force_type))
# for pol in self.polarizations])
# else:
# maps=dict([(pol, np.fft.ifft2(self.pol_maps[pol].map * k_window_func))
# for pol in self.polarizations])
# return PolarizedMap(maps=dict([(pol, (np.fft.ifft2(self.pol_maps[pol].map * k_window_func)).astype(force_type))
# for pol in self.polarizations]),
# **self.initArgs() )
# else:
# return PolarizedMap(maps=dict([(pol, np.fft.ifft2(self.pol_maps[pol].map * k_window_func))
# for pol in self.polarizations]),
# **self.initArgs() )

    ########################################################################################################

    def addFTMap(self, pol, map, **new_args):
        """Adds a new map, of polarization pol, to this object. The input "map" is assumed to represent
        the FT of an existing map, and we store it in the self.ft_maps struct. If map is a Map object,
        add it directly. If map is an array or list, create a new Map object for it.

        INPUTS
            pol: (string) The polarization of the input map.

            map: (Map or array) The map information.

            **new_args: Any extra arguments, set as attributes of the newly-created map object.

        OUTPUT
            None, but this object is modified.

        EXCEPTIONS
            None

        AUTHOR
            Stephen Hoover, 11 June 2013
        """
        try:
            map.getResoRad()  # A Map object will have this method.
        except AttributeError:
            map = Map(map=map, **self.initArgs(polarization=pol, **new_args))

        if not hasattr(self, 'ft_maps'):
            self.ft_maps = struct()

        self.ft_maps[pol] = map

    ########################################################################################################

#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
    def whatsDifferent(self, other, verbose=False):
        """
        Compares two maps to see which important properties are different. This is the comparision
        which is run before adding two maps. If this function reports anything different, the maps
        mayn't be added.

        INPUTS
            other: (Map) The map we're comparing this one with.

            verbose [False]: (bool) If True, print details to the screen about what's different.

        OUTPUT
            A list of names of attributes which differ between the two maps.

        EXCEPTIONS
            None

        AUTHOR
            Stephen Hoover, 24 May 2013
        """
        different_attributes = []

        # Are the map shapes the same?
        if not (self.shape == other.shape).all():
            different_attributes.append('shape')
            if verbose:
                print "This map has shape %s, and the other map has shape %s." % (str(self.shape), str(other.shape))
        if not (self.active_shape == other.active_shape).all():
            different_attributes.append('active_shape')
            if verbose:
                print "This map has active shape %s, and the other map has active shape %s." % (str(self.active_shape), str(other.active_shape))

        # Are the two maps using the same projection?
        if self.projection != other.projection:
            different_attributes.append('projection')
            if verbose:
                print "This map uses projection %d, and the other map uses projection %d." % (self.projection, other.projection)

        # Compare the map centers.
        if not (self.center == other.center).all():
            different_attributes.append('center')
            if verbose:
                print "This map has center %s, and the other map has center %s." % (str(self.center), str(other.center))

        # Both maps need the same resolution.
        if self.reso_arcmin != other.reso_arcmin:
            different_attributes.append('reso_arcmin')
            if verbose:
                print "This map has a resolution of %f arcmin/pixel, and the other map has %f arcmin/pixel." % (self.reso_arcmin, other.reso_arcmin)

        # Check if the two maps are both weighted.
        if getattr(self, 'weighted_map', False) != getattr(other, 'weighted_map', False):
            different_attributes.append('weighted_map')
            if verbose:
                print "This map is%s weighted, and the other map is%s." % (('' if getattr(self, 'weighted_map', False) else ' not'),
                                                                           ('' if getattr(other, 'weighted_map', False) else ' not'))

        return different_attributes

    ########################################################################################################
    def setMapAttr(self, name, value):
        """
        Repeated here so we can call this function without knowing whether we have a Map or PolarizedMap.
        """
        setattr(self, name, value)

    def __getitem__(self, key):
        """
        Accesses the map data array.
        """
        return self.map[key]

    def __setitem__(self, key, value):
        """
        Accesses the map data array.
        """
        self.map[key] = value

    def __eq__(self, other):
        """
        Tests whether two Maps have the same shape and resolution.
        """
        try:
            # The "getattr" calls deal with Maps made before the weighted_map attribute was present.
            # Assume all of those have already had weights divided out.
            return ((self.shape == other.shape).all() and self.projection == other.projection
                    and (self.center == other.center).all()
                    and (self.active_shape == other.active_shape).all() and self.reso_arcmin == other.reso_arcmin
                    and getattr(self, 'weighted_map', False) == getattr(other, 'weighted_map', False))
        except AttributeError:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return str(self.map)

    # Define addition and subtraction: add and subtract maps, but only if the maps have the same parameters.
    def __add__(self, other):
        if isinstance(other, Map):
            assert other == self, "I can only add two maps if they have the same parameters!"
            if self.weight is None or other.weight is None:
                raise ValueError("I can't add maps unless I know their weights")

            summed_weights = self.weight + other.weight
            if getattr(self, 'weighted_map', False):
                summed_maps = self.map + other.map
            else:
                summed_maps = deepcopy(self.map)
                summed_maps[other.weight > 0] = (self.map[other.weight > 0] * self.weight[other.weight > 0] +
                                                 other.map[other.weight > 0] * other.weight[other.weight > 0]) / summed_weights[other.weight > 0]
            return type(self)(map=summed_maps, **self.initArgs(weight=(summed_weights)))
        else:
            warnings.warn('Adding two things, one of which is not a Map.', RuntimeWarning)
            return type(self)(map=(self.map + other), **self.initArgs())

    def __radd__(self, other):
        if isinstance(other, Map):
            raise ValueError("Why am I in radd when both objects are Maps? I should be using __add__.")
        else:
            return type(self)(map=(other + self.map), **self.initArgs())

    def __iadd__(self, other):
        if isinstance(other, Map):
            assert other == self, "I can only add two maps if they have the same parameters!"
            if self.weight is None or other.weight is None:
                raise ValueError("I can't add maps unless I know their weights!")

            new_map = self + other
            self.map = new_map.map
            self.weight = new_map.weight
        else:
            self.map += other
        return self

    def __sub__(self, other):
        if isinstance(other, Map):
            return self.subtract(other, noise_estimation=True, sub_with_weight=False)
        else:
            warnings.warn('Subtracting two things, one of which is not a Map.', RuntimeWarning)
            return type(self)(map=(self.map - other), **self.initArgs())

    def __rsub__(self, other):
        if isinstance(other, Map):
            return other.subtract(self, noise_estimation=True, sub_with_weight=False)
        else:
            return type(self)(map=(other - self.map), **self.initArgs())

    def __isub__(self, other):
        if isinstance(other, Map):
            # Does diffmap = (map1-map2)/2
            assert other == self, "I can only subtract two maps if they have the same parameters!"
            if self.weight is None or other.weight is None:
                raise ValueError("I can't subtract maps unless I know their weights!")

            if getattr(self, 'weighted_map', False):
                # If these maps have weights multiplied in, we actually need to remove them before we can subtract.
                difference_map = np.zeros_like(self.map)
                nonzero_weight = ((other.weight != 0) & (self.weight != 0))
                difference_map[nonzero_weight] = other.map[nonzero_weight] / \
                    other.weight[nonzero_weight] - self.map[nonzero_weight] / self.weight[nonzero_weight]

                # Multiply back in the weights.
                self.weight += other.weight
                difference_map *= self.weight

                # Multiply by 0.5 so that the difference map is representative of
                # the noise properties of the self+other coadd.
                self.map = 0.5 * difference_map
            else:
                self.map -= other.map
                # Multiply by 0.5 so that the difference map is representative of
                # the noise properties of the self+other coadd.
                self.map *= 0.5

                # Set to zero any pixel not represented in both maps.
                self.map[(other.weight == 0) | (self.weight == 0)] = 0.
                self.weight += other.weight
        else:
            self.map -= other
        return self

    def __mul__(self, other):
        if isinstance(other, Map):
            assert other == self, "I can only multiply two maps if they have the same parameters!"
            return type(self)(map=(self.map * other.map), **self.initArgs())
        else:
            return type(self)(map=(self.map * other), **self.initArgs())

    def __rmul__(self, other):
        if isinstance(other, Map):
            assert other == self, "I can only multiply two maps if they have the same parameters!"
            return type(self)(map=(other.map * self.map), **self.initArgs())
        else:
            return type(self)(map=(other * self.map), **self.initArgs())

    def __imul__(self, other):
        if isinstance(other, Map):
            assert other == self, "I can only multiply two maps if they have the same parameters!"
            self.map *= other.map
        else:
            self.map *= other
        return self

    def __abs__(self):
        return type(self)(map=np.abs(self.map), **self.initArgs())


class Mask(Map):
    pass


class TrueSky(Map):

    @classmethod
    def noiseMap(cls, dimensions, rms=1.0, **map_init_keys):
        return cls(dimensions, map=(np.random.standard_normal(list(dimensions)) * rms), **map_init_keys)

    def __init__(self, *args, **keyargs):
        super(TrueSky, self).__init__(*args, **keyargs)


class ObservedSky(Map):

    def __init__(self, *args, **keyargs):
        super(ObservedSky, self).__init__(*args, **keyargs)


class PolarizedTrueSky(PolarizedMap):

    @classmethod
    def noiseMap(cls, dimensions, rms=1.0, **map_init_keys):
        maps = []
        for pol in tqu:
            maps.append(TrueSky.noiseMap(dimensions, rms, polarization=pol, **map_init_keys))
        return cls(maps=maps, polarizations=tqu)

    def __init__(self, *args, **keyargs):
        super(PolarizedTrueSky, self).__init__(*args, **keyargs)


class PolarizedObservedSky(PolarizedMap):

    def __init__(self, *args, **keyargs):
        super(PolarizedObservedSky, self).__init__(*args, **keyargs)


###########################################################################################
###########################################################################################
class TestProjections(unittest.TestCase):
    """
    Defines unit tests on ang2Pix and pix2Ang.
    """

    def setUp(self):
        """
        Define the test cases for the unit tests.

        Attributes needed for ang2Pix and pix2Ang:

            tolerance_arcsec: (float) In arcseconds. The difference between the
                pixel coordinate from ang2Pix and the RA/dec from IDL's ang2pix must be less
                than this in both x and y.
                (The difference in pixel coordinates will be transformed to arcseconds using
                the test_resoarcmin.)

            test_radec: (2-element tuple of arrays) A tuple or list of arrays,
                [ra, dec]. Can also be a PointingSequence object. In degrees.
                Defaults to np.array([[352.00738525], [-49.68125153]]) if not provided.

            test_pix: (2-element tuple of arrays) The input pixel coordinates.
                A tuple or list of arrays, [y_coord, x_coord].
                Note the order! The first element is the "y", the mostly-dec coordinate, and the
                second element is the "x", the mostly-RA coordinate.
                Defaults to np.array([[523.], [1723.]]).

            test_shape: (2-element array) The map shape, in pixels_y, pixels_x.
                Defaults to np.array([3600, 3600]).

            test_map_center: (2-element array) The [RA, declination] of the center of the map.

            test_resoarcmin: (float) The width of each pixel in arcminutes, assumed to be the same for
                both x and y directions.
        """

        self.tolerance_arcsec = 0.1

        # This section is for testing ang2Pix and pix2Ang
        self.proj_index_in_idl = [0, 1, 2, 4, 5]
        self.proj_to_test = [0, 1, 2, 4, 5, 7]

        self.test_resoarcmin = 0.25  # Arcminutes per pixel.
        self.test_pix = np.array([[523., 911.1, 2131.], [1723., 1., 2131.]])
        self.test_shape = np.array([4500, 3600])
        self.test_map_center = np.array([352.5,  -55.])
        self.test_radec = np.array([[352.00738525, 346., 352.5, 4.2], [-49.68125153, -53.123, -55., -59.318]])

        #####################
        # Now variables for testing applyPointingOffset and calculatePointingOffset
        self.test_boresight_radec = np.array([[349., 357., 349., 357.], [-49.13, -49., -60.31, -59.32]])
        # In arcminutes azimuth offset (degrees on sky), arcminutes el offset (on sky)
        self.test_azel_offset = np.array([[28.2], [64.1]])

    ##########################################################################################
    def test_Pix2AngVsIDL(self, test_proj=None, test_pix=None, test_shape=None,
                          test_map_center=None):
        """
        Compare the output of pix2Ang with the equivalent IDL procedures. Raise an exception if
        the outputs are too different from each other.

        INPUTS
            test_proj: (int or str) Either the integer index of the projection to test, or its string designator.

            test_pix [None]: (2-element tuple of arrays) The input pixel coordinates.
                A tuple or list of arrays, [y_coord, x_coord].
                Note the order! The first element is the "y", the mostly-dec coordinate, and the
                second element is the "x", the mostly-RA coordinate.
                Defaults to np.array([[523.], [1723.]]).

            test_shape [None]: (2-element array) The map shape, in pixels_y, pixels_x.
                Defaults to np.array([3600, 3600]).

            test_map_center [None]: (2-element array) The [RA, declination] of the center of the map.

        OUTPUT
            None
        """
        from sptpol_software.util import idl

        # If we didn't get a specific projection to test, then test all of them.
        if test_proj is None:
            for proj in self.proj_index_in_idl:
                self.test_Pix2AngVsIDL(proj, test_pix=test_pix,
                                       test_shape=test_shape,
                                       test_map_center=test_map_center)
        else:
            # Set defaults if inputs weren't provided.
            if test_pix is None:
                test_pix = self.test_pix
            if test_shape is None:
                test_shape = self.test_shape
            if test_map_center is None:
                test_map_center = self.test_map_center

            # Find IDL's RA and dec.
            idlout = idl.idlRun("pix2ang_proj%d, npixels, radec0, reso_arcmin, ra, dec, xpix=xpix, ypix=ypix" % test_proj, return_all=True,
                                xpix=test_pix[1], ypix=test_pix[0], npixels=test_shape[::-1],
                                radec0=test_map_center, reso_arcmin=self.test_resoarcmin)

            # Now run the Python code.
            radec = pix2Ang(test_pix, test_map_center, self.test_resoarcmin, test_shape, proj=test_proj)

            # Compare the answers, and complain if they're too different.
            ra_error = radec[0] - idlout['ra']
            dec_error = radec[1] - idlout['dec']

            self.assertFalse((np.abs(ra_error * 3600) > self.tolerance_arcsec).any(),
                             ("proj %d: max ra_error = %f arcseconds, max dec_error = %f arcseconds" %
                              (test_proj, ra_error.max() * 3600, dec_error.max() * 3600)))
            self.assertFalse((np.abs(dec_error * 3600) > self.tolerance_arcsec).any(),
                             ("proj %d: max ra_error = %f arcseconds, max dec_error = %f arcseconds" %
                              (test_proj, ra_error.max() * 3600, dec_error.max() * 3600)))

    ##########################################################################################
    def test_Ang2PixVsIDL(self, test_proj=None, test_radec=None, test_shape=None,
                          test_map_center=None):
        """
        Compare the output of ang2Pix with the equivalent IDL procedures. Raise an exception if
        the outputs are too different from each other.

        INPUTS
            test_proj: (int or str) Either the integer index of the projection to test, or its string designator.

            test_radec [None] : (2-element tuple of arrays) A tuple or list of arrays,
                [ra, dec]. Can also be a PointingSequence object. In degrees.
                Defaults to np.array([[352.00738525], [-49.68125153]]) if not provided.

            test_shape [None]: (2-element array) The map shape, in pixels_y, pixels_x.
                Defaults to np.array([3600, 3600]).

            test_map_center [None]: (2-element array) The [RA, declination] of the center of the map.

        OUTPUT
            None
        """
        from sptpol_software.util import idl

        # If we didn't get a specific projection to test, then test all of them.
        if test_proj is None:
            for proj in self.proj_index_in_idl:
                self.test_Ang2PixVsIDL(proj, test_radec=test_radec,
                                       test_shape=test_shape, test_map_center=test_map_center)
        else:
            # Set defaults if inputs weren't provided.
            if test_radec is None:
                test_radec = self.test_radec
            if test_shape is None:
                test_shape = self.test_shape
            if test_map_center is None:
                test_map_center = self.test_map_center

            # Find IDL's pixel coordinates. Note that the IDL ang2pix_proj1 doesn't take the "/noround"
            # argument, so don't use it there. The output of that function is still unrounded pixel coordinates.
            idlout = idl.idlRun("ang2pix_proj%d, ra, dec, npixels, radec0, reso_arcmin, ipix, xpix=xpix, ypix=ypix%s" % (test_proj, ', /noround' if test_proj != 1 else ''),
                                return_all=True,
                                ra=test_radec[0], dec=test_radec[1], npixels=test_shape[::-1],
                                radec0=test_map_center, reso_arcmin=self.test_resoarcmin)

            # Now run the Python code.
            pixout = ang2Pix(test_radec, test_map_center, self.test_resoarcmin, test_shape,
                             proj=test_proj, round=False, return_validity=False)

            # Compare the answers and complain if they're too different.
            x_error = pixout[1] - idlout['xpix']
            y_error = pixout[0] - idlout['ypix']

            self.assertTrue((np.abs(x_error * self.test_resoarcmin * 60) < self.tolerance_arcsec).all(),
                            ("proj %d: max x_error = %f pixels, %f arcseconds, max y_error = %f pixels, %f arcseconds" %
                             (test_proj, x_error.max(), x_error.max() * self.test_resoarcmin * 60, y_error.max(), y_error.max() * self.test_resoarcmin * 60)))
            self.assertTrue((np.abs(y_error * self.test_resoarcmin * 60) < self.tolerance_arcsec).all(),
                            ("proj %d: max x_error = %f pixels, %f arcseconds, max y_error = %f pixels, %f arcseconds" %
                             (test_proj, x_error.max(), x_error.max() * self.test_resoarcmin * 60, y_error.max(), y_error.max() * self.test_resoarcmin * 60)))

    ##########################################################################################
    def test_CircularPixelMappings(self, test_proj=None, test_pix=None, test_shape=None,
                                   test_map_center=None):
        """
        At a bare minimum, when the RA and dec calculated from pix2Ang are provided to ang2Pix,
        the result should be the same as the input pixel coordinate, up to float precision errors.
        This function tests that.

        INPUTS
            test_proj: (int or str) Either the integer index of the projection to test, or its string designator.

            test_pix [None]: (2-element tuple of arrays) The input pixel coordinates.
                A tuple or list of arrays, [y_coord, x_coord].
                Note the order! The first element is the "y", the mostly-dec coordinate, and the
                second element is the "x", the mostly-RA coordinate.
                Defaults to np.array([[523.], [1723.]]).

            test_shape [None]: (2-element array) The map shape, in pixels_y, pixels_x.
                Defaults to np.array([3600, 3600]).

            test_map_center [None]: (2-element array) The [RA, declination] of the center of the map.

        OUTPUT
            None
        """

        # If we didn't get a specific projection to test, then test all of them.
        if test_proj is None:
            for proj in self.proj_to_test:
                self.test_CircularPixelMappings(proj, test_pix=test_pix,
                                                test_shape=test_shape, test_map_center=test_map_center)
        else:
            # Set defaults if inputs weren't provided.
            if test_pix is None:
                test_pix = self.test_pix
            if test_shape is None:
                test_shape = self.test_shape
            if test_map_center is None:
                test_map_center = self.test_map_center

            # Run pix2Ang.
            radec = pix2Ang(test_pix, test_map_center, self.test_resoarcmin, test_shape, proj=test_proj)

            # Put the results ot pix2Ang into ang2Pix.
            yout, xout = ang2Pix(radec, test_map_center, self.test_resoarcmin, test_shape,
                                 proj=test_proj, round=False, return_validity=False)

            # Compute the difference between the input to pix2Ang and the output of ang2Pix.
            if len(yout) == 1:
                yout = yout[0]
            if len(xout) == 1:
                xout = xout[0]
            error = (test_pix - np.array([yout, xout]))
            error_arcsec = error * self.test_resoarcmin * 60

            # Check for too-large errors.
            self.assertTrue((np.abs(error_arcsec) < self.tolerance_arcsec).all(),
                            ("proj %d: max x_error = %f pixels, %f arcseconds, max y_error = %f pixels, %f arcseconds" %
                             (test_proj, error[1].max(), error_arcsec[1].max(), error[0].max(), error_arcsec[0].max())))

    ##########################################################################################
    def test_pointingOffsets(self):
        """
        Tests the applyPointingOffset and calculatePointingOffset functions by making sure that
        they are consistent with each other.
        """

        offset_pointing = applyPointingOffset(self.test_boresight_radec, self.test_azel_offset, as_azel=False)
        new_elaz_offset = calculatePointingOffset(self.test_boresight_radec, offset_pointing, as_azel=False)
        # Note that the output of calculatePointingOffset is reversed in order from the
        # offset given to applyPointingOffset.

        self.assertTrue((np.abs(self.test_azel_offset - new_elaz_offset[::-1]) * 60 < self.tolerance_arcsec).all())

    ##########################################################################################
    def test_PointingOffsetVsIDL(self, boresight_pointing=None, detector_pointing=None):
        """
        Compare outputs of calculatePointingOffset with the equivalent IDL procedure, rotate_boresight_to_zero_zero.pro.

        INPUTS
            boresight_pointing [None]: (2-element iterable) Telescope boresight pointing, (RA, dec), in degrees.
                The RA and dec may be arrays. Defaults to self.test_boresight_radec.

            detector_pointing [None]: (2-element iterable) A different pointing, here named under the assumption
                that it's the pointing of an individual detector element, (RA, dec), in degrees.
                The RA and dec may be arrays. Defaults to self.test_boresight_radec + self.test_azel_offset[::-1]/60.

        OUTPUT
            None
        """
        from sptpol_software.util import idl

        if boresight_pointing is None:
            boresight_pointing = self.test_boresight_radec
        if detector_pointing is None:
            detector_pointing = self.test_boresight_radec + self.test_azel_offset[::-1] / 60.

        # First get the Python version.
        python_offset = calculatePointingOffset(
            boresight_pointing, detector_pointing, offset_units='degrees', as_azel=False)

        ###################################
        # That was easy, no? Now try IDL.

        # Unpack the inputs.
        # bs = boresight, det = detector
        bs_ra, bs_dec = boresight_pointing[0], boresight_pointing[1]
        det_ra, det_dec = detector_pointing[0], detector_pointing[1]

        idlout = idl.idlRun("rotate_boresight_to_zero_zero, bs_ra, bs_dec, det_ra, det_dec, offset_az, offset_el",
                            return_all=True, bs_ra=bs_ra, bs_dec=bs_dec, det_ra=det_ra, det_dec=det_dec)

        idl_offset = np.array([-idlout['offset_el'], idlout['offset_az']])
        offset_error = python_offset.squeeze() - idl_offset.squeeze()

        self.assertTrue((np.abs(offset_error) * 3600 < self.tolerance_arcsec).all())
