# This is a translation of spt_analysis/sources/calc_optband_2d.pro
#  which was made to:
#      "Calculate optimal filter for source detection in single map
#       with many different assumed source profiles."

import numpy as np
import pylab as pl
import sptpol_software.constants as constants
import sptpol_software.analysis.maps as maps 
import sptpol_software.analysis.plotting as plotting
from sptpol_software.constants import DTOR 
from sptpol_software.util.tools import struct
from sptpol_software.util.files import readCAMBCls
from sptpol_software.util.math import makeEllGrids
from sptpol_software.simulation.beams import calc_gauss_Bl
from sptpol_software.observation.sky import filterMap
import sptpol_software.util.idl as idl 
import scipy as sp
import cPickle as pkl
import sptpol_software.util.files as files
from glob import glob
import os as os
import sptpol_software.util.time as spttime
import multiprocessing
import itertools as itools


def calc_optfilt_oneband_2d(profiles, skysigvar, noisepsd, reso_arcmin,
                            sigspect = 1,
                            fwhm_beam_arcmin = 1E-6,# 1E-6 is an infinitely small beam
                            ell_highpass = None,
                            ell_lowpass = None,
                            f_highpass = None, 
                            f_lowpass = None,
                            scanspeed = None, 
                            trans_func_grid= None, 
                            beam_grid = None,
                            keepdc = False, 
                            arcmin_sigma = None,
                            area_eff = None,
                            whitenoise1d = None,
                            pointsource = False,
                            norm_unfilt = None,
                            external_profile_norm = None,
                            bfilt = None,
                            debug1d = False,
                            debug = False,
                            ):
    '''
    This is a translation of spt_analysis/sources/calc_optband_2d.pro
    which was made to:
    "Calculate optimal filter for source detection in single map
       with many different assumed source profiles."
    
    ** There is some ambiguity in returning x,y or y,x **
    INPUT:
      profiles: (array) 1d ell-space profiles of objects you want to detect.
             This input is not used when pointsource = True. 
      skysigvar: (array) Assumed 1D variance as a function of ell (isotropic)
              of all sky signals other than the signal of
              interest.  (E.g., CMB + point sources for an SZ
              measurement.)
     noisepsd: (ndarray): N_pts-by-N_pts array of 2d Fourier
              representation of noise in map.
              Should be normalized such that
              sqrt(int_tabulated(uv_arr,noisepsd[*,*,j]^2)) =
              noise rms in the map.
              **** Need to confirm this normalization ****
     reso_arcmin: (float) Pixel size in arcminutes of maps to be filtered.
     sigspect[1] - sensitivity to signal of interest in supplied map.
              For clusters, this should be equal to fxsz(band).
              Leave this equal to 1 for single band filtering. 
     fwhm_beam_arcmin[1E-6]: (float) Size (in full-width at half-max in units of
              arcminutes) of beams in each band.  Beams
              assumed to be Gaussian.  Default is 10^-6
              (infinitely small beams).  Ignored if
              TRANS_FUNC_GRID is supplied.
                          
     beam_grid - set this to a 2d array with the 2d beam response
              (if you want to specify a non-Gaussian or
              asymmetric beam but use default filters).
     bfilt[None]: (array) Hand the program a 1-d beam window function, 
              which will be used in place of the default assumed
              Gaussian beam for spatial filtering
     ell_highpass[None]: (float) Ell at which timestreams were high-passed
              before maps were made.  Default is no filtering.
              Ignored if TRANS_FUNC_GRID is supplied.
              *Do not supply an ell_highpass and f_highpass.*
     ell_lowpass[None]: (float) Ell at which timestreams were low-passed
              before maps were made.  Default is no filtering.
              Ignored if TRANS_FUNC_GRID is supplied.
              *Do not supply an ell_lowpass and f_lowpass.*
     f_highpass[None]: (float) Frequency at which timestreams were high-passed
              before maps were made.  Default is no filtering.
              Ignored if TRANS_FUNC_GRID is supplied.
              *Do not supply an ell_highpass and f_highpass.*
     f_lowpass[None]: (float) Frequency at which timestreams were low-passed
              before maps were made.  Default is no filtering.
              Ignored if TRANS_FUNC_GRID is supplied.
              *Do not supply an ell_lowpass and f_lowpass.*
     scanspeed[None]: (float) what speed were we scanning at when the maps were
              made?  (Need to know this to translate high- and
              low-pass temporal frequencies into spatial
              frequencies.)  Default is 0.25 (deg/s on the
              sky). 
              *This must be supplied to use f_highpass or f_lowpass*
     trans_func_grid[None]: (ndarray) Array the same size as MAP containing the
              effective delta-function response (in 2d
              fourier space) of all the filtering we've
              done to the signal in the map
              (INCLUDING THE BEAM).  If TRANS_FUNC_GRID
              is set, other filtering keywords (including
              FWHM_BEAM_ARCMIN) are ignored.
     keepdc[False]: (bool) Set this to keep the DC information along the scan
              direction.  Otherwise, the u=0 modes
              are set to zero even if F_HIGHPASS isn't
              set. 
     arcmin_sigma - SIGMA scaled to what it would be for a
              1-arcminute FWHM beam.  (SIGMA is reported for
              the smallest beam used.)
     area_eff - effective area of sources with input profile and
              unit amplitude, convolved with smallest beam.
     whitenoise1d[False]: (bool)Set this to ignore all anisotropic noise and
              filtering (mostly for debugging). 
              ****** This is currently not working*******
     pointsource[False]: (bool)Set this to True to tell the routine that you want the
              optimal filter for a point source. This effectively makes the
              profile the beam scale. Setting pointsource=True is equivalent
              to setting profiles = np.ones(ellmax)
     norm_unfilt - set this to normalize the output such that 
              sources of the shape of one of the input
              profiles will have amplitude in the output map
              equal to the amplitude of the input source
              before filtering or beam-smoothing.
     external_profile_norm - set this to the known area (in sr)
              under the real-space version of your
              source profiles, if you don't want the
              area calculated in ell space.  Only
              appropriate if NORM_UNFILT is set.
    OUTPUTS:
       2D Fourirer-space optimal filters, one for each input profile
    HISTORY:
       Big Bang + 300,000 years: Tom Crawford writes the initial IDL function. 
       Dec 15th, 2014: Tyler Natoli translates the function to python. 
    '''

    ellmax = len(skysigvar)-1 # 
    ell = np.arange(0,ellmax+1)

    ngridx,ngridy = np.shape(noisepsd)

    if pointsource:
        # This makes the optimal filtering for a point source
        #   which just makes the profile an array of 1s for now.
        #   later the beam will be multiplied into the profile, 
        #   which means the profile will effectively just be the beam sky
        profiles = np.ones(ellmax +1) 

    if external_profile_norm:
        ##################### I dont really understand what this does at all
        do_ext_norm = True

    if f_highpass or f_lowpass:
        if not scanspeed:
            print 'You must supply a scanspeed to use f_highpass or f_lowpass'
            print '....but you did not, so I am returning without doing a damn thing.'
            return
    
    # just making sure this not an integer
    reso_arcmin = float(reso_arcmin)
    reso_rad = reso_arcmin/60.*constants.DTOR

    # get a 2d grid of values appropriate fo 2d fft map
    ellg,ellx,elly = makeEllGrids([ngridx,ngridy],reso_rad)
    # create a flat ellg with values above ellmax set to ellmax
    #     this will be used later to make a 2d things in Fourier space
    flat_ellg = np.round(ellg.flatten()).astype(int)
    flat_ellg[flat_ellg>=ellmax] = ellmax -1
    
    # array of residual noise to use in optimal filter &
    #   variance calculation
    ivarr = np.zeros([ngridx,ngridy])+1E20

    # get 2d Fourier representations of timestream filters and beams
    # (if TRANS_FUNC_GRID is set, these will be ignored except for 1d
    #     debugging case.)
    #
    # trans_func_grid is the 2D transfer function (beam+filtering+etc..) in ell-space
    if trans_func_grid == None:
        trans_func_grid = np.zeros([ngridx,ngridy])
        hplp1d = np.ones(len(ellg[:,0]))# high-pass low-pass 1 dimmensional

        # make a gaussian beam to use if a beam is not supplied
        if bfilt == None:
            bfilt = calc_gauss_Bl(ell,fwhm_beam_arcmin)## move this gaussian beam maker somewhere else
        
        if f_highpass or f_lowpass:
            # the effective frequency is used to turn the high & low pass
            # frequency numbers into ell ranges
            f_eff = ellg[:,0]*scanspeed*DTOR/2./np.pi # f_eff is the effective frequency 

            if f_highpass:
                hplp1d *= np.e**(-1.*(f_highpass/f_eff)**6)
            if f_lowpass:
                hplp1d *= np.e**(-1.*(f_eff/f_lowpass)**6)

        if ell_highpass or ell_lowpass:
            if ell_highpass:
                hplp1d *= np.e**(-1.*(ell_highpass/ellg[:,0])**6)
            if ell_lowpass:
                hplp1d *= np.e**(-1.*(ellg[:,0]/ell_lowpass)**6)
            #****** the exact filter used is around line 214 of 
            #              spt_cdevel/mapping/decon_scan.c

        if not keepdc:
            # kill the dc component
            hplp1d[ellg[:,0] == 0] = 0 # ell=0 is the dc component

        for _num in range(0,ngridy):
            trans_func_grid[_num,:] = hplp1d
        if beam_grid != None:
            trans_func_grid *= beam_grid
        else:

            # multiply the beam into the trans_func_grid at every ell in ellg
            trans_func_grid *= np.reshape(bfilt[flat_ellg],np.shape(ellg))
    
    # get 2D ell-space representation of unwanted 1D spectra
    skysigvar2d = np.reshape(skysigvar[flat_ellg],np.shape(ellg))# skysigvar is 1D, make it 2D
    # skysigvar2d is in power, so multipy it by the trans function squared
    skysigvar2d *= abs(trans_func_grid)**2

    # calculate noise + cotaminant vaiance at all grid points
    skysigvar2d += noisepsd**2

    #     ivarr is the noise spectra (cmb+point source+pixel noise)
    ivarr = skysigvar2d/sigspect**2 # Im not really sure what sigspect is, 
    szbftemp = abs(trans_func_grid*sigspect)**2# sz beam temp?
    
    whsig = [szbftemp/np.max(szbftemp) > 1.E-4][0]# where the values are over 1/10,000
    nsig = whsig.tolist().count(True)
    # only use ell values up to where the transfer function has values over 1/10,000
    ellgmax = max(abs(ellg[whsig]))
    ellmax2use = ellgmax
    if ellgmax < ellmax:
        ellmax2use = ellgmax
    ell2use = np.arange(0,ellmax2use+1) # these are the good ells
    
    # 1d case for debugging  - interp(xneed,xhave,yhave)
    if debug1d:
        noisepsd1d = np.interp(ell2use,ellg[0:ngridx/2.,0],noisepsd[:ngridx/2,0])
        whgellg = np.where(ell2use > max(ellg[0:ngridx/2.,0]))[0]
        ngellg = len(whgellg)
        if ngellg > 0:
            noisepsd1d = noisepsd1d[:whgellg[0]-1]
        ivarr1d = (skysigvar[:int(ellmax2use)]*bfilt[:int(ellmax2use)]**2 + noisepsd1d**2)/sigspect**2 
        ivarr1d[0] = 1.E20
        #  note: ivarr1d only has the beam in the transfer function
        
    ivarr[0,0] = 1.E20# kill the dc component (the first bin)


#----------------------------------------------------------------
    # up until this point all we have made is the noise and unwanted signal convolved with the 
    #   transfer function, now we will start making the optimal filter
#----------------------------------------------------------------
    # make optimal filter(s) to use on synthesized map (using
    # Haehnelt-Tegmark prescription).
    try:
        nprofs,profile_len = np.shape(profiles) # nprofs = the number of profiles            
    except ValueError:# above will crash if profiles is a single array
        nprofs = 1
    if whitenoise1d:
        optfilts = np.zeros([ellmax2use+1.,nprofs],dtype = np.complex)
    else:
        optfilts = np.zeros([ngridx,ngridy,nprofs],dtype = np.complex)

    results = struct({})

    for _nprof in range(0,nprofs):
        if nprofs==1:
            prof_now = profiles
        else: 
            prof_now = profiles[_nprof]
       
        results[_nprof] = struct({'profile':prof_now})

        # If we want to get the 1D case working:
        # we must take extra care to ensure the profile, beam, and transfer function 
        #    are all the same length, they will be truncated to whatever the shortest
        #    one of those variables is
        #max(len(prof_now),len(bfilt))
        
        if debug1d:
            tauprime = prof_now*bfilt[:len(prof_now)] # tauprime is the profile*beam
            if norm_unfilt: # this sets the normalization....not really sure why we do this
                if do_ext_norm:
                    area_eff_1d = external_profile_norm[_n_prof]
                else:
                    area_eff_1d = 1./np.sum((2.*ell+1)/4./np.pi*prof_now)
            else:
                area_eff_1d = 1./np.sum((2.*ell+1)/4./np.pi*tauprime)
            tauprime = tauprime*area_eff_1d # 1d version of tau*filtering*beam
        

        area_eff_1am = 1./np.sum((2.*ell+1)/4./np.pi*prof_now*calc_gauss_Bl(ell,1.))
        # make 2d versions np.reshape(skysigvar[flat_ellg],np.shape(ellg))
        prof2d = np.reshape(prof_now[flat_ellg],np.shape(ellg)) # takes the 1d profile and makes it 2d
        tauprime_2d = prof2d*trans_func_grid # this is multipying the beam*transfer_func into the profile, both are in ell space
        du = (ellg[0,1] - ellg[0,0])/2./np.pi # du = dy -- although it really does not matter for this
        dv = (ellg[1,0] - ellg[0,0])/2./np.pi # dv = dx ---- anytime we use a du we use dv also

        if norm_unfilt:# this sets the normalization....not really sure why we do this
            if do_ext_norm:
                area_eff = external_profile_norm
            else:
                area_eff = 1./np.sum((2.*ell+1.)/4./np.pi*prof_now)
        else:
            area_eff = 1./np.sum(abs(tauprime_2d)*du*dv)

        tauprime_2d = tauprime_2d*area_eff

        optfilt = tauprime_2d/ivarr
        optfilt_norm = np.sum((np.conj(tauprime_2d)*optfilt).real)*du*dv
        optfilt = optfilt/optfilt_norm
        
        # 1d case for checking things and debugging
        if debug1d:
            optfilt1d = tauprime/ivarr1d
            optfilt1d_norm = np.sum(2*(ell+1)/4./np.pi*optfilt1d*tauprime)
            optfilt1d = optfilt1d/optfilt1d_norm
            sigma_1d = 1./sqrt(optfilt1d_norm)
            arcmin_sigma_1d = sigma_1d*area_eff_1d/area_1am                             
        # calculate variance of central signal value of features detected in
        # synthesized map using optimal filter (this is the variance on the
        # central value of the features after they have been smoothed by the
        # smallest beam among the bands).
        sigma = 1./np.sqrt(optfilt_norm)
        # scale to that same value if the minimum beam were 1 arcminute
        arcmin_sigma = sigma*area_eff/area_eff_1am
        # stuff values in results object
        if whitenoise1d: # do something different with this---------------------------------------------------------------------------------------------------------------------
            optfilts[:,_nprof] = optfilt1d
            sigmas[_nprof] = sigma_1d
            arcmin_sigmas[_nprof] = arcmin_sigma_1d
            area_effs[_nprof] = area_eff_1d
        
        results[_nprof]['optfilt'] = optfilt
        results[_nprof]['sigma'] = sigma
        results[_nprof]['arcmin_sigma'] = arcmin_sigma
        results[_nprof]['area_eff'] = area_eff
        results[_nprof]['trans_func_grid'] = trans_func_grid
        results[_nprof]['ell_grid'] = ellg
        results[_nprof]['beam_profile'] = bfilt
        results[_nprof]['skysigvar'] = skysigvar2d
        results[_nprof]['prof2d'] = prof2d
    if debug:
        print asdf
    return results
#    return optfilts

def getFTMap(map_data, sky_window_func):
    fft_shape = map_data.shape
    padded_shape = np.array(fft_shape)

    padded_map = np.zeros(padded_shape, dtype=map_data.dtype)
    padded_map[:map_data.shape[0], :map_data.shape[1]] = map_data
    if (sky_window_func.shape != padded_shape).any():
        # Make sure that the window function is compatible with the padding,
        # and pad it with zeros if necesary.
        if (sky_window_func.shape > padded_shape).any():
            raise ValueError("The input mask is larger than the padded map size!")

        padded_window = np.zeros(padded_shape, dtype=map_data.dtype)
        padded_window[:sky_window_func.shape[0], :sky_window_func.shape[1]] = sky_window_func
        sky_window_func = padded_window

    window_norm_factor = 1.

    #window_norm_factor = np.sqrt(np.sum(sky_window_func**2) / np.product(sky_window_func.shape)) - QYT: NORMALIZATION FACTOR THAT WE PROBABLY? DON'T NEED

    # Take the FFT and attach it to this map. If we padded, set the entire
    # area of the new map as "active" -- FTs don't have blank areas around
    # the edges.
    #fft_map = np.fft.fft2(padded_map * sky_window_func / window_norm_factor) - QYT see above
    print "not using the mask when FFT-ing the map"
    fft_map = np.fft.fft2(padded_map)
    fft_map = np.complex64(fft_map)
    return fft_map



def k_to_mjy(beamsize = 66, freq = 150, omega_b_in = None):
    if omega_b_in != None:
        idl_k_cmb_to_mjy_call = ["k_cmb_to_mjy = mjy_to_uk(uk,freq=freq,beamsize=beamsize,omega_b_in=omega_b_in,invert=invert)"]
        k_to_mjy = idl.idlRun(idl_k_cmb_to_mjy_call,
                              idl_startup_file='/home/tnatoli/idl/startx.pro',
                              uk = 1E6, # this is 1Kelvin in uKelvin
                              beamsize = beamsize,#FWHM in arcseconds
                              freq = freq, # frequency in GHz
                              omega_b_in = omega_b_in, # this should be area_eff, opening angle in steradians
                              # leaving omega_b as None seems to be alright also
                              invert = True)
    else:# the call below does not hand the IDL code a value for omega_b_in
        idl_k_cmb_to_mjy_call = ["k_cmb_to_mjy = mjy_to_uk(uk,freq=freq,beamsize=beamsize,invert=invert)"]
        k_to_mjy = idl.idlRun(idl_k_cmb_to_mjy_call,
                              idl_startup_file='/home/tnatoli/idl/startx.pro',
                              uk = 1E6, # this is 1Kelvin in uKelvin
                              beamsize = beamsize,#FWHM in arcseconds
                              freq = freq, # frequency in GHz
                              #omega_b_in = omega_b_in, # this should be area_eff, opening angle in steradians
                              # leaving omega_b as None seems to be alright also
                              invert = True)

    return k_to_mjy['k_cmb_to_mjy']
        
def k_to_mjy_py(year = 2012, freq = 150, alpha = -0.5, beamsize=1.1, omega_b=None):
    '''
    Returns the # to multiply a map in Kelvin by to get a map in mJy
    year[2012]: (int) The version of the array used to take data, either 2012 of 2013+
    
    freq[150]: (int) The detector frequency to use for the FTS spectra. The center frequency
                 of the bandpass will be calculated based on the FTS spectra for the respective
                 band and year. 
    
    alpha[-0.5]: (float) The power for the power law of the population spectra. 
                  -0.5 = radio
                     2 = Rayleigh-Jeans
                   3.5 = Dusty
    beamsize[1.1]: (float) Beam FWHM in arcmin. 
    omega_b[None]: (float) The solid angle of the detector in steradians. 
                 If None than a gaussian approximation of the beam will be integrated
                 over for the solid angle of the detector. If using a matched source
                 (optimal) filter the filter creation code will spit out 'area_eff'
                 which is the source profile (times the beam and filter tranfer function)
                 times the optimal filter integrated over is k_x and k_y. 
                 
    '''
    k_b = constants.k_b #1.3806503e-23, boltzmann constant in m^2 kg /(s^2 K)
    c = constants.c  #speed of light in m/s
    h = constants.h #6.626068e-34 m^2 kg/s
    Tcmb = 2.725 # in Kelvin

    # choose the right bandpass file
    if year < 2012:
        print "Don't be a dummy, SPTpol did not exist earlier than 2012. Try again. "
        return 
    if freq == 150:
        if year == 2012:
            spec_file = '/home/tnatoli/gits_sptpol/sptpol_software/scratch/tnatoli/fts/sptpol_150GHz_2012season.txt'
        if year >= 2013:
            # this will need an end cap on it if SPT-3G uses the same functions
            spec_file = '/home/tnatoli/gits_sptpol/sptpol_software/scratch/tnatoli/fts/sptpol_150GHz_2013season.txt'
    if freq == 90:
        if year == 2012:
            spec_file = '/home/tnatoli/gits_sptpol/sptpol_software/scratch/tnatoli/fts/sptpol_90GHz_2012season.txt'
        if year >= 2013:
            # this will need an end cap on it if SPT-3G uses the same functions
            spec_file = '/home/tnatoli/gits_sptpol/sptpol_software/scratch/tnatoli/fts/sptpol_90GHz_2013season.txt'
    # read in the appropriate bandpass file
    freqs,amps = files.readColumns(spec_file)
    freqs, amps = np.array(freqs),np.array(amps)
    if freqs[0] < 1E9:
        # freqs was probably supplied in GHz already, so make it in GHz now
        freqs *= 1E9
    dnu = freqs[1:]-freqs[:-1]
    if len(set(dnu)) ==1:
        dnu = dnu[0]
        # Calculate the band center
    else: # this is used if freqs are not evenly spaced
        amps = amps[:-1]
        freqs = freqs[:-1]
    nu0 = np.sum(freqs*amps*dnu)/np.sum(amps*dnu)
    print 'band center = ',nu0/1E9, 'GHz'

    # there are no A*Omega (entendue) factors below because that factor is already in the FTS spectrum data.
    def term(nu): return (h*nu/(k_b*Tcmb)) # this is just a shortcut for the exponential term in Planck's law
    def exp_term(nu): return np.exp(h*nu/(k_b*Tcmb)) # this is just a shortcut for the exponential term in Planck's law
    def dB_dT(nu): return ( (2*(h**2)*(nu**4))/( k_b*(c**2)*(Tcmb**2) ) ) * (exp_term(nu)/((exp_term(nu)-1)**2)) # derivative of Planck's law
    #   not actually using dB_dT
    def source_spec(nu,nu0,alpha): return (nu/nu0)**alpha
    def final(nu): return (2*k_b / c**2)*( (k_b*Tcmb/h)**2) * (term(nu)**4) * exp_term(nu)/ ((exp_term(nu)-1)**2)*1E26# 1E26 is for W/m**2/Hz to Jy

    if omega_b == None:
        omega_b = ((beamsize/60. * np.pi/180.)**2 )* 1.1331 # gaussian beam solid angle, beamsize is FWHM
       # The 1.1331 above is 4*ln2 which is part of the conversion between sigma and FWHM
       #  For a tophat beam you would use omega_b = np.pi*((theta/2)**2) if theta = FWHM

    #k_div_jy = 1./((np.sum(dB_dT(freqs)*amps*dnu)/np.sum(source_spec(freqs,nu0,alpha)*amps*dnu)) * 1.0E26 * omega_b) # Jy/K
    # the seemingly random 1E29 above is the conversion between W/m**2/Hz to mJy
    jy_div_k = np.sum(final(freqs)*amps*dnu)/np.sum(source_spec(freqs,nu0,alpha)*amps*dnu)

    print 'omega_b = ',omega_b
    print 'MJy/K/sr = ',jy_div_k*1E-6#,', or   mJy/K/sr = ',jy_div_k*1E3
    print 'For point source, Jy/K= ',jy_div_k*omega_b, ',  or  mJy/K/sr = ',jy_div_k*omega_b*1E3
    # dB_dT_0 = dB_dT(nu0)
##### dB_dt = (2*(h_p^2)*(nu_o^4.0)/(k_b*(Tcmb*c)^2))*exp_term/((exp_term-1)^2)*1e20 # from lindsey

    return jy_div_k*omega_b

def convert_map_to_mjy(Map, beamsize=66, omega_b = None, conv_factor=None):
    '''
    the conv_factor arguement overrules all others and just uses that number as the conversion factor.
    '''
    if conv_factor == None:
        freq = Map.getTOnly().band
    
        times = spttime.filenameToTime(Tmap.maps_included)
        year = times[0].year
 
        conv_factor = k_to_mjy_py(beamsize=beamsize, freq=freq, omega_b = omega_b)
    else:
        pass
    
    for _pol in Map.pol_maps:
        Map.pol_maps[_pol].map *= conv_factor
        Map.pol_maps[_pol].units = '~mJy'
        try:
            Map.pol_maps[_pol].map_sigma *= conv_factor
        except:
            pass
            
    Map.setMapAttr('units','mjy')
    Map.setMapAttr('k_to_mjy_conversion_factor',conv_factor)

     

def find_group_wrapper(Map, sigma_noise, offset=0, nsigma=5, mask=None, return_raw=False):
    
    Tmap = Map.getTOnly()
    if mask == None:
        mask = np.ones(Tmap.active_shape)
        print 'Not using a mask, this is not advised.'

    idl_find_groups_call = ['find_groups_funk,Map, signoise, offset, plist, gbounds, nsigma=nsigma,xcen=xcen,ycen=ycen,maxvals=maxvals,sigvals=sigvals']

    plist,gbounds,xpeak_ps,ypeak_ps,maxvals_ps,sigvals_ps=[],[],[],[],[],[] 

    call_output = idl.idlRun(idl_find_groups_call,
                             idl_startup_file='/home/tnatoli/idl/startx.pro',
                             #print_screen=True,
                             return_all = True,
                             Map = Map.pol_maps.T.map*mask,
                             signoise = sigma_noise,
                             plist = plist, # list of all pixels above threshold in groups
                             offset = offset,
                             nsigma = nsigma, 
                             gbounds = gbounds, # group boundaries
                             xcen = xpeak_ps, # x-location of group centers
                             ycen = ypeak_ps, # y-location of group centers
                             maxvals = maxvals_ps, # height in map units of found objects
                             sigvals = sigvals_ps, # significance value of found objects
                             )
    if return_raw:
        return call_ouput
    else:
        return make_finder_output_nice(call_output,Map,nsigma)

def make_finder_output_nice(finder_output,Map,nsigma):

    Tmap = Map.getTOnly()
    times = spttime.filenameToTime(Tmap.maps_included)
    decs, ras = Tmap.pix2Ang([finder_output['ycen'],finder_output['xcen']])
    nice_output = dict({'maxvals':finder_output['maxvals'],
                        'plist':finder_output['plist'],
                        'sigvals':finder_output['sigvals'],
                        'map_xpix':finder_output['xcen'],
                        'map_ypix':finder_output['ycen'],
                        'sigma':finder_output['signoise'],
                        'dec':decs,
                        'ra':ras,
                        'band':Tmap.band,
                        'units':Tmap.units,
                        'map_filename':Map.from_filename,
                        'n_sigma':nsigma,
                        # get a start,end, and median time for the coadd
                        'start_time':min(times),
                        'stop_time':max(times),
                        'avg_time':min(times)+(max(times)-min(times))/2,
                        })
    return nice_output

def find_sources_in_map(filtered_map,filt_output, nsigma=5, mjy=True,
                        beamsize=108, omega_b=None,
                        mask=None, 
                        txt_filename=None,
                        pkl_filename=None,
                        map_out_filename=None
                        ):
                        
    '''
    beamsize - only used for the mjy conversion
    omefa_b - only used for the mjy conversion
    
    txt_file - set this equal to a filename to deposit of txt file to 
                that file with the sources in it
    '''
    Map = filtered_map.copy()
    print 'I am only using the first filter in filt_output for now.'
    filt_output = filt_output[0]
    map_sigma = filt_output['sigma']

    if mjy:
        convert_map_to_mjy(Map, beamsize=beamsize, omega_b = omega_b)
        print 'k_to_mjy =  ',k_to_mjy( freq=Map.getTOnly().band, omega_b = omega_b, beamsize = beamsize)
        map_sigma *= k_to_mjy( freq=Map.getTOnly().band, omega_b = omega_b, beamsize = beamsize)

    output = find_group_wrapper(Map, map_sigma, nsigma=nsigma, mask=mask)
    
    if pkl_filename:
        pkl.dump(output,open(pkl_filename,'w'))
    
    if txt_filename:
        make_txt_file_from_source_list( output,txt_filename, Map=filtered_map)
        
    if map_out_filename:
        Map.writeToHDF5(map_out_filename)

    return output


def make_txt_file_from_source_list( info,txt_filename, Map=None ):
    '''
    a Map only needs to be supplied if the info given is in the raw format from the 
       source finder (the Map is only used to go from pix to ra & dec
    txt_filename is the filename of the output text file
    Right now this assumes the map (and pt sources in the info) are in mJy already
    '''

    # find the format of info
    try:
        # if info is already conditioned it will have an 'ra' field
        info['ra']
    except KeyError:
        info = make_finder_output_nice(info,Map)
        
    #####################################
    # I don't know how to find the radius, so setting it equal to 0 for now
    info['radius'] = np.zeros(len(info['sigvals']))
    ####################################

    txt_file = open(txt_filename,'w')

    txt_file.write('###########################################################\n')
    txt_file.write('#  {}\n'.format(info['map_filename']))
    txt_file.write('#  This file contains RA & dec & mask radius for \n')
    txt_file.write('#  point sources detected above {} sigma \n'.format(info['n_sigma']))
    txt_file.write('#  in the {}.\n'.format(info['band']))
    txt_file.write('#  The error on the flux is {} mJy.\n'.format(info['sigma']))
    txt_file.write('###########################################################\n')
    txt_file.write('# Index RA              DEC             Radius      S/N      Flux\n')
    txt_file.write('#       (deg)           (deg)           (deg)       --        mJy\n')
    

    temp = zip(info['sigvals'],info['ra'],info['dec'],info['radius'],info['maxvals'])
    temp.sort()
    sigvals,ra,dec,radius,maxvals = zip(*temp)

    for _sig,_ra,_dec,_rad,_max in temp:
#        txt_file.write('{}\t{}\t{}\t{}\t{}'.format(info.ra[_i],
        txt_file.write('{}\t{}\t{}\t{}\t{}\n'.format(_ra,_dec,_rad,_sig,_max))
    txt_file.close()
    



def cycle_over_dir_maps(directory, pre_name='smap_set_',
                        beam=None, nsigma = 5 ,output_dir = None,
                        mask = None,removeWeight=True,
                        noisepsd = None):
    '''
    This script cycles over every map in a directory doing the following:
          - filters the map with a point source matched filter
          - finds sources in the filtered map
          - saves the sources found to a pkl and txt
          - saves the filtered map
    directory: (filepath) This is the directory that contains the maps to be 
               filtered. 
    
    pre_name['smap_set_': (str) This is the pre-phase in front of every
               to be filtered map (usually 'smap_set_' is right for bundles)
    beam[None]: (filepath or beam dict) If you want to use a custom beam for 
               the matched source filter you can specify the location of the 
               custom beam or just pass this function a beam that has already
               been read in and is in standard beam format. You can usually get
               away with setting this to None, which means a default gaussian beam
               with a fwhm of 1.1 arcmin is used.  (1.1 arcmin is for 150GHz, 
               1.8 arcmin is for 90GHz)
    nsigma[5]: (int) The signal to noise to look for sources down to. 
    output_dir[None]: (filepath) The directory where the filtered maps and lists
               of sources found will be saved to. 
    mask[None]: (filepath or ndarray) Either the filepath to a pkl file holding the 
               mask as an ndarray or an ndarray (which is the mask) which has already
               been read in. If mask = None a mask will be looked for in the 'directory'
               given with the expected name of 'apod_mask.pkl'.
    removeWeight[True]: (Bool) Remove weight prior to filtering the map, this should
               almost always be set to True. 
    noisepsd[None]: (ndarray) If not None this is used as the 2d noisepsd instead of the 
               fft of the given map. 
    '''
    map_filenames = glob(directory+'/'+pre_name+'*.h5')
    if mask == None:
        mask = files.read(directory+'/apod_mask.pkl')
        print "I am using the apod mask I found in %s named 'apod_mask.pkl'"%(directory)
    else:
        # if mask is a string then assume it is a path to a mask pkl
        if type(mask) == str:
            mask = files.read(mask)
        # if mask is already read in then nothing will happen to it

    # create the output directory if given and it does not exist
    #   or use my own naming scheme if output_dir not supplied
    if not output_dir:
        output_dir = directory+'/sources_found/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_dir+'/txt_files/'):
        os.makedirs(output_dir+'/txt_files/')
    if not os.path.exists(output_dir+'/filtered_maps/'):
        os.makedirs(output_dir+'/filtered_maps/')

    # check if the 2d noisepsd is supplied, if not use the 
    #    fft as the noise model
    if noisepsd != None:
        use_fft_as_noise = False
    else:
        use_fft_as_noise = True

    # read the beam in once so we do not need to read in in for every map
    if beam:
        # make sure it is a pkl file and has not already been read in
        try:
            beam['B_ell']
            # if this works the beam has already been read in
        except TypeError:
            # read in the beam now, assuming we were handed a pkl
            beam = files.read(beam,'r')
        # this is the 150 GHz 2012 beam
        #beam = '/data/sptdat/analysis_products/deep_field_bb/beams/beam_150ghz_2012_abscal.pkl'
    
    for _num,_file in enumerate(map_filenames):
        print '----reading {} of {} files, {}'.format(_num,len(map_filenames),_file)
        _map = files.read(_file)
        map_num = _file.rpartition(pre_name)[-1].rpartition('.')[0]

        txt_filename = output_dir+'/txt_files/ps_in_'+str(map_num)+'.txt'
        pkl_filename = output_dir+'/ps_results_'+str(map_num)+'.pkl'
        map_out_filename = output_dir+'/filtered_maps/filt_map_'+str(map_num)+'.h5'

        output,filt_map = call_matched_filt_point(_map,mask=mask,
                                                  ell_lowpass=20000,ell_highpass=400,
                                                  beam = beam,
                                                  use_fft_as_noise = use_fft_as_noise,
                                                  noisepsd = noisepsd,
                                                  plot_map=False,return_filtered_map=True,
                                                  use_idl=False,
                                                  fft_smooth_arcmin = 2,
                                                  removeWeight=removeWeight)

        junk = find_sources_in_map(filt_map,output, nsigma=nsigma, 
                                   mjy=True,
                                   mask=mask, txt_filename=txt_filename,
                                   pkl_filename=pkl_filename,
                                   map_out_filename=map_out_filename)

def dir_noise_2d(directory, pre_name = 'smap_set_',
                 output_name = 'dir_noise_2d.pkl',
                 mask = None, fft_smooth_arcmin=2.0,
                 plot=False):
    '''
    This fuction opens up every map within a directory and creates
    an average noise in 2d
    The average FFT is saved as the given 'output_name' in the map
       directory given.
    '''
    map_filenames = glob(directory+'/'+pre_name+'*.h5')
    map_filenames.sort()

    if type(mask) == str:
        # if mask is a string then assume it is a path to a mask pkl
        mask = files.read(mask)
    elif type(mask) == np.ndarray:
        pass
    else:
        mask = files.read(directory+'/apod_mask.pkl')
        print "I am using the apod mask I found in %s named 'apod_mask.pkl'"%(directory)


    pl.figure(10)
    pl.clf()
        
    for _num,_file in enumerate(map_filenames):
        print '----reading {} of {} files, {}'.format(_num,len(map_filenames),_file)
        Map = files.read(_file)
        Map.removeWeight()
        map_num = _file.rpartition(pre_name)[-1].rpartition('.')[0]

        if _num == 0:
            # the variables below should not change map to map, so we will only 
            #   set there values once. 
            Tmap = Map.getTOnly()
            reso_rad = Tmap.reso_rad
            reso_arcmin = Tmap.reso_arcmin
            ngridx, ngridy = Tmap.active_shape
            map_size = ngridx*ngridy
            ellg,ellx,elly = makeEllGrids([ngridx,ngridy],reso_rad)

        # take the fft of the T map and return the fft scaled by the pixel size
        map_fft = Map.getFTMap(return_type='T', sky_window_func=mask)
        map_fft = np.sqrt((np.conjugate(map_fft)*map_fft).real)*(reso_rad/np.sqrt(map_size))
        
        #### Below is getting a single number from each map so I can roughly 
        ###       see how uniform the noise level is
        # convert the fft_smooth_arcmin to the number of pixels to smooth
        smooth_pixels = fft_smooth_arcmin/reso_arcmin
        smoothed_fft = sp.ndimage.gaussian_filter(map_fft, smooth_pixels, mode='wrap')
        flat_ellg = ellg.flatten()
        flat_smoothed_fft = smoothed_fft.flatten() 
        fft_rms = np.sqrt(np.mean(flat_smoothed_fft[(flat_ellg>4000) & (flat_ellg<6000)]**2))

        pl.plot(ellx[0][0:ngridx/2],np.average(map_fft,axis=0)[0:ngridx/2],alpha=0.6)

        if _num == 0:
            avg_fft = map_fft
            map_count=1
            all_rms = [fft_rms]
            maps_in = [len(Map.getTOnly().maps_included)]
        
        else:
            avg_fft += map_fft
            map_count+=1
            all_rms.append(fft_rms)
            maps_in.append(len(Map.getTOnly().maps_included))

    avg_fft /=map_count
    pl.plot(ellx[0][0:ngridx/2],np.average(avg_fft,axis=0)[0:ngridx/2],'k',lw=2)
    pkl.dump(avg_fft,open(directory+'/'+output_name,'w'))

    return avg_fft, all_rms,maps_in


def filt_maps_in_dir(directory, filter_to_apply,
                     pre_name='smap_set_',
                     #beam=None, 
                     output_dir = None,
                     mask = None,removeWeight=True,
                     #beamsize=1.1, # beamsize in arcmin
                     in_mjy=True, # convert the output map to mjy using a hard-coded conversion!!!!
                     processes = 6,
                     ):
    '''
    First this function gets a noise estimate from every 
    This script cycles over every map in a directory doing the following:
          - filters the map with a point source matched filter
          - saves the filtered map
    directory: (filepath) This is the directory that contains the maps to be 
               filtered. 
    
    pre_name['smap_set_': (str) This is the pre-phase in front of every
               to be filtered map (usually 'smap_set_' is right for bundles)
#    beam[None]: (filepath or beam dict) If you want to use a custom beam for 
               the matched source filter you can specify the location of the 
               custom beam or just pass this function a beam that has already
               been read in and is in standard beam format. You can usually get
               away with setting this to None, which means a default gaussian beam
               with a fwhm of 1.1 arcmin is used. (1.1 arcmin is for 150GHz, 
               1.8 arcmin is for 90GHz)
    output_dir[None]: (filepath) The directory where the filtered maps and lists
               of sources found will be saved to. 
    mask[None]: (filepath or ndarray) Either the filepath to a pkl file holding the 
               mask as an ndarray or an ndarray (which is the mask) which has already
               been read in. If mask = None a mask will be looked for in the 'directory'
               given with the expected name of 'apod_mask.pkl'.
    removeWeight[True]: (Bool) Remove weight prior to filtering the map, this should
               almost always be set to True. 
    processes[6]: (int) The number of parallel processes to use to filter the maps.
    '''
    map_filenames = glob(directory+'/'+pre_name+'*.h5')
    map_filenames.sort()
    if mask == None:
        mask = files.read(directory+'/apod_mask.pkl')
        print "I am using the apod mask I found in %s named 'apod_mask.pkl'"%(directory)
    else:
        # if mask is a string then assume it is a path to a mask pkl
        if type(mask) == str:
            mask = files.read(mask)
        # if mask is already read in then nothing will happen to it

    # create the output directory if given and it does not exist
    #   or use my own naming scheme if output_dir not supplied
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # save a copy of the filter used to the output directory
    pkl.dump(filter_to_apply,open(output_dir+'/filter_used.pkl','w'))

    # # read the beam in once so we do not need to read in in for every map
    # if beam:
    #     # make sure it is a pkl file and has not already been read in
    #     try:
    #         beam['B_ell']
    #         # if this works the beam has already been read in
    #     except TypeError:
    #         # read in the beam now, assuming we were handed a pkl
    #         beam = files.read(beam,'r')
    #     # this is the 150 GHz 2012 beam
    #     #beam = '/data/sptdat/analysis_products/deep_field_bb/beams/beam_150ghz_2012_abscal.pkl'

    # the filter handed to us was probably the output from the calc_optfilt_2d,
    #   if it was, extract what we want
    try:
        optfilt = filter_to_apply[0].optfilt
        filt_sigma = filter_to_apply[0].sigma
    except AttributeError:
        optfilt = filter_to_apply
        filt_sigma = 0.0

    # multiprocess the actual map filtering by the processes requested
    pool = multiprocessing.Pool(processes=processes)
    nmaps = len(map_filenames)
    junk_output = pool.map(single_filter_save_map,
                           zip(map_filenames,
                               itools.repeat(mask,nmaps),
                               itools.repeat(optfilt,nmaps),
                               itools.repeat(filt_sigma,nmaps),
                               itools.repeat(in_mjy,nmaps),
                               itools.repeat(output_dir,nmaps),
                               itools.repeat(nmaps,nmaps),
                               range(0,nmaps,1),
                               itools.repeat(pre_name,nmaps),
                               ))
    return 

def single_filter_save_map(input_params):
    ''' 
    This function is just a worker for filt_maps_in_dir so that processes can 
        be ran in parallel. Please refer any questions about this code to filt_maps_in_dir.
    '''
    _file,mask,optfilt,optfilt_sigma,in_mjy,output_dir,num_files_total,_num,pre_name = input_params
    print '----reading {} of {} files, {}'.format(_num,num_files_total,_file)
    # read in map, will read in both T only and Pol maps 
    #    and then treat everything as a pol map from now on
    _map = files.readTOnlyMapAsPolMap(_file)
    _map.removeWeight()
    # extract/create some useful items for later
    map_num = _file.rpartition(pre_name)[-1].rpartition('.')[0]
    map_out_filename = output_dir+'/filt_map_'+str(map_num)+'.h5'




    filt_map = filterMap(_map,optfilt, apod_mask=mask)
    filt_map.setMapAttr('filt_sigma',optfilt_sigma)
    filt_map.setMapAttr('source_filtered_map',True)

    # convert the map to mjy
    if in_mjy:
        # below is the correct conversion for 150 GHz in 2012, 2013 should be 47.966
        conv_factor = 47780 # this is for 150 GHz in 2012
        # below is the old INCORRECT coversion used
        # conv_factor = 123764.81294631958
        print ' -- I am using a hardcoded value of %.5g K/mJy for the mjy conversion! --'%(conv_factor)
        for _pol in filt_map.pol_maps:
            filt_map.pol_maps[_pol].map *= conv_factor
            filt_map.pol_maps[_pol].units = '~mJy'
        filt_map.setMapAttr('units','mjy')
        filt_map.setMapAttr('k_to_mjy_conversion_factor',conv_factor)
    # compute map sigma, only spots where the mask is exactly 1 to prevent the masking from 
    #    messig with the statistics of the map
    bi_mask = np.ones(np.shape(mask))
    bi_mask[mask != 1.0] = 0

    map_vals = (filt_map.getTOnly().map.flatten()*bi_mask.flatten())[np.isfinite(filt_map.getTOnly().map.flatten())]
    # get rid of any values exactly zero also, these are just dead areas of the map
    map_vals = map_vals[map_vals!=0.0]
    map_vals.sort()
    
    med = np.median(map_vals)
    _std = np.std(map_vals[int(len(map_vals)*.005):-int(len(map_vals)*.005)]) # just getting an esitmate for the stddev
    
    pl.ioff()
    pl.clf()
    pl.title(_file,fontsize=17)
    hist_out = pl.hist(map_vals,bins=np.linspace(med-5*_std,med+5*_std,100))
    fit_center,fit_sigma = np.abs(plotting.gaussFit(hist_out,return_values = True))
    pl.ion()
    # stick the calculated sigma value into the map
    #     getTOnly retuns a copy of the map, so it is not helpful for us here
    #   filt_map.getTOnly().setMapAttr('map_sigma',fit_sigma)
    filt_map.pol_maps.T.setMapAttr('map_sigma',fit_sigma)
    print map_out_filename
    filt_map.writeToHDF5(map_out_filename)
    return 'done'


def filter_one_map(_map,optfilt,optfilt_sigma,mask,in_mjy=True):
    '''
    this essentially is the same as single_filter_save_map, but this assumes map is already read in. 
    '''

    filt_map = filterMap(_map,optfilt, apod_mask=mask)
    filt_map.setMapAttr('filt_sigma',optfilt_sigma)
    filt_map.setMapAttr('source_filtered_map',True)
    
    # convert the map to mjy
    if in_mjy:
        conv_factor = 123764.81294631958#10.610835488478187 <- I dont know where the 10 factor came from
        print ' -- I am using a hardcoded value of %.5g K/mJy for the mjy conversion! --'%(conv_factor)
        for _pol in filt_map.pol_maps:
            filt_map.pol_maps[_pol].map *= conv_factor
            filt_map.pol_maps[_pol].units = '~mJy'
        filt_map.setMapAttr('units','mjy')
        filt_map.setMapAttr('k_to_mjy_conversion_factor','mjy')
    # compute map sigma, only spots where the mask is exactly 1 to prevent the masking from 
    #    messig with the statistics of the map
    bi_mask = np.ones(np.shape(mask))
    bi_mask[mask != 1.0] = 0

    map_vals = (filt_map.getTOnly().map.flatten()*bi_mask.flatten())[np.isfinite(filt_map.getTOnly().map.flatten())]
    # get rid of any values exactly zero also, these are just dead areas of the map
    map_vals = map_vals[map_vals!=0.0]
    map_vals.sort()
    
    med = np.median(map_vals)
    _std = np.std(map_vals[int(len(map_vals)*.005):-int(len(map_vals)*.005)]) # just getting an esitmate for the stddev
    
    pl.ioff()
    pl.clf()
    #pl.title(_file,fontsize=17)
    hist_out = pl.hist(map_vals,bins=np.linspace(med-5*_std,med+5*_std,100))
    fit_center,fit_sigma = np.abs(plotting.gaussFit(hist_out,return_values = True))
    pl.ion()
    # stick the calculated sigma value into the map
    #     getTOnly retuns a copy of the map, so it is not helpful for us here
    #   filt_map.getTOnly().setMapAttr('map_sigma',fit_sigma)
    filt_map.pol_maps.T.setMapAttr('map_sigma',fit_sigma)
    return filt_map


def make_white_noise_map(Map = None, mask=None, directory = None,
                         noise_level = None):
    '''
    directory is the bundle directory, it also looks to this directory if no mask is given
    '''

    if Map:
        if type(Map)==str:
            Map = files.read(Map)
    else:
        # it does not matter the exact map, we are just stealing parameters out of it
        #   and using it for a noise estimate
        Map = files.read(directory+'/smap_set_000.h5')

    if mask == None:
        mask = files.read(directory+'/apod_mask.pkl')
    else:
        # if mask is a string then assume it is a path to a mask pkl
        if type(mask) == str:
            mask = files.read(mask)
        # if mask is already read in then nothing will happen to it
        
    noise_map = Map.copy()
    noise_map.removeWeight()
    if not noise_level:
        psd = maps.calculateNoise(noise_map.getTOnly(),mask = mask,ell_range=[4000,6000])
        noise_level = np.sqrt(np.mean(psd['TT'][np.where((psd.ell['TT']>4000) & (psd.ell['TT']<6000))[0]]))
    
    noise_map.pol_maps.T.map = np.random.normal(loc=0,size=noise_map.pol_maps.T.active_shape,scale = noise_level)
    noise_map.pol_maps.T.setMapAttr('white_noise_level',noise_level)

    return noise_map

def filt_weighted_map(Map,mask=None):
    '''
    This is a shoty attempt to filter weighted maps.
    Do not use this, it is shit, there is no way to remove the weights after filtering the maps. 
    '''
    if type(Map)==str:
        Map = files.read(Map)

    if mask==None:
        # make an apod mask if one not provided
        print 'making border apodization mask'
        mask = maps.makeBorderApodization(Map, apod_type='cos', weight_threshold=0.3, radius_arcmin=30.,
                                          zero_border_arcmin=4., smooth_weights_arcmin=10., verbose=True)
    else:
        if type(Map)==str:
            mask = files.read(mask)
    if not Map.weighted_map:
        Map.addPixelWeight()
    outputw,filt_mapw_obj = call_matched_filt_point(Map,mask=mask, ell_lowpass=20000,ell_highpass=400,use_fft_as_noise=True,
                                                    plot_map=True,
                                                    return_filtered_map=True,fft_smooth_arcmin = 2,removeWeight=False)

    Map.removeWeight()

    output,filt_map_obj = call_matched_filt_point(Map,mask=mask, ell_lowpass=20000,ell_highpass=400,use_fft_as_noise=True,
                                                  plot_map=True,
                                                  return_filtered_map=True,fft_smooth_arcmin = 2,removeWeight=False)
    filtw = outputw[0].optfilt

    weight_obj = Map.copy()
    weight_obj.pol_maps.T.map = Map.getTOnly().weight
    filt_weight_obj = filterMap(weight_obj,filtw, apod_mask=mask)
    
    filt_mapw = filt_mapw_obj.pol_maps.T.map
    filt_weight = filt_weight_obj.pol_maps.T.map

    pl.figure(1)
    pl.clf()
    plotting.spt_imshow(filt_mapw/filt_weight,mask=mask)
    print asdf

def call_matched_filt_point(Map,
                            cmb = True,
                            ellmax = 30000,
                            skysigvar = None,
                            white_noise_level = None, # given in K**2
                            use_fft_as_noise = True, # if true the fft of the given map is used as the noise
                            noisepsd = None, 
                            beam_grid = None, #2D beam grid
                            beam = None, #assuming that beam array has ell as its index
                            fwhm_beam_arcmin = 1.1,# this was the beam for 150GHz in 2012, 1.8 is for 90GHz
                            ell_highpass = 400, # this is what tyler cluster maps are
                            ell_lowpass = 20000, # this is what tyler cluster maps are
                            trans_func_grid = None,
                            mask = None, # used to calculate the noise
                            arcmin_sigma = None,
                            norm_unfilt = None,
                            fft_smooth_arcmin = 2, # the smoothing factor in arcmin
                            external_profile_norm = None,
                            plot_map = False,
                            debug = False,
                            return_filtered_map = False,
                            use_idl = False,
                            removeWeight = True,
                            verbose = False,
                            reso_arcmin = 0.25,
                            ngridx = 3000,
                            ngridy = 3000,
                            f_highpass = None, 
                            f_lowpass = None,
                            scanspeed = None,   
                            ):

    ''' This function is just a wrapper for calc_optfilt_oneband_2d above. 
        
    You feed this function a map and it extracts parameters 
       to make the matched source filter.
    This code assumes your profile is for a point source. 
    INPUTS:
       Map: (path to .h5 or map object) A map that parameters 
          will be taken from to make the matched filter. 
          This map should be the same shape 
          and resolution as all maps the output filter will be applied to. 
          If white_noise_level = True, then the white noise level will be the
          average(map_psd[6000 < ell < 10000]) of this map. 
       cmb[True]: (bool or file) If True use the nominal Planck (first release)
          cmb spectra in the unwanted map signal.
          If cmb = file, it is assumed that file is a camb
          file and will use this file as the cmb spectra in the unwanted
          map signal
       white_noise_level[None]: (bool or float) If a float, the float given 
          will be the white noise level given to the filter maker. If True, 
          then the white_noise_level will be taken from the Map supplied
          and will be the average(map_psd[6000 < ell <10000])
       skysigvar[None]: 
       
       plot_map[False]: bool
       
       debug[False]: bool
       
       fft_smooth_arcmin[2]: (int)Number of arcmins to smooth the fft
       
       returned_filtered_map[False]: (bool) Returns the m_filt structure but also returns
          a map object where the maps have been filtered (but only the T map has been 
          properly filtered). 
       use_idl[False]: (bool) Be a dummy and use the IDL script to create the filter.
       verbose[False]: (bool) Print extra information to the screen.
       
       removeWeight[True]: (bool) for testing.
    OUTPUT:
       The optimal filter for a point source and the noise in the given map. If 
       return_filtered_map = True then the filtered map object is also returned.
    '''

    if type(Map)==str:
        Map = files.read(Map)
        # otherwise assume the map is already read in

    # get the resolution from the T map
    reso_rad = reso_arcmin * constants.DTOR / 60.

    map_size = ngridx*ngridy
    ellg,ellx,elly = makeEllGrids([ngridx,ngridy],reso_rad)

    if skysigvar != None:
        ellmax = len(skysigvar)-1
    ell = np.arange(0,ellmax)

    if skysigvar == None:
        skysigvar = np.zeros(ellmax)
        if cmb != False:
            if cmb == True:
                cls = readCAMBCls('/data/tnatoli/camb/planck_lensing_wp_highL_bestFit_20130627_massless0p046_massive3_lensedtotCls.dat',
                                  as_cls = True, extrapolate = True, 
                                  as_uksq = False, max_ell = ellmax-1)
            elif type(cmb) == str:
                # units on cls will be K**2
                cls = readCAMBCls(cmb, as_cls = True, extrapolate = True, 
                                  as_uksq = False, max_ell = ellmax-1)
            else:
                # assume cmb is a spectra that has already been read in and is the 
                #   standard form ---------------- make this more general
                cls = cmb
    
        skysigvar[cls['ell']] = cls['cl']['TT']

    if white_noise_level:
        if white_noise_level==True:
            # get the noise level from the input map
            if removeWeight:
                Map.removeWeight()
            psd = maps.calculateNoise(Map,mask = mask,ell_range=[4000,6000])
            # having noise at every ell may become interesting later
            ## noise_ells = np.interp(np.arange(0,psd['ell']['TT'][-1]),psd['ell']['TT'],psd['TT'])
            # take the white noise level to be the noise between 4000<ell<6000
            white_noise_level = np.sqrt(np.mean(psd['TT'][np.where((psd.ell['TT']>4000) & (psd.ell['TT']<6000))[0]]))
            if verbose:
                print 'white noise level is  '+str(white_noise_level)+ ' K-rad      or  ' +str(white_noise_level*60/DTOR*1E6)+' uK-arcmin'

            # below was a different way to calculate the white noise level 
            #      I did it orignally because I did not trust the above calculation
            #      but it turned out it is actually fine
            # ellg,ellx,elly = makeEllGrids([ngridx,ngridy],reso_rad)
            # map_fft = Map.getFTMap(return_type='T', sky_window_func=mask)
            # map_fft = np.sqrt((np.conjugate(map_fft)*map_fft).real)*reso_rad
            # flat_ellg = ellg.flatten()
            # flat_map_fft = map_fft.flatten() 
            # if verbose:
            #     print 'white noise level is  {} K-rad '.format(np.sqrt(np.mean(flat_map_fft[(flat_ellg>4000) & (flat_ellg<6000)]**2))*DTOR/60)
     

        # if white_noise_level is not a bool, assume it is a float of the noise level
        noisepsd = np.ones(Map.getTOnly().active_shape)*white_noise_level

    elif use_fft_as_noise:
        # this assumes that the map is dominated by noise
        if removeWeight:
            Map.removeWeight()

        # take the fft of the T map and return the fft scaled by the pixel size
        map_fft = Map.getFTMap(return_type='T', sky_window_func=mask)
#        map_fft = np.sqrt((np.conjugate(map_fft)*map_fft).real)*(Map.reso_rad**2) # - pretty sure this squared factor should notbe here
        map_fft = np.sqrt((np.conjugate(map_fft)*map_fft).real)*(reso_rad/np.sqrt(map_size))
        
        # ellg,ellx,elly = makeEllGrids([ngridx,ngridy],reso_rad) - I have moved this defintion earlier
        # convert the fft_smooth_arcmin to the number of pixels to smooth
        smooth_pixels = fft_smooth_arcmin/reso_arcmin
        smoothed_fft = sp.ndimage.gaussian_filter(map_fft, smooth_pixels, mode='wrap')
        flat_ellg = ellg.flatten()
        flat_smoothed_fft = smoothed_fft.flatten() 
        fft_rms = np.sqrt(np.mean(flat_smoothed_fft[(flat_ellg>4000) & (flat_ellg<6000)]**2))
        if verbose:
            print 'rms of fft for 4,000< rms <6,000 = ',fft_rms,' K-rad    or ',fft_rms/DTOR*60*1E6,' uK-arcmin'

        noisepsd = smoothed_fft

    elif white_noise_level==False and noisepsd==False:
        print 'you must either set white_noise_level or supply a noisepsd'
        return
        
    ##### not dealin with beam_grid right now #####
    # we want the beam_grid to have the same shape as the 
    #    trans_func_grid in calc_optfilt_oneband_2d
    #######################################################
    #get the beam in the same ells as everything else
    if (trans_func_grid == None) and (beam != None):
        beam_1d_ell = np.loadtxt(beam)[:,0]
        beam_1d = np.loadtxt(beam)[:,1]   
        bfilt = np.interp(ell, beam_1d_ell, beam_1d)
    else:
        bfilt = None
        #calc_gauss_Bl(ell,fwhm_beam_arcmin)
    if use_idl:
        idl_to_run = ['filt_idl = calc_optfilt_oneband_2d_funk(profiles,skysigvar,noisepsd,reso_arcmin,fwhm_beam_arcmin=beam_arcmin, ell_highpass=ell_highpass, ell_lowpass=ell_lowpass, bfilt=bfilt, pointsource=True, sigspect=sigspect, keepdc=keepdc, sigma=sigma, tf_out=tf_out, a_eff_out=a_eff_out)']

        filt_idl = idl.idlRun(idl_to_run,
                              idl_startup_file='/home/tnatoli/idl/startx.pro',
                              profiles = -1,# 
                              skysigvar = skysigvar,# THis is the 1d spectra of all unwanted sky signal (like the CMB signal)
                              noisepsd = noisepsd,
                              reso_arcmin = reso_arcmin,
                              beam_arcmin = fwhm_beam_arcmin,
                              ell_highpass = ell_highpass,
                              ell_lowpass = ell_lowpass,
                              bfilt = bfilt,
                              pointsource = True,
                              sigspect = 1,
                              keepdc = False,
                              sigma = 1,
                              )
    else:
        m_filt =calc_optfilt_oneband_2d([0,0],# profiles is not used with pointsource=True
                                        skysigvar, noisepsd, reso_arcmin,
                                        fwhm_beam_arcmin = fwhm_beam_arcmin, 
                                        ell_highpass = ell_highpass,
                                        ell_lowpass = ell_lowpass,
                                        trans_func_grid = trans_func_grid,
                                        bfilt = bfilt,
                                        beam_grid = beam_grid,
                                        # keeping below for all calls from the function
                                        pointsource = True,
                                        sigspect = 1,
                                        keepdc = False, 
                                        arcmin_sigma = None,
                                        norm_unfilt = None,
                                        external_profile_norm = None,
                                        f_highpass = f_highpass, 
                                        f_lowpass = f_lowpass,
                                        scanspeed = scanspeed, 
                                        )

    if return_filtered_map or plot_map:
        filtered_map = Map.copy()
        uv_filter = m_filt[0].optfilt

        # Create a default "mask" of all ones if we didn't get an input mask,
        # and calculate the mask normalization.

        # window_norm_factor = np.sqrt(np.sum(mask**2) / np.product(mask.shape))

        # Go through the T maps and filter them.
        ft_map = getFTMap(Map, mask)

            # Apply the k-space filter.
        ft_map *= uv_filter

          # Transform back to real space, and remove the apodization mask.
        filtered_map = np.fft.ifft2(ft_map).real

        filt_map = filtered_map
#        filt_map.setMapAttr('filt_sigma',m_filt[0].sigma)
#        filt_map.setMapAttr('source_filtered_map',True)
        if plot_map:
            filt_map.pol_maps.T.drawImage(figure=1,mask=mask)
            filt_map.pol_maps.T.drawImage(figure=2,mask=mask*(1./m_filt[0].sigma),)
            pl.suptitle('units in S/N, not K_CMB')
            #plotting.kspaceImshow(m_filt[0].optfilt,reso_arcmin=reso_arcmin,figure=3,log=True,title='ell-space optfilt')
            plotting.kspaceImshow(m_filt[0].optfilt,reso_arcmin=reso_arcmin,figure=4,log=False,title='ell-space optfilt')
            pl.figure(5);pl.clf()
            plotting.plotRadialProfile(np.fft.fftshift(m_filt[0].optfilt), center_bin=[ngridx/2,ngridy/2], 
                                       resolution=(ellx[0,1]-ellx[0,0]), 
                                       histogram=True, range=[0,20000],
                                       plot=True, bin_resolution_factor=0.1, 
                                       figure=5,new=False)
            pl.title('1D optfilt')
            pl.xlabel('ell')

    print 'sigma of map = ',m_filt[0].sigma
    if debug:
        print adsf

    if return_filtered_map:
        return m_filt, filt_map
    return m_filt
