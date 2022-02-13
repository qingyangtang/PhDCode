;
;----------------------------------------------------------------------------- 
; NAME:
;       CALC_OPTFILT_ONEBAND_2D.PRO
; 
; PURPOSE: 
;       Calculate optimal filter for source detection in single map
;       with many different assumed source profiles.  
;
; CALLING SEQUENCE: 
;       RESULT = CALC_OPTFILT_ONEBAND_2D, profiles, skysigvar, noisepsd, reso_arcmin, $
;                                  sigspect=sigspect, $
;                                  profile=profile, $
;                                  fwhm_beam_arcmin=fwhm_beam_arcmin, $
;                                  f_highpass=f_highpass, f_lowpass=f_lowpass, $
;                                  trans_func_grid=trans_func_grid, $
;                                  keepdc=keepdc, $
;                                  scanspeed=scanspeed, elscan=elscan, $
;                                  sigma=sigma, $
;                                  arcmin_sigma=arcmin_sigma, $
;                                  maxcoeff=maxcoeff, $
;                                  area_eff=area_eff, $
;                                  whitenoise1d=whitenoise1d, $
;                                  bfilt = bfilt, $
;                                  stopit=stopit
;
; INPUT:
;       PROFILES - 1d ell-space profiles of objects you want to detect.
;
;       SKYSIGVAR - assumed variance as a function of ell (isotropic)
;                   of all sky signals other than the signal of
;                   interest.  (E.g., CMB + point sources for an SZ
;                   measurement.)
;
;       FNOISEMAPS - N_pts-by-N_pts array of 2d Fourier
;                    representation of noise in map.
;                    Should be arranged like standard IDL 2d ffts.
;                    Should be normalized such that
;                    sqrt(int_tabulated(uv_arr,fnoisemaps[*,*,j]^2)) =
;                    noise rms in the map.
;
;       RESO_ARCMIN - pixel size in arcminutes of maps to be combined.
;
; OUTPUT: 
;       RESULT - 2d Fourier-space optimal filters, one for each input
;                profile. 
;
; KEYWORDS: 
;       SIGSPECT - sensitivity to signal of interest in supplied map.
;                  For clusters, this should be equal to fxsz(band).
;
;       FWHM_BEAM_ARCMIN - size (in full-width at half-max in units of
;                          arcminutes) of beams in each band.  Beams
;                          assumed to be Gaussian.  Default is 10^-6
;                          (infinitely small beams).  Ignored if
;                          TRANS_FUNC_GRID is supplied.
;
;       BEAM_GRID - set this to a 2d array with the 2d beam response
;                   (if you want to specify a non-Gaussian or
;                   asymmetric beam but use default filters).
;
;       BFILT - Hand the program a 1-d beam window function, which will be used in place of 
;               the default assumed Gaussian beam for spatial filtering
;
;       F_HIGHPASS - frequency at which timestreams were high-passed
;                    before maps were made.  Default is no filtering.
;                    Ignored if TRANS_FUNC_GRID is supplied.
;
;       F_LOWPASS - frequency at which timestreams were low-passed
;                    before maps were made.  Default is no filtering.
;                    Ignored if TRANS_FUNC_GRID is supplied.
;
;       TRANS_FUNC_GRID - array the same size as MAP containing the
;                          effective delta-function response (in 2d
;                          fourier space) of all the filtering we've
;                          done to the signal in the map
;                          (INCLUDING THE BEAM).  If TRANS_FUNC_GRID
;                          is set, other filtering keywords (including
;                          FWHM_BEAM_ARCMIN) are ignored.
;
;       TF_OUT - will contain transfer functions used on output.
;
;       KEEPDC - set this to keep the DC information along the scan
;                direction.  Otherwise, the u=0 modes (or v=0 if we're
;                el scanning) are set to zero even if F_HIGHPASS isn't
;                set. 
;
;       SCANSPEED - what speed were we scanning at when the maps were
;                   made?  (Need to know this to translate high- and
;                   low-pass temporal frequencies into spatial
;                   frequencies.)  Default is 0.25 (deg/s on the
;                   sky). 
;
;       ELSCAN - set this if we were el scanning rather than az
;                scanning.
;
;       SIGMA - square root of (expected) variance of central value of features
;               detected in map using OPTFILT.
;
;       ARCMIN_SIGMA - SIGMA scaled to what it would be for a
;                      1-arcminute FWHM beam.  (SIGMA is reported for
;                      the smallest beam used.)
;
;       AREA_EFF - effective area of sources with input profile and
;                  unit amplitude, convolved with smallest beam.
;
;       WHITENOISE1D - set this to ignore all anisotropic noise and
;                      filtering (mostly for debugging).
;
;       POINTSOURCE - set this to tell the routine that you want the
;                     optimal filter for a point source.  Equivalent
;                     to setting profiles = fltarr(nlmax+1) + 1.
;
;       NORM_UNFILT - set this to normalize the output such that 
;                     sources of the shape of one of the input
;                     profiles will have amplitude in the output map
;                     equal to the amplitude of the input source
;                     before filtering or beam-smoothing.
;
;       EXTERNAL_PROFILE_NORM - set this to the known area (in sr)
;                               under the real-space version of your
;                               source profiles, if you don't want the
;                               area calculated in ell space.  Only
;                               appropriate if NORM_UNFILT is set.
;
; HISTORY: 
;	Created 10Sep08, TC, from combine_sptmaps_get2dfcoeffs.
;       Added POINTSOURCE keyword, 28Oct08, TC.
;       Added BEAM_GRID, TF_OUT, NORM_UNFILT, 01Jun09, TC.
;       Changed NORM_UNFILT calculation to use (full) 1d profile even
;         in 2d filtering case, 28Apr10, TC.
;       Tweak to accept non-square maps, 17Sep10, TC.
;
;-----------------------------------------------------------------------------

function calc_optfilt_oneband_2d, profiles, skysigvar, noisepsd, reso_arcmin, $
                                  sigspect=sigspect, $
                                  fwhm_beam_arcmin=fwhm_beam_arcmin, $
                                  f_highpass=f_highpass, f_lowpass=f_lowpass, $
                                  trans_func_grid=trans_func_grid, $
                                  beam_grid=beam_grid, $
                                  keepdc=keepdc, $
                                  scanspeed=scanspeed, elscan=elscan, $
                                  sigma=sigma, $
                                  arcmin_sigma=arcmin_sigma, $
                                  maxcoeff=maxcoeff, $
                                  area_eff=area_eff, $
                                  whitenoise1d=whitenoise1d, $
                                  pointsource=pointsource, $
                                  tf_out=tf_out, $
                                  norm_unfilt=norm_unfilt, $
                                  external_profile_norm=external_profile_norm, $
                                  bfilt = bfilt, $
                                  stopit=stopit

; define constants & process keywords
ellmax = n_elements(skysigvar) - 1L
ell = lindgen(ellmax+1L)
if not(keyword_set(fwhm_beam_arcmin)) then fwhm_beam_arcmin = 1.d-6
;ngrid = n_elements(noisepsd[*,0])
ngridx = n_elements(noisepsd[*,0])
ngridy = n_elements(noisepsd[0,*])
if n_elements(sigspect) eq 0 then sigspect = 1.
if keyword_set(pointsource) then profiles = fltarr(ellmax+1L) + 1.
do_ext_norm = n_elements(external_profile_norm) gt 0

; get 2d grid of ell values appropriate for 2d fft of map
;ellg = make_fft_grid(reso_arcmin/60.*!dtor,ngrid,ngrid)*2.*!pi
ellg = make_fft_grid(reso_arcmin/60.*!dtor,ngridx,ngridy)*2.*!pi

; array of residual noise to use in optimal filter &
; variance calculation
;ivarr = dblarr(ngrid,ngrid) + 1.d20
ivarr = dblarr(ngridx,ngridy) + 1.d20

; get 2d Fourier representations of timestream filters and beams
; (if TRANS_FUNC_GRID is set, these will be ignored except for 1d
; debugging case.)
if keyword_set(bfilt) then bfilt = bfilt else bfilt = beamfilt(ell,fwhm_beam_arcmin[0])
if n_elements(trans_func_grid) eq 0 then begin
;    trans_func_grid = dblarr(ngrid,ngrid)
    trans_func_grid = dblarr(ngridx,ngridy)
    if n_elements(scanspeed) eq 0 then scanspeed = 0.25 ; deg/s
    f_eff = ellg[*,0]*scanspeed*!dtor/2./!pi
    hplp1d = fltarr(n_elements(f_eff)) + 1.
    whn0 = where(f_eff ne 0.,nn0,compl=wh0)
    if n_elements(f_highpass) gt 0. then begin
        if nn0 gt 0 then hplp1d[whn0] *= exp(-1.*(f_highpass[0]/f_eff[whn0])^6)
    endif
    if n_elements(f_lowpass) gt 0. then hplp1d *= exp(-1.*(f_eff/f_lowpass[0])^6)
    if keyword_set(keepdc) eq 0 and nn0 ne n_elements(f_eff) then hplp1d[wh0] = 0.
;    for i=0,ngrid-1 do trans_func_grid[*,i] = hplp1d
    for i=0,ngridy-1 do trans_func_grid[*,i] = hplp1d
;    if keyword_set(elscan) then trans_func_grid = transpose(trans_func_grid)
    if keyword_set(elscan) then trans_func_grid = congrid(transpose(trans_func_grid),ngridx,ngridy)
    if n_elements(beam_grid) gt 0 then trans_func_grid *= beam_grid else trans_func_grid *= bfilt[ellg]
endif

; get 2d Fourier representations of covariance matrices
skysigvar2d = skysigvar[ellg]
; assume timestream filters have rolled off astrophysical contaminants
skysigvar2d *= abs(trans_func_grid)^2

; calculate noise + contaminant variance at all ell grid points
skysigvar2d += noisepsd^2
ivarr = skysigvar2d/sigspect[0]^2
szbftemp = abs(trans_func_grid*sigspect[0])^2
whsig = where(szbftemp/max(szbftemp) ge 1.d-4,nsig)
ellgmax = max(abs(ellg[whsig]))
if ellgmax lt ellmax then ellmax2use = ellgmax + 1 else ellmax2use = ellmax
ell2use = lindgen(ellmax2use+1L)

; 1d case for debugging
noisepsd1d = fltarr(ellmax2use+1)
;whgellg = where(ell2use ge max(ellg[0:ngrid/2L,0]),ngellg)
whgellg = where(ell2use ge max(ellg[0:ngridx/2L,0]),ngellg)
;noisepsd1d = interpol(noisepsd[0:ngrid/2L,0],ellg[0:ngrid/2L,0],ell2use)
noisepsd1d = interpol(noisepsd[0:ngridx/2L,0],ellg[0:ngridx/2L,0],ell2use)
if ngellg gt 0 then noisepsd1d[whgellg] = noisepsd1d[whgellg[0]-1L]
ivarr1d = (skysigvar*bfilt^2 + noisepsd1d^2)/sigspect[0]^2

ivarr[0,0] = 1.d20
ivarr1d[0] = 1.d20
; kludge to agree with gauss_optfilt
if max(abs(skysigvar)) ge 1.e-10 and keyword_set(keepdc) eq 0 then ivarr1d[1] = 1.d20

; make optimal filter(s) to use on synthesized map (using
; Haehnelt-Tegmark prescription).
nprofs = n_elements(profiles[0,*])
if keyword_set(whitenoise1d) then begin
    optfilts = dblarr(ellmax2use+1L,nprofs)
endif else begin
;    optfilts = dcomplexarr(ngrid,ngrid,nprofs)
    optfilts = dcomplexarr(ngridx,ngridy,nprofs)
endelse
sigmas = dblarr(nprofs)
arcmin_sigmas = dblarr(nprofs)
area_effs = dblarr(nprofs)
for i=0,nprofs-1 do begin
    tauprime = profiles[*,i]*bfilt
    if keyword_set(norm_unfilt) then begin
        if do_ext_norm then $
          area_eff_1d = external_profile_norm[i] $
        else $
          area_eff_1d = 1./total((2.*ell+1.)/4./!pi*profiles[*,i]) 
    endif else begin
        area_eff_1d = 1./total((2.*ell+1.)/4./!pi*tauprime)
    endelse
    area_eff_1am = 1./total((2.*ell+1.)/4./!pi*profiles[*,i]*beamfilt(ell,1.))
    tauprime = tauprime*area_eff_1d
; make 2d versions
    prof1d = profiles[*,i]
    prof2d = prof1d[ellg]
    tauprime_2d = prof2d*trans_func_grid
    du = (ellg[0,1] - ellg[0,0])/2./!pi
    dv = (ellg[1,0] - ellg[0,0])/2./!pi
;    if keyword_set(norm_unfilt) then $
;;      area_eff = 1./total(abs(prof2d)*du^2) $
;      area_eff = 1./total((2.*ell+1.)/4./!pi*profiles[*,i]) $
;    else $
;      area_eff = 1./total(abs(tauprime_2d)*du^2)
    if keyword_set(norm_unfilt) then begin
        if do_ext_norm then $
          area_eff = external_profile_norm[i] $
        else $
          area_eff = 1./total((2.*ell+1.)/4./!pi*profiles[*,i]) 
    endif else begin
;        area_eff = 1./total(abs(tauprime_2d)*du^2)
        area_eff = 1./total(abs(tauprime_2d)*du*dv)
    endelse
    tauprime_2d = tauprime_2d*area_eff
    optfilt = tauprime_2d/ivarr
;    optfilt_norm = total(abs(optfilt*conj(tauprime_2d)))*du^2
    optfilt_norm = total(abs(optfilt*conj(tauprime_2d)))*du*dv
    optfilt = optfilt/optfilt_norm
; for debugging
;ivarr1d=interpol(ivarr[0:250,0],ellg[0:250,0],ell)
    optfilt1d=tauprime/ivarr1d
    optfilt1d_norm = total(2.*(ell+1.)/4./!pi*optfilt1d*tauprime)
    optfilt1d = optfilt1d/optfilt1d_norm
    sigma_1d = 1./sqrt(optfilt1d_norm)
    arcmin_sigma_1d = sigma_1d*area_eff_1d/area_eff_1am
; calculate variance of central signal value of features detected in
; synthesized map using optimal filter (this is the variance on the
; central value of the features after they have been smoothed by the
; smallest beam among the bands).
    sigma = 1./sqrt(optfilt_norm)
; scale to that same value if the minimum beam were 1 arcminute
    arcmin_sigma = sigma*area_eff/area_eff_1am
; stuff values in arrays
    if keyword_set(whitenoise1d) then begin
        optfilts[*,i] = optfilt1d
        sigmas[i] = sigma_1d
        arcmin_sigmas[i] = arcmin_sigma_1d
        area_effs[i] = area_eff_1d
    endif else begin
        optfilts[*,*,i] = optfilt
        sigmas[i] = sigma
        arcmin_sigmas[i] = arcmin_sigma
        area_effs[i] = area_eff
    endelse
endfor

if nprofs eq 1 then begin
    if keyword_set(whitenoise1d) then begin
        optfilts = optfilts[*,0]
    endif else begin
        optfilts = optfilts[*,*,0]
    endelse
    sigma = sigmas[0]
    arcmin_sigma = arcmin_sigmas[0]
    area_eff = area_effs[0]
endif else begin
    sigma = sigmas
    arcmin_sigma = arcmin_sigmas
    area_eff = area_effs
endelse

tf_out = trans_func_grid

if keyword_set(stopit) then stop

return, optfilts

end
