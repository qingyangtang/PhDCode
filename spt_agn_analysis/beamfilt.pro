;----------------------------------------------------------------------------- 
; NAME:
;	BEAMFILT.PRO
; 
; PURPOSE: 
;       calculates l-space window function of Gaussian beam. 
;
; CALLING SEQUENCE: 
;       RESULT = BEAMFILT(ell, fwhm_arcmin)
;
; INPUT:
;       ELL - vector of l values for which window function should be
;             calculated. 
;
;       FWHM_ARCMIN - full-width at half-max of beam
;
; OUTPUT: 
;       RESULT - vector of window-function values.
;
; KEYWORDS: 
;
; ROUTINES USED:
;
; HISTORY: 
;	Created 15-Feb-2005, TC 
;
;-----------------------------------------------------------------------------

function beamfilt, ell, fwhm_arcmin

sig_smooth = fwhm_arcmin/SQRT(8.*ALOG(2.))/(60.*180.)* !PI
filter = exp(-ell*(ell+1.)*sig_smooth^2/2.)

return, filter

end
