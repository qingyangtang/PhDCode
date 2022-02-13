import pickle as pk
from scipy.interpolate import UnivariateSpline
from scipy.integrate import quad
import numpy as np
from scipy.optimize import fsolve

Tcmb = 2.725
# g_nu is the relative of tSZ to TCMB at each frequency
def g_nu(freq):
    x = 6.626e-34 * freq * 1e9 / (1.381e-23 * Tcmb)
    g_nu = x * 1.0 / np.tanh(x / 2.0) - 4
    return g_nu

def dB_dT(freq):
    # derivative of blackbody spectrum wrt T at Tcmb
    f = freq * 1e9 # assume input freq in GHz
    h = 6.626e-34
    k = 1.381e-23
    Tcmb = 2.725
    c = 299792458
    exp_fac = np.exp(h * f / k / Tcmb)
    return 2 * h**2 * f**4 / k / Tcmb**2 / c**2 * exp_fac / (exp_fac - 1)**2

# load up the detector frequency bands
# they are stored with the detector A Omega and Lyot stop 

band90_f = UnivariateSpline(freq90, trans90, s=0)
band150_f = UnivariateSpline(freq150, trans150, s=0)
band220_f = UnivariateSpline(freq220, trans220, s=0)

def f_dust(freq):
    freq = freq * 1e9
    return freq ** 3.5

def f_radio(freq):
    freq = freq * 1e9
    return freq ** -0.5

def f_tsz(freq):
    freq = freq # no need to convert to Hz here
    return dB_dT(freq) * g_nu(freq)

def dB_dT_over_f_dust(freq, *ratio):
    return dB_dT(freq) / f_dust(freq) - ratio

def dB_dT_over_f_radio(freq, *ratio):
    return dB_dT(freq) / f_radio(freq) - ratio

def dB_dT_over_f_tsz(freq, *ratio):
    return dB_dT(freq) / f_tsz(freq) - ratio

# for 90ghz

ratio_dust_90 = quad(lambda freq: dB_dT(freq)*band90_f(freq), 66.03, 124.07)[0]/ quad(lambda freq: f_dust(freq)*band90_f(freq), 66.03, 124.07)[0]
f_eff_dust_90 = fsolve(dB_dT_over_f_dust, 90, args = ratio_dust_90)[0]

ratio_radio_90 = quad(lambda freq: dB_dT(freq)*band90_f(freq), 66.03, 124.07)[0]/ quad(lambda freq: f_radio(freq)*band90_f(freq), 66.03, 124.07)[0]
f_eff_radio_90 = fsolve(dB_dT_over_f_radio, 90, args = ratio_radio_90)[0]

ratio_tsz_90 = quad(lambda freq: dB_dT(freq)*band90_f(freq), 66.03, 124.07)[0]/ quad(lambda freq: f_tsz(freq)*band90_f(freq), 66.03, 124.07)[0]
f_eff_tsz_90 = fsolve(dB_dT_over_f_tsz, 90, args = ratio_tsz_90)[0]

# for 150ghz                                                        
ratio_dust_150 = quad(lambda freq: dB_dT(freq)*band150_f(freq), 106.06, 198.11, limit=100)[0]/ quad(lambda freq: f_dust(freq)*band150_f(freq), 106.06, 198.11, limit=100)[0]
f_eff_dust_150 = fsolve(dB_dT_over_f_dust, 150, args = ratio_dust_150)[0]

ratio_radio_150 = quad(lambda freq: dB_dT(freq)*band150_f(freq), 106.06, 198.11, limit=100)[0]/ quad(lambda freq: f_radio(freq)*band150_f(freq), 106.06, 198.11, limit=100)[0]
f_eff_radio_150 = fsolve(dB_dT_over_f_radio, 150, args = ratio_radio_150)[0]

ratio_tsz_150 = quad(lambda freq: dB_dT(freq)*band150_f(freq), 106.06, 198.11, limit=100)[0]/ quad(lambda freq: f_tsz(freq)*band150_f(freq), 106.06, 198.11, limit=100)[0]
f_eff_tsz_150 = fsolve(dB_dT_over_f_tsz, 150, args = ratio_tsz_150)[0]

# for 220ghz                                                        
ratio_dust_220 = quad(lambda freq: dB_dT(freq)*band220_f(freq), 160.09, 298.17, limit=100)[0]/ quad(lambda freq: f_dust(freq)*band220_f(freq), 160.09, 298.17, limit=100)[0]
f_eff_dust_220 = fsolve(dB_dT_over_f_dust, 220, args = ratio_dust_220)[0]

ratio_radio_220 = quad(lambda freq: dB_dT(freq)*band220_f(freq), 160.09, 298.17, limit=100)[0]/ quad(lambda freq: f_radio(freq)*band220_f(freq), 160.09, 298.17, limit=100)[0]
f_eff_radio_220 = fsolve(dB_dT_over_f_radio, 220, args = ratio_radio_220)[0]

ratio_tsz_220 = quad(lambda freq: dB_dT(freq)*band220_f(freq), 160.09, 298.17, limit=100)[0]/ quad(lambda freq: f_tsz(freq)*band220_f(freq), 160.09, 298.17, limit=100)[0]
f_eff_tsz_220 = fsolve(dB_dT_over_f_tsz, 220, args = ratio_tsz_220)[0]

print('Dusty source band center for 90, 150, and 220GHz bands are, %.3f, %.3f, %.3f' %(f_eff_dust_90, f_eff_dust_150, f_eff_dust_220))
print('Radio source band center for 90, 150, and 220GHz bands are, %.3f, %.3f, %.3f' %(f_eff_radio_90, f_eff_radio_150, f_eff_radio_220))
print('tSZ band center for 90, 150, and 220GHz bands are, %.3f, %.3f, %.3f' %(f_eff_tsz_90, f_eff_tsz_150, f_eff_tsz_220))

# Christian's effective frequencies
f_eff_dust_sz = np.array([96.9, 153.4, 221.6])
f_eff_radio_sz = np.array([93.5, 149.5, 215.8])
f_eff_tsz_sz = np.array([96.6, 152.3, 220.1])

# 3G's effective frequencies
f_eff_dust_3g = np.array([f_eff_dust_90, f_eff_dust_150, f_eff_dust_220])
f_eff_radio_3g = np.array([f_eff_radio_90, f_eff_radio_150, f_eff_radio_220])
f_eff_tsz_3g = np.array([f_eff_tsz_90, f_eff_tsz_150, f_eff_tsz_220])

# ratio of T_3G to T_sptsz/sptpol
# Delta T prop to f(f_eff)/ B'(f_eff), that is the inverse of dB_dT_over_f_dust
ratios_dust = dB_dT_over_f_dust(f_eff_dust_sz,0) / dB_dT_over_f_dust(f_eff_dust_3g,0)
ratios_radio = dB_dT_over_f_radio(f_eff_radio_sz,0) / dB_dT_over_f_radio(f_eff_radio_3g,0)
ratios_tsz = dB_dT_over_f_tsz(f_eff_tsz_sz,0) / dB_dT_over_f_tsz(f_eff_tsz_3g,0)

print('T-3G/T-reichardt ratios for dusty sources', ratios_dust)
print('T-3G/T-reichardt ratios for radio sources', ratios_radio)
print('T-3G/T-reichardt ratios for tSZ', ratios_tsz)

print('Cl-3G/Cl-reichardt ratios for dusty sources', ratios_dust**2)
print('Cl-3G/Cl-reichardt ratios for radio sources', ratios_radio**2)
print('Cl-3G/Cl-reichardt ratios for tSZ', ratios_tsz**2)

# George's effective frequencies                                                                                                                                                                                                         
f_eff_dust_sz = np.array([97.9, 154.1, 219.6])
f_eff_radio_sz = np.array([95.3, 150.5, 214.0])
f_eff_tsz_sz = np.array([97.6, 153.1, 218.1])

# 3G's effective frequencies
f_eff_dust_3g = np.array([f_eff_dust_90, f_eff_dust_150, f_eff_dust_220])
f_eff_radio_3g = np.array([f_eff_radio_90, f_eff_radio_150, f_eff_radio_220])
f_eff_tsz_3g = np.array([f_eff_tsz_90, f_eff_tsz_150, f_eff_tsz_220])

# ratio of T_3G to T_sptsz/sptpol
# Delta T prop to f(f_eff)/ B'(f_eff), that is the inverse of dB_dT_over_f_dust 

ratios_dust = dB_dT_over_f_dust(f_eff_dust_sz,0) / dB_dT_over_f_dust(f_eff_dust_3g,0)
ratios_radio = dB_dT_over_f_radio(f_eff_radio_sz,0) / dB_dT_over_f_radio(f_eff_radio_3g,0)
ratios_tsz = dB_dT_over_f_tsz(f_eff_tsz_sz,0) / dB_dT_over_f_tsz(f_eff_tsz_3g,0)

print('T-3G/T-george ratios for dusty sources', ratios_dust)
print('T-3G/T-george ratios for radio sources', ratios_radio)
print('T-3G/T-george ratios for tSZ', ratios_tsz)

print('Cl-3G/Cl-george ratios for dusty sources', ratios_dust**2)
print('Cl-3G/Cl-george ratios for radio sources', ratios_radio**2)
print('Cl-3G/Cl-george ratios for tSZ', ratios_tsz**2)
