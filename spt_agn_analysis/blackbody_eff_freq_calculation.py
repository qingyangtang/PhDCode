import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
import math
from scipy.optimize import curve_fit
import astropy.io.fits as fits
import scipy.constants as const
from scipy.interpolate import interp1d
from scipy import integrate
import pickle as pk
from scipy.interpolate import UnivariateSpline
from scipy.integrate import quad
import numpy as np
from scipy.optimize import fsolve

print ("This needs to be run on python3")

c = const.c
k = const.k
h = const.h

def mod_BB_curve_with_z(v, A, T, z):
    beta = 2.
    v0 = v*(1+z)
    Bv = (2*h*v0**3/c**2)*(1/(np.exp(h*v0/(k*T))-1))
    Sv = A*v0**beta*Bv
    return Sv

def dbdt(f,T):#      #t in K, f in Hz; returns SI units
    x = h*f/(k*T)
    return (2*h*f**3/c**2) / (np.exp(x)-1.0)**2 * (x/T) * np.exp(x)

def dcmbrdt(f):
    return dbdt(f, 2.726)

def calc_nu_eff1(nu, transmission, source_spectrum, nonusq=True):
    nu_interp = np.arange(1e4)/1e4*(np.max(nu)-np.min(nu)+20.) + np.min(nu) - 10.

    dbdt = dcmbrdt(nu*1.e9)

    if nonusq:
        bw_source = integrate.simps(transmission*source_spectrum,nu)
        bw_cmb = integrate.simps(transmission*dbdt,nu)
    else:
        bw_source = integrate.simps(transmission*source_spectrum/nu**2,nu)
        bw_cmb = integrate.simps(transmission*dbdt/nu**2,nu)
    f = interp1d(nu, source_spectrum,fill_value="extrapolate")
    ss_interp = f(nu_interp)
    dbdt_interp = dcmbrdt(nu_interp*1e9)
    whlow = np.where(nu_interp <= 40.)[0]
    if np.any(nu_interp<= 40.) == True:
        dbdt_interp[whlow] = 1e12
    ratio1 = bw_source/bw_cmb
    ratio2 = ss_interp/dbdt_interp
    dss = np.abs(ratio1-ratio2)
    from scipy.signal import argrelmin
    minima_ind = argrelmin(np.log(dss),order=10)[0]
    if len(minima_ind) > 1:
        band_start = np.min(nu[transmission>0.2])
        band_end = np.max(nu[transmission>0.2])
        band_cent = (band_start+band_end)/2.
        true_min = np.where(np.abs(band_cent-nu_interp[minima_ind]) == np.min(np.abs(band_cent-nu_interp[minima_ind])))[0][0]
        nu_eff = nu_interp[minima_ind[true_min]]
    else:
        nu_eff = nu_interp[minima_ind]
    return nu_eff

#### HERSCHEL ####
hdu = fits.open('SPIRE_phot_RSRF_14_3.fits')
freq = hdu[1].data['frequency']
wavenumber = hdu[1].data['wavenumber']
psw = hdu[1].data['psw']
pmw = hdu[1].data['pmw']
plw = hdu[1].data['plw']

plt.figure()
plt.plot(freq, psw)
plt.plot(freq,pmw)
plt.plot(freq, plw)
plt.xlabel('Freq, GHz')
plt.ylabel('Transmission')
plt.title('Herschel Frequency Transmission Curves')

freq = np.concatenate((np.linspace(np.min(freq)-200,np.min(freq)-1, 99), freq))
trans = np.asarray([plw,pmw,psw])
trans = np.hstack((np.zeros((3,99)),trans))
wavelength = c/(freq*1.e9)
plt.figure()
z_set = np.arange(0.01,3,0.01)
T_set = np.arange(3,50,0.1)
eff_bands_250=np.zeros((len(z_set), len(T_set)))
eff_bands_350=np.zeros((len(z_set), len(T_set)))
eff_bands_500=np.zeros((len(z_set), len(T_set)))
for m in range(len(z_set)):
    for n in range(len(T_set)):
        eff_bands_500[m,n] = calc_nu_eff1(freq, trans[0],mod_BB_curve_with_z(wavelength,1, T_set[n], z_set[m]))
        eff_bands_350[m,n] = calc_nu_eff1(freq, trans[1],mod_BB_curve_with_z(wavelength,1, T_set[n], z_set[m]))
        eff_bands_250[m,n] = calc_nu_eff1(freq, trans[2],mod_BB_curve_with_z(wavelength,1, T_set[n], z_set[m]))

##### SPT-SZ ####

spt90 = np.loadtxt('spt_95.txt')
spt150 = np.loadtxt('spt_150.txt')
spt220 = np.loadtxt('spt_220.txt')
data = np.asarray([spt90, spt150, spt220])
freq90 = data[0][:,0]
trans90 = data[0][:,1]
freq150 = data[1][:,0]
trans150 = data[1][:,1]
freq220 = data[2][:,0]
trans220 = data[2][:,1]

plt.figure()
plt.plot(freq90,trans90)
plt.plot(freq150,trans150)
plt.plot(freq220,trans220)

freq90 = np.concatenate((np.linspace(np.min(freq90)-200,np.min(freq90)-1, 99), freq90))
freq150 = np.concatenate((np.linspace(np.min(freq150)-200,np.min(freq150)-1, 99), freq150))
freq220 = np.concatenate((np.linspace(np.min(freq220)-200,np.min(freq220)-1, 99), freq220))

trans90 = np.hstack((np.zeros((99)),trans90))
trans150 = np.hstack((np.zeros((99)),trans150))
trans220 = np.hstack((np.zeros((99)),trans220))

wavelength90 = c/(freq90*1.e9)
wavelength150 = c/(freq150*1.e9)
wavelength220 = c/(freq220*1.e9)

plt.figure()
z_set = np.arange(0.01,3,0.01)
T_set = np.arange(3,50,0.1)

eff_bands_90=np.zeros((len(z_set), len(T_set)))
eff_bands_150=np.zeros((len(z_set), len(T_set)))
eff_bands_220=np.zeros((len(z_set), len(T_set)))

for m in range(len(z_set)):
    for n in range(len(T_set)):
        eff_bands_90[m,n] = calc_nu_eff1(freq90, trans90,mod_BB_curve_with_z(wavelength90,1, z_set[m], T_set[n]))
        eff_bands_150[m,n] = calc_nu_eff1(freq150, trans150,mod_BB_curve_with_z(wavelength150,1, z_set[m], T_set[n]))
        eff_bands_220[m,n] = calc_nu_eff1(freq220, trans220,mod_BB_curve_with_z(wavelength220,1, z_set[m], T_set[n]))

### SPTPOL #####

freq90 = fits.open('/Users/amytang/spt_agn_analysis/sptpol_90_bandpass.fits')[1].data['freqs90'] 
trans90 = fits.open('/Users/amytang/spt_agn_analysis/sptpol_90_bandpass.fits')[1].data['trans90']
freq150 = fits.open('/Users/amytang/spt_agn_analysis/sptpol_150_bandpass.fits')[1].data['freqs150']
trans150 = fits.open('/Users/amytang/spt_agn_analysis/sptpol_150_bandpass.fits')[1].data['trans150']

plt.figure()
plt.plot(freq90,trans90)
plt.plot(freq150,trans150)

freq90 = np.concatenate((np.linspace(np.min(freq90)-200,np.min(freq90)-1, 99), freq90))
freq150 = np.concatenate((np.linspace(np.min(freq150)-200,np.min(freq150)-1, 99), freq150))

trans90 = np.hstack((np.zeros((99)),trans90))
trans150 = np.hstack((np.zeros((99)),trans150))

wavelength90 = c/(freq90*1.e9)
wavelength150 = c/(freq150*1.e9)

plt.figure()
z_set = np.arange(0.01,3,0.01)
T_set = np.arange(3,50,0.1)

eff_bands_90=np.zeros((len(z_set), len(T_set)))
eff_bands_150=np.zeros((len(z_set), len(T_set)))

for m in range(len(z_set)):
    for n in range(len(T_set)):
        eff_bands_90[m,n] = calc_nu_eff1(freq90, trans90,mod_BB_curve_with_z(wavelength90,1, z_set[m], T_set[n]))
        eff_bands_150[m,n] = calc_nu_eff1(freq150, trans150,mod_BB_curve_with_z(wavelength150,1, z_set[m], T_set[n]))

look_up_table = np.zeros((len(z_set)+1, len(T_set)+1,2))
look_up_table[1:,0,0] = z_set
look_up_table[1:,0,1] = z_set
look_up_table[0,1:,0] = T_set
look_up_table[0,1:,1] = T_set
look_up_table[1:,1:,0] = eff_bands_90
look_up_table[1:,1:,1] = eff_bands_150
np.save('SPTpol_effective_bands.npy', look_up_table)

### checking with t-sz eff freq values
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

def f_tsz(freq):
    freq = freq # no need to convert to Hz here
    return dB_dT(freq) * g_nu(freq)
def dB_dT_over_f_tsz(freq, *ratio):
    return dB_dT(freq) / f_tsz(freq) - ratio

band90_f = UnivariateSpline(freq90, trans90, s=0)
band150_f = UnivariateSpline(freq150, trans150, s=0)

ratio_tsz_90 = quad(lambda freq: dB_dT(freq)*band90_f(freq), 66.03, 124.07)[0]/ quad(lambda freq: f_tsz(freq)*band90_f(freq), 66.03, 124.07)[0]
f_eff_tsz_90 = fsolve(dB_dT_over_f_tsz, 90, args = ratio_tsz_90)[0]
ratio_tsz_150 = quad(lambda freq: dB_dT(freq)*band150_f(freq), 106.06, 198.11, limit=100)[0]/ quad(lambda freq: f_tsz(freq)*band150_f(freq), 106.06, 198.11, limit=100)[0]
f_eff_tsz_150 = fsolve(dB_dT_over_f_tsz, 150, args = ratio_tsz_150)[0]
