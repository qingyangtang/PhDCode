import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
import math
from scipy.optimize import curve_fit

c = const.c
k = const.k
h = const.h
freq = c/(np.asarray([500, 350, 250])*1.e-6)

### data pulled from Viero et al ###
bands = np.asarray([24., 70., 100., 160., 250., 350., 500., 1100.])*1.e-6 #in m
freqs = c/bands
stacked_flux = np.asarray([6.9, 1.3, 3.3, 3.6, 2.5, 1.5, 5.9, 2.0]) #in units nW/m^2/sr
stacked_flux_err = np.asarray([0.2, 0.1, 0.1, 0.3, 0.1, 0.1, 0.3, 0.2])

def mod_BB_curve_with_z(v, A, beta, z, T):
    v0 = v*(1+z)
    Bv = (2*h*v0**3/c**2)*(1/(np.exp(h*v0/(k*T))-1))
    Sv = A*v0**beta*Bv
    return Sv

i = 1
for i in range(10):
    ydata = np.asarray([good500[i], good350[i], good250[i]])
    popt, pcov = curve_fit(mod_BB_curve_with_z, freq, ydata, bounds = ([0, 1.999, 0.2, 0], [10., 2, 1.5, 100]),maxfev=100000)
    print (popt)
    plt.figure()
    plt.plot(freq, ydata,'.')
    xvalues = np.linspace(min(freq), max(freq), 100)
    plt.plot(xvalues, mod_BB_curve_with_z(xvalues, *popt),label='amp: '+str(popt[0])+'\n beta:'+str(popt[1])+'\n z:' + str(popt[2]) + '\n T:'+ str(popt[3]))
    plt.legend()
    plt.title('Source ' + str(i))


popt, pcov = curve_fit(mod_BB_curve_with_z, freq, ydata, bounds = ([0, 1.999, 0.2, 0], [10., 2, 1.5, 100]))



def spec_fit(x, A, f0, alpha):
    return A*(x/f0)**alpha

from scipy.optimize import curve_fit


freqs = np.array([90.e9, 150.e9, 220.e9])
ymed = np.asarray([ 0.20596054,  0.29023269,  0.        ])
yerr = np.asarray([ 0.10792972,  0.04768894,  0.02347652])
popt, pcov = curve_fit(spec_fit, freqs, ymed, sigma = yerr, p0 = [2.e-1, 1.e11, -2.5], bounds=((0, 0, -3), (np.inf, np.inf, -1)),maxfev=10000)
print (popt)
plt.errorbar(freqs, ymed, yerr=yerr, fmt='o', label='median flux')
plt.plot(freqs, spec_fit(freqs, popt[0], popt[1], popt[2]))
plt.errorbar(freqs, ymed, yerr=yerr, fmt='o', label='median flux')
plt.plot(freqs, spec_fit(freqs, 2.e-1, 1.e11, -3))