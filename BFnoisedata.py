
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import matplotlib.backends.backend_pdf


# Loading the low res data. 

# In[2]:
#for KID in ["14", "15", "16", "17", "18", "19", "20", "21","22", "23","24", "25","26", "27"]:
for KID in ["02", "03", "04", "06","10", "11", "12"]:
	KIDID = KID
	Temp = "030"
	atten = "40"
	datadate = "20181111"
	datadir = "/home/tangq/Downloads/"+ datadate +"/KID_K0" + KIDID +"/Set_Temperature_"+Temp+"_mK/Set_Attenuation_"+atten+"p0dB/"
	hdulist = fits.open(datadir + "Sweep.fits")
	data = hdulist[1].data
	freq = data.field(0)
	I = data.field(1)
	Q = data.field(2)
	amp = np.sqrt(I**2 + Q**2)
	phase = np.arctan2(Q,I)
	savedir = "/home/tangq/Downloads/"+datadate+"/KID_K0" + KIDID + "_Temp" + Temp+"_Atten" + atten
	pdf = matplotlib.backends.backend_pdf.PdfPages(savedir + "cal_output.pdf")

	# Getting the off resonance part of sweep and checking to make sure it looks okay

	# In[3]:
	figonoff = plt.figure()
	plt.plot(I,Q,'.',label='onres data')
	fresindex = np.where(amp==min(amp))[0][0]
	offmargin = len(amp)/4
	Ioffres = np.concatenate((I[10:fresindex-offmargin],I[fresindex+offmargin:-10]))
	Qoffres = np.concatenate((Q[10:fresindex-offmargin],Q[fresindex+offmargin:-10]))
	foffres = np.concatenate((freq[10:fresindex-offmargin],freq[fresindex+offmargin:-10]))
	plt.plot(Ioffres,Qoffres,'.', label='offres data')
	plt.xlabel('I')
	plt.ylabel('Q')
	plt.legend(loc='best')
	plt.title('Comparing raw on and off resonance data')
	plt.gca().set_aspect('equal')
	#plt.show()
	pdf.savefig(figonoff)
	plt.close()


	# Getting the electronic delay $\tau$ with the latter half of the off resonance data and fitting the phase as a straight line

	# In[4]:


	#getting the latter half of the offresonance data
	fresindex = np.where(amp==min(amp))[0][0]
	Ioffres = I[fresindex+offmargin:-10]
	Qoffres = Q[fresindex+offmargin:-10]
	foffres = freq[fresindex+offmargin:-10]
	phaseoffres = np.arctan2(Qoffres,Ioffres)
	plt.plot(foffres,phaseoffres,'.')


	#doing fits
	from scipy.optimize import curve_fit, least_squares

	def phase(x, a,b):
	    y = -2.*np.pi*x*a+b
	    return y

	popt, pcov = curve_fit(phase, foffres, phaseoffres, bounds=((1.e-8,-np.inf),(1.e-6,np.inf)))

	print (popt[0],popt[1])
	tau = popt[0]
	#plotting the fit and data

	plt.clf()
	figcabledelay =  plt.figure()
	plt.plot(foffres, phaseoffres,'.',label='data')
	plt.plot(foffres, -2*np.pi*foffres*popt[0]+popt[1], label='fit')
	plt.legend()
	plt.xlabel('Freq')
	plt.ylabel('Phase')
	plt.title('Fitting to the phase to remove cable delay, tau = ' + str(tau))
	pdf.savefig(figcabledelay)
	#plt.show()
	plt.close()


	# Removing cable delay from data by multiplying $e^{2\pi jf \tau}$, the off resonance tails should now map to a dot

	# In[5]:


	data1 = (I+1.j*Q)*np.exp(1.j*2.*np.pi*freq*popt[0])
	plt.clf()
	fignocabledelay = plt.figure()
	plt.plot(data1.real, data1.imag,'.', label='cable delay removed')
	plt.plot(I,Q,'.',label='original data')
	plt.gca().set_aspect('equal')
	plt.xlabel('I')
	plt.ylabel('Q')
	plt.title('Comparing the low res raw data with the cable delay removed')
	plt.legend(loc=9, bbox_to_anchor=(1.5, 0.5))

	pdf.savefig(fignocabledelay)
	#plt.show()
	plt.close()


	# Loading the high resolution data and comparing to low resolution data with and without cable delay (sanity check).

	# In[6]:


	#plotting raw data of both HR and LR data
	hdulist = fits.open(datadir + "Sweep_HR.fits")
	datahr = hdulist[1].data
	freqhr = np.zeros((len(datahr)))
	Ihr = np.zeros((len(datahr)))
	Qhr = np.zeros((len(datahr)))
	for i in range(len(datahr)):
	    freqhr[i] = np.asarray(datahr[i])[0]
	    Ihr[i] = np.asarray(datahr[i])[1]
	    Qhr[i] = np.asarray(datahr[i])[2]
	amphr = np.sqrt(Ihr**2 + Qhr**2)
	phasehr = np.arctan2(Qhr,Ihr)
	plt.clf()
	figlrhr = plt.figure()
	plt.plot(Ihr,Qhr,'.',label='raw HR data')
	plt.plot(I,Q,'.', label='raw LR data')
	plt.ylabel('Q')
	plt.xlabel('I')
	plt.legend(loc='best')
	plt.gca().set_aspect('equal')

	plt.title('Comparing the RAW HR and LR data, making sure they overlap')

	pdf.savefig(figlrhr)
	plt.close()


	#removing cable delay for both data
	sweepdatahr = Ihr+1.j*Qhr
	datahr1 = sweepdatahr*np.exp(1.j*2.*np.pi*freqhr*tau)#-popt[1])
	plt.clf()
	figlrhrncd = plt.figure()
	Ihr1 = datahr1.real
	Qhr1 = datahr1.imag
	plt.plot(data1.real, data1.imag,'.', label='low res data')
	plt.plot(Ihr1,Qhr1, label='high res data')
	plt.xlabel('I')
	plt.ylabel('Q')
	plt.legend(loc='best')
	plt.title('After cable delay is remove for both atasets')
	plt.gca().set_aspect('equal')

	pdf.savefig(figlrhrncd)
	plt.close()


	# Doing circle fitting with least squares

	# In[7]:


	from scipy import optimize
	def calc_R(xc,yc):
	    return np.sqrt((Ihr1-xc)**2+(Qhr1-yc)**2)

	def f_2(c):
	    Ri = calc_R(*c)
	    return Ri- Ri.mean()

	center_estimate = np.mean([max(Ihr1), min(Ihr1)]), np.mean([max(Qhr1), min(Qhr1)])
	center_2, ier = optimize.leastsq(f_2,center_estimate)

	xc_2, yc_2 = center_2
	Ri_2 = calc_R(*center_2)
	R_2 = Ri_2.mean()
	residu_2 = sum((Ri_2-R_2)**2)

	#Plotting fit and the data
	circle1 = plt.Circle((xc_2,yc_2), R_2)
	fig, ax=plt.subplots()
	ax.add_artist((circle1))
	ax.plot(Ihr1,Qhr1,'.', color='r', label='Data')
	ax.set_aspect('equal')
	ax.set_ylabel('Q')
	ax.set_xlabel('I')
	ax.set_title('Circle fit with HR data')
	pdf.savefig(fig)


	# Rotating and translating the circle via equation E.10 from Gao

	# In[8]:


	zc = xc_2 + 1.j*yc_2
	alpha = np.angle(zc)
	datahr2 = (zc-datahr1)*np.exp(-1.j*alpha)
	Ihr2 = datahr2.real
	Qhr2 = datahr2.imag

	#plotting
	circle2 = plt.Circle((0,0), R_2)
	fig2, ax2=plt.subplots()
	ax2.add_artist((circle2))
	ax2.plot(Ihr2,Qhr2,'.',color='r',label='data')
	ax2.set_aspect('equal')
	ax2.set_ylabel('Q')
	ax2.set_xlabel('I')
	ax2.set_title('Rotated and translated HR data, overlaid with a circle centred at 0,0')
	pdf.savefig(fig2)


	# Phase angle fit via eqn E.11 from Gao

	# In[9]:



	def phasefit(x, a, b, c):
	    y = -a + 2*np.arctan(2*b*(1-(x/c)))
	    return y
	phasehr2 = np.arctan2(Qhr2,Ihr2)
	popt, pcov = curve_fit(phasefit, freqhr, phasehr2, bounds=((-3, 1.e4,min(freqhr)),(3, 5.e6,max(freqhr))))
	theta0,Qr,fr = popt[0], popt[1], popt[2]

	#plotting
	print (theta0,Qr,fr)
	plt.clf()
	figphasefit1 = plt.figure()
	plt.plot(freqhr,phasehr2,'.', label='Data')
	plt.plot(freqhr, phasefit(freqhr, theta0, Qr, fr), label='fit')
	plt.legend()
	plt.xlabel('Freq')
	plt.ylabel('Phase (rad)')
	plt.title('Initial phase fitting to entire HR data')
	pdf.savefig(figphasefit1)
	#plt.show()
	plt.close()


	# Trying to fit to less data

	# In[10]:


	#popt, pcov = curve_fit(phasefit, freqhr[150:-150], phasehr2[150:-150], bounds=((-3, 1.e4,min(freqhr)),(3, 5.e6,max(freqhr))))
	#theta0,Qr,fr = popt[0], popt[1], popt[2]

	#print (theta0,Qr,fr)
	#plt.clf()
	#figphasefit2 = plt.figure()
	#plt.plot(freqhr,phasehr2,'.', label='Data')
	#plt.plot(freqhr, phasefit(freqhr, theta0, Qr, fr), label='fit')
	#plt.legend()
	#plt.xlabel('Freq')
	#plt.ylabel('Phase (rad)')
	#plt.title('Fitting to the middle 100 points')
	#pdf.savefig(figphasefit2)
	#plt.show()
	#plt.close()



	# Loading timestream data

	# In[11]:

	hdulist = fits.open(datadir + "TS_200000_Hz_OFF_RES_x020.fits")
	freqoff = hdulist[1].header['SYNTHFRE']
	data_off1 = hdulist[1].data
	Ioff1 = np.zeros((len(data_off1.field(0)),20))
	Qoff1 = np.zeros((len(data_off1.field(1)),20))
	fastmask = np.zeros((2, 20),dtype=bool)
	slowmask = np.zeros((2, 12),dtype=bool)
	for i in range(20):
	    Ioff1[:,i] = data_off1.field(i*2)
	    Qoff1[:,i] = data_off1.field(i*2+1)
	    Iof1indices = np.where(abs(Ioff1[:,i]-np.median(Ioff1[:,i])) > 3*np.std(Ioff1[:,i]))[0]
	    Ioff1[Iof1indices] = np.median(Ioff1[:,i])
	    Qof1indices = np.where(abs(Qoff1[:,i]-np.median(Qoff1[:,i])) > 3*np.std(Qoff1[:,i]))[0]
	    Qoff1[Qof1indices] = np.median(Qoff1[:,i])
	    if len(Iof1indices)/float(len(Ioff1[:,i])) > 0.01:
		fastmask[1, i] = True
	    if len(Qof1indices)/float(len(Qoff1[:,i])) > 0.01:
		fastmask[1, i] = True
	hdulist = fits.open(datadir + "TS_200000_Hz_ON_RES_x020.fits")
	data_on1 = hdulist[1].data
	freqon = hdulist[1].header['SYNTHFRE']
	Ion1 = np.zeros((len(data_on1.field(0)),20))
	Qon1 = np.zeros((len(data_on1.field(1)),20))
	for i in range(20):
	    Ion1[:,i] = data_on1.field(i*2)
	    Qon1[:,i] = data_on1.field(i*2+1)
	    Ion1indices = np.where(abs(Ion1[:,i]-np.median(Ion1[:,i])) > 3*np.std(Ion1[:,i]))[0]
	    Ion1[Ion1indices] = np.median(Ion1[:,i])
	    Qon1indices = np.where(abs(Qon1[:,i]-np.median(Qon1[:,i])) > 3*np.std(Qon1[:,i]))[0]
	    Qon1[Qon1indices] = np.median(Qon1[:,i])
	    if len(Ion1indices)/float(len(Ion1[:,i])) > 0.01:
		fastmask[0, i] = True
	    if len(Qon1indices)/float(len(Qon1[:,i])) > 0.01:
		fastmask[0, i] = True
	hdulist = fits.open(datadir + "TS_2000_Hz_OFF_RES_x012.fits")
	data_off2 = hdulist[1].data
	Ioff2 = np.zeros((len(data_off2.field(0)),12))
	Qoff2 = np.zeros((len(data_off2.field(1)),12))
	for i in range(12):
	    Ioff2[:,i] = data_off2.field(i*2)
	    Qoff2[:,i] = data_off2.field(i*2+1)
	    Iof2indices = np.where(abs(Ioff2[:,i]-np.median(Ioff2[:,i])) > 3*np.std(Ioff2[:,i]))[0]
	    Ioff2[Iof2indices] = np.median(Ioff2[:,i])
	    Qof2indices = np.where(abs(Qoff2[:,i]-np.median(Qoff2[:,i])) > 3*np.std(Qoff2[:,i]))[0]
	    Qoff2[Qof2indices] = np.median(Qoff2[:,i])
	    if len(Iof2indices)/float(len(Ioff2[:,i])) > 0.01:
		slowmask[1, i] = True
	    if len(Qof2indices)/float(len(Qoff2[:,i])) > 0.01:
		slowmask[1, i] = True
	hdulist = fits.open(datadir + "TS_2000_Hz_ON_RES_x012.fits")
	data_on2 = hdulist[1].data
	Ion2 = np.zeros((len(data_on2.field(0)),12))
	Qon2 = np.zeros((len(data_on2.field(1)),12))
	for i in range(12):
	    Ion2[:,i] = data_on2.field(i*2)
	    Qon2[:,i] = data_on2.field(i*2+1)
	    Ion2indices = np.where(abs(Ion2[:,i]-np.median(Ion2[:,i])) > 3*np.std(Ion2[:,i]))[0]
	    Ion2[Ion2indices] = np.median(Ion2[:,i])
	    Qon2indices = np.where(abs(Qon2[:,i]-np.median(Qon2[:,i])) > 3*np.std(Qon2[:,i]))[0]
	    Qon2[Qon2indices] = np.median(Qon2[:,i])
	    if len(Ion2indices)/float(len(Ion2[:,i])) > 0.01:
		slowmask[0, i] = True
	    if len(Qon2indices)/float(len(Qon2[:,i])) > 0.01:
		slowmask[0, i] = True
	Ion1f = Ion1.flatten()
	Qon1f = Qon1.flatten()
	Ion2f = Ion2.flatten()
	Qon2f = Qon2.flatten()
	Ioff1f = Ioff1.flatten()
	Qoff1f = Qoff1.flatten()
	Ioff2f = Ioff2.flatten()
	Qoff2f = Qoff2.flatten()

	figalliqplots = plt.figure()
	plt.plot(Ion2f,label='Onres, slow',rasterized=True)
	plt.plot(Qon2f,label='Onres, slow',rasterized=True)
	plt.plot(Ion1f,label='Onres, fast',rasterized=True)
	plt.plot(Qon1f,label='Onres, fast',rasterized=True)
	plt.plot(Ioff2f,label='Offres, slow',rasterized=True)
	plt.plot(Qoff2f,label='Offres, slow',rasterized=True)
	plt.plot(Ioff1f,label='Offres, fast',rasterized=True)
	plt.plot(Qoff1f,label='Offres, fast',rasterized=True)
	plt.xlabel('I')
	plt.ylabel('Q')
	#plt.legend(loc='best')
	plt.title('Plotting timestreams')
	#plt.show()
	pdf.savefig(figalliqplots)
	plt.close()

	plt.clf()
	figalltimestream = plt.figure()
	plt.plot(I,Q)
	plt.plot(Ion2f,Qon2f,label='Onres, slow',rasterized=True)
	plt.plot(Ion1f,Qon1f,label='Onres, fast',rasterized=True)
	plt.plot(Ioff2f,Qoff2f,label='Offres, slow',rasterized=True)
	plt.plot(Ioff1f,Qoff1f,label='Offres, fast',rasterized=True)
	plt.xlabel('I')
	plt.ylabel('Q')
	plt.legend(loc='best')
	plt.title('After cable delay is remove for all timestreams')
	plt.gca().set_aspect('equal')
	#plt.show()
	pdf.savefig(figalltimestream)
	plt.close()

	plt.clf()
	figamp = plt.figure()
	plt.plot(freq,amp)
	plt.plot(np.ones(len(Ion1f))*freqon, np.sqrt(Ion1f**2+Qon1f**2),rasterized=True)
	plt.plot(np.ones(len(Ioff1f))*freqoff, np.sqrt(Ioff1f**2+Qoff1f**2),rasterized=True)
	plt.xlabel('Freq (Hz)')
	plt.ylabel('S21 Amp')
	plt.title('Checking On/Off res timestreams in amplitude space')
	#plt.show()
	pdf.savefig(figamp)
	plt.close()


	# In[12]:


	#applying transformations to the timestreams and checking:
	tson1 = Ion1+1.j*Qon1
	tson2 = Ion2+1.j*Qon2
	tsoff1 = Ioff1+1.j*Qoff1
	tsoff2 = Ioff2+1.j*Qoff2
	# removing cable delay 
	tson1_1 = tson1*np.exp(1.j*2.*np.pi*freqon*tau)
	tson2_1 = tson2*np.exp(1.j*2.*np.pi*freqon*tau)
	tsoff1_1 = tsoff1*np.exp(1.j*2.*np.pi*freqoff*tau)
	tsoff2_1 = tsoff2*np.exp(1.j*2.*np.pi*freqoff*tau)
	#rotating and translating the circle
	tson1_2 = (zc-tson1_1)*np.exp(-1.j*alpha)
	tson2_2 = (zc-tson2_1)*np.exp(-1.j*alpha)
	tsoff1_2 = (zc-tsoff1_1)*np.exp(-1.j*alpha)
	tsoff2_2 = (zc-tsoff2_1)*np.exp(-1.j*alpha)

	#plotting
	plt.clf()
	figmodts = plt.figure(dpi=200)
	plt.plot(Ihr2,Qhr2)
	for i in range(20):
	    plt.plot(tson1_2[:,i].real,tson1_2[:,i].imag, color='r',rasterized=True)
	    plt.plot(tsoff1_2[:,i].real,tsoff1_2[:,i].imag, color='g',rasterized=True)
	plt.plot(tson1_2[:,i].real,tson1_2[:,i].imag, color='r',label='onres fast',rasterized=True)
	plt.plot(tsoff1_2[:,i].real,tsoff1_2[:,i].imag, color='g', label='onres fast',rasterized=True)
	for i in range(12):
	    plt.plot(tson2_2[:,i].real,tson2_2[:,i].imag, color = 'b',rasterized=True)
	    plt.plot(tsoff2_2[:,i].real,tsoff2_2[:,i].imag, color='k',rasterized=True)
	plt.plot(tson2_2[:,i].real,tson2_2[:,i].imag, color = 'b',label='onres fast',rasterized=True)
	plt.plot(tsoff2_2[:,i].real,tsoff2_2[:,i].imag, color='k',label='onres fast',rasterized=True)
	plt.gca().set_aspect('equal')
	plt.legend(loc='best')
	plt.ylabel('Q')
	plt.xlabel('I')
	plt.title('Checking on/off res timestream still make sense after transformations')
	#plt.show()
	pdf.savefig(figmodts)
	plt.close()


	# In[13]:




	def freqfit(x, a, b, c):
	    y = c*(1.-(1./(2*b))*np.tan((x+a)/2.))
	    return y



	#moving the IQ noiseball off resonance to where on resonance is
	avg_on1 = np.mean(tson1_2.real) +1.j*np.mean(tson1_2.imag) 
	avg_on2 = np.mean(tson2_2.real) +1.j*np.mean(tson2_2.imag)
	avg_off1 = np.mean(tsoff1_2.real)+1.j*np.mean(tsoff1_2.imag)
	avg_off2 = np.mean(tsoff2_2.real) + 1.j*np.mean(tsoff2_2.imag)
	tsoff1_3 = tsoff1_2 - avg_off1 + avg_on1
	tsoff2_3 = tsoff2_2 - avg_off2 + avg_on2


	phase_on1 = np.zeros((tson1_2.shape))
	phase_on2 = np.zeros((tson2_2.shape))
	phase_off1 = np.zeros((tsoff1_3.shape))
	phase_off2 = np.zeros((tsoff2_3.shape))
	freq_on1 = np.zeros((phase_on1.shape))
	freq_on2 = np.zeros((phase_on2.shape))
	freq_off1 = np.zeros((phase_off1.shape))
	freq_off2 = np.zeros((phase_off2.shape))


	for i in range(20):
	    phase_on1[:,i] = np.arctan2(tson1_2[:,i].imag,tson1_2[:,i].real)
	    phase_off1[:,i] = np.arctan2(tsoff1_3[:,i].imag,tsoff1_3[:,i].real)
	    freq_on1[:,i] = freqfit(phase_on1[:,i], theta0, Qr, fr)
	    freq_off1[:,i] = freqfit(phase_off1[:,i], theta0, Qr, fr)
	    
	for i in range(12):
	    phase_on2[:,i] = np.arctan2(tson2_2[:,i].imag,tson2_2[:,i].real)
	    phase_off2[:,i] = np.arctan2(tsoff2_3[:,i].imag,tsoff2_3[:,i].real)
	    freq_on2[:,i] = freqfit(phase_on2[:,i], theta0, Qr, fr)
	    freq_off2[:,i] = freqfit(phase_off2[:,i], theta0, Qr, fr)

	plt.clf()
	figphasecheck = plt.figure()
	plt.plot(freqhr,phasehr2,'.', label='Data')
	plt.plot(freqhr, phasefit(freqhr, theta0, Qr, fr), label='fit')
	for i in range(20):
	    plt.plot(freq_on1[:,i], phase_on1[:,i],rasterized=True)
	    plt.plot(freq_off1[:,i], phase_off1[:,i],rasterized=True)
	for i in range(12):
	    plt.plot(freq_on2[:,i], phase_on2[:,i],rasterized=True)
	    plt.plot(freq_off2[:,i], phase_off2[:,i],rasterized=True)
	plt.xlabel('Freq')
	plt.ylabel('Phase')
	plt.title('Checking the phase fitting for on and off resonance (offres shifted to onres data)')
	pdf.savefig(figphasecheck)
	#plt.show()
	pdf.close()
	plt.close()




	sr1 = 200000.
	sr2 = 2000.
	psd_on1 = np.zeros((freq_on1.shape))
	psd_on2 = np.zeros((freq_on2.shape))
	psd_off1 = np.zeros((freq_off1.shape))
	psd_off2 = np.zeros((freq_off2.shape))

	psd_onfreq1 = np.fft.fftfreq(len(freq_on1[:,0]), d=1./sr1)
	psd_onfreq2 = np.fft.fftfreq(len(freq_on2[:,0]), d=1./sr2)
	psd_offfreq1 = np.fft.fftfreq(len(freq_off1[:,0]), d=1./sr1)
	psd_offfreq2 = np.fft.fftfreq(len(freq_off2[:,0]), d=1./sr2)

	plt.clf()
	figpsdone = plt.figure()
	figpsd1, axarr1 = plt.subplots(5, 4,sharex=True, sharey=True)
	for i in range(20):
	    psd_on1[:,i] = abs(np.fft.fft(freq_on1[:,i]))**2/(len(freq_on1[:,i])*sr1)/freqon**2
	    psd_off1[:,i] = abs(np.fft.fft(freq_off1[:,i]))**2/(len(freq_off1[:,i])*sr1)/freqoff**2
	    axarr1[int(i/4), i % 4].plot(psd_onfreq1, psd_on1[:,i], color='r')
	    axarr1[int(i/4), i % 4].plot(psd_offfreq1, psd_off1[:,i], color = 'g')
	axarr1[int(i/4), i % 4].plot(psd_onfreq1, psd_on1[:,i], color='r', label='Onres')
	axarr1[int(i/4), i % 4].plot(psd_offfreq1, psd_off1[:,i], color = 'g', label='Offres')

	plt.semilogy()
	plt.semilogx()
	axarr1[0,0].set_ylabel('Sxx (Hz$^{-1}$)')
	axarr1[4,3].set_xlabel('Freq (Hz)')
	plt.legend(loc=9, bbox_to_anchor=(2, 0.5))
	axarr1[0,0].set_title('Noise PSDs of on vs off for all timestreams (fast)')
	#plt.show()
	plt.savefig(savedir+'PSDfast.pdf')
	plt.close()

	plt.clf()
	figpsdtwo = plt.figure(dpi=200)
	figpsd2, axarr2 = plt.subplots(4, 3,sharex=True, sharey=True)
	for i in range(12):
	    psd_on2[:,i] = abs(np.fft.fft(freq_on2[:,i]))**2/(len(freq_on2[:,i])*sr2)/freqon**2
	    psd_off2[:,i] = abs(np.fft.fft(freq_off2[:,i]))**2/(len(freq_off2[:,i])*sr2)/freqoff**2
	    axarr2[int(i/3), i % 3].plot(psd_onfreq2, psd_on2[:,i], color='r')
	    axarr2[int(i/3), i % 3].plot(psd_offfreq2, psd_off2[:,i], color='g')
	axarr2[int(i/3), i % 3].plot(psd_onfreq2, psd_on2[:,i], color='r',label='Onres')
	axarr2[int(i/3), i % 3].plot(psd_offfreq2, psd_off2[:,i], color='g',label='Offres')
	plt.semilogy()
	plt.semilogx()
	axarr2[0,0].set_ylabel('Sxx (Hz$^{-1}$)')
	axarr2[3,2].set_xlabel('Freq (Hz)')
	plt.legend(loc=9, bbox_to_anchor=(2, 0.5))
	axarr2[0,0].set_title('Noise PSDs of on vs off for all timestreams (slow)')
	#plt.show()
	plt.savefig(savedir+'PSDslow.pdf')

	plt.close()


	# Applying transformations to the timestreams and checking

	# In[15]:
	fastmask0 = np.where(fastmask[0,:]==True)[0]
	if np.any(fastmask0) == True:
	    psd_on1 = np.delete(psd_on1, fastmask0,1)
	    print ("masking out part timestream " + str(fastmask0))
	fastmask1 = np.where(fastmask[1,:]==True)[0]	
	if np.any(fastmask1) == True:
	    psd_off1 = np.delete(psd_off1, fastmask1, 1)
	    print ("masking out part timestream " + str(fastmask1))
	slowmask0 = np.where(slowmask[0,:]==True)[0]
	if np.any(slowmask0) == True:
	    psd_on2 = np.delete(psd_on2, slowmask0, 1)
	    print ("masking out part timestream " + str(slowmask0))
	slowmask1 = np.where(slowmask[1,:]==True)[0]	
	if np.any(slowmask1) == True:
	    psd_off2 = np.delete(psd_off2, slowmask1, 1)
	    print ("masking out part timestream " + str(slowmask0))

	psd_on1mean = psd_on1.mean(axis=1)
	psd_off1mean = psd_off1.mean(axis=1)
	psd_fast = np.vstack((psd_onfreq1, psd_on1mean, psd_off1mean))

	psd_on2mean = psd_on2.mean(axis=1)
	psd_off2mean = psd_off2.mean(axis=1)
	psd_slow = np.vstack((psd_onfreq2, psd_on2mean, psd_off2mean))
	plt.plot(psd_fast[0,:],psd_fast[1,:])
	plt.plot(psd_fast[0,:],psd_fast[2,:])
	plt.plot(psd_slow[0,:], psd_slow[1,:])
	plt.plot(psd_slow[0,:], psd_slow[2,:])
	plt.semilogy()
	plt.semilogx()
	plt.xlabel('Freq (Hz)')
	plt.ylabel('Sxx (Hz$^{-1}$)')
	plt.title('Noise PSD, averaged over all timestreams')
	plt.savefig(savedir+'PSDaveraged.pdf')

	np.savetxt(savedir+'PSDfast.txt', psd_fast)
	np.savetxt(savedir+'PSDslow.txt', psd_slow)

	#Log binning data
	if 3 < 1:
		psd_fast_mask = np.where(psd_fast[0,:] > 1.e3)[0]
		psd_slow_mask = np.where(psd_slow[0,:] > 0)[0]
		freqson = np.hstack((psd_fast[0,psd_fast_mask],psd_slow[0,psd_slow_mask]))
		dataon = np.hstack((psd_fast[1,psd_fast_mask],psd_slow[1,psd_slow_mask]))

		bins = np.logspace(np.log10(1.e0),np.log10(1.e5),num=50)
		lowfreqsindex = np.where(freqson <=2.0)[0]
		newfreqson = freqson[lowfreqsindex]
		newdataon = dataon[lowfreqsindex]
		print (newfreqson)
		for i in range(len(bins)-1):
		    freqsinbin = np.where((freqson >= bins[i]) & (freqson < bins[i+1]))[0]
		    if (len(freqsinbin) < 30) and (len(freqsinbin) > 0):
			newfreqson = np.append(newfreqson,freqson[freqsinbin])
			newdataon = np.append(newdataon,dataon[freqsinbin])        
		    elif ((len(freqsinbin) >= 30) and (len(freqsinbin) < 100)):
			if np.all(np.diff(freqson[freqsinbin])>0):
			    modfreqlist = np.log10(freqson[freqsinbin[:len(freqsinbin)/5*5]])
			    mod2freqlist = np.mean(modfreqlist.reshape(-1, 5), axis=1)
			    moddata = np.log10(dataon[freqsinbin[:len(freqsinbin)/5*5]])
			    mod2data = np.mean(moddata.reshape(-1, 5), axis=1)
			    newfreqson = np.append(newfreqson,10**(mod2freqlist))
			    newdataon = np.append(newdataon,10**(mod2data))
			else:
			    newfreqson = np.append(newfreqson,freqson[freqsinbin])
			    newdataon = np.append(newdataon,dataon[freqsinbin])
		    elif ((len(freqsinbin) >= 100) and (len(freqsinbin) < 200)):
			if np.all(np.diff(freqson[freqsinbin])>0):
			    modfreqlist = np.log10(freqson[freqsinbin[:len(freqsinbin)/15*15]])
			    mod2freqlist = np.mean(modfreqlist.reshape(-1, 15), axis=1)
			    moddata = np.log10(dataon[freqsinbin[:len(freqsinbin)/15*15]])
			    mod2data = np.mean(moddata.reshape(-1, 15), axis=1)
			    newfreqson = np.append(newfreqson,10**(mod2freqlist))
			    newdataon = np.append(newdataon,10**(mod2data))
		    elif (len(freqsinbin) >= 200):
			if np.all(np.diff(freqson[freqsinbin])>0):
			    modfreqlist = np.log10(freqson[freqsinbin[:len(freqsinbin)/30*30]])
			    mod2freqlist = np.mean(modfreqlist.reshape(-1, 30), axis=1)
			    moddata = np.log10(dataon[freqsinbin[:len(freqsinbin)/30*30]])
			    mod2data = np.mean(moddata.reshape(-1, 30), axis=1)
			    newfreqson = np.append(newfreqson,10**(mod2freqlist))
			    newdataon = np.append(newdataon,10**(mod2data))

		def func_log(x,A,B,n,C,tau, tau_f):
		    return np.log10(((A+B*(10**x)**(-n))/(1+(2*np.pi*10**x*tau)**2) + C )*(1/(1+(2*np.pi*10**x*tau_f)**2)))
		A,B,n,C, tau, tau_f = curve_fit(func_log, np.log10(newfreqson), np.log10(newdataon), bounds=([1.e-20,1.e-21,0.1,1.e-23,1.e-5,1.e-7], [1.e-17, 5.e-17,2.5, 1.e-20,5.e-3,5.e-5]))[0]

		print (A,B,n,C,tau,tau_f)

		plt.plot(freqson, dataon)
		plt.plot(newfreqson, newdataon)
		plt.plot(freqson, 10**func_log(np.log10(freqson),A,B,n,C, tau, tau_f))
		plt.semilogy()
		plt.semilogx()
		plt.xlabel('Freq (Hz)')
		plt.ylabel('Sxx (Hz$^{-1}$)')
		plt.title('Fit to Noise PSD')
		plt.savefig(savedir+'PSDaveraged_with_fit.pdf')
		fitvals = np.asarray([A,B,C,n,C,tau,tau_f])
		np.savetxt(savedir+'NoiseFit_params.txt', fitvals)

