##################Signal Characteristics###########################
import pandas as pd
Signals = pd.read_csv('Signals/071309_w_21-BL1-081_bio.csv',sep = 	'	').iloc[:,:].values
Signals_Filtered = pd.read_csv('Signals/071309_w_21-BL1-081_bio.csv',sep = 	'	').iloc[:,:].values


N = len(Signals[:,0])            # Number of samplepoints
Fs = len(Signals[:,0])//(Signals[:,0][-1]/1000000)          # Sample Frequency
Ts = 1.0 / Fs       # Sample Period
Nyq = 0.5 * Fs  # Nyquist Frequency
import numpy as np
signal = Signals[:,4]
######################Signal Spectrum##############################  
import scipy.fftpack
signal_freq = scipy.fftpack.fft(signal)
######################Filter Bandwidth#############################
from scipy.signal import butter, lfilter, freqz
LowCutoff = 20           # desired low cutoff frequency of the filter, Hz
HighCutoff = 250          # desired high cutoff frequency of the filter, Hz
b, a = butter(N = 4, Wn = [LowCutoff/Nyq, HighCutoff/Nyq], btype='band', analog=False)
########################Apply Filter###############################
signal_filtered = lfilter(b, a, signal)
##################Filtered Signal Spectrum#########################
import scipy.fftpack
signal_filtered_freq = scipy.fftpack.fft(signal_filtered)



###################################################################
###################################################################
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker 
plt.subplot(2, 2, 1)
plt.title("Bandpass Filter Frequency Response")
plt.xlabel('Frequency [Hz]')
plt.ylabel('Gain')
plt.legend(loc='best')
plt.grid()
for order in [4]:
    b, a = butter(N = order, Wn = [LowCutoff/Nyq, HighCutoff/Nyq], btype='band', analog=False)
    w, h = freqz(b, a, worN=8000)
    plt.plot(Nyq*w/np.pi, np.abs(h), label="order = %d" % order)
plt.plot(LowCutoff, 0.5*np.sqrt(2), 'ko')   # Intersection Point
plt.plot(HighCutoff, 0.5*np.sqrt(2), 'ko')  # Intersection Point
plt.axvline(LowCutoff, color='k')           # Ideal low cutoff frequency
plt.axvline(HighCutoff, color='k')          # Ideal high cutoff frequency


plt.subplot(2, 2, 2)
plt.subplots_adjust(hspace=0.35)
plt.title('Signals Visualisation')
plt.xlabel('Time [sec]')
plt.ylabel('Magnitude')
plt.grid()
plt.legend()
plt.plot(Signals[:,0][:], signal[:], 'b-', label='data')
plt.plot(Signals[:,0][:], signal_filtered[:], 'g-', linewidth=2, label='filtered data')


plt.subplot(2, 2, 3)
plt.title('Signal Spectrum')
plt.xlabel('Frequency')
plt.ylabel('Magnitude')
plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%d Hz'))
plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
Freq = np.linspace(0.0, 1.0/(2.0*Ts), N/2)
Amp = 2/N * np.abs(signal_freq[:N//2])
plt.plot(Freq,Amp)
plt.show()

plt.subplot(2, 2, 4)
plt.title('Filtered Signal Spectrum')
plt.xlabel('Frequency')
plt.ylabel('Magnitude')
plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%d Hz'))
plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
Freq = np.linspace(0.0, 1.0/(2.0*Ts), N/2)
Amp = 2/N * np.abs(signal_filtered_freq[:N//2])
plt.plot(Freq,Amp)
plt.show()
###################################################################
###################################################################