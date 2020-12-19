"""
Created on Tue Apr  4 12:54:04 2017
http://stackoverflow.com/questions/21052052/bandwidth-of-an-eeg-signal
@author: Amr
"""

#from numpy.fft import fft
#from numpy.fft.fftpack import fft, ifft
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
import numpy as np

N = 64
T = 1/64.0
x = np.linspace(0.0, 2*np.pi*N*T ,N)

y1 = np.cos(20*x)
y2 = np.sin(10*x)
y3 = 0
y = y1 + y2 + y3

fy = fft(y)

plt.figure(1)
plt.plot(np.linspace(0.0, 1.0/(2.0*T) , N/2), (2.0/N)*np.abs(fy[0:N/2]))

plt.figure(2)
y4 = ifft(fy)

plt.plot(x,y4, 'r')
plt.plot(x, y, 'b')

>>> from scipy.fftpack import fft
>>> # Number of sample points
>>> N = 600
>>> # sample spacing
>>> T = 1.0 / 800.0
>>> x = np.linspace(0.0, N*T, N)
>>> y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
>>> yf = fft(y)
>>> import matplotlib.pyplot as plt
>>> plt.plot( np.linspace(0.0, 1.0/(2.0*T), N//2) , 2.0/N * np.abs(yf[0:N//2]) )
>>> plt.grid()
>>> plt.show()